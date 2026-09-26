import math
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from socceraction.spadl import config as spadl
from torch_geometric.nn import SAGEConv

from fbrecruit import shrinkage as sh
from fbrecruit import style
from fbrecruit.logs import Tee, key, show
from fbrecruit.paths import PROCESSED
from fbrecruit.sources.statsbomb import LEAGUES

KEYS = style.KEYS
PAIR = ["league", "team_id", "player_a", "player_b"]
PASSES = ["pass", "cross"]
SUCCESS = spadl.results.index("success")
TOGETHER_MS = 5000
PARTNERS = 5
MIN_COUNT = 3
KINDS = {"left_only": "pass", "right_only": "press", "both": "both"}
HIDDEN = 32
DIM = 16
LR = 0.01
STEPS = 200
SCORED = 0.3
REPORT_AT = [1, 50, 100, 150, 200]
SEED = 0
CHECK_SEEDS = [1, 2, 3, 4]
NEIGHBOURS = 10
RESAMPLES = 2000
E = [f"e{i}" for i in range(DIM)]
# the hidden-link areas of the first run, trained with a scored share of 0
FIRST_AREAS = {"with links": 0.6762783544790465, "without links": 0.7581252043482103}
# the float64 sum of the first run's pooled embeddings
FIRST_SUM = -336.4133332394995


def links_path():
    return PROCESSED / "graph_links_w1.parquet"


def embeddings_path():
    return PROCESSED / "embeddings_w1.parquet"


def in_window1(frame, games):
    ids = pd.MultiIndex.from_frame(games[["league", "game_id"]])
    return pd.MultiIndex.from_frame(frame[["league", "game_id"]]).isin(ids)


def receivers(a):
    """The player of the next action if it is in the same game and period, by the same team and
    another player; a must be in game, period, action order."""
    nxt = a[["game_id", "period_id", "team_id", "player_id"]].shift(-1)
    ok = (
        (nxt.game_id == a.game_id)
        & (nxt.period_id == a.period_id)
        & (nxt.team_id == a.team_id)
        & (nxt.player_id != a.player_id)
    )
    return nxt.player_id.where(ok).astype("Int64")


def unordered(frame, x, y):
    lo, hi = np.minimum(frame[x], frame[y]), np.maximum(frame[x], frame[y])
    return pd.DataFrame(
        {
            "league": frame.league.to_numpy(),
            "team_id": frame.team_id.to_numpy(),
            "player_a": lo.to_numpy(),
            "player_b": hi.to_numpy(),
        }
    )


def completed_passes(a):
    done = (a.type_name.isin(PASSES) & (a.result_id == SUCCESS)).to_numpy()
    return a[done].assign(receiver=receivers(a)[done])


def pass_counts(passes):
    r = passes[passes.receiver.notna()]
    pairs = unordered(r.assign(receiver=r.receiver.astype("int64")), "player_id", "receiver")
    return pairs.groupby(PAIR).size().rename("n_pass").reset_index()


def press_counts(press):
    """Pairs of pressure events by two players of one team in one game and period, at most 5 s
    apart, per unordered pair of players."""
    cols = ["league", "game_id", "period_id", "team_id"]
    # timestamps are whole milliseconds, so the 5-second rule is compared exactly
    p = press.assign(ms=np.round(press.seconds.to_numpy() * 1000).astype("int64"))
    p = p.sort_values([*cols, "ms"], kind="stable").reset_index(drop=True)
    group = p.groupby(cols, sort=False).ngroup().to_numpy()
    ms, player = p.ms.to_numpy(), p.player_id.to_numpy()
    left, right = [np.zeros(0, "int64")], [np.zeros(0, "int64")]
    k = 1
    while k < len(p):
        near = (group[k:] == group[:-k]) & (ms[k:] - ms[:-k] <= TOGETHER_MS)
        if not near.any():
            break
        i = np.flatnonzero(near & (player[k:] != player[:-k]))
        left.append(i)
        right.append(i + k)
        k += 1
    i, j = np.concatenate(left), np.concatenate(right)
    pairs = unordered(p.iloc[i].assign(other=player[j]), "player_id", "other")
    return pairs.groupby(PAIR).size().rename("n_press").reset_index()


def select(counts, n):
    """Each node keeps up to five partners with a count of at least 3, highest count first and
    ties to the lower player id; a pair is linked if either end keeps the other."""
    c = counts[counts[n] >= MIN_COUNT]
    ends = pd.concat(
        [
            c.assign(node=c.player_a, partner=c.player_b),
            c.assign(node=c.player_b, partner=c.player_a),
        ]
    )
    ends = ends.sort_values(
        ["league", "team_id", "node", n, "partner"], ascending=[True, True, True, False, True]
    )
    kept = ends[ends.groupby(["league", "team_id", "node"]).cumcount() < PARTNERS]
    return kept[PAIR].drop_duplicates().sort_values(PAIR).reset_index(drop=True)


def merge_kinds(passes, presses):
    m = select(passes, "n_pass").merge(
        select(presses, "n_press"), on=PAIR, how="outer", indicator=True
    )
    m["kind"] = m._merge.astype(str).map(KINDS)
    m = m[[*PAIR, "kind"]].merge(passes, on=PAIR, how="left").merge(presses, on=PAIR, how="left")
    m[["n_pass", "n_press"]] = m[["n_pass", "n_press"]].fillna(0).astype("int64")
    return m.sort_values(PAIR).reset_index(drop=True)


def on_nodes(counts, nodes):
    idx = pd.MultiIndex.from_frame(nodes[KEYS])
    ok = np.ones(len(counts), bool)
    for end in ("player_a", "player_b"):
        ends = pd.MultiIndex.from_arrays([counts.league, counts[end], counts.team_id], names=KEYS)
        ok &= ends.isin(idx)
    return ok


def link_table(m):
    out = {}
    for s, end in (("a", "player_a"), ("b", "player_b")):
        out |= {f"league_{s}": m.league, f"player_id_{s}": m[end], f"team_id_{s}": m.team_id}
    return pd.DataFrame(out).assign(n_pass=m.n_pass, n_press=m.n_press, kind=m.kind)


def end_keys(links, s):
    return links[[f"{c}_{s}" for c in KEYS]].set_axis(KEYS, axis=1)


def build_links(nodes):
    games, a, press, _ = style.load_inputs()
    press = press[in_window1(press, games)]
    outside = int((~in_window1(a, games)).sum()) + int((~in_window1(press, games)).sum())
    key(f"2: window-1 games {len(games)}; actions {len(a)}, pressure events {len(press)}")
    key(f"2: actions or pressure events from a game outside window 1: {outside}")
    assert outside == 0, "HARD STOP: an action or pressure event is not from a window-1 game"

    a = a.sort_values(["game_id", "period_id", "action_id"], kind="stable").reset_index(drop=True)
    passes = completed_passes(a)
    counts = {"pass": pass_counts(passes), "press": press_counts(press)}
    kept, dropped = {}, {}
    for kind, c in counts.items():
        ok = on_nodes(c, nodes)
        kept[kind], dropped[kind] = c[ok], c[~ok]
    links = link_table(merge_kinds(kept["pass"], kept["press"]))
    a_end, b_end = end_keys(links, "a"), end_keys(links, "b")
    across = int(((a_end.league != b_end.league) | (a_end.team_id != b_end.team_id)).sum())
    itself = int((a_end.player_id == b_end.player_id).sum())
    key(f"2: links between teams or leagues {across}, links from a node to itself {itself}")
    assert across == 0 and itself == 0, "HARD STOP: a link crosses teams or joins a node to itself"
    return passes, kept, dropped, links


def degrees(nodes, links):
    ends = pd.concat([end_keys(links, "a"), end_keys(links, "b")])
    per = ends.value_counts().rename("links").reset_index()
    return nodes[KEYS].merge(per, on=KEYS, how="left").links.fillna(0).to_numpy()


def part(frame, lg, col="league"):
    return frame if lg == "total" else frame[frame[col] == lg]


def link_stats(nodes, passes, kept, dropped, links):
    deg = degrees(nodes, links)
    rows = {}
    for lg in [*LEAGUES, "total"]:
        k = part(links, lg, "league_a").kind
        d = deg if lg == "total" else deg[(nodes.league == lg).to_numpy()]
        rows[lg] = {
            "nodes": len(part(nodes, lg)),
            "passes": len(part(passes, lg)),
            "with_receiver": part(passes, lg).receiver.notna().mean(),
            "pairs_pass": len(part(kept["pass"], lg)),
            "pairs_press": len(part(kept["press"], lg)),
            "pass_only": int((k == "pass").sum()),
            "press_only": int((k == "press").sum()),
            "both": int((k == "both").sum()),
            "links": len(k),
            "mean_links": d.mean(),
            "no_link": int((d == 0).sum()),
            "dropped_pass": len(part(dropped["pass"], lg)),
            "dropped_press": len(part(dropped["press"], lg)),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def links():
    nodes = pd.read_parquet(style.player_path())
    z = [c for c in nodes.columns if c.startswith("z_")]
    key(f"2: nodes {len(nodes)}, keyed by {KEYS} as stored:")
    key(nodes[KEYS].dtypes.to_string())
    key(f"2: z_ columns {len(z)}: {z}")
    key(f"2: duplicate node keys {int(nodes[KEYS].duplicated().sum())}")
    assert len(nodes) == 2073 and len(z) == 8, "HARD STOP: node table is not 2,073 rows, 8 z_"
    assert not nodes[KEYS].duplicated().any()
    names = set(spadl.actiontypes)
    key(f"2: pass types {PASSES} in the spadl config: {all(t in names for t in PASSES)}")
    key(f"2: result id of success {SUCCESS}")

    passes, kept, dropped, table = build_links(nodes)
    table.to_parquet(links_path(), index=False)
    stats = link_stats(nodes, passes, kept, dropped, table)
    key("\n2: per league")
    key(show(stats, 6))
    for lg, r in stats.iterrows():
        key(
            f"2 {lg}: share of completed passes with a receiver {r.with_receiver!r}, "
            f"mean links per node {r.mean_links!r}"
        )
    key(f"\n2: wrote {links_path()} with {len(table)} rows, columns {list(table.columns)}")


def features(nodes):
    """The z_ scores in file order with missing ones at 0, then a one-hot of the group."""
    z = [c for c in nodes.columns if c.startswith("z_")]
    # the group table stores a missing group as UNKNOWN
    groups = sorted(set(nodes.group) - {"UNKNOWN"}) + ["UNKNOWN"]
    onehot = {f"group_{g}": (nodes.group == g).astype(float) for g in groups}
    x = nodes[z].fillna(0.0).assign(**onehot)
    return list(x.columns), np.ascontiguousarray(x.to_numpy(np.float32))


def node_pairs(nodes, links):
    """Row positions of both ends of every link, lower position first."""
    pos = pd.Series(np.arange(len(nodes)), index=pd.MultiIndex.from_frame(nodes[KEYS]))
    ends = [pos.loc[pd.MultiIndex.from_frame(end_keys(links, s))].to_numpy() for s in "ab"]
    return np.sort(np.column_stack(ends), axis=1)


def team_pairs(nodes):
    """Every pair of different nodes on one team, as row positions, lower first."""
    out = [np.zeros((0, 2), "int64")]
    for idx in nodes.groupby(["league", "team_id"]).indices.values():
        i, j = np.triu_indices(len(idx), 1)
        out.append(np.column_stack([idx[i], idx[j]]))
    pairs = np.concatenate(out)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def unlinked_pairs(nodes, linked):
    pairs = team_pairs(nodes)
    n = len(nodes)
    return pairs[~np.isin(pairs[:, 0] * n + pairs[:, 1], linked[:, 0] * n + linked[:, 1])]


def negatives(pool, n, rng):
    return pool[rng.integers(len(pool), size=n)]


def both_ways(pairs):
    return torch.from_numpy(np.ascontiguousarray(np.concatenate([pairs, pairs[:, ::-1]]).T))


class Sage(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.conv1 = SAGEConv(n_features, HIDDEN)
        self.conv2 = SAGEConv(HIDDEN, DIM)

    def forward(self, x, edge_index):
        return self.conv2(self.conv1(x, edge_index).relu(), edge_index)


def split(links, share, rng):
    """The links scored at one step and the links messages pass along: with a share above 0, a
    random share of the links, rounded down, and the rest; with a share of 0, all links for both."""
    if share == 0:
        return links, links
    order = rng.permutation(len(links))
    n = math.floor(share * len(links))
    return links[order[:n]], links[order[n:]]


def train(x, links, pool, seed, share, messages=True):
    """Full-batch Adam telling the scored links from as many pairs drawn from the pool, passing
    messages along the other links, or along none."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = Sage(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    x = torch.from_numpy(x)
    losses = []
    for _ in range(STEPS):
        scored, passed = split(links, share, rng)
        passed = passed if messages else passed[:0]
        pairs = torch.from_numpy(np.concatenate([scored, negatives(pool, len(scored), rng)]))
        labels = torch.cat([torch.ones(len(scored)), torch.zeros(len(scored))])
        z = model(x, both_ways(passed))
        loss = F.binary_cross_entropy_with_logits((z[pairs[:, 0]] * z[pairs[:, 1]]).sum(1), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.isfinite(losses).all(), "HARD STOP: a loss is not finite"
    return model, {"losses": losses, "positives": len(scored), "messages": len(passed)}


def embed(model, x, messages):
    with torch.no_grad():
        z = model(torch.from_numpy(x), both_ways(messages)).numpy()
    assert np.isfinite(z).all(), "HARD STOP: an embedding value is not finite"
    return z


def run(nodes, x, linked, seed, share, keep=None):
    """Train on the kept rows and the links between them, then embed every row over every link."""
    keep = np.ones(len(nodes), bool) if keep is None else keep
    rows = np.flatnonzero(keep)
    new = np.full(len(nodes), -1)
    new[rows] = np.arange(len(rows))
    sub = new[linked[keep[linked[:, 0]] & keep[linked[:, 1]]]]
    pool = unlinked_pairs(nodes.iloc[rows], sub)
    start = time.perf_counter()
    model, stats = train(x[rows], sub, pool, seed, share)
    z = embed(model, x, linked)
    info = {"rows": rows, "links": len(sub)} | stats
    return z, info | {"seconds": time.perf_counter() - start}


def graph_inputs():
    nodes = pd.read_parquet(style.player_path())
    names, x = features(nodes)
    return nodes, names, x, node_pairs(nodes, pd.read_parquet(links_path()))


def report(name, info):
    loss = ", ".join(f"step {s} {info['losses'][s - 1]!r}" for s in REPORT_AT)
    key(
        f"{name}: trained on nodes {len(info['rows'])}, links {info['links']}; per step positives "
        f"{info['positives']}, message links {info['messages']}; "
        f"loss {loss}; seconds {info['seconds']:.2f}"
    )


def embedding_table(share, seed=SEED):
    nodes, names, x, linked = graph_inputs()
    key(f"3: feature columns {len(names)}: {names}")
    key(f"3: nodes {len(nodes)}, links {len(linked)}")
    z, info = run(nodes, x, linked, seed, share)
    report("3 pooled", info)
    # every lolo network embeds all rows, stored under the league it held out
    out = [("pooled", None, z)]
    for lg in LEAGUES:
        held = (nodes.league == lg).to_numpy()
        z, info = run(nodes, x, linked, seed, share, ~held)
        leaked = int(held[info["rows"]].sum())
        key(f"3 lolo {lg}: held-out rows in the training set {leaked}")
        assert leaked == 0, f"HARD STOP: the {lg} training set holds a {lg} row"
        report(f"3 lolo {lg}", info)
        out.append(("lolo", lg, z))
    return pd.concat(
        [
            pd.concat(
                [
                    nodes[KEYS].assign(branch=b, fold=f),
                    pd.DataFrame(z.astype(np.float32), columns=E),
                ],
                axis=1,
            )
            for b, f, z in out
        ],
        ignore_index=True,
    )


def embeddings(share):
    table = embedding_table(share)
    table.to_parquet(embeddings_path(), index=False)
    key(f"\n3: wrote {embeddings_path()} with {len(table)} rows")
    key(table.dtypes.to_string())
    norms = table.assign(norm=np.linalg.norm(table[E].to_numpy(np.float64), axis=1))
    key("\n3: embedding norms per branch (std with ddof 1)")
    for b, s in norms.groupby("branch").norm:
        key(f"3 {b}: rows {len(s)}, mean norm {s.mean()!r}, std {s.std()!r}")
    key("\n3: lolo embedding norms per fold (std with ddof 1)")
    for f, s in norms[norms.branch == "lolo"].groupby("fold").norm:
        key(f"3 lolo {f}: rows {len(s)}, mean norm {s.mean()!r}, std {s.std()!r}")


def saved_pooled(nodes):
    saved = pd.read_parquet(embeddings_path())
    saved = saved[saved.branch == "pooled"].reset_index(drop=True)
    assert saved[KEYS].equals(nodes[KEYS])
    return saved[E].to_numpy()


def rerun(share):
    nodes, _, x, linked = graph_inputs()
    saved = saved_pooled(nodes)
    fresh, info = run(nodes, x, linked, SEED, share)
    report("4a pooled", info)
    same = np.array_equal(saved, fresh)
    key(f"4a: dtypes saved {saved.dtype}, recomputed {fresh.dtype}, shape {fresh.shape}")
    key(f"4a: recomputed pooled seed-0 embeddings identical to the saved ones: {same}")
    key(f"4a: largest absolute difference {np.abs(saved - fresh).max()!r}")
    assert same, "HARD STOP: the rerun does not reproduce the saved pooled embeddings"


def pair_scores(z, pairs):
    """Dot products of the pairs' embeddings, in float64."""
    z = z.astype(np.float64)
    return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(1)


def auc(scores):
    """Area under the ROC curve of pair scores, the first half of them for linked pairs."""
    n = len(scores) // 2
    return roc_auc_score(np.r_[np.ones(n), np.zeros(n)], scores)


def area(z, test):
    """Area under the ROC curve of the pairs' dot products, the first half of the pairs linked."""
    return auc(pair_scores(z, test))


def hidden_scores(nodes, x, linked, share):
    """Scores of a tenth of the links and as many unlinked same-team pairs, all kept out of
    training, from networks passing messages along the remaining links and along none. Also
    returns the network trained with links, the hidden links, the training links and the pairs."""
    rng = np.random.default_rng(SEED)
    n = len(linked) // 10
    pool = unlinked_pairs(nodes, linked)
    hidden = rng.choice(len(linked), size=n, replace=False)
    hidden_neg = rng.choice(len(pool), size=n, replace=False)
    rest = np.delete(linked, hidden, axis=0)
    train_pool = np.delete(pool, hidden_neg, axis=0)
    key(f"4b: links {len(linked)}, unlinked same-team pairs {len(pool)}")
    key(f"4b: training links {len(rest)}, training negative pool {len(train_pool)}")
    test = np.concatenate([linked[hidden], pool[hidden_neg]])
    models, scores = {}, {}
    for name, messages in (("with links", True), ("without links", False)):
        start = time.perf_counter()
        models[name], stats = train(x, rest, train_pool, SEED, share, messages)
        scores[name] = pair_scores(embed(models[name], x, rest if messages else rest[:0]), test)
        key(
            f"4b {name}: per step message links {stats['messages']}, positives "
            f"{stats['positives']}, loss at step {STEPS} {stats['losses'][-1]!r}, "
            f"seconds {time.perf_counter() - start:.2f}"
        )
    return scores, models["with links"], linked[hidden], rest, test


def hidden_runs(nodes, x, linked, share):
    """The areas under the ROC curve of hidden_scores' two networks, then the rest it returns."""
    scores, model, hidden, rest, test = hidden_scores(nodes, x, linked, share)
    return {name: auc(s) for name, s in scores.items()}, model, hidden, rest, test


def hidden_links(nodes, x, linked, share):
    aucs, _, hidden, _, _ = hidden_runs(nodes, x, linked, share)
    return len(hidden), aucs


def visible_area(model, x, hidden, rest, test):
    """The area when the network passes messages along the training links plus the hidden links."""
    return area(embed(model, x, np.concatenate([rest, hidden])), test)


def nearest(z):
    """Each row's ten nearest other rows by Euclidean distance, ties to the earlier row."""
    d = np.sqrt(sum((z[:, None, k] - z[None, :, k]) ** 2 for k in range(z.shape[1])))
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind="stable")[:, :NEIGHBOURS]


def neighbour_shares(nodes, x, linked, share):
    regular = (nodes.window1_minutes >= sh.W1_MINUTES).to_numpy()
    base = nearest(saved_pooled(nodes)[regular].astype(np.float64))
    shares = {}
    for seed in CHECK_SEEDS:
        z, info = run(nodes, x, linked, seed, share)
        report(f"4c seed {seed}", info)
        near = nearest(z[regular].astype(np.float64))
        shares[seed] = float((near[:, :, None] == base[:, None, :]).any(2).mean(1).mean())
    return int(regular.sum()), shares


def check(share):
    nodes, _, x, linked = graph_inputs()
    n, aucs = hidden_links(nodes, x, linked, share)
    a, b = aucs["with links"], aucs["without links"]
    key(f"\n4b: hidden links {n}, hidden negatives {n}")
    key(f"4b: area under the ROC curve with links {a!r}, without links {b!r}")
    higher = "with links" if a > b else "without links" if b > a else "neither, they are equal"
    key(f"4b: the higher area is {higher}")
    m, shares = neighbour_shares(nodes, x, linked, share)
    key("")
    for seed, s in shares.items():
        key(f"4c seed {seed}: mean share of the seed-0 ten nearest neighbours {s!r}, nodes {m}")


def diagnose():
    """The first run, trained with a scored share of 0: its pooled embeddings and hidden-link
    areas reproduced, then its network with links scored with the hidden links visible."""
    nodes, _, x, linked = graph_inputs()
    fresh, info = run(nodes, x, linked, SEED, 0)
    report("(1) pooled", info)
    total = float(fresh.astype(np.float64).sum())
    same = total == FIRST_SUM
    key(f"(1): float64 sum of the pooled seed-0 embeddings with share 0 {total!r}")
    key(f"(1): equal to the first run's sum {FIRST_SUM!r}: {same}")
    assert same, "HARD STOP: share 0 does not reproduce the first run's pooled embedding sum"

    aucs, model, hidden, rest, test = hidden_runs(nodes, x, linked, 0)
    a, b = aucs["with links"], aucs["without links"]
    key(f"(2): hidden links {len(hidden)}; area with links {a!r}, without links {b!r}")
    same = aucs == FIRST_AREAS
    key(f"(2): both areas equal the first run's: {same}")
    assert same, "HARD STOP: share 0 does not reproduce the first run's hidden-link areas"

    seen = visible_area(model, x, hidden, rest, test)
    key(f"(3): message links {len(rest) + len(hidden)}, of them hidden {len(hidden)}")
    key(f"(3): area with links, hidden links not visible {a!r}, visible {seen!r}")


def margin(share):
    """Report only: the hidden-link check's area with links less its area without, and a paired
    bootstrap drawing the hidden links and the unlinked pairs separately, both networks per draw."""
    nodes, _, x, linked = graph_inputs()
    scores, _, hidden, _, test = hidden_scores(nodes, x, linked, share)
    n = len(hidden)
    a, b = auc(scores["with links"]), auc(scores["without links"])
    key(f"1e: hidden links {n}, unlinked pairs {len(test) - n}")
    key(f"1e: area with links {a!r}, without links {b!r}")
    assert n == len(test) - n == 969, "HARD STOP: not 969 pairs of each kind"
    assert (round(a, 4), round(b, 4)) == (0.7676, 0.7602), "HARD STOP: not the second run's areas"

    rng = np.random.default_rng(SEED)
    diffs = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        drawn = np.r_[rng.choice(n, size=n), n + rng.choice(n, size=n)]
        got = {name: auc(s[drawn]) for name, s in scores.items()}
        diffs[r] = got["with links"] - got["without links"]
    q = sh.intervals(
        pd.DataFrame({"subset": "hidden", "predictor": "margin", "metric": "area", "value": diffs})
    ).iloc[0]
    key(f"1e: margin, area with links less area without {a - b!r}")
    key(f"1e: {RESAMPLES} resamples, 2.5 and 97.5 percentiles {q.p2_5!r}, {q.p97_5!r}")
    key(f"1e: resamples with a margin at or below zero {int((diffs <= 0).sum())}")


def main(argv):
    step = argv[0]
    style.LOGS.mkdir(parents=True, exist_ok=True)
    full = open(style.LOGS / f"p5_{step}.log", "w", encoding="utf-8", errors="replace")
    brief = open(style.LOGS / f"p5_{step}_summary.log", "w", encoding="utf-8", errors="replace")
    sys.stdout = Tee(full, brief)
    try:
        if step == "links":
            links()
        elif step == "embed":
            embeddings(SCORED)
        elif step == "rerun":
            rerun(SCORED)
        elif step == "check":
            check(SCORED)
        elif step == "diagnose":
            diagnose()
        elif step == "margin":
            margin(SCORED)
    finally:
        sys.stdout = sys.__stdout__
        full.close()
        brief.close()


if __name__ == "__main__":
    main(sys.argv[1:])
