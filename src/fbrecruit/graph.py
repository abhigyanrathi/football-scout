import sys

import numpy as np
import pandas as pd
from socceraction.spadl import config as spadl

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


def links_path():
    return PROCESSED / "graph_links_w1.parquet"


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


def main(argv):
    step = argv[0]
    style.LOGS.mkdir(parents=True, exist_ok=True)
    full = open(style.LOGS / f"p5_{step}.log", "w", encoding="utf-8", errors="replace")
    brief = open(style.LOGS / f"p5_{step}_summary.log", "w", encoding="utf-8", errors="replace")
    sys.stdout = Tee(full, brief)
    try:
        if step == "links":
            links()
    finally:
        sys.stdout = sys.__stdout__
        full.close()
        brief.close()


if __name__ == "__main__":
    main(sys.argv[1:])
