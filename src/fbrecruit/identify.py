import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from fbrecruit import shrinkage as sh
from fbrecruit import style
from fbrecruit.logs import Tee, key
from fbrecruit.paths import PROCESSED
from fbrecruit.sources.statsbomb import LEAGUES

KEYS = sh.KEYS
TEAM_GROUP = ["league", "team_id", "group"]
Z = [f"z_{d}" for d in style.DIMENSIONS]
MINUTES = 450
TOLERANCE = 1e-9
REDRAWS = 2000
SEEDS = [0, 1, 2, 3, 4]
E = [f"e{i}" for i in range(16)]
DISTANCES = ["style", "embedding", "combined"]
TOPS = [1, 3, 5, 10]


def proxy_path():
    return PROCESSED / "identify_proxy_redraws.parquet"


def redraws_path():
    return PROCESSED / "identify_redraws.parquet"


def players_path():
    return PROCESSED / "identify_players.parquet"


def seeds_path():
    return PROCESSED / "embeddings_seeds.parquet"


def links2_path():
    return PROCESSED / "graph_links_w2.parquet"


def window_minutes():
    """The pooled v2 player-window table read as keys, window and minutes alone."""
    return pd.read_parquet(
        PROCESSED / "player_window_vaep_pooled_v2.parquet", columns=[*KEYS, "window", "minutes"]
    )


def estimation_keys(t):
    """The keys of phase 4c's estimation population: window-1 rows with at least 450 minutes."""
    return t[(t.window == 1) & (t.minutes >= sh.W1_MINUTES)][KEYS]


def pool(player, t):
    """The outfield rows of the window-1 style table with at least 450 minutes for their team in
    each window of t, with both windows' minutes."""
    rows = player[player.group.isin(sh.OUTFIELD)]
    for w in (1, 2):
        m = t[t.window == w][[*KEYS, "minutes"]].rename(columns={"minutes": f"minutes_w{w}"})
        rows = rows.merge(m, on=KEYS, validate="one_to_one")
    return rows[(rows.minutes_w1 >= MINUTES) & (rows.minutes_w2 >= MINUTES)].reset_index(drop=True)


def vectors(frame):
    """The eight standard scores in float64, a missing one counted as 0, the group mean."""
    return frame[Z].fillna(0.0).to_numpy(np.float64)


def distances(first, second):
    """Euclidean distances in float64 from every row of first to every row of second."""
    first, second = np.asarray(first, np.float64), np.asarray(second, np.float64)
    return np.sqrt(((first[:, None, :] - second[None, :, :]) ** 2).sum(axis=2))


def ranks(d):
    """For distances from first vectors to second vectors, each row's own on the diagonal: one plus
    the number of other columns at the same or a smaller distance than its own."""
    return (d <= d.diagonal()[:, None]).sum(axis=1)


def team_hits(first, second, rows):
    """Per row, whether its first vector is nearer its own second vector than every other second
    vector of its league, team and group, an equal distance a miss, and k, the rows there."""
    hit, k = np.zeros(len(rows), bool), np.zeros(len(rows), np.int64)
    for b in rows.groupby(TEAM_GROUP).indices.values():
        hit[b] = ranks(distances(first[b], second[b])) == 1
        k[b] = len(b)
    return hit, k


def statistic(hit, k):
    """The mean over the rows with k of 2 or more of the hit, 1 or 0, less 1/k."""
    q = k >= 2
    return float(np.mean(hit[q] - 1 / k[q]))


def team_draws(rows):
    """Each league's team_ids among the rows, sorted, and the row positions of each team."""
    teams = {lg: np.sort(rows.team_id[rows.league == lg].unique()) for lg in LEAGUES}
    return teams, rows.groupby(["league", "team_id"]).indices


def draw(rng, teams, at):
    """One redraw's row positions: each league's teams drawn with replacement, as many as it has,
    and each drawn team's rows in draw order, a team drawn twice giving its rows twice."""
    out = []
    for lg in LEAGUES:
        out += [at[lg, t] for t in rng.choice(teams[lg], size=len(teams[lg]), replace=True)]
    return np.concatenate(out)


def half_hits(rows, inputs, mom):
    """The rows with minutes on both the odd and the even game days of the inputs' games, the hits
    of their odd profiles against their even ones, and k."""
    odd, even = style.split_halves(*inputs, rows[[*KEYS, "group"]], mom)
    hit, k = team_hits(vectors(odd), vectors(even), odd)
    return odd, hit, k


def proxy_redraws(rows, hit, k, n=REDRAWS):
    """The statistic over n redraws of the rows' teams from a new generator seeded 0."""
    rng = np.random.default_rng(0)
    teams, at = team_draws(rows)
    out = np.empty(n)
    for r in range(n):
        idx = draw(rng, teams, at)
        out[r] = statistic(hit[idx], k[idx])
    return out


def proxy():
    start = time.perf_counter()
    t = window_minutes()
    player = pd.read_parquet(style.player_path())
    inputs = style.window_inputs(1)
    games, a, press, lineups = inputs
    key(
        f"4: windows of the games read {sorted(games.window.unique().tolist())}; "
        f"games {len(games)}, actions {len(a)}, pressure events {len(press)}, "
        f"lineup rows {len(lineups)}"
    )
    outside = sum(int((~f.game_id.isin(games.game_id)).sum()) for f in (a, press, lineups))
    key(f"4: actions, pressure events or lineup rows from a game outside window 1: {outside}")
    assert outside == 0, "HARD STOP: a row read is not from a window-1 game"

    pop = estimation_keys(t)
    same = style.player_table(*inputs, pop).equals(player)
    key(
        "4: the player table recomputed with the changed style.py equals style_player_w1.parquet "
        f"in every column, NaN matching NaN: {same}"
    )
    assert same, "HARD STOP: style_player_w1.parquet is not reproduced"

    rows = pool(player, t)
    gap = float((rows.minutes_w1 - rows.window1_minutes).abs().max())
    key(f"\n4a: pool rows {len(rows)}")
    key(f"4a: largest difference of the v2 window-1 minutes from window1_minutes {gap!r}")
    assert gap <= TOLERANCE, "HARD STOP: a pool row's window-1 minutes differ from window1_minutes"
    key(rows.league.value_counts().reindex(list(LEAGUES)).to_string())
    key(rows.group.value_counts().reindex(sh.OUTFIELD).to_string())

    odd, hit, k = half_hits(rows, inputs, style.moments(player, pop))
    q = k >= 2
    key(f"\n4b: pool players with minutes on both odd and even game days of window 1 {len(odd)}")
    key(f"4b: N0 {int(q.sum())!r}, A0 {int((k == 1).sum())!r}")
    key(f"4b: hits {int(hit[q].sum())!r}, sum of 1/k {float((1 / k[q]).sum())!r}")
    key(f"4b: statistic {statistic(hit, k)!r}")

    values = proxy_redraws(odd, hit, k)
    table = pd.DataFrame({"replicate": np.arange(REDRAWS), "statistic": values})
    table.to_parquet(proxy_path(), index=False)
    key(f"\n4c: wrote {proxy_path()} with {len(table)} rows, columns {list(table.columns)}")
    teams, _ = team_draws(odd)
    key("4c: teams drawn from per league " + ", ".join(f"{lg} {len(v)}" for lg, v in teams.items()))
    lo, hi = np.percentile(values, [2.5, 97.5])
    sd = float(values.std(ddof=1))
    z = norm.ppf(0.975) + norm.ppf(0.8)
    key(f"4d: 2.5th percentile {lo!r}, 97.5th percentile {hi!r}")
    key(f"4d: standard deviation of the redraws (ddof 1) {sd!r}")
    key(f"4d: norm.ppf(0.975) + norm.ppf(0.8) {z!r}; MDE {z * sd!r}")
    key(f"4: {time.perf_counter() - start:.1f} s")


def expected(m):
    """(1 + 1/2 + ... + 1/m) / m for each m: the expected reciprocal rank of one of m candidates put
    in random order."""
    m = np.asarray(m)
    return np.cumsum(1 / np.arange(1, m.max() + 1))[m - 1] / m


def spread(blocks):
    """The standard deviation, ddof 1, of the distances of every pair in the blocks."""
    return float(np.concatenate([b.ravel() for b in blocks]).std(ddof=1))


def embeddings(rows):
    """The rows' embeddings in float64 per window and seed, from embeddings_seeds.parquet, missing
    where a row has none."""
    e = pd.read_parquet(seeds_path())
    out = {}
    for w in (1, 2):
        for s in SEEDS:
            one = e[(e.window == w) & (e.seed == s)]
            got = rows[KEYS].merge(one, on=KEYS, how="left", validate="one_to_one")
            out[w, s] = got[E].to_numpy(np.float64)
    return out


def search(rows, s1, s2, e):
    """Per row: k, n, the hit, and under the style, embedding and combined distances the rank of
    its own window-2 row among its group's and the distance to it; and the standard deviations
    that scale the distances."""
    groups = list(rows.groupby("group").indices.values())
    seed = {s: [distances(e[1, s][b], e[2, s][b]) for b in groups] for s in SEEDS}
    sd = {f"seed {s}": spread(seed[s]) for s in SEEDS}
    d = {"style": [distances(s1[b], s2[b]) for b in groups]}
    d["embedding"] = [
        np.mean([seed[s][i] / sd[f"seed {s}"] for s in SEEDS], axis=0) for i in range(len(groups))
    ]
    sd["style"], sd["embedding"] = spread(d["style"]), spread(d["embedding"])
    d["combined"] = [
        x / sd["style"] + y / sd["embedding"]
        for x, y in zip(d["style"], d["embedding"], strict=True)
    ]
    hit, k = team_hits(s1, s2, rows)
    n = rows.groupby("group").group.transform("size").to_numpy()
    out = rows[[*KEYS, "group", "minutes_w1", "minutes_w2"]].assign(k=k, n=n, hit=hit)
    rank = {name: np.zeros(len(rows), np.int64) for name in DISTANCES}
    own = {name: np.zeros(len(rows)) for name in DISTANCES}
    for name in DISTANCES:
        for b, x in zip(groups, d[name], strict=True):
            rank[name][b], own[name][b] = ranks(x), x.diagonal()
    out = out.assign(**{f"rank_{name}": rank[name] for name in DISTANCES})
    return out.assign(**{f"distance_{name}": own[name] for name in DISTANCES}), sd


def statistics(p, w1, w2, idx):
    """Over the players at positions idx of p, with w1 and w2 their window-1 and window-2 standard
    scores: the test statistic, the mean reciprocal ranks and their differences, and the eight
    window correlations."""
    k = p.k.to_numpy()[idx]
    rr = {name: 1 / p[f"rank_{name}"].to_numpy()[idx] for name in DISTANCES}
    out = {
        "statistic": statistic(p.hit.to_numpy()[idx], k),
        "mrr_style": float(rr["style"].mean()),
        "style_less_team": float(np.mean(rr["style"] - expected(k))),
    }
    for c in style.half_correlations(w1.iloc[idx], w2.iloc[idx], "windows"):
        out[f"r_{c['dimension']}"] = c["r"]
    out["mrr_embedding"] = float(rr["embedding"].mean())
    out["mrr_combined"] = float(rr["combined"].mean())
    out["combined_less_style"] = out["mrr_combined"] - out["mrr_style"]
    return out


def redraw_table(p, w1, w2, n=REDRAWS):
    """The statistics over n redraws of the players' teams from a new generator seeded 0."""
    rng = np.random.default_rng(0)
    teams, at = team_draws(p)
    return pd.DataFrame(
        [{"replicate": r, **statistics(p, w1, w2, draw(rng, teams, at))} for r in range(n)]
    )


def test():
    start = time.perf_counter()
    t = window_minutes()
    player = pd.read_parquet(style.player_path())
    rows = pool(player, t)
    gap = float((rows.minutes_w1 - rows.window1_minutes).abs().max())
    key(f"7a: pool rows {len(rows)}; largest window-1 minutes difference {gap!r}")
    assert gap <= TOLERANCE, "HARD STOP: a pool row's window-1 minutes differ from window1_minutes"
    w2 = pd.read_parquet(style.window2_path())
    m = rows[KEYS].merge(w2, on=KEYS, how="left", validate="one_to_one", indicator=True)
    no_profile = int((m._merge != "both").sum())
    e = embeddings(rows)
    no_embedding = {ws: int(np.isnan(v).any(axis=1).sum()) for ws, v in e.items()}
    key(f"7a: pool rows with no window-2 profile {no_profile}")
    key(f"7a: pool rows with no embedding, per (window, seed) {no_embedding}")
    assert no_profile == 0 and not any(no_embedding.values()), "HARD STOP: a pool row lacks data"

    p, sd = search(rows, vectors(rows), vectors(m), e)
    pairs = int((p.groupby("group").size() ** 2).sum())
    for name, v in sd.items():
        key(f"7a: standard deviation (ddof 1) over the {pairs} pairs, {name}: {v!r}")
    scales = np.array(list(sd.values()))
    assert np.isfinite(scales).all() and (scales != 0).all(), "HARD STOP: a scale is 0 or infinite"

    w1z, w2z = rows[Z], m[Z]
    across = style.half_correlations(w1z, w2z, "windows")
    mom = style.moments(player, estimation_keys(t))
    halves = {}
    for w in (1, 2):
        odd, even = style.split_halves(*style.window_inputs(w), rows[[*KEYS, "group"]], mom)
        key(f"\n7b: pool players with minutes on both halves of window {w} {len(odd)}")
        halves[w] = style.half_correlations(odd, even, f"window {w}")
    for i, d in enumerate(style.DIMENSIONS):
        r = across[i]["r"]
        up = {w: 2 * h[i]["r"] / (1 + h[i]["r"]) for w, h in halves.items()}
        key(f"7b {d}: windows r {r!r}, n {across[i]['n']}")
        for w, h in halves.items():
            key(f"7b {d}: window {w} odd-even r {h[i]['r']!r}, n {h[i]['n']}, stepped up {up[w]!r}")
        key(f"7b {d}: corrected stability {r / np.sqrt(up[1] * up[2])!r}")

    table = redraw_table(p, w1z, w2z)
    table.to_parquet(redraws_path(), index=False)
    key(f"\n7c: wrote {redraws_path()} with {len(table)} rows, columns {list(table.columns)}")
    p.to_parquet(players_path(), index=False)
    key(f"7d: wrote {players_path()} with {len(p)} rows, columns {list(p.columns)}")

    full = statistics(p, w1z, w2z, np.arange(len(p)))
    ends = {s: np.percentile(table[s], [2.5, 97.5]) for s in full}
    key("\n7e: full pool, then the 2.5th and 97.5th percentiles over the redraws")
    for s, v in full.items():
        key(f"7e {s}: {v!r}, {ends[s][0]!r} to {ends[s][1]!r}")
    q = p.k >= 2
    key(
        f"7e: pool {len(p)}; with a teammate in their group {int(q.sum())}, their hits "
        f"{int(p.hit[q].sum())}, expected by chance {float((1 / p.k[q]).sum())!r}; alone "
        f"{int((~q).sum())}"
    )
    key("7e: pool by group " + ", ".join(f"{g} {int((p.group == g).sum())}" for g in sh.OUTFIELD))
    key("7e: ranked within " + ", ".join(f"{t} {int((p.rank_style <= t).sum())}" for t in TOPS))
    key(f"7e: mean reciprocal rank by chance {float(expected(p.n).mean())!r}")
    key(f"7e: team-and-group searcher's mean reciprocal rank {float(expected(p.k).mean())!r}")
    passes = ends["statistic"][0] > 0
    enter = ends["combined_less_style"][0] > 0
    key(f"7e: 2.5th percentile of the statistic above zero, the claim passes: {passes}")
    key(f"7e: 2.5th percentile of combined less style above zero, the embeddings enter: {enter}")

    links = pd.read_parquet(links2_path())
    first = w2[KEYS].merge(player[KEYS], on=KEYS, how="left", indicator=True)._merge == "both"
    unknown = w2.group == "UNKNOWN"
    key(
        f"7e: window-2 graph rows {len(w2)}, links {len(links)}; groups from window 2 "
        f"{int((~first & ~unknown).sum())}, UNKNOWN {int(unknown.sum())}"
    )
    key(f"7: {time.perf_counter() - start:.1f} s")


def main(argv):
    step = argv[0]
    run = {"proxy": proxy, "test": test}[step]
    style.LOGS.mkdir(parents=True, exist_ok=True)
    full = open(style.LOGS / f"p9a_{step}.log", "w", encoding="utf-8", errors="replace")
    brief = open(style.LOGS / f"p9a_{step}_summary.log", "w", encoding="utf-8", errors="replace")
    sys.stdout = Tee(full, brief)
    try:
        run()
    finally:
        sys.stdout = sys.__stdout__
        full.close()
        brief.close()


if __name__ == "__main__":
    main(sys.argv[1:])
