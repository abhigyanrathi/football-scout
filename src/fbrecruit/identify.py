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


def proxy_path():
    return PROCESSED / "identify_proxy_redraws.parquet"


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


def main(argv):
    step = argv[0]
    run = {"proxy": proxy}[step]
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
