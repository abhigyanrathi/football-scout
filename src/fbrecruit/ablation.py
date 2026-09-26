import sys

import numpy as np
import pandas as pd
from socceraction.spadl import config as spadl

from fbrecruit import graph, style
from fbrecruit import shrinkage as sh
from fbrecruit.logs import Tee, key, show
from fbrecruit.paths import PROCESSED
from fbrecruit.sources.statsbomb import LEAGUES, OUT

KEYS = sh.KEYS
TEAM_KEYS = style.TEAM_KEYS
CELL = ["league", "group"]
DIMENSIONS = style.DIMENSIONS
Z = [f"z_{d}" for d in DIMENSIONS]
TEAM_Z = [f"team_z_{d}" for d in DIMENSIONS]
FIT = [f"fit_{d}" for d in DIMENSIONS]
TYPES = ["shot", "shot_freekick", "shot_penalty", "pass", "cross"]
RESULTS = ["success", "owngoal"]
NP_SHOTS = ["shot", "shot_freekick"]
PASSES = ["pass", "cross"]
PER90 = {"np_goal": "np_goals_per90", "np_shot": "np_shots_per90", "assist": "assists_per90"}
BASIC = ["log_minutes", *PER90.values()]
SEEDS = [1, 2, 3, 4]


def without_path():
    return PROCESSED / "style_team_without_w1.parquet"


def inputs_path():
    return PROCESSED / "ablation_inputs.parquet"


def seeds_path():
    return PROCESSED / "embeddings_w1_seeds.parquet"


def key_set(frame, cols):
    return set(frame[cols].itertuples(index=False, name=None))


def rows():
    """The evaluation pairs checked against style_player_w1. Returns the player and team tables
    and the outfield estimation rows."""
    cols = [*KEYS, "group", "minutes_w1"]
    ev = {b: pd.read_parquet(sh.evaluation_path(b), columns=cols) for b in sh.BRANCHES}
    player = pd.read_parquet(style.player_path())
    team = pd.read_parquet(style.team_path())
    pop = player[player.window1_minutes >= sh.W1_MINUTES]
    est = pop[pop.group.isin(sh.OUTFIELD)].reset_index(drop=True)

    kinds, found, same_minutes = {}, {}, {}
    for b, e in ev.items():
        kinds[b] = e.group.where(~e.group.isin(sh.OUTFIELD), "outfield").value_counts().to_dict()
        out = e[e.group.isin(sh.OUTFIELD)].merge(
            est[[*KEYS, "group"]], on=KEYS, how="left", suffixes=("", "_est")
        )
        found[b] = int((out.group_est == out.group).sum()), len(out)
        m = e.merge(player[[*KEYS, "window1_minutes"]], on=KEYS, how="left", validate="one_to_one")
        same_minutes[b] = int((m.minutes_w1 == m.window1_minutes).sum()), len(m)
        key(f"2a {b}: rows {len(e)}, distinct keys {len(key_set(e, KEYS))}, groups {kinds[b]}")
        key(f"2a {b}: outfield pairs that are outfield estimation rows of their group {found[b]}")
        key(f"2a {b}: pairs whose minutes_w1 equals window1_minutes {same_minutes[b]}")
    teams = team.groupby("league").size()
    same_teams = key_set(player, TEAM_KEYS) == key_set(team, TEAM_KEYS)
    key(f"2a: estimation population {len(pop)}, outfield estimation rows {len(est)}")
    key(f"2a: teams per league {teams.to_dict()}; same teams in the player table {same_teams}")

    checks = {
        "1,144 rows in each evaluation table, same keys": all(
            len(e) == len(key_set(e, KEYS)) == 1144 for e in ev.values()
        )
        and key_set(ev["pooled"], KEYS) == key_set(ev["lolo"], KEYS),
        "groups GK 76, UNKNOWN 1, outfield 1,067": all(
            k == {"outfield": 1067, "GK": 76, "UNKNOWN": 1} for k in kinds.values()
        ),
        "estimation population 1,401 rows": len(pop) == 1401,
        "every outfield pair an outfield estimation row": all(f == n for f, n in found.values()),
        "minutes_w1 equal to window1_minutes": all(f == n for f, n in same_minutes.values()),
        "20 teams in every league": bool((teams == 20).all()) and same_teams,
    }
    for name, passed in checks.items():
        key(f"2a {name}: {passed}")
    assert all(checks.values()), "HARD STOP: a row check failed"
    key("2a: outfield estimation rows by league and group")
    key(est.groupby(CELL).size().unstack(fill_value=0).to_string())
    return player, team, est


def flags(a):
    """Per action, a in action order with style.py's sequences: non-penalty goal and shot,
    penalty goal, own goal, and assist, a completed pass or cross after which every later action
    of the sequence is by one teammate, the last being his non-penalty goal."""
    success = (a.result_id == spadl.results.index("success")).to_numpy()
    shot = a.type_name.isin(NP_SHOTS).to_numpy()
    goal = shot & success
    seq, player = a.seq.to_numpy(), a.player_id.to_numpy()
    pos = np.arange(len(a))
    last = pd.Series(pos).groupby(seq).transform("max").to_numpy()
    same_seq = np.r_[seq[1:] == seq[:-1], False]
    other = np.r_[player[1:] != player[:-1], False]
    runs = np.cumsum(np.r_[True, other[:-1] | ~same_seq[:-1]])
    run_end = pd.Series(pos).groupby(runs).transform("max").to_numpy()
    nxt = np.minimum(pos + 1, len(a) - 1)
    passes = a.type_name.isin(PASSES).to_numpy() & success
    return pd.DataFrame(
        {
            "np_goal": goal,
            "np_shot": shot,
            "penalty_goal": (a.type_name == "shot_penalty").to_numpy() & success,
            "own_goal": (a.result_id == spadl.results.index("owngoal")).to_numpy(),
            "assist": passes & same_seq & other & (run_end[nxt] == last) & goal[last],
        },
        index=a.index,
    )


def basic_numbers(games, a, est):
    """Per outfield estimation row, from the player's window-1 actions for that team: goals,
    shots and assists per 90 window-1 minutes, and the log of those minutes."""
    missing = [t for t in TYPES if t not in spadl.actiontypes]
    missing += [r for r in RESULTS if r not in spadl.results]
    key(f"2b: type and result names missing from the installed spadl config: {missing}")
    assert not missing, "HARD STOP: names missing from the spadl config"
    f = flags(a)

    per_league = pd.concat([a.league, f], axis=1).groupby("league").sum()
    key("2b: over all window-1 actions, per league")
    key(per_league[["np_goal", "penalty_goal", "own_goal", "np_shot", "assist"]].to_string())
    over = per_league.index[per_league.assist > per_league.np_goal].tolist()
    key(f"2b: leagues with more assists than non-penalty goals {over}")
    assert not over, "HARD STOP: more assists than non-penalty goals"
    cols = ["game_id", "home_score", "away_score"]
    scores = pd.concat(
        pd.read_parquet(OUT / lg / "games.parquet", columns=cols).assign(league=lg)
        for lg in LEAGUES
    )
    scores = games[["league", "game_id"]].merge(
        scores, on=["league", "game_id"], validate="one_to_one"
    )
    goals = (scores.home_score + scores.away_score).groupby(scores.league).sum()
    events = per_league.np_goal + per_league.penalty_goal + per_league.own_goal
    key("2b: goals from the window-1 scores, and non-penalty, penalty and own goals, report only")
    both = {"games": scores.groupby("league").size(), "scores": goals, "events": events}
    key(pd.DataFrame(both).to_string())

    counts = pd.concat([a[KEYS], f[list(PER90)]], axis=1).groupby(KEYS, as_index=False).sum()
    out = est[[*KEYS, "group", "window1_minutes"]].merge(
        counts, on=KEYS, how="left", validate="one_to_one"
    )
    key(f"2b: outfield estimation rows with no window-1 actions {int(out.np_goal.isna().sum())}")
    out[list(PER90)] = out[list(PER90)].fillna(0)
    key("2b: counts summed over the outfield estimation rows")
    key(out[list(PER90)].sum().to_string())
    for c, name in PER90.items():
        out[name] = out[c] / out.window1_minutes * 90
    out["log_minutes"] = np.log(out.window1_minutes)
    return out[[*KEYS, "group", "window1_minutes", *BASIC]]


def team_parts(games, a, press):
    """Per team: its window-1 actions, window-1 pressure events and opponent shares."""
    press = press[press.game_id.isin(set(games.game_id))]
    frames = [dict(list(f.groupby(TEAM_KEYS))) for f in (a, press, style.opponent_share(a, games))]
    return {k: tuple(f[k] for f in frames) for k in frames[0]}


def profile_without(parts, times, player_id, m, total):
    """A team's profile less one player's units: his actions; for build-up, the sequences he took
    part in through any of them; his pressure events, the rest of the pressing times
    total / (total - m). Sequences, build-up times and creation flags stay those of the full
    stream. A player_id of None removes nobody."""
    ta, tp, ts = parts
    mine = (ta.player_id == player_id).to_numpy()
    kept = ~times.seq.isin(ta.seq[mine])
    out = style.action_dimensions(ta[~mine], TEAM_KEYS, times[kept])
    pressing = style.team_pressing(tp[(tp.player_id != player_id).to_numpy()], ts).sb_pressing
    return out.assign(sb_pressing=pressing * (total / (total - m)))[DIMENSIONS]


def full_profiles(parts, times, totals, team):
    """Every team's profile with nobody removed, standardized as style.py standardizes teams."""
    out = [profile_without(p, times, None, 0, totals[k]) for k, p in parts.items()]
    return style.standardize_teams(pd.concat(out), team).reset_index()


def reproduction(full, team):
    """Largest absolute difference per column from the stored team profiles."""
    cols = [*DIMENSIONS, *Z]
    got = full.set_index(TEAM_KEYS).reindex(pd.MultiIndex.from_frame(team[TEAM_KEYS]))[cols]
    assert got.notna().all().all(), "HARD STOP: a team profile value is missing"
    return (got - team.set_index(TEAM_KEYS)[cols]).abs().max()


def team_profiles(games, a, press, player, team, est):
    parts = team_parts(games, a, press)
    times = style.buildup_times(a)
    totals = player.groupby(TEAM_KEYS).window1_minutes.sum()
    diff = reproduction(full_profiles(parts, times, totals, team), team)
    key("2c: nobody removed against style_team_w1, largest absolute difference per column")
    key(diff.to_string())
    key(f"2c: largest over every column {diff.max()!r}")
    assert (diff <= 1e-12).all(), "HARD STOP: the stored team profiles are not reproduced"

    out = []
    for r in est.itertuples(index=False):
        k = (r.league, r.team_id)
        out.append(profile_without(parts[k], times, r.player_id, r.window1_minutes, totals[k]))
    z = style.standardize_teams(pd.concat(out), team).reset_index(drop=True)
    without = est[[*KEYS, "group"]].join(z[[*DIMENSIONS, *Z]])
    missing = int(without[[*DIMENSIONS, *Z]].isna().sum().sum())
    key(f"2c: missing values in the profiles without the player {missing}")
    assert missing == 0, "HARD STOP: a value of a profile without the player is missing"

    full = without[TEAM_KEYS].merge(team, on=TEAM_KEYS, how="left")
    change = without[[*DIMENSIONS, *Z]] - full[[*DIMENSIONS, *Z]]
    key("2c: change from the full team profile, per column")
    summary = {"median": change.median(), "median_abs": change.abs().median()}
    key(show(pd.DataFrame(summary | {"max_abs": change.abs().max()}), 6))
    total = est.merge(totals.rename("total").reset_index(), on=TEAM_KEYS, how="left").total
    factor = total / (total - est.window1_minutes)
    key(f"2c: T / (T - m) min {factor.min()!r}, median {factor.median()!r}, max {factor.max()!r}")
    without.to_parquet(without_path(), index=False)
    key(f"2c: wrote {without_path()} with {len(without)} rows")
    return without


def centre(frame, cols):
    """Each column less its mean over the frame's rows of the same league and position group."""
    return frame[cols] - frame.groupby(CELL)[cols].transform("mean")


def fit_products(frame):
    """Per dimension, the centred player score times the centred score of his team without him,
    the products then centred the same way."""
    c = centre(frame, [*Z, *TEAM_Z])
    products = {f: c[z] * c[t] for f, z, t in zip(FIT, Z, TEAM_Z, strict=True)}
    return centre(frame[CELL].assign(**products), FIT)


def team_spread(team):
    """Across the stored team profiles: each dimension's between-league share of the sum of
    squares, and the absolute correlations among possession, build-up and verticality."""
    d = team[DIMENSIONS]
    between = ((team.groupby("league")[DIMENSIONS].transform("mean") - d.mean()) ** 2).sum()
    share = between / ((d - d.mean()) ** 2).sum()
    return share, team[["possession", "buildup", "verticality"]].corr().abs()


def centred_inputs(basic, player, without, team):
    frame = basic.merge(player[[*KEYS, *Z]], on=KEYS, how="left", validate="one_to_one").merge(
        without[[*KEYS, *Z]].set_axis([*KEYS, *TEAM_Z], axis=1), on=KEYS, validate="one_to_one"
    )
    missing = frame[Z].isna().sum()
    key(f"2d: missing player z_ scores set to 0, over {len(frame)} outfield estimation rows")
    key(missing.to_string())
    key(f"2d: missing player z_ scores in all {int(missing.sum())}")
    frame[Z] = frame[Z].fillna(0.0)

    c = centre(frame, [*BASIC, *Z, *TEAM_Z]).join(fit_products(frame)).add_prefix("c_")
    means = frame[CELL].join(c).groupby(CELL).mean().abs().max()
    key("2d: largest absolute cell mean per centred column")
    key(means.to_string())
    assert (means <= 1e-12).all(), "HARD STOP: a centred column has a cell mean away from zero"
    out = frame[[*KEYS, "group", "window1_minutes", *PER90.values(), "log_minutes"]].join(c)
    out.to_parquet(inputs_path(), index=False)
    key(f"2d: wrote {inputs_path()} with {len(out)} rows, columns {list(out.columns)}")

    share, corr = team_spread(team)
    key("\n2d: across the 80 stored team profiles, between-league share of the sum of squares")
    key(share.to_string())
    key("2d: absolute correlations among possession, build-up and verticality")
    key(corr.to_string())


def embedding_seeds():
    stored = pd.read_parquet(graph.embeddings_path())
    key("\n2e: seed 0 rebuilt through graph.embedding_table, nothing written")
    table = graph.embedding_table(graph.SCORED)
    diff = np.abs(table[graph.E].to_numpy(np.float64) - stored[graph.E].to_numpy(np.float64))
    same = table.equals(stored)
    key(f"2e: seed 0 rows {len(table)}, equal to the stored file {same}, largest {diff.max()!r}")
    assert same, "HARD STOP: seed 0 does not rebuild the stored embeddings"
    tables = []
    for seed in SEEDS:
        key(f"\n2e: seed {seed}")
        tables.append(graph.embedding_table(graph.SCORED, seed).assign(seed=seed))
    out = pd.concat(tables, ignore_index=True)
    key(f"2e: rows {len(out)}, per seed {out.seed.value_counts().sort_index().to_dict()}")
    assert len(out) == 41460, "HARD STOP: the seed table does not have 41,460 rows"
    out.to_parquet(seeds_path(), index=False)
    key(f"2e: wrote {seeds_path()}, columns {list(out.columns)}")


def inputs():
    player, team, est = rows()
    games, a, press, _ = style.load_inputs()
    key(f"\n2b: window-1 games {len(games)}, actions {len(a)}")
    basic = basic_numbers(games, a, est)
    without = team_profiles(games, a, press, player, team, est)
    centred_inputs(basic, player, without, team)
    embedding_seeds()


def participation():
    """Per outfield estimation row: the change in his team's build-up standard score when he is
    left out, against his share of the team's build-up sequences."""
    without = pd.read_parquet(without_path(), columns=[*KEYS, "z_buildup"])
    player = pd.read_parquet(style.player_path(), columns=[*KEYS, "n_buildup"])
    team = pd.read_parquet(style.team_path(), columns=[*TEAM_KEYS, "z_buildup", "n_buildup"])
    m = without.merge(player, on=KEYS, how="left", validate="one_to_one").merge(
        team, on=TEAM_KEYS, how="left", suffixes=("", "_team"), validate="many_to_one"
    )
    change = m.z_buildup - m.z_buildup_team
    share = m.n_buildup / m.n_buildup_team
    missing = int(change.isna().sum() + share.isna().sum())
    key(f"1b: outfield estimation rows {len(m)}, missing changes or shares {missing}")
    assert len(m) == 1299 and missing == 0, "HARD STOP: not 1,299 complete rows"
    above = int((change > 0).sum())
    rho = sh.spearman(change.to_numpy(), share.to_numpy())
    key(f"1b: rows whose team build-up score rises when he is left out {above} of {len(m)}")
    key(f"1b: change in the team's z_buildup, median {change.median()!r}, largest {change.max()!r}")
    key(f"1b: Spearman correlation of the change with the share {rho!r}")
    assert above > len(m) / 2 and rho > 0, "HARD STOP: the shift is not mostly up with the share"


def main(argv):
    step = argv[0]
    style.LOGS.mkdir(parents=True, exist_ok=True)
    full = open(style.LOGS / f"p6_{step}.log", "w", encoding="utf-8", errors="replace")
    brief = open(style.LOGS / f"p6_{step}_summary.log", "w", encoding="utf-8", errors="replace")
    sys.stdout = Tee(full, brief)
    try:
        if step == "inputs":
            inputs()
        elif step == "participation":
            participation()
    finally:
        sys.stdout = sys.__stdout__
        full.close()
        brief.close()


if __name__ == "__main__":
    main(sys.argv[1:])
