import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from socceraction.spadl import config as spadl
from threadpoolctl import threadpool_info, threadpool_limits

from fbrecruit import graph, style
from fbrecruit import shrinkage as sh
from fbrecruit.logs import Tee, key, show
from fbrecruit.paths import INTERIM, PROCESSED
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
TEAM_SIDE = [d for d in DIMENSIONS if d != "buildup"]
C_E = [f"c_{e}" for e in graph.E]
# each tier adds one family to the tier below, so T1 to T4 are prefixes of T4's columns
TIERS = {"T1": [*(f"group_{g}" for g in sh.OUTFIELD), *(f"c_{b}" for b in BASIC)]}
TIERS["T2"] = [*TIERS["T1"], "P3"]
TIERS["T3"] = [*TIERS["T2"], *(f"c_{z}" for z in Z), *(f"c_team_z_{d}" for d in TEAM_SIDE)]
TIERS["T4"] = [*TIERS["T3"], *(f"c_fit_{d}" for d in TEAM_SIDE)]
TIERS["T5"] = [*TIERS["T4"], *C_E]
BASELINES = ["P0", "P2", "P3"]
COMPARISONS = [("T1", "P0"), ("T2", "T1"), ("T3", "T2"), ("T4", "T3"), ("T5", "T4"), ("T5", "T2")]
BRANCHES = sh.BRANCHES
SUBSETS = ["outfield", *LEAGUES]
OUTER = 5
INNER = 5
ORDER = {"pooled": list(range(OUTER)), "lolo": list(LEAGUES)}
L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
REDRAWS = 2000
PART = 100
WORKERS = 8
CHECK_REDRAWS = 16
LIMIT_S = 6 * 3600
PASS_AT = 1950
RECORD = ["replicate", "branch", "subset", "model", "sse", "psi2_sum", "rows"]


def without_path():
    return PROCESSED / "style_team_without_w1.parquet"


def inputs_path():
    return PROCESSED / "ablation_inputs.parquet"


def seeds_path():
    return PROCESSED / "embeddings_w1_seeds.parquet"


def predictions_path():
    return PROCESSED / "ablation_predictions.parquet"


def draws_path():
    return PROCESSED / "ablation_draws.parquet"


def part_path(k):
    return INTERIM / "ablation_bootstrap" / f"part_{k}.parquet"


def bootstrap_path():
    return PROCESSED / "ablation_bootstrap.parquet"


def summary_path():
    return PROCESSED / "ablation_summary.parquet"


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


def pairs():
    """Per branch, the outfield pairs in key order: target, psi2, P0, P2 and P3 joined to their
    inputs, with the five group indicators."""
    inputs = pd.read_parquet(inputs_path())
    cols = [*KEYS, "group", "target", "psi2", *BASELINES]
    out = {}
    for b in BRANCHES:
        ev = pd.read_parquet(sh.evaluation_path(b), columns=cols)
        ev = ev[ev.group.isin(sh.OUTFIELD)]
        m = ev.merge(inputs, on=KEYS, how="left", suffixes=("", "_inputs"), validate="one_to_one")
        missing = int(m.group_inputs.isna().sum())
        other = int((m.group_inputs.notna() & (m.group_inputs != m.group)).sum())
        key(f"2a {b}: outfield pairs {len(m)}, without inputs {missing}, of another group {other}")
        assert len(m) == 1067 and missing == other == 0, "HARD STOP: a pair has no inputs"
        indicators = {f"group_{g}": (m.group == g).astype(float) for g in sh.OUTFIELD}
        out[b] = m.assign(**indicators).sort_values(KEYS).reset_index(drop=True)
    assert out["pooled"][KEYS].equals(out["lolo"][KEYS]), "the branches' pairs differ"
    return out


def centred_embeddings(table, est, keys):
    """A table's 16 columns in float64, centred within league and position group by their mean
    over the outfield estimation rows, for keys' rows; and the largest absolute cell mean of a
    centred column over those 1,299 rows."""
    e = est.merge(table[[*KEYS, *graph.E]], on=KEYS, how="left", validate="one_to_one")
    assert e[graph.E].notna().all().all(), "HARD STOP: an estimation row has no embedding"
    c = centre(e.astype(dict.fromkeys(graph.E, np.float64)), graph.E).set_axis(C_E, axis=1)
    worst = float(e[CELL].join(c).groupby(CELL).mean().abs().max().max())
    rows = keys.merge(e[KEYS].join(c), on=KEYS, how="left", validate="one_to_one")
    return rows[C_E].to_numpy(), worst


def fold_embeddings(table, est, keys, seed):
    """Per branch and outer fold, the centred embeddings of keys' rows: the pooled network's for
    every pooled group; for each held-out league, those of the network that held it out."""
    parts = {"pooled": table[table.branch == "pooled"]}
    parts |= {lg: table[(table.branch == "lolo") & (table.fold == lg)] for lg in LEAGUES}
    got = {name: centred_embeddings(t, est, keys) for name, t in parts.items()}
    worst = {name: g[1] for name, g in got.items()}
    key(f"2a seed {seed}: largest absolute cell mean over the 1,299 rows, per table {worst}")
    assert max(worst.values()) <= 1e-12, "HARD STOP: a centred embedding cell mean is not zero"
    out = {"pooled": dict.fromkeys(ORDER["pooled"], got["pooled"][0])}
    return out | {"lolo": {lg: got[lg][0] for lg in LEAGUES}}


def team_codes(frame):
    """Each league's team ids, sorted, and a code per team: leagues in LEAGUES order, then id."""
    teams = {lg: np.sort(frame.team_id[frame.league == lg].unique()) for lg in LEAGUES}
    order = [(lg, int(t)) for lg in LEAGUES for t in teams[lg]]
    return teams, {k: i for i, k in enumerate(order)}


def fit_data(frames, emb, code):
    """The arrays a fit needs, in the frames' row order: team codes, leagues, and per branch the
    inputs of T4, the fold embeddings, target, psi2 and the baselines."""
    f = frames["pooled"]
    team = np.array([code[k] for k in zip(f.league, f.team_id, strict=True)])
    data = {"team": team, "league": f.league.to_numpy()}
    for b in BRANCHES:
        d = frames[b]
        data[b] = {c: d[c].to_numpy(np.float64) for c in ["target", "psi2", *BASELINES]}
        data[b] |= {"x": d[TIERS["T4"]].to_numpy(np.float64), "emb": emb[b]}
    return data


def deal(teams, rng):
    """Each league's teams sorted, permuted by rng and dealt in turn to groups 0 to 4, the deal
    continuing from one league to the next; teams maps each league to its team ids."""
    group, k = {}, 0
    for lg in LEAGUES:
        for t in rng.permutation(np.sort(teams[lg])):
            group[(lg, int(t))] = k % OUTER
            k += 1
    return group


def outer_groups(frame, teams):
    """The pooled outer group of every row: its team's in the deal from seed 0."""
    group = deal(teams, np.random.default_rng(sh.SEED))
    g = pd.Series(group, name="group").rename_axis(TEAM_KEYS).reset_index()
    per = g.groupby(["league", "group"]).size().unstack(fill_value=0).reindex(list(LEAGUES))
    rows = frame[TEAM_KEYS].merge(g, on=TEAM_KEYS, how="left", validate="many_to_one").group
    key("2b: teams per league and pooled group")
    key(per.to_string())
    key(f"2b: rows per group {rows.value_counts().sort_index().to_dict()}")
    assert per.shape == (len(LEAGUES), OUTER) and (per == 4).all().all(), "HARD STOP: the deal"
    return rows.to_numpy()


def setup():
    """The pairs, the outfield estimation rows' keys and groups, the seed-0 fit data, the outer
    folds of every row, each league's teams and their codes."""
    frames = pairs()
    est = pd.read_parquet(inputs_path(), columns=[*KEYS, "group"])
    teams, code = team_codes(frames["pooled"])
    table = pd.read_parquet(graph.embeddings_path())
    data = fit_data(frames, fold_embeddings(table, est, frames["pooled"][KEYS], 0), code)
    folds = {"pooled": outer_groups(frames["pooled"], teams), "lolo": data["league"]}
    return frames, est, data, folds, teams, code


def inner_folds(teams):
    """Five inner folds of whole teams, as (training, held-out) row positions; teams are codes."""
    return list(GroupKFold(n_splits=INNER).split(teams, groups=teams))


def fit(x, y, teams):
    """ElasticNetCV on x rescaled over its own rows, tuned over five inner folds of whole teams;
    also the numbers of ConvergenceWarnings and of other warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler = StandardScaler().fit(x)
        model = ElasticNetCV(l1_ratio=L1_RATIOS, alphas=100, cv=inner_folds(teams))
        model.fit(scaler.transform(x), y)
    convergence = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return scaler, model, convergence, len(caught) - convergence


def tier_inputs(d, tier, fold):
    """A prefix of T4's columns, or for T5 all of them and the fold's embeddings."""
    return np.hstack([d["x"], d["emb"][fold]]) if tier == "T5" else d["x"][:, : len(TIERS[tier])]


def out_of_fold(d, teams, folds, branch, tier):
    """Each outer fold's rows predicted by the tier fitted on the other folds' rows; and per fold
    the rows, alpha, mix, nonzero coefficients and warnings."""
    pred, fits = np.empty(len(folds)), []
    for f in ORDER[branch]:
        test = folds == f
        x = tier_inputs(d, tier, f)
        scaler, model, convergence, other = fit(x[~test], d["target"][~test], teams[~test])
        pred[test] = model.predict(scaler.transform(x[test]))
        fits.append(
            {"fold": f, "train": int((~test).sum()), "held_out": int(test.sum())}
            | {"alpha": model.alpha_, "mix": model.l1_ratio_}
            | {"nonzero": int((model.coef_ != 0).sum()), "convergence": convergence}
            | {"other": other}
        )
    return pred, fits


def predictions(data, folds, branches=BRANCHES, tiers=TIERS):
    """Out-of-fold predictions and fits per branch and tier with BLAS held to one thread, and the
    BLAS thread counts seen while fitting."""
    with threadpool_limits(limits=1, user_api="blas"):
        blas = sorted({i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"})
        out = {
            (b, t): out_of_fold(data[b], data["team"], folds[b], b, t)
            for b in branches
            for t in tiers
        }
    return out, blas


def warned(fits):
    """ConvergenceWarnings and other warnings summed over fits' folds."""
    per = [f for _, fs in fits.values() for f in fs]
    return sum(f["convergence"] for f in per), sum(f["other"] for f in per)


def sse(target, pred):
    return float(((target - pred) ** 2).sum())


def mask(league, subset):
    return np.ones(len(league), bool) if subset == "outfield" else league == subset


def records(league, target, psi2, preds):
    """Per subset, all rows and each league's, and per model: SSE, psi2 sum and rows."""
    out = []
    for s in SUBSETS:
        m = mask(league, s)
        for name, p in preds.items():
            out.append((s, name, sse(target[m], p[m]), float(psi2[m].sum()), int(m.sum())))
    return out


def ratios(t, index):
    """From one row per model with sse and psi2_sum: TSE and the three ratios, one column per
    model, and the six differences in SSE, one column per comparison."""
    sse_ = t.pivot(index=index, columns="model", values="sse")
    tse = sse_ - t.pivot(index=index, columns="model", values="psi2_sum")
    return {
        "tse": tse,
        "tse_ratio_p2": tse.div(tse.P2, axis=0),
        "sse_ratio_p2": sse_.div(sse_.P2, axis=0),
        "tse_ratio_p3": tse.div(tse.P3, axis=0),
        "sse_diff": pd.DataFrame({f"{a}-{b}": sse_[a] - sse_[b] for a, b in COMPARISONS}),
    }


def long(frame, name):
    """A frame with one column per model or comparison, as rows."""
    return frame.reset_index().melt(id_vars=frame.index.names, var_name="item", value_name=name)


def scores(data, preds):
    """2d: per branch, subset and model on all rows."""
    rows = []
    for b in BRANCHES:
        d = data[b]
        for s, name, e, psi2, n in records(data["league"], d["target"], d["psi2"], preds[b]):
            m = mask(data["league"], s)
            p, t = preds[b][name][m], d["target"][m]
            rows.append(
                {"branch": b, "subset": s, "model": name, "sse": e, "psi2_sum": psi2, "rows": n}
                | {"spearman": sh.spearman(p, t), "slope": sh.ols_line(p, t)[0]}
            )
    t = pd.DataFrame(rows)
    r = ratios(t, ["branch", "subset"])
    table = t.rename(columns={"model": "item"})
    for name in ["tse", "tse_ratio_p2", "sse_ratio_p2", "tse_ratio_p3"]:
        table = table.merge(long(r[name], name), on=["branch", "subset", "item"])
    cols = ["sse", "psi2_sum", "tse", "tse_ratio_p2", "sse_ratio_p2", "tse_ratio_p3"]
    for b in BRANCHES:
        key(f"\n2d {b}: all rows; Spearman and slope from shrinkage.spearman and ols_line")
        part = table[table.branch == b].set_index(["subset", "item"])
        key(show(part[[*cols, "spearman", "slope", "rows"]], 6))
        key(f"2d {b}: differences in SSE")
        key(show(r["sse_diff"].loc[b], 6))
    for x in table.itertuples():
        print(
            f"2d {x.branch} {x.subset} {x.item}: sse {x.sse!r}, psi2 sum {x.psi2_sum!r}, "
            f"rows {x.rows}, spearman {x.spearman!r}, slope {x.slope!r}"
        )
    for (b, s), x in r["sse_diff"].iterrows():
        print(f"2d {b} {s} differences in SSE: " + ", ".join(f"{c} {v!r}" for c, v in x.items()))
    head = table.set_index(["branch", "subset", "item"]).loc[("pooled", "outfield")]
    got = [head.tse_ratio_p2["P0"], head.tse_ratio_p2["P3"], head.sse_ratio_p2["P3"]]
    key(f"2d pooled outfield: TSE(P0)/TSE(P2), TSE(P3)/TSE(P2), SSE(P3)/SSE(P2) {got}")
    assert [round(v, 3) for v in got] == [0.793, 0.469, 0.809], "HARD STOP: baselines differ"


def describe(data):
    """2g, description only: each tier fitted on all pooled rows, rescaled over all of them."""
    d = data["pooled"]
    with threadpool_limits(limits=1, user_api="blas"):
        for t, cols in TIERS.items():
            # every pooled group holds the pooled network's embeddings
            _, model, convergence, other = fit(tier_inputs(d, t, 0), d["target"], data["team"])
            coef = pd.Series(model.coef_, index=cols)
            kept = coef[coef != 0].sort_values(key=np.abs, ascending=False)
            key(
                f"\n2g {t}: alpha {model.alpha_!r}, mix {model.l1_ratio_!r}, nonzero {len(kept)} "
                f"of {len(cols)}, ConvergenceWarnings {convergence}, other warnings {other}"
            )
            key(show(kept.to_frame("coefficient"), 6))


def fit_tiers():
    frames, est, data, folds, _, code = setup()
    counts = {t: len(c) for t, c in TIERS.items()}
    key(f"2a: inputs per tier {counts}")
    assert list(counts.values()) == [9, 10, 25, 32, 48], "HARD STOP: inputs per tier"
    seeds = pd.read_parquet(seeds_path())
    keys = frames["pooled"][KEYS]
    emb = {s: fold_embeddings(seeds[seeds.seed == s], est, keys, s) for s in SEEDS}

    fits, blas = predictions(data, folds)
    key(f"\n2c: BLAS threads while fitting {blas}")
    for (b, t), (_, per) in fits.items():
        for f in per:
            key(
                f"2c {b} {t} fold {f['fold']}: training rows {f['train']}, held out "
                f"{f['held_out']}, alpha {f['alpha']!r}, mix {f['mix']!r}, "
                f"nonzero coefficients {f['nonzero']}"
            )
        key(f"2c {b} {t}: ConvergenceWarnings, other warnings {warned({(b, t): fits[(b, t)]})}")
    key(f"2c: ConvergenceWarnings, other warnings, all tiers {warned(fits)}")
    preds = {b: {t: fits[(b, t)][0] for t in TIERS} for b in BRANCHES}
    scores(data, {b: preds[b] | {p: data[b][p] for p in BASELINES} for b in BRANCHES})

    again, _ = predictions(data, folds, ["pooled"], ["T5"])
    same = np.array_equal(again[("pooled", "T5")][0], preds["pooled"]["T5"])
    key(f"\n2e: pooled T5 fitted again, predictions identical {same}, warnings {warned(again)}")
    assert same, "HARD STOP: the pooled T5 predictions differ on a second fit"

    seed_preds, diffs = {}, {}
    for s in [0, *SEEDS]:
        if s:
            got, _ = predictions(fit_data(frames, emb[s], code), folds, tiers=["T5"])
            seed_preds |= {(b, s): got[(b, "T5")][0] for b in BRANCHES}
            key(f"2f seed {s}: ConvergenceWarnings, other warnings {warned(got)}")
        for b in BRANCHES:
            t5 = seed_preds[(b, s)] if s else preds[b]["T5"]
            y = data[b]["target"]
            diffs[(b, s)] = sse(y, t5) - sse(y, preds[b]["T4"])
    key("\n2f: SSE(T5) less SSE(T4) over the outfield rows, by embedding seed")
    key(show(pd.Series(diffs).unstack(), 6))
    for (b, s), v in diffs.items():
        key(f"2f {b} seed {s}: {v!r}")

    describe(data)

    out = [
        keys.assign(branch=b, tier=t, seed=0, fold=folds[b].astype(str), prediction=preds[b][t])
        for b in BRANCHES
        for t in TIERS
    ]
    out += [
        keys.assign(branch=b, tier="T5", seed=s, fold=folds[b].astype(str), prediction=p)
        for (b, s), p in seed_preds.items()
    ]
    out = pd.concat(out, ignore_index=True)
    out.to_parquet(predictions_path(), index=False)
    key(f"\n2h: wrote {predictions_path()} with {len(out)} rows, columns {list(out.columns)}")


def draws(teams):
    """Every redraw from one generator seeded 0: each league's 20 teams drawn with replacement, in
    LEAGUES order, then the distinct drawn teams dealt to the pooled groups."""
    rng = np.random.default_rng(sh.SEED)
    rows = []
    for r in range(REDRAWS):
        drawn = {}
        for lg in LEAGUES:
            ids = teams[lg]
            drawn[lg] = np.unique(rng.choice(ids, size=len(ids)), return_counts=True)
        group = deal({lg: ids for lg, (ids, _) in drawn.items()}, rng)
        for lg, (ids, counts) in drawn.items():
            rows += [
                (r, lg, int(t), int(n), group[(lg, int(t))])
                for t, n in zip(ids, counts, strict=True)
            ]
    return pd.DataFrame(rows, columns=["replicate", "league", "team_id", "count", "group"])


def draw_arrays(frame, code):
    """Per redraw, count and pooled group by team code, 0 and -1 for teams not drawn."""
    out = []
    for _, d in frame.groupby("replicate"):
        c = [code[k] for k in zip(d.league, d.team_id, strict=True)]
        count, group = np.zeros(len(code), "int64"), np.full(len(code), -1)
        count[c], group[c] = d["count"].to_numpy(), d.group.to_numpy()
        out.append((count, group))
    return out


def copies(team, count, group):
    """A redraw's rows, each row once per draw of its team, as positions; and each row's pooled
    group. count and group are indexed by team code."""
    idx = np.repeat(np.arange(len(team)), count[team])
    return idx, group[team[idx]]


def redraw(data, count, group):
    """One redraw: every tier refitted out of fold on the drawn rows of both branches, the copies
    of a team keeping its code. Returns the records, ConvergenceWarnings, other warnings and the
    BLAS thread counts seen."""
    idx, pooled = copies(data["team"], count, group)
    d = {"team": data["team"][idx], "league": data["league"][idx]}
    for b in BRANCHES:
        d[b] = {c: v[idx] for c, v in data[b].items() if c != "emb"}
        d[b]["emb"] = {f: e[idx] for f, e in data[b]["emb"].items()}
    fits, blas = predictions(d, {"pooled": pooled, "lolo": d["league"]})
    out = []
    for b in BRANCHES:
        preds = {t: fits[(b, t)][0] for t in TIERS} | {p: d[b][p] for p in BASELINES}
        out += [(b, *r) for r in records(d["league"], d[b]["target"], d[b]["psi2"], preds)]
    return out, *warned(fits), blas


def run_redraws(data, arrays, replicates, workers):
    jobs = (delayed(redraw)(data, *arrays[r]) for r in replicates)
    return Parallel(n_jobs=workers, backend="loky")(jobs)


def bootstrap():
    _, _, data, _, teams, code = setup()
    d = draws(teams)
    d.to_parquet(draws_path(), index=False)
    per = d.groupby(["replicate", "league"])["count"].sum()
    distinct = d.groupby("replicate").size()
    key(
        f"\n3a: wrote {draws_path()} with {len(d)} rows over {d.replicate.nunique()} redraws; "
        f"distinct teams per redraw min {distinct.min()}, max {distinct.max()}"
    )
    assert (per == 20).all(), "a league's draw does not hold 20 teams"
    arrays = draw_arrays(d, code)

    start = time.perf_counter()
    eight = run_redraws(data, arrays, range(CHECK_REDRAWS), WORKERS)
    per_redraw = (time.perf_counter() - start) / CHECK_REDRAWS
    key(
        f"\n3c: redraws 0 to {CHECK_REDRAWS - 1} with 8 workers: seconds per redraw "
        f"{per_redraw:.2f}, projected for {REDRAWS} {per_redraw * REDRAWS / 3600:.2f} h; "
        f"ConvergenceWarnings {sum(g[1] for g in eight)}, other warnings "
        f"{sum(g[2] for g in eight)}, BLAS threads {sorted({t for g in eight for t in g[3]})}"
    )
    assert per_redraw * REDRAWS <= LIMIT_S, "HARD STOP: the projection exceeds 6 hours"
    start = time.perf_counter()
    one = run_redraws(data, arrays, range(CHECK_REDRAWS), 1)
    same = [g[0] for g in one] == [g[0] for g in eight]
    key(
        f"3c: the same redraws with 1 worker: {time.perf_counter() - start:.1f} s, BLAS threads "
        f"{sorted({t for g in one for t in g[3]})}; every record identical {same}"
    )
    assert same, "HARD STOP: the records differ between 1 and 8 workers"

    part_path(0).parent.mkdir(parents=True, exist_ok=True)
    for k in range(REDRAWS // PART):
        path = part_path(k)
        if path.exists():
            key(f"3d part {k}: exists, skipped")
            continue
        start = time.perf_counter()
        reps = range(k * PART, (k + 1) * PART)
        got = run_redraws(data, arrays, reps, WORKERS)
        rows = [(r, *rec) for r, g in zip(reps, got, strict=True) for rec in g[0]]
        tmp = path.with_suffix(".tmp")
        pd.DataFrame(rows, columns=RECORD).to_parquet(tmp, index=False)
        tmp.replace(path)
        key(
            f"3d part {k}: redraws {reps.start} to {reps.stop - 1}, ConvergenceWarnings "
            f"{sum(g[1] for g in got)}, other warnings {sum(g[2] for g in got)}, BLAS threads "
            f"{sorted({t for g in got for t in g[3]})}, {time.perf_counter() - start:.1f} s"
        )

    out = pd.concat([pd.read_parquet(part_path(k)) for k in range(REDRAWS // PART)])
    out = out.reset_index(drop=True)
    out.to_parquet(bootstrap_path(), index=False)
    key(f"\n3d: wrote {bootstrap_path()} with {len(out)} rows, columns {list(out.columns)}")
    key(f"3d: records per redraw {out.groupby('replicate').size().unique().tolist()}")
    assert len(out) == 160000, "HARD STOP: the bootstrap table does not have 160,000 rows"


def passes(diff):
    """The rule of the entry "Ablation success rule": the difference in SSE below zero in at least
    1,950 of the 2,000 redraws, an equal SSE counting against."""
    return int((np.asarray(diff) < 0).sum()) >= PASS_AT


def saved_prediction(saved, frame, branch, tier, seed):
    s = saved[(saved.branch == branch) & (saved.tier == tier) & (saved.seed == seed)]
    return frame[KEYS].merge(s, on=KEYS, how="left", validate="one_to_one").prediction.to_numpy()


def all_rows(frames):
    """From the stored predictions, per branch, subset and model on all rows: SSE, psi2 sum and
    rows; and per branch and seed, SSE(T5) less SSE(T4) over the outfield rows."""
    saved = pd.read_parquet(predictions_path())
    recs, diffs = [], {}
    for b in BRANCHES:
        f = frames[b]
        y = f.target.to_numpy()
        preds = {t: saved_prediction(saved, f, b, t, 0) for t in TIERS}
        preds |= {p: f[p].to_numpy() for p in BASELINES}
        recs += [(b, *r) for r in records(f.league.to_numpy(), y, f.psi2.to_numpy(), preds)]
        for s in [0, *SEEDS]:
            t5 = saved_prediction(saved, f, b, "T5", s)
            diffs[(b, s)] = sse(y, t5) - sse(y, preds["T4"])
    return pd.DataFrame(recs, columns=RECORD[1:]), diffs


def intervals(frame, name):
    """Per branch, subset and column of frame, indexed by branch, subset and replicate: the 2.5
    and 97.5 percentiles and median over the redraws, as shrinkage.intervals computes them."""
    rows = long(frame, "value").rename(columns={"item": "predictor"}).assign(metric=name)
    out = [sh.intervals(rows[rows.branch == b]).reset_index().assign(branch=b) for b in BRANCHES]
    return pd.concat(out).rename(columns={"predictor": "item"})


def verdicts():
    frames = pairs()
    point, seed_diffs = all_rows(frames)
    boot = pd.read_parquet(bootstrap_path())
    key(f"4: redraws {boot.replicate.nunique()}, records {len(boot)}")
    key(
        "4: the intervals are conditional on the action-value models, calibrators, shrinkage "
        "fits, league levels, style profiles, input centres and embeddings, held fixed"
    )
    at, over = ratios(point, ["branch", "subset"]), ratios(boot, ["branch", "subset", "replicate"])
    index = ["branch", "subset", "item"]
    parts = []
    for name in ["sse_diff", "tse_ratio_p2", "sse_ratio_p2", "tse_ratio_p3"]:
        q = intervals(over[name], name).merge(long(at[name], "all_rows"), on=index)
        parts.append(q)
    summary = pd.concat(parts, ignore_index=True)
    diffs = long(over["sse_diff"], "value")
    below = diffs.assign(below=diffs.value < 0).groupby(index).below.sum().reset_index()
    ok = diffs.groupby(index).value.apply(passes).rename("passes").reset_index()
    summary = summary.merge(below, on=index, how="left")
    models = ["T1", "T2", "T3", "T4", "T5", "P3", "P0"]
    keep = (summary.metric == "sse_diff") | summary.item.isin(models)
    summary = summary[keep][[*index, "metric", "all_rows", "below", "p2_5", "p97_5", "median"]]
    summary = summary.reset_index(drop=True).astype({"below": "Int64"})
    summary.to_parquet(summary_path(), index=False)

    order = [f"{a}-{b}" for a, b in COMPARISONS]
    d = summary[summary.metric == "sse_diff"].merge(ok, on=index)
    d = d.set_index(["branch", "subset", "item"])[["all_rows", "below", "passes", "p2_5", "p97_5"]]
    for b in BRANCHES:
        key(f"\n4a {b}, outfield: differences in SSE, redraws below zero, verdict (pass at 1,950)")
        key(show(d.loc[(b, "outfield")].reindex(order), 4))
    for b in BRANCHES:
        for lg in LEAGUES:
            key(f"\n4a {b}, {lg}: rests on that league's twenty teams alone; no verdict")
            key(show(d.loc[(b, lg)].reindex(order).drop(columns="passes"), 4))
    for x in d.itertuples():
        print(
            f"4a {' '.join(map(str, x.Index))}: below {x.below}, p2_5 {x.p2_5!r}, "
            f"p97_5 {x.p97_5!r}, all rows {x.all_rows!r}"
        )

    r = summary[summary.metric != "sse_diff"].set_index([*index, "metric"])
    r = r[["all_rows", "p2_5", "p97_5"]].unstack("metric")
    for b in BRANCHES:
        for s in SUBSETS:
            flag = "" if s == "outfield" else "; rests on that league's twenty teams alone"
            key(f"\n4b {b}, {s}: ratios on all rows and 95% intervals{flag}")
            key(show(r.loc[(b, s)].reindex(models), 3))
    for x in summary[summary.metric != "sse_diff"].itertuples():
        print(
            f"4b {x.branch} {x.subset} {x.item} {x.metric}: all rows {x.all_rows!r}, "
            f"p2_5 {x.p2_5!r}, p97_5 {x.p97_5!r}"
        )

    verdict = {(b, c): bool(d.loc[(b, "outfield", c), "passes"]) for b in BRANCHES for c in order}
    steps = order[:5]
    failing = [c for c in steps if not verdict[("pooled", c)]]
    key(
        f"\n4c: every step from T1 to T5 passes on the pooled branch {not failing}; not passing "
        f"{failing}"
    )
    only = [c for c in order if verdict[("pooled", c)] and not verdict[("lolo", c)]]
    key(f"4c: passing on the pooled branch but not on the leave-one-league-out branch {only}")
    later = [seed_diffs[("pooled", s)] for s in SEEDS]
    if not verdict[("pooled", "T5-T4")]:
        label = "T5 against T4 does not pass, so no seed label applies"
    elif any(v >= 0 for v in later):
        label = "the pass of T5 against T4 is seed-dependent"
    else:
        label = "the pass of T5 against T4 holds for seeds 1 to 4"
    key(f"4c: seed label: {label}")
    key("\n4c: SSE(T5) less SSE(T4) over the outfield rows on all rows, by embedding seed")
    key(show(pd.Series(seed_diffs).unstack(), 6))
    key(f"\n4c: wrote {summary_path()} with {len(summary)} rows, columns {list(summary.columns)}")


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
        elif step == "fit":
            fit_tiers()
        elif step == "bootstrap":
            bootstrap()
        elif step == "verdicts":
            verdicts()
    finally:
        sys.stdout = sys.__stdout__
        full.close()
        brief.close()


if __name__ == "__main__":
    main(sys.argv[1:])
