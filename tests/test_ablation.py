import numpy as np
import pandas as pd
import pytest
from socceraction.spadl import config as spadl

from fbrecruit import ablation, graph, style
from fbrecruit import shrinkage as sh

SUCCESS = spadl.results.index("success")
FAIL = spadl.results.index("fail")
OWNGOAL = spadl.results.index("owngoal")
OUTFIELD_ROWS = 1299
CELLS = {
    ("la_liga", "CB"): 64, ("la_liga", "FB"): 65, ("la_liga", "FWD"): 45,
    ("la_liga", "MID"): 68, ("la_liga", "WIDE"): 93,
    ("ligue_1", "CB"): 66, ("ligue_1", "FB"): 72, ("ligue_1", "FWD"): 44,
    ("ligue_1", "MID"): 70, ("ligue_1", "WIDE"): 79,
    ("premier_league", "CB"): 60, ("premier_league", "FB"): 59, ("premier_league", "FWD"): 38,
    ("premier_league", "MID"): 70, ("premier_league", "WIDE"): 80,
    ("serie_a", "CB"): 66, ("serie_a", "FB"): 67, ("serie_a", "FWD"): 49,
    ("serie_a", "MID"): 88, ("serie_a", "WIDE"): 56,
}  # fmt: skip
# SSE over the 1,067 outfield pairs of the out-of-fold predictions, per branch and tier
ALL_ROWS_SSE = {
    ("pooled", "T1"): 22.318767443729293,
    ("pooled", "T2"): 21.401271464610907,
    ("pooled", "T3"): 20.823640172024373,
    ("pooled", "T4"): 20.79994710607643,
    ("pooled", "T5"): 20.911792783287282,
    ("lolo", "T1"): 22.262971973636237,
    ("lolo", "T2"): 21.211370908701618,
    ("lolo", "T3"): 20.869155955457956,
    ("lolo", "T4"): 20.635384644614884,
    ("lolo", "T5"): 20.62709202275976,
}
# redraw 0 of the 2,000: rows per subset, psi2 sum per branch and subset, SSE per model
REDRAW_ZERO_ROWS = {
    "outfield": 1085,
    "la_liga": 300,
    "premier_league": 246,
    "serie_a": 269,
    "ligue_1": 270,
}
REDRAW_ZERO_PSI2 = {
    ("pooled", "outfield"): 17.650244842594322,
    ("pooled", "la_liga"): 5.381317002162959,
    ("pooled", "premier_league"): 3.758539066113658,
    ("pooled", "serie_a"): 4.402887662397664,
    ("pooled", "ligue_1"): 4.10750111192004,
    ("lolo", "outfield"): 17.620179330248014,
    ("lolo", "la_liga"): 5.373386552148009,
    ("lolo", "premier_league"): 3.7519972892658213,
    ("lolo", "serie_a"): 4.39444075421399,
    ("lolo", "ligue_1"): 4.100354734620196,
}
REDRAW_ZERO_SSE = {
    ("pooled", "outfield", "T1"): 24.22714510211608,
    ("pooled", "outfield", "T2"): 23.305002363454765,
    ("pooled", "outfield", "T3"): 22.148749942890298,
    ("pooled", "outfield", "T4"): 21.975098374456287,
    ("pooled", "outfield", "T5"): 21.996762887009822,
    ("pooled", "outfield", "P0"): 28.161504895042988,
    ("pooled", "outfield", "P2"): 28.70710313562629,
    ("pooled", "outfield", "P3"): 24.10746501919703,
    ("pooled", "la_liga", "T1"): 7.213109634939958,
    ("pooled", "la_liga", "T2"): 6.74379278003434,
    ("pooled", "la_liga", "T3"): 6.4756273816710035,
    ("pooled", "la_liga", "T4"): 6.244333386639115,
    ("pooled", "la_liga", "T5"): 6.182363763905629,
    ("pooled", "la_liga", "P0"): 9.501281138273601,
    ("pooled", "la_liga", "P2"): 7.6434489368613665,
    ("pooled", "la_liga", "P3"): 6.7461094480167825,
    ("pooled", "premier_league", "T1"): 4.790391253235569,
    ("pooled", "premier_league", "T2"): 4.49731940520474,
    ("pooled", "premier_league", "T3"): 4.59415880494785,
    ("pooled", "premier_league", "T4"): 4.661547785275497,
    ("pooled", "premier_league", "T5"): 4.692263920117849,
    ("pooled", "premier_league", "P0"): 5.378285803498022,
    ("pooled", "premier_league", "P2"): 6.15139464291711,
    ("pooled", "premier_league", "P3"): 4.374371508230386,
    ("pooled", "serie_a", "T1"): 4.916973778967902,
    ("pooled", "serie_a", "T2"): 4.835078594012251,
    ("pooled", "serie_a", "T3"): 4.3535216682146896,
    ("pooled", "serie_a", "T4"): 4.472953199665331,
    ("pooled", "serie_a", "T5"): 4.433491351098269,
    ("pooled", "serie_a", "P0"): 5.329000170684864,
    ("pooled", "serie_a", "P2"): 7.217314800530536,
    ("pooled", "serie_a", "P3"): 5.204400465475957,
    ("pooled", "ligue_1", "T1"): 7.306670434972652,
    ("pooled", "ligue_1", "T2"): 7.228811584203433,
    ("pooled", "ligue_1", "T3"): 6.725442088056758,
    ("pooled", "ligue_1", "T4"): 6.596264002876347,
    ("pooled", "ligue_1", "T5"): 6.688643851888074,
    ("pooled", "ligue_1", "P0"): 7.952937782586499,
    ("pooled", "ligue_1", "P2"): 7.694944755317281,
    ("pooled", "ligue_1", "P3"): 7.782583597473906,
    ("lolo", "outfield", "T1"): 24.08234673576788,
    ("lolo", "outfield", "T2"): 23.308324089077843,
    ("lolo", "outfield", "T3"): 22.85583795855356,
    ("lolo", "outfield", "T4"): 22.72974127884851,
    ("lolo", "outfield", "T5"): 24.18848781178579,
    ("lolo", "outfield", "P0"): 28.658109880553454,
    ("lolo", "outfield", "P2"): 29.26152251801578,
    ("lolo", "outfield", "P3"): 24.532951173529078,
    ("lolo", "la_liga", "T1"): 6.540198770382895,
    ("lolo", "la_liga", "T2"): 6.36177721908915,
    ("lolo", "la_liga", "T3"): 6.112807041839503,
    ("lolo", "la_liga", "T4"): 6.06240825929787,
    ("lolo", "la_liga", "T5"): 7.255727173859549,
    ("lolo", "la_liga", "P0"): 9.586483053194861,
    ("lolo", "la_liga", "P2"): 7.909368795112316,
    ("lolo", "la_liga", "P3"): 6.861089247613337,
    ("lolo", "premier_league", "T1"): 4.9263995277289805,
    ("lolo", "premier_league", "T2"): 4.62408766207527,
    ("lolo", "premier_league", "T3"): 4.64521000958543,
    ("lolo", "premier_league", "T4"): 4.663163166561166,
    ("lolo", "premier_league", "T5"): 4.681669123756128,
    ("lolo", "premier_league", "P0"): 5.421996690565026,
    ("lolo", "premier_league", "P2"): 6.174421071902821,
    ("lolo", "premier_league", "P3"): 4.409318283036347,
    ("lolo", "serie_a", "T1"): 5.4061214018728085,
    ("lolo", "serie_a", "T2"): 5.131754260577374,
    ("lolo", "serie_a", "T3"): 5.043627001294626,
    ("lolo", "serie_a", "T4"): 4.9937877738798,
    ("lolo", "serie_a", "T5"): 5.238535692285398,
    ("lolo", "serie_a", "P0"): 5.733427766814282,
    ("lolo", "serie_a", "P2"): 7.425807672129579,
    ("lolo", "serie_a", "P3"): 5.501961214496737,
    ("lolo", "ligue_1", "T1"): 7.209627035783195,
    ("lolo", "ligue_1", "T2"): 7.1907049473360445,
    ("lolo", "ligue_1", "T3"): 7.054193905834003,
    ("lolo", "ligue_1", "T4"): 7.010382079109675,
    ("lolo", "ligue_1", "T5"): 7.012555821884715,
    ("lolo", "ligue_1", "P0"): 7.916202369979286,
    ("lolo", "ligue_1", "P2"): 7.751924978871065,
    ("lolo", "ligue_1", "P3"): 7.760582428382657,
}


def need(path):
    if not path.exists():
        pytest.skip(f"local data cache missing: {path}")
    return path


def fit_setup():
    for branch in sh.BRANCHES:
        need(sh.evaluation_path(branch))
    need(ablation.inputs_path())
    need(graph.embeddings_path())
    return ablation.setup()


@pytest.mark.slow
def test_row_counts():
    for branch in sh.BRANCHES:
        need(sh.evaluation_path(branch))
    need(style.player_path())
    need(style.team_path())
    _, team, est = ablation.rows()
    assert len(est) == OUTFIELD_ROWS
    assert est.groupby(["league", "group"]).size().to_dict() == CELLS
    assert team.groupby("league").size().to_dict() == dict.fromkeys(graph.LEAGUES, 20)


@pytest.mark.slow
def test_team_profiles_reproduced_with_nobody_removed():
    for lg in graph.LEAGUES:
        need(style.OUT / lg / "actions.parquet")
        need(style.OUT / lg / "pressures.parquet")
    player = pd.read_parquet(need(style.player_path()))
    team = pd.read_parquet(need(style.team_path()))
    games, a, press, _ = style.load_inputs()
    parts = ablation.team_parts(games, a, press)
    totals = player.groupby(style.TEAM_KEYS).window1_minutes.sum()
    full = ablation.full_profiles(parts, style.buildup_times(a), totals, team)
    assert (ablation.reproduction(full, team) <= 1e-12).all()


@pytest.mark.slow
def test_seed_zero_rebuilds_the_stored_embeddings():
    need(style.player_path())
    need(graph.links_path())
    stored = pd.read_parquet(need(graph.embeddings_path()))
    assert graph.embedding_table(graph.SCORED).equals(stored)


@pytest.mark.slow
def test_all_rows_sse_of_every_tier():
    _, _, data, folds, _, _ = fit_setup()
    fits, blas = ablation.predictions(data, folds)
    assert blas == [1]
    got = {(b, t): ablation.sse(data[b]["target"], p) for (b, t), (p, _) in fits.items()}
    assert got == ALL_ROWS_SSE


@pytest.mark.slow
def test_redraw_zero_records():
    _, _, data, _, teams, code = fit_setup()
    draws = ablation.draws(teams)
    count, group = ablation.draw_arrays(draws[draws.replicate == 0], code)[0]
    records, _, _, blas = ablation.redraw(data, count, group)
    assert blas == [1]
    expected = {
        (b, s, m): (e, REDRAW_ZERO_PSI2[(b, s)], REDRAW_ZERO_ROWS[s])
        for (b, s, m), e in REDRAW_ZERO_SSE.items()
    }
    assert {(b, s, m): (e, p, n) for b, s, m, e, p, n in records} == expected


def actions(rows):
    """Rows of game, team, player, type and result, in action order within one period."""
    a = pd.DataFrame(rows, columns=["game_id", "team_id", "player_id", "type_name", "result_id"])
    return style.with_sequences(a.assign(period_id=1, time_seconds=np.arange(len(a), dtype=float)))


def test_goals_shots_and_assists():
    a = actions(
        [
            # an assist through the scorer's carries
            (1, 1, 10, "pass", SUCCESS), (1, 1, 11, "dribble", SUCCESS),
            (1, 1, 11, "dribble", SUCCESS), (1, 1, 11, "shot", SUCCESS),
            # a penalty goal
            (2, 2, 20, "shot_penalty", SUCCESS),
            # an own goal, after a completed pass of the other team
            (3, 2, 21, "pass", SUCCESS), (3, 1, 12, "clearance", OWNGOAL),
            # a corner headed in
            (4, 1, 13, "corner_crossed", SUCCESS), (4, 1, 14, "shot", SUCCESS),
            # a goal after two teammates' passes
            (5, 1, 10, "pass", SUCCESS), (5, 1, 11, "pass", SUCCESS), (5, 1, 15, "shot", SUCCESS),
            # a missed shot and a free kick scored
            (6, 1, 14, "shot", FAIL), (7, 1, 16, "shot_freekick", SUCCESS),
        ]
    )  # fmt: skip
    f = ablation.flags(a)
    got = pd.concat([a.player_id, f], axis=1).groupby("player_id").sum().astype(int)
    # non-penalty goals, non-penalty shots, penalty goals, own goals, assists
    assert {p: tuple(r) for p, r in got.iterrows()} == {
        10: (0, 0, 0, 0, 1),
        11: (1, 1, 0, 0, 1),
        12: (0, 0, 0, 1, 0),
        13: (0, 0, 0, 0, 0),
        14: (1, 2, 0, 0, 0),
        15: (1, 1, 0, 0, 0),
        16: (1, 1, 0, 0, 0),
        20: (0, 0, 1, 0, 0),
        21: (0, 0, 0, 0, 0),
    }


def team_parts():
    """One team's actions, pressure events and opponent share in one game, and the build-up times
    of every sequence of that game."""
    rows = [
        # time, team, player, type, start_x, end_x
        (0.0, 1, 10, "pass", 30.0, 50.0),
        (4.0, 1, 11, "pass", 50.0, 75.0),  # reaches the final third after 4 s
        (5.0, 2, 20, "pass", 40.0, 45.0),
        (10.0, 1, 12, "pass", 20.0, 60.0),
        (16.0, 1, 12, "dribble", 60.0, 72.0),  # reaches it after 6 s
        (17.0, 2, 20, "tackle", 30.0, 30.0),
        (18.0, 1, 11, "interception", 40.0, 40.0),
    ]
    cols = ["time_seconds", "team_id", "player_id", "type_name", "start_x", "end_x"]
    a = pd.DataFrame(rows, columns=cols).assign(
        league="x", game_id=1, period_id=1, start_y=20.0, end_y=30.0, result_id=SUCCESS
    )
    a = style.with_sequences(a)
    press = pd.DataFrame(
        {"league": "x", "team_id": 1, "player_id": [10, 11, 11, 12], "x": [70.0, 80, 65, 30]}
    )
    share = pd.DataFrame({"league": ["x"], "game_id": [1], "team_id": [1], "opp_share": [0.25]})
    return (a[a.team_id == 1], press, share), style.buildup_times(a)


def test_team_profile_with_nobody_removed_is_the_teams():
    parts, times = team_parts()
    ta, tp, ts = parts
    team = style.action_dimensions(ta, style.TEAM_KEYS, times).join(style.team_pressing(tp, ts))
    got = ablation.profile_without(parts, times, None, 0, 900)
    pd.testing.assert_frame_equal(got, team[style.DIMENSIONS], check_exact=True)


def test_build_up_drops_the_sequences_he_took_part_in():
    parts, times = team_parts()
    assert ablation.profile_without(parts, times, None, 0, 900).buildup.item() == -5.0
    # 11 took part in the 4-second sequence, 12 made the 6-second one alone
    assert ablation.profile_without(parts, times, 11, 300, 900).buildup.item() == -6.0
    assert ablation.profile_without(parts, times, 12, 300, 900).buildup.item() == -4.0


def test_pressing_without_him_is_scaled_by_the_team_minutes():
    parts, times = team_parts()
    # three pressure events beyond x = 60 in one game, opponent share 0.25
    assert ablation.profile_without(parts, times, None, 0, 900).sb_pressing.item() == 6.0
    # 11 had two of them: one left, times 900 / (900 - 300)
    assert ablation.profile_without(parts, times, 11, 300, 900).sb_pressing.item() == 3.0


def test_centring_and_fit_products():
    frame = pd.DataFrame(
        {
            "league": ["x", "x", "x", "x", "x"],
            "group": ["CB", "CB", "FB", "FB", "FB"],
            "z": [1.0, 3.0, 0.0, 4.0, 8.0],
            "team_z": [2.0, 6.0, 1.0, 1.0, 4.0],
        }
    )
    c = ablation.centre(frame, ["z", "team_z"])
    assert c.z.tolist() == [-1.0, 1.0, -4.0, 0.0, 4.0]
    assert c.team_z.tolist() == [-2.0, 2.0, -1.0, -1.0, 2.0]
    assert (frame[ablation.CELL].join(c).groupby(ablation.CELL).mean() == 0).all().all()

    scores = frame[ablation.CELL].assign(
        **dict.fromkeys(ablation.Z, frame.z), **dict.fromkeys(ablation.TEAM_Z, frame.team_z)
    )
    fit = ablation.fit_products(scores)
    # centred products 2, 2 and 4, 0, 8; from uncentred scores they would be 2, 18 and 0, 4, 32
    for col in ablation.FIT:
        assert fit[col].tolist() == [0.0, 0.0, 0.0, -4.0, 4.0]


def test_the_deal_gives_four_teams_per_league_per_group_and_copies_share_one():
    teams = {lg: np.arange(20) + 100 * i for i, lg in enumerate(ablation.LEAGUES)}
    group = ablation.deal(teams, np.random.default_rng(0))
    per = pd.Series(group).groupby(level=0).value_counts()
    assert per.to_dict() == {(lg, g): 4 for lg in ablation.LEAGUES for g in range(5)}
    # rows of three teams: the first drawn three times, the second not at all, the third twice
    idx, rows = ablation.copies(
        np.array([0, 0, 1, 2, 2, 2]), np.array([3, 0, 2]), np.array([4, -1, 1])
    )
    assert idx.tolist() == [0, 0, 0, 1, 1, 1, 3, 3, 4, 4, 5, 5]
    assert rows.tolist() == [4] * 6 + [1] * 6


def test_inner_folds_keep_a_teams_copies_together():
    team = np.repeat(np.arange(12), [1, 2, 3] * 4)
    count = np.array([2, 1, 0, 3, 1, 1, 2, 1, 1, 2, 1, 1])
    idx, _ = ablation.copies(team, count, np.zeros(12, "int64"))
    rows = team[idx]
    folds = ablation.inner_folds(rows)
    assert len(folds) == 5
    assert np.sort(np.concatenate([test for _, test in folds])).tolist() == list(range(len(rows)))
    for train, test in folds:
        assert not set(rows[train]) & set(rows[test])


def test_the_pass_rule_needs_1950_redraws_below_zero_and_counts_a_tie_against():
    assert ablation.passes(np.r_[-np.ones(1950), np.ones(50)])
    assert not ablation.passes(np.r_[-np.ones(1949), np.ones(51)])
    assert not ablation.passes(np.r_[-np.ones(1949), np.zeros(51)])
