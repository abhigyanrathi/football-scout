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


def need(path):
    if not path.exists():
        pytest.skip(f"local data cache missing: {path}")
    return path


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
