import numpy as np
import pandas as pd
import pytest

from fbrecruit import identify, style
from fbrecruit import shrinkage as sh
from fbrecruit.paths import PROCESSED
from fbrecruit.sources.statsbomb import LEAGUES, OUT

# from p9a_proxy_summary.log
POOL = 929
POOL_LEAGUES = {"la_liga": 240, "premier_league": 229, "serie_a": 228, "ligue_1": 232}
POOL_GROUPS = {"CB": 202, "FB": 179, "MID": 221, "WIDE": 204, "FWD": 123}
# players with minutes in both halves of window 1, N0, A0 and the hits of the N0
PROXY_COUNTS = (929, 863, 66, 602)
PROXY_STATISTIC = 0.32213209733487835
PROXY_PERCENTILES = [0.29710898150965087, 0.3475296049990633]
# from p9a_style_summary.log and p9a_test_summary.log
W2_ROWS = 1902
# N, their hits and A
TEST_COUNTS = (863, 621, 66)
STATISTIC = 0.34414831981460026
STATISTIC_PERCENTILES = [0.3157814603500224, 0.37201232553524577]
MRR = {
    "style": 0.2693689868518743,
    "embedding": 0.21223629422583212,
    "combined": 0.2858603812770043,
}
COMBINED_LESS_STYLE = [-0.002200416126587932, 0.036115022075151305]


def need(path):
    if not path.exists():
        pytest.skip(f"local data cache missing: {path}")
    return path


def test_an_equal_distance_is_a_miss_and_a_rank_against_the_player():
    rows = pd.DataFrame({"league": "x", "team_id": 1, "group": "MID", "player_id": [0, 1, 2]})
    first = np.array([[0.0, 0.0], [0.0, 5.0], [9.0, 9.0]])
    second = np.array([[1.0, 0.0], [0.0, 1.0], [9.0, 8.0]])
    # player 0 is 1 from his own second vector and 1 from player 1's; 1 and 2 are nearest their own
    assert identify.ranks(identify.distances(first, second)).tolist() == [2, 1, 1]
    hit, k = identify.team_hits(first, second, rows)
    assert hit.tolist() == [False, True, True]
    assert k.tolist() == [3, 3, 3]


def test_a_player_alone_in_his_team_and_group_is_left_out():
    rows = pd.DataFrame(
        {"league": "x", "team_id": [1, 1, 1, 2], "group": ["MID", "MID", "FWD", "MID"]}
    )
    first = second = np.eye(4)
    hit, k = identify.team_hits(first, second, rows)
    assert k.tolist() == [2, 2, 1, 1]
    assert hit.all()
    # the two teammates in MID count, 1 - 1/2 each; the forward and team 2's player do not
    assert identify.statistic(hit, k) == 0.5


class FirstTeamEveryTime:
    """Stands in for the generator: every team a league draws is its first."""

    def choice(self, teams, size, replace):
        assert replace
        return np.repeat(teams[:1], size)


def test_a_team_drawn_twice_gives_its_players_twice():
    rows = pd.DataFrame(
        {"league": ["la_liga", "la_liga", "la_liga", "serie_a"], "team_id": [7, 5, 7, 9]}
    )
    teams, at = identify.team_draws(rows)
    assert {lg: v.tolist() for lg, v in teams.items()} == {
        "la_liga": [5, 7],
        "premier_league": [],
        "serie_a": [9],
        "ligue_1": [],
    }
    # la_liga draws team 5 twice, so its one row twice; serie_a draws team 9
    assert identify.draw(FirstTeamEveryTime(), teams, at).tolist() == [1, 1, 3]


def test_chance_and_team_and_group_values():
    # (1 + ... + 1/m) / m for m of 1 to 4
    got = identify.expected([1, 2, 3, 4])
    assert np.allclose(got, [1, 3 / 4, 11 / 18, 25 / 48], rtol=0, atol=1e-15)
    # three MID of team 1, one MID of team 2 and a FWD of team 1
    rows = pd.DataFrame(
        {
            "league": "x",
            "player_id": range(5),
            "team_id": [1, 1, 1, 2, 1],
            "group": ["MID", "MID", "MID", "MID", "FWD"],
            "minutes_w1": 900,
            "minutes_w2": 900,
        }
    )
    rng = np.random.default_rng(0)
    e = {(w, s): rng.normal(size=(5, 16)) for w in (1, 2) for s in identify.SEEDS}
    p, _ = identify.search(rows, np.eye(5), np.eye(5), e)
    assert p.n.tolist() == [4, 4, 4, 4, 1]
    assert p.k.tolist() == [3, 3, 3, 1, 1]
    assert np.allclose(identify.expected(p.n), [25 / 48] * 4 + [1], rtol=0, atol=1e-15)
    assert np.allclose(identify.expected(p.k), [11 / 18] * 3 + [1, 1], rtol=0, atol=1e-15)


def pool_inputs():
    need(PROCESSED / "player_window_vaep_pooled_v2.parquet")
    return pd.read_parquet(need(style.player_path())), identify.window_minutes()


@pytest.mark.slow
def test_pool_and_proxy_statistic_recomputed_from_the_caches():
    for lg in LEAGUES:
        for name in ["games", "actions", "pressures", "lineups"]:
            need(OUT / lg / f"{name}.parquet")
    need(sh.group_path())
    player, t = pool_inputs()
    rows = identify.pool(player, t)
    assert len(rows) == POOL
    assert rows.league.value_counts().to_dict() == POOL_LEAGUES
    assert rows.group.value_counts().to_dict() == POOL_GROUPS

    mom = style.moments(player, identify.estimation_keys(t))
    odd, hit, k = identify.half_hits(rows, style.window_inputs(1), mom)
    q = k >= 2
    assert (len(odd), int(q.sum()), int((k == 1).sum()), int(hit[q].sum())) == PROXY_COUNTS
    assert identify.statistic(hit, k) == PROXY_STATISTIC

    stored = pd.read_parquet(need(identify.proxy_path()))
    assert np.array_equal(identify.proxy_redraws(odd, hit, k, 5), stored.statistic.to_numpy()[:5])
    assert np.percentile(stored.statistic, [2.5, 97.5]).tolist() == PROXY_PERCENTILES


@pytest.mark.slow
def test_window2_profile_rows():
    assert len(pd.read_parquet(need(style.window2_path()))) == W2_ROWS


def pool_and_window2():
    player, t = pool_inputs()
    rows = identify.pool(player, t)
    w2 = pd.read_parquet(need(style.window2_path()))
    return rows, rows[identify.KEYS].merge(w2, on=identify.KEYS, how="left", validate="one_to_one")


@pytest.mark.slow
def test_players_table_recomputed_from_the_profiles_and_embeddings():
    rows, m = pool_and_window2()
    need(identify.seeds_path())
    e = identify.embeddings(rows)
    p, _ = identify.search(rows, identify.vectors(rows), identify.vectors(m), e)
    stored = pd.read_parquet(need(identify.players_path()))
    pd.testing.assert_frame_equal(p, stored, check_exact=True)
    q = p.k >= 2
    assert (int(q.sum()), int(p.hit[q].sum()), int((~q).sum())) == TEST_COUNTS


@pytest.mark.slow
def test_statistics_and_the_first_redraws_match_the_stored_ones():
    rows, m = pool_and_window2()
    p = pd.read_parquet(need(identify.players_path()))
    w1, w2 = rows[identify.Z], m[identify.Z]
    full = identify.statistics(p, w1, w2, np.arange(len(p)))
    assert full["statistic"] == STATISTIC
    assert {name: full[f"mrr_{name}"] for name in identify.DISTANCES} == MRR
    stored = pd.read_parquet(need(identify.redraws_path()))
    again = identify.redraw_table(p, w1, w2, 5)
    pd.testing.assert_frame_equal(again, stored.head(5), check_exact=True)
    assert np.percentile(stored.statistic, [2.5, 97.5]).tolist() == STATISTIC_PERCENTILES
    assert np.percentile(stored.combined_less_style, [2.5, 97.5]).tolist() == COMBINED_LESS_STYLE
