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
