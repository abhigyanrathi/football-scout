import itertools

import numpy as np
import pandas as pd
import pytest

from fbrecruit import shrinkage as sh
from fbrecruit import squad
from fbrecruit.sources.statsbomb import LEAGUES, OUT

SIZES = {"CB": 3, "FB": 3, "MID": 4, "WIDE": 3, "FWD": 2}
# from p7_pool_summary.log and p7_budgets_summary.log
PAIR_REASONS = {
    "no Transfermarkt id": 9,
    "shared Transfermarkt id": 0,
    "no valuation": 4,
    "priced": 1054,
}
ROW_REASONS = {
    "no Transfermarkt id": 14,
    "shared Transfermarkt id": 0,
    "no valuation": 5,
    "priced": 1280,
}
CUTOFF = pd.Timestamp("2016-02-02")
BUDGETS = {0.25: 15_000_000, 0.5: 35_000_000, 0.75: 90_000_000}
CHEAPEST = 2_050_000
# from p7_full_summary.log
HEADLINE_SQUADS = {
    "P3": [3248, 3404, 3625, 3814, 5656, 6732, 7475, 23349, 40367, 40427],
    "P2": [3404, 3990, 4597, 5554, 5656, 6847, 19299, 23349, 40247, 40427],
}
HEADLINE_ACTUAL = {"P3": 1.3033749390409892, "P2": 0.4944632023913539}
RECOMMENDED = [3204, 3404, 3625, 3814, 5656, 6732, 7475, 23349, 40367, 40427]
RANDOM_BELOW = {15_000_000: 9305, 35_000_000: 9988, 90_000_000: 9834}
# from p7_tally_summary.log
WINS = {15_000_000: 1154, 35_000_000: 1339, 90_000_000: 1085}
IDENTICAL = {15_000_000: 1, 35_000_000: 0, 90_000_000: 0}


def need(path):
    if not path.exists():
        pytest.skip(f"local data cache missing: {path}")
    return path


def made_up(seed):
    """SIZES rows of each group, one player each, values from a standard normal and prices in
    whole euros."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(list(SIZES), list(SIZES.values()))
    values = rng.normal(0.0, 1.0, len(groups))
    prices = rng.integers(500_000, 50_000_001, len(groups))
    return values, prices, groups, np.arange(len(groups))


def every_squad(groups):
    """Every squad the quotas allow, as ascending positions."""
    per = [itertools.combinations(np.flatnonzero(groups == g), k) for g, k in squad.QUOTAS.items()]
    return [tuple(sorted(itertools.chain(*c))) for c in itertools.product(*per)]


def cost_range(prices, groups):
    costs = [prices[list(s)].sum() for s in every_squad(groups)]
    return min(costs), max(costs)


@pytest.mark.parametrize("seed", range(5))
def test_pick_matches_brute_force(seed):
    values, prices, groups, players = made_up(seed)
    squads = every_squad(groups)
    low, high = cost_range(prices, groups)
    for budget in [low, high, (low + high) // 2]:
        feasible = [s for s in squads if prices[list(s)].sum() <= budget]
        best = sorted(((values[list(s)].sum(), s) for s in feasible), reverse=True)
        got = squad.pick(values, prices, groups, players, budget)
        assert abs(values[got].sum() - best[0][0]) <= 1e-9
        if len(best) == 1 or best[0][0] - best[1][0] > 1e-6:
            assert tuple(got) == best[0][1]


def test_a_player_with_two_rows_is_picked_once():
    values = np.array([5.0, 5.0, 1.0, 0.5])
    groups = np.array(["CB", "FB", "CB", "FB"])
    players = np.array([7, 7, 8, 9])
    quotas = {"CB": 1, "FB": 1}
    got = squad.pick(values, np.full(4, 1_000_000), groups, players, 10_000_000, quotas)
    # both of player 7's rows would be worth 10; taking him once, his FB row with 8 is worth 6
    assert got.tolist() == [1, 2]


def test_a_budget_below_the_cheapest_squad_raises():
    values, prices, groups, players = made_up(0)
    with pytest.raises(RuntimeError):
        squad.pick(values, prices, groups, players, squad.cheapest_cost(prices, groups) - 1)


def test_pick_returns_the_same_squad_twice():
    values, prices, groups, players = made_up(0)
    low, high = cost_range(prices, groups)
    first = squad.pick(values, prices, groups, players, (low + high) // 2)
    assert np.array_equal(first, squad.pick(values, prices, groups, players, (low + high) // 2))


def test_fill_completes_at_the_cheapest_squads_cost():
    _, prices, groups, players = made_up(0)
    budget = squad.cheapest_cost(prices, groups)
    rng = np.random.default_rng(0)
    for _ in range(200):
        got = squad.fill(rng.permutation(len(groups)), prices, groups, players, budget)
        assert {g: int((groups[got] == g).sum()) for g in squad.QUOTAS} == squad.QUOTAS
        assert prices[got].sum() <= budget


def test_fill_by_value_skips_the_row_that_leaves_too_little():
    values = np.array([3.0, 2.0, 1.5, 1.0, 0.5, 0.2])
    prices = np.array([9, 3, 2, 4, 1, 1]) * 1_000_000
    groups = np.array(["FWD", "MID", "CB", "FWD", "MID", "CB"])
    quotas = {"CB": 1, "MID": 1, "FWD": 1}
    order = np.argsort(-values)
    # the 9m forward would leave 1m for a midfielder and a centre-back, who cost 2m at least
    got = squad.fill(order, prices, groups, np.arange(6), 10_000_000, quotas)
    assert got.tolist() == [1, 2, 3]


@pytest.mark.slow
def test_pool_reasons_cutoff_budgets_and_cheapest_squad():
    for lg in LEAGUES:
        need(OUT / lg / "games.parquet")
    pool = pd.read_parquet(need(squad.pool_path()))
    for part, expected in [(pool[pool.pair], PAIR_REASONS), (pool, ROW_REASONS)]:
        assert part.reason.value_counts().reindex(squad.REASONS, fill_value=0).to_dict() == expected
    assert squad.cutoff(squad.first_window2()) == CUTOFF
    priced = squad.priced_pairs(pool)
    assert squad.budgets(priced.price) == BUDGETS
    assert squad.cheapest_cost(priced.price, priced.group) == CHEAPEST


@pytest.mark.slow
def test_pool_has_no_window_two_column_and_the_pairs_values():
    pool = pd.read_parquet(need(squad.pool_path()))
    assert not {"target", "vaep_per90_w2", "minutes_w2"} & set(pool.columns)
    cols = [*sh.KEYS, "group", "P2", "P3"]
    ev = pd.read_parquet(need(sh.evaluation_path("pooled")), columns=cols)
    ev = ev[ev.group.isin(sh.OUTFIELD)]
    m = ev.merge(pool, on=sh.KEYS, how="left", suffixes=("", "_pool"), validate="one_to_one")
    assert len(m) == 1067 and m.pair.eq(True).all() and pool.pair.sum() == 1067
    assert (m.P2 == m.P2_pool).all() and (m.P3 == m.P3_pool).all()


def full_inputs():
    pool = pd.read_parquet(need(squad.pool_path()))
    need(sh.evaluation_path("pooled"))
    return squad.with_target(squad.priced_pairs(pool)), squad.priced_rows(pool)


@pytest.mark.slow
def test_full_pool_squads_are_the_stored_ones():
    stored = pd.read_parquet(need(squad.full_path()))
    pairs, rows = full_inputs()
    got = squad.squad_table(pairs, rows, squad.full_squads(pairs, rows))
    pd.testing.assert_frame_equal(got, stored, check_exact=True)
    headline = got[got.budget == squad.BUDGETS[squad.HEADLINE]]
    for kind, ids in HEADLINE_SQUADS.items():
        players = headline[headline.kind == kind]
        assert sorted(players.player_id) == ids
        assert abs(players.target.to_numpy().sum() - HEADLINE_ACTUAL[kind]) <= 1e-12
    assert sorted(headline[headline.kind == "recommended"].player_id) == RECOMMENDED


@pytest.mark.slow
def test_random_squads_below_the_p3_squad_and_the_first_twenty_rebuilt():
    stored = pd.read_parquet(need(squad.random_path()))
    full = pd.read_parquet(need(squad.full_path()))
    assert len(stored) == 30_000
    for b, below in RANDOM_BELOW.items():
        p3 = full[(full.kind == "P3") & (full.budget == b)].target.to_numpy().sum()
        assert (stored[stored.budget == b].actual < p3).sum() == below
    pairs, _ = full_inputs()
    b = squad.BUDGETS[squad.HEADLINE]
    first = stored[(stored.budget == b) & (stored.draw < 20)].reset_index(drop=True)
    pd.testing.assert_frame_equal(squad.random_squads(pairs, b, 20), first, check_exact=True)


@pytest.mark.slow
def test_redraw_counts_and_redraws_zero_to_two_solved_again():
    stored = pd.read_parquet(need(squad.redraws_path()))
    assert len(stored) == 12_000
    counts = squad.outcomes(stored)
    assert counts.wins.to_dict() == WINS and counts.identical.to_dict() == IDENTICAL
    pairs, _ = full_inputs()
    draws = pd.read_parquet(need(squad.draws_path()), columns=["replicate", "league", "team_id"])
    got = squad.redraw_table(range(3), pairs, draws)
    want = stored[stored.replicate < 3].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_exact=True)
