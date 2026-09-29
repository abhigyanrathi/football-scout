import numpy as np
import pandas as pd
import pytest

from fbrecruit import gems, squad
from fbrecruit import shrinkage as sh
from fbrecruit.paths import INTERIM, PROCESSED
from fbrecruit.sources.statsbomb import LEAGUES, OUT

# from p8_value_summary.log, p8_prices_summary.log and p8_scores_summary.log
ROWS = 1247
TAU2 = {
    "CB": 0.0012847875521513595,
    "FB": 0.0019273895949810335,
    "FWD": 0.007542038039772019,
    "MID": 0.002411942041151048,
    "WIDE": 0.007274907612556252,
}
REASONS = {
    "no Transfermarkt id": 13,
    "shared Transfermarkt id": 0,
    "no summer valuation": 67,
    "no date of birth": 1,
    "priced": 1166,
}
FIT_ROWS = {"gem": 1166, "gem_P2": 1166, "gem_stats": 1166, "momentum": 1130}
# from p8_outcomes_summary.log, p8_test_summary.log and p8_squads_summary.log
HOW = {
    1: {"window": 1129, "carried": 30, "price": 7},
    2: {"window": 1076, "carried": 87, "price": 3},
}
STATISTICS = {
    "gem_1": 0.06636542795227626,
    "P2_1": 0.038234886070613115,
    "stats_1": 0.07636985421896036,
    "momentum_1": 0.19589872275398498,
    "diff_1": 0.028130541881663147,
    "gem_2": 0.0911994618761879,
    "P2_2": 0.05239233100140419,
    "stats_2": 0.10324765040634543,
    "momentum_2": 0.15328439605277958,
    "diff_2": 0.038807130874783706,
    "low_1": 0.05026822572719912,
    "high_1": 0.12109371730783478,
}
GEM_1 = [0.006785671851836903, 0.12538372900239794]
PICK = [3372, 3404, 3625, 3672, 3674, 7770, 11342, 23349, 40367, 40427]
RATIO = 1.082810539523212


def need(path):
    if not path.exists():
        pytest.skip(f"local data cache missing: {path}")
    return path


def test_full_season_rate_weights_each_window_less_its_level():
    t = pd.DataFrame(
        {
            "league": "la_liga",
            "player_id": [1, 1, 2],
            "team_id": 10,
            "window": [1, 2, 2],
            "minutes": [600, 400, 900],
            "vaep_sum": [1.2, 0.4, 1.8],
        }
    )
    index = pd.MultiIndex.from_tuples([("la_liga", 1), ("la_liga", 2)], names=["league", "window"])
    c = pd.Series([0.1, 0.2], index=index)
    got = gems.full_season(t, c)
    # player 1: (90 * 1.6 - 600 * 0.1 - 400 * 0.2) / 1000; player 2 has window 2 only:
    # (90 * 1.8 - 900 * 0.2) / 900
    assert got.player_id.tolist() == [1, 2]
    assert got.minutes.tolist() == [1000, 900]
    assert np.allclose(got.y, [0.004, -0.02], rtol=0, atol=1e-12)


def made_up_decision():
    """Player 1 is priced, 2 has no summer valuation, 3 no date of birth, 4 shares his
    Transfermarkt id with player 6 and 5 has none; la_liga's last game is on 2016-05-15."""
    value = pd.DataFrame({"league": "la_liga", "player_id": [1, 2, 3, 4, 5], "team_id": 10})
    links = pd.DataFrame({"player_id": [1, 2, 3, 4, 6], "tm_player_id": [101, 102, 103, 104, 104]})
    v = pd.DataFrame(
        [
            (101, "2015-04-30", 1_000_000),
            (101, "2015-05-01", 1_500_000),
            (101, "2015-08-31", 2_000_000),
            (101, "2015-09-01", 2_500_000),
            (101, "2016-05-15", 7_000_000),
            (101, "2016-05-16", 5_000_000),
            (101, "2016-06-01", 6_000_000),
            (101, "2016-07-01", 0),
            (101, "2016-09-01", 9_000_000),
            (102, "2015-05-01", 800_000),
            (102, "2015-06-01", 0),
            (102, "2016-05-15", 3_000_000),
            (103, "2015-04-30", 1_200_000),
            (103, "2016-06-01", 2_000_000),
            (104, "2016-06-01", 4_000_000),
        ],
        columns=["player_id", "date", "market_value_in_eur"],
    )
    v["date"] = pd.to_datetime(v.date)
    dates = pd.to_datetime(["1990-07-01", "1994-01-01", "1992-01-01"])
    born = pd.DataFrame({"player_id": [101, 102, 104], "date_of_birth": dates})
    apps = pd.DataFrame(
        {
            "player_id": [101, 101, 102],
            "goals": [1, 1, 0],
            "assists": [1, 0, 0],
            "minutes_played": [90, 180, 0],
        }
    )
    last = pd.Series({"la_liga": pd.Timestamp("2016-05-15 19:30")})
    return value, links, v, born, apps, last


def test_price_is_the_latest_positive_valuation_after_the_last_game():
    got = gems.build_decision(*made_up_decision()).set_index("player_id")
    assert got.reason.to_dict() == {
        1: "priced",
        2: "no summer valuation",
        3: "no date of birth",
        4: "shared Transfermarkt id",
        5: "no Transfermarkt id",
    }
    # player 1: not the valuation on the last game's day, the zero, or the one after 2016-08-31
    assert got.loc[1, "price"] == 6_000_000
    assert got.loc[1, "price_date"] == pd.Timestamp("2016-06-01")
    assert got.loc[3, "price"] == 2_000_000
    assert got.price[[2, 4, 5]].isna().all()
    # 26 years with seven 29 Februaries
    assert got.loc[1, "age"] == 9497 / 365.25
    assert got.loc[1, "ga90"] == 1.0 and np.isnan(got.loc[2, "ga90"])


def test_value_2015_is_the_latest_positive_valuation_from_may_to_august_2015():
    got = gems.build_decision(*made_up_decision()).set_index("player_id")
    assert got.loc[1, "value_2015"] == 2_000_000
    assert got.loc[1, "value_2015_date"] == pd.Timestamp("2015-08-31")
    # the first day of the range counts and the zero after it does not
    assert got.loc[2, "value_2015"] == 800_000
    assert got.loc[2, "value_2015_date"] == pd.Timestamp("2015-05-01")
    # the day before the range does not count
    assert pd.isna(got.loc[3, "value_2015"])


def test_score_residuals_are_orthogonal_to_the_controls():
    rng = np.random.default_rng(0)
    n = 500
    rows = pd.DataFrame(
        {
            "league": rng.choice(list(LEAGUES), n),
            "group": rng.choice(sh.OUTFIELD, n),
            "price": pd.array(rng.integers(100_000, 100_000_001, n), dtype="Int64"),
            "age": rng.uniform(17.0, 38.0, n),
        }
    )
    x = gems.controls(rows)
    assert x.shape == (n, len(gems.CONTROLS))
    _, r, _ = gems.ols(rng.normal(0.0, 0.1, n), x)
    assert np.abs(x.T @ r).max() <= 1e-9


def test_one_row_per_player_keeps_the_row_with_more_minutes():
    rows = pd.DataFrame(
        {
            "league": ["la_liga", "la_liga", "serie_a"],
            "player_id": [1, 2, 1],
            "team_id": [10, 10, 20],
            "minutes": [1000, 950, 1500],
        }
    )
    got = gems.one_per_player(rows).sort_values("player_id")
    assert got[["player_id", "team_id", "minutes"]].to_numpy().tolist() == [
        [1, 20, 1500],
        [2, 10, 950],
    ]


def test_icc_matches_an_analysis_of_variance_by_hand():
    # teams of 2, 3 and 4 rows with means 2, 7 and 5 and a grand mean of 5: SSB 30 on 2 degrees
    # of freedom, SSW 2 + 8 + 20 = 30 on 6, k0 = (9 - 29 / 9) / 2 = 26 / 9, and
    # ICC = (15 - 5) / (15 + (26 / 9 - 1) 5) = 9 / 22
    rows = pd.DataFrame(
        {
            "league": "la_liga",
            "team_id": [1, 1, 2, 2, 2, 3, 3, 3, 3],
            "v": [1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 8.0],
        }
    )
    msb, msw, k, raw = gems.icc(rows, "v")
    assert abs(msb - 15.0) <= 1e-12 and abs(msw - 5.0) <= 1e-12
    assert abs(k - 26 / 9) <= 1e-12
    assert abs(raw - 9 / 22) <= 1e-12
    # equal team means: MSB 0, MSW (16 + 16 + 9 + 9) / 2 = 25, k0 (4 - 8 / 4) / 1 = 2, ICC -1
    rows = pd.DataFrame({"league": "la_liga", "team_id": [1, 1, 2, 2], "v": [1.0, 9.0, 2.0, 8.0]})
    assert gems.icc(rows, "v") == (0.0, 25.0, 2.0, -1.0)


def value_inputs():
    for lg in LEAGUES:
        need(OUT / lg / "teams.parquet")
    need(PROCESSED / "player_window_vaep_pooled_v2.parquet")
    need(sh.group_path())
    need(sh.shrinkage_path("pooled"))
    return gems.value_inputs()


def decision_inputs():
    for lg in LEAGUES:
        need(OUT / lg / "games.parquet")
    for name in ["player_valuations", "players", "appearances"]:
        need(INTERIM / "transfermarkt" / f"{name}.parquet")
    need(squad.links_path())
    need(gems.value_path())
    return gems.decision_inputs()


@pytest.mark.slow
def test_value_step_reproduces_the_stored_table_and_each_groups_tau2():
    stored = pd.read_parquet(need(gems.value_path()))
    got, fits = gems.build_value(*value_inputs())
    pd.testing.assert_frame_equal(got, stored, check_exact=True)
    assert len(got) == ROWS
    assert sorted(fits) == sorted(TAU2)
    for g, tau2 in TAU2.items():
        assert abs(fits[g]["tau2"] - tau2) <= 1e-12


@pytest.mark.slow
def test_prices_step_reproduces_the_stored_table_and_its_reasons():
    stored = pd.read_parquet(need(gems.decision_path()))
    got = gems.build_decision(*decision_inputs())
    pd.testing.assert_frame_equal(got, stored, check_exact=True)
    assert len(got) == ROWS
    assert got.reason.value_counts().reindex(gems.REASONS, fill_value=0).to_dict() == REASONS


@pytest.mark.slow
def test_scores_step_reproduces_the_stored_table_and_its_fits():
    stored = pd.read_parquet(need(gems.scores_path()))
    got, fits = gems.build_scores(pd.read_parquet(need(gems.decision_path())))
    pd.testing.assert_frame_equal(got, stored, check_exact=True)
    assert len(got) == ROWS
    assert {name: f["rows"] for name, f in fits.items()} == FIT_ROWS


@pytest.mark.slow
def test_no_valuation_after_the_summer_window_is_loaded():
    need(INTERIM / "transfermarkt" / "player_valuations.parquet")
    assert gems.valuations().date.max() <= pd.Timestamp("2016-08-31")


def outcome_inputs():
    need(gems.scores_path())
    need(INTERIM / "transfermarkt" / "player_valuations.parquet")
    return gems.priced_rows(), gems.later_valuations()


def with_outcomes():
    need(gems.scores_path())
    need(gems.outcomes_path())
    return gems.with_outcomes()


def squad_rows():
    need(gems.scores_path())
    need(gems.outcomes_path())
    return gems.squad_rows()


@pytest.mark.slow
def test_outcomes_step_reproduces_the_stored_table_and_its_counts():
    stored = pd.read_parquet(need(gems.outcomes_path()))
    got = gems.build_outcomes(*outcome_inputs())
    pd.testing.assert_frame_equal(got, stored, check_exact=True)
    for h, counts in HOW.items():
        assert got[f"how_{h}"].value_counts().reindex(gems.HOW, fill_value=0).to_dict() == counts


@pytest.mark.slow
def test_full_data_statistics_and_the_first_redraws_match_the_stored_ones():
    rows = with_outcomes()
    got = gems.statistics(rows)
    assert list(got) == list(STATISTICS)
    for s, value in STATISTICS.items():
        assert abs(got[s] - value) <= 1e-12
    stored = pd.read_parquet(need(gems.redraws_path()))
    pd.testing.assert_frame_equal(gems.redraw_table(rows, 5), stored.head(5), check_exact=True)
    assert np.percentile(stored.gem_1, [2.5, 97.5]).tolist() == GEM_1


@pytest.mark.slow
def test_pick_and_the_first_random_squads_match_the_stored_ones():
    rows = squad_rows()
    pos = squad.pick(rows.P3_full, rows.price, rows.group, rows.player_id, gems.BUDGET)
    assert rows.player_id.to_numpy()[pos].tolist() == PICK
    price, later = rows.price.to_numpy(), rows.value_1.to_numpy()
    assert abs(later[pos].sum() / price[pos].sum() - RATIO) <= 1e-12
    stored = pd.read_parquet(need(gems.squads_path()))
    again = gems.random_squads(rows, gems.BUDGET, 20)
    pd.testing.assert_frame_equal(again, stored.head(20), check_exact=True)
