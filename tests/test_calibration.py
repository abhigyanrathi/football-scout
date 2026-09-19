import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from fbrecruit.actionvalue import labels_path, pred_path
from fbrecruit.calibration import (
    CALIBRATORS_PATH,
    FOLDS_PATH,
    apply,
    assign_folds,
    fit_calibrator,
    load_calibrators,
    raw_rows,
    spread,
)
from fbrecruit.paths import PROCESSED

LEAGUES = ["la_liga", "premier_league", "serie_a", "ligue_1"]
FOLDS = {
    # league: [(games, actions) for folds 0 to 4]
    "la_liga": [(46, 90961), (46, 91747), (46, 90602), (46, 92711), (46, 89208)],
    "premier_league": [(46, 92120), (46, 90898), (46, 90305), (46, 90193), (46, 92906)],
    "serie_a": [(46, 89272), (46, 92653), (46, 92378), (46, 91837), (46, 93803)],
    "ligue_1": [(46, 96082), (46, 91854), (46, 90274), (45, 91011), (45, 91927)],
}
TRAINING_ROWS = [1464307, 1465590, 1469183, 1466990, 1464898]
CALIBRATORS = {
    # (branch, league, label): (a, b)
    ("pooled", "la_liga", "scores"): (-0.5250137788485826, 0.8578776438160345),
    ("pooled", "la_liga", "concedes"): (-1.5006892166128265, 0.6949964440130489),
    ("pooled", "premier_league", "scores"): (-0.6496121010085096, 0.8381593731005273),
    ("pooled", "premier_league", "concedes"): (-1.0935181112316485, 0.7888390716817599),
    ("pooled", "serie_a", "scores"): (-0.6108818824046965, 0.8431895366629677),
    ("pooled", "serie_a", "concedes"): (-0.9603274331192855, 0.8118541177203975),
    ("pooled", "ligue_1", "scores"): (-0.5637963923673313, 0.8656404375260744),
    ("pooled", "ligue_1", "concedes"): (-1.1873288672967108, 0.7728593752611848),
    ("lolo", "la_liga", "scores"): (-0.5384937998466297, 0.8525135760938578),
    ("lolo", "la_liga", "concedes"): (-1.5257815752727508, 0.6827598969003836),
    ("lolo", "premier_league", "scores"): (-0.6276237870382227, 0.8440125941526456),
    ("lolo", "premier_league", "concedes"): (-1.0716877793159576, 0.7943194396817063),
    ("lolo", "serie_a", "scores"): (-0.6268632392856386, 0.8355720101103244),
    ("lolo", "serie_a", "concedes"): (-1.0095350055745802, 0.8020452344652218),
    ("lolo", "ligue_1", "scores"): (-0.608351404603787, 0.8563653601926867),
    ("lolo", "ligue_1", "concedes"): (-1.3714682214853633, 0.739344919711209),
}
POOLED_W2_BRIER = {"scores": 0.009091654208855271, "concedes": 0.002197706528734231}
TABLE_ROWS = {
    # league: (window 1, window 2), the same for both calibrated tables
    "la_liga": (502, 464),
    "premier_league": (491, 477),
    "serie_a": (537, 474),
    "ligue_1": (543, 487),
}
MEAN_PER90 = {
    # branch: {league: (window 1, window 2)}, players with 450+ season minutes
    "pooled": {
        "la_liga": (0.1566858882027673, 0.17241551804162902),
        "premier_league": (0.15863140071393886, 0.17405360761208097),
        "serie_a": (0.15267712009745044, 0.14485540485265888),
        "ligue_1": (0.146919224825044, 0.13846983607513771),
    },
    "lolo": {
        "la_liga": (0.16039866525701188, 0.17640300297551925),
        "premier_league": (0.16105311178209478, 0.17855271986869833),
        "serie_a": (0.14980800706951922, 0.14630052197025953),
        "ligue_1": (0.1527755293253843, 0.14599122464773118),
    },
}


def need(path):
    if not path.exists():
        pytest.skip(f"local data cache missing: {path}")
    return path


def schedule():
    rows = []
    for league, days, per_day, first_id in (("a", 4, 3, 100), ("b", 3, 3, 200)):
        for day in range(1, days + 1):
            for j in range(per_day):
                date = pd.Timestamp("2015-08-01") + pd.Timedelta(days=7 * day + j)
                rows.append((league, first_id + 10 * day + j, day, date))
    g = pd.DataFrame(rows, columns=["league", "game_id", "game_day", "game_date"])
    return g.sample(frac=1, random_state=0)


def test_every_game_gets_exactly_one_fold():
    g = schedule()
    out = assign_folds(g)
    assert len(out) == len(g) and not out.duplicated(["league", "game_id"]).any()
    assert set(out.game_id) == set(g.game_id)
    assert set(out[out.league == "a"].fold) == {0, 1, 2, 3, 4}


def test_fold_sizes_within_a_league_differ_by_at_most_one():
    sizes = assign_folds(schedule()).groupby("league").fold.value_counts()
    for league in ("a", "b"):
        assert sizes[league].max() - sizes[league].min() <= 1


def test_folds_follow_matchday_order_and_repeat():
    g = schedule()
    out = assign_folds(g).sort_values("game_id").reset_index(drop=True)
    again = assign_folds(g.sample(frac=1, random_state=1)).sort_values("game_id")
    assert out.equals(again.reset_index(drop=True))
    a = out[out.league == "a"].set_index("game_id").fold
    assert a[110] == 0 and a[111] == 1 and a[112] == 2 and a[120] == 3 and a[121] == 4
    assert a[122] == 0


def synthetic(a, b, n=1_000_000):
    rng = np.random.default_rng(0)
    x = rng.normal(-4, 1.5, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(a + b * x))))
    return 1 / (1 + np.exp(-x)), y


def test_calibrator_recovers_intercept_and_slope():
    p, y = synthetic(-0.5, 0.8)
    c = fit_calibrator(p, y, np.ones(len(p), dtype=int))
    assert abs(c["a"] - -0.5) < 0.1 and abs(c["b"] - 0.8) < 0.1


def test_identity_calibrator_returns_the_input():
    p = np.random.default_rng(0).uniform(1e-6, 1 - 1e-6, 1000)
    assert np.abs(apply({"a": 0.0, "b": 1.0}, p) - p).max() < 1e-12


def test_calibrator_refuses_window_two_rows():
    p, y = synthetic(-0.5, 0.8, n=1000)
    window = np.ones(len(p), dtype=int)
    window[-1] = 2
    with pytest.raises(ValueError, match="window-1"):
        fit_calibrator(p, y, window)


def test_calibrator_stops_on_a_slope_that_is_not_positive():
    p, y = synthetic(-0.5, -0.8)
    with pytest.raises(ValueError, match="not positive"):
        fit_calibrator(p, y, np.ones(len(p), dtype=int))


@pytest.mark.slow
def test_fold_games_actions_and_training_rows():
    folds = pd.read_parquet(need(FOLDS_PATH))
    actions = pd.concat(
        [
            pd.read_parquet(need(labels_path(lg)), columns=["game_id"]).assign(league=lg)
            for lg in LEAGUES
        ]
    )
    t = actions.merge(folds, on=["league", "game_id"])
    games = folds.groupby(["league", "fold"]).size()
    rows = t.groupby(["league", "fold"]).size()
    for lg, expected in FOLDS.items():
        assert [(games[(lg, k)], rows[(lg, k)]) for k in range(5)] == expected
    per_fold = rows.groupby("fold").sum()
    assert (per_fold.sum() - per_fold).tolist() == TRAINING_ROWS


@pytest.mark.slow
def test_calibrator_intercepts_and_slopes():
    need(CALIBRATORS_PATH)
    cals = load_calibrators()
    assert {key: (c["a"], c["b"]) for key, c in cals.items()} == CALIBRATORS


@pytest.mark.slow
@pytest.mark.parametrize("label", ["scores", "concedes"])
def test_pooled_calibrated_window_2_brier(label):
    need(CALIBRATORS_PATH)
    for lg in LEAGUES:
        need(pred_path("pooled", lg))
    cals = load_calibrators()
    ys, ps = [], []
    for lg in LEAGUES:
        p, y = raw_rows("pooled", lg, 2)
        ys.append(y[label].to_numpy().astype(int))
        ps.append(apply(cals[("pooled", lg, label)], p[f"p_{label}"]))
    assert brier_score_loss(np.concatenate(ys), np.concatenate(ps)) == POOLED_W2_BRIER[label]


@pytest.mark.slow
@pytest.mark.parametrize("branch", ["pooled", "lolo"])
def test_calibrated_table_rows_and_means(branch):
    table = pd.read_parquet(need(PROCESSED / f"player_window_vaep_{branch}_cal.parquet"))
    counts = table.groupby(["league", "window"]).size()
    for lg, (w1, w2) in TABLE_ROWS.items():
        assert (counts[(lg, 1)], counts[(lg, 2)]) == (w1, w2)
    players = pd.read_parquet(need(PROCESSED / "minutes_player_statsbomb.parquet"))
    regular = players[players.minutes >= 450][["league", "player_id"]]
    means = spread(table, regular)["mean"]
    for lg, (w1, w2) in MEAN_PER90[branch].items():
        assert (means[(lg, 1)], means[(lg, 2)]) == (w1, w2)
