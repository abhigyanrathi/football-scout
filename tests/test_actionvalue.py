import pandas as pd
import pytest

from fbrecruit import actionvalue
from fbrecruit.actionvalue import (
    COLUMNS,
    FITS,
    Rows,
    labels_path,
    per90,
    rating_fit,
    train_labels,
    window_minutes,
)
from fbrecruit.minutes import TEAM_KEYS, aggregate
from fbrecruit.paths import PROCESSED

LABEL_COUNTS = {
    # league: (rows, scores positives, concedes positives)
    "la_liga": (757310, 8354, 1782),
    "premier_league": (758434, 8272, 1602),
    "serie_a": (761745, 7890, 1620),
    "ligue_1": (766071, 7480, 1677),
}
BASE_RATES = {
    # league: (scores, concedes), as printed to six decimals
    "la_liga": (0.011031, 0.002353),
    "premier_league": (0.010907, 0.002112),
    "serie_a": (0.010358, 0.002127),
    "ligue_1": (0.009764, 0.002189),
}
TRAINING = {
    # fit: (window-1 rows, scores positives, concedes positives)
    "pooled": (1832742, 18984, 3928),
    "lolo_la_liga": (1377513, 14032, 2843),
    "lolo_premier_league": (1376320, 14119, 2996),
    "lolo_serie_a": (1372799, 14213, 2969),
    "lolo_ligue_1": (1371594, 14588, 2976),
}
TABLE_ROWS = {
    # league: (window 1, window 2), the same for the pooled and lolo tables
    "la_liga": (502, 464),
    "premier_league": (491, 477),
    "serie_a": (537, 474),
    "ligue_1": (543, 487),
}


def need(path):
    if not path.exists():
        pytest.skip(f"local data cache missing: {path}")
    return path


def lineups():
    rows = [
        # game, team, player, minutes, starter
        (1, 1, 10, 90, True),
        (2, 1, 10, 60, True),
        (3, 1, 10, 0, False),
        (3, 1, 11, 30, False),
        (4, 2, 10, 90, True),
        (4, 2, 11, 45, False),
    ]
    df = pd.DataFrame(
        rows, columns=["game_id", "team_id", "player_id", "minutes_played", "is_starter"]
    )
    return df.assign(league="liga", season="2015/16")


def wins():
    return pd.DataFrame({"league": "liga", "game_id": [1, 2, 3, 4], "window": [1, 1, 2, 2]})


def test_window_minutes_sum_to_season():
    by_window = window_minutes(lineups(), wins())
    total = by_window.groupby(TEAM_KEYS).minutes.sum()
    season = aggregate(lineups(), TEAM_KEYS).set_index(TEAM_KEYS).minutes
    assert total.sort_index().equals(season.sort_index())


def test_window_minutes_split_by_window():
    out = window_minutes(lineups(), wins()).set_index(["team_id", "player_id", "window"])
    assert out.loc[(1, 10, 1), "minutes"] == 150
    assert out.loc[(1, 10, 2), "minutes"] == 0
    assert out.loc[(1, 10, 2), "games"] == 0
    assert out.loc[(2, 10, 2), "minutes"] == 90


def test_per90_arithmetic_and_zero_minutes():
    df = pd.DataFrame({"minutes": [90, 45, 0, 270], "vaep_sum": [0.3, 0.3, 0.5, -0.6]})
    out = per90(df)
    assert list(out.minutes) == [90, 45, 270]
    assert out.vaep_per90.tolist() == pytest.approx([0.3, 0.6, -0.2])


def test_lolo_values_come_from_the_model_that_excluded_the_league():
    assert rating_fit("lolo", "serie_a") == "lolo_serie_a"
    assert rating_fit("pooled", "serie_a") == "pooled"


def feed(monkeypatch, labels):
    keys = pd.DataFrame({"game_id": [1, 1, 2], "action_id": [0, 1, 0]})
    x = pd.concat([keys, pd.DataFrame(0.0, index=keys.index, columns=COLUMNS)], axis=1)
    monkeypatch.setattr(actionvalue, "batches", lambda league, games: iter([(x, None)]))
    rows = Rows([("liga", 1)], labels)
    rows.reset()
    fed = {}
    return rows.next(lambda **kw: fed.update(kw)), fed


def test_rows_accepts_aligned_keys(monkeypatch):
    labels = pd.DataFrame(
        {"game_id": [1, 1, 2], "action_id": [0, 1, 0], "scores": [True, False, False]}
    )
    assert feed(monkeypatch, labels)[1]["label"].tolist() == [True, False, False]


def test_rows_rejects_misaligned_keys(monkeypatch):
    # The same positives in a different row order: a positive-count check would pass.
    labels = pd.DataFrame(
        {"game_id": [1, 1, 2], "action_id": [1, 0, 0], "scores": [True, False, False]}
    )
    with pytest.raises(AssertionError):
        feed(monkeypatch, labels)


def test_lolo_fits_exclude_their_league():
    for league in TABLE_ROWS:
        assert league not in FITS[f"lolo_{league}"]
    assert sorted(FITS["pooled"]) == sorted(TABLE_ROWS)


@pytest.mark.slow
@pytest.mark.parametrize("league", LABEL_COUNTS)
def test_label_base_rates(league):
    labels = pd.read_parquet(need(labels_path(league)))
    rows, scores, concedes = LABEL_COUNTS[league]
    assert (len(labels), labels.scores.sum(), labels.concedes.sum()) == (rows, scores, concedes)
    assert (round(labels.scores.mean(), 6), round(labels.concedes.mean(), 6)) == BASE_RATES[league]


@pytest.mark.slow
@pytest.mark.parametrize("fit", TRAINING)
def test_training_rows(fit):
    for league in FITS[fit]:
        need(labels_path(league))
    y = train_labels([(lg, 1) for lg in FITS[fit]])
    assert (len(y), y.scores.sum(), y.concedes.sum()) == TRAINING[fit]


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["pooled", "lolo"])
def test_player_window_table_rows(kind):
    table = pd.read_parquet(need(PROCESSED / f"player_window_vaep_{kind}.parquet"))
    counts = table.groupby(["league", "window"]).size()
    for league, (w1, w2) in TABLE_ROWS.items():
        assert (counts[(league, 1)], counts[(league, 2)]) == (w1, w2)
    assert (table.minutes > 0).all()
