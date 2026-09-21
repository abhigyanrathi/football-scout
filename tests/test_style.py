import pandas as pd
import pytest

from fbrecruit import minutes, style

GAMES = {"la_liga": 380, "premier_league": 380, "serie_a": 380, "ligue_1": 377}
PRESSURES = {"la_liga": 114081, "premier_league": 115402, "serie_a": 130888, "ligue_1": 130733}


def need(path):
    if not path.exists():
        pytest.skip(f"local data cache missing: {path}")
    return path


def pressures(league):
    return pd.read_parquet(need(style.OUT / league / "pressures.parquet"))


@pytest.mark.slow
@pytest.mark.parametrize("league", list(GAMES))
def test_games_read_and_pressure_events(league):
    summary = pd.read_parquet(need(style.OUT / league / "pressures_summary.parquet"))
    assert len(summary) == GAMES[league]
    assert summary.pressures.sum() == PRESSURES[league]
    assert len(pressures(league)) == PRESSURES[league]


@pytest.mark.slow
def test_every_pressure_has_a_lineup_row():
    lu = minutes.load_lineups("statsbomb")[["league", "game_id", "team_id", "player_id"]]
    lu = lu.astype({"team_id": "int64"}).drop_duplicates().assign(found=True)
    pr = pd.concat(pressures(lg).assign(league=lg) for lg in GAMES)
    assert pr.merge(lu, how="left").found.isna().sum() == 0


def test_pressed_player_is_the_first_related_event_of_the_other_team():
    events = pd.DataFrame(
        {
            "event_id": ["p1", "a", "b", "c", "p2", "p3"],
            "team_id": [1, 1, 2, 2, 1, 2],
            "player_id": [10.0, 11.0, 20.0, 21.0, 12.0, 22.0],
        }
    )
    pressures = pd.DataFrame(
        {
            "team_id": [1, 1, 2],
            "related_events": [["a", "c", "b"], ["a", "missing"], []],
        }
    )
    got = style.pressed_player(pressures, events)
    assert got.tolist() == [21, pd.NA, pd.NA]
