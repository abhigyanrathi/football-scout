import pandas as pd
import pytest

from fbrecruit.minutes import TEAM_KEYS, aggregate, load_lineups
from fbrecruit.paths import INTERIM, RAW

pytestmark = pytest.mark.slow

TRANSFERMARKT = {
    # table: (rows, columns, bytes)
    "competitions": (65, 11, 2242),
    "clubs": (796, 17, 51281),
    "players": (50149, 26, 4389958),
    "games": (88958, 23, 4995595),
    "appearances": (1894350, 13, 45253638),
    "player_valuations": (656301, 6, 7197935),
    "transfers": (175165, 10, 5809514),
}
STATSBOMB = {
    # league: (games, player-team rows, lineup minutes)
    "la_liga": (380, 546, 780443),
    "premier_league": (380, 561, 799881),
    "serie_a": (380, 584, 786907),
    "ligue_1": (377, 588, 778295),
}
WYSCOUT_GAMES = {"italy": 380, "england": 380, "spain": 380, "france": 380, "germany": 306}
SPAIN_FIRST_TEN = [2565922, 2565925, 2565919, 2565924, 2565927]
SPAIN_FIRST_TEN += [2565920, 2565921, 2565923, 2565926, 2565918]


def need(path):
    if not path.exists():
        pytest.skip(f"local data cache missing: {path}")
    return path


def table(provider, league, name):
    return pd.read_parquet(need(INTERIM / provider / league / f"{name}.parquet"))


@pytest.mark.parametrize("name", TRANSFERMARKT)
def test_transfermarkt(name):
    rows, cols, size = TRANSFERMARKT[name]
    df = pd.read_parquet(need(INTERIM / "transfermarkt" / f"{name}.parquet"))
    assert df.shape == (rows, cols)
    assert need(RAW / "transfermarkt" / f"{name}.csv.gz").stat().st_size == size


@pytest.mark.parametrize("league", STATSBOMB)
def test_statsbomb_league(league):
    games, player_team_rows, minutes = STATSBOMB[league]
    assert len(table("statsbomb", league, "games")) == games
    lineups = load_lineups("statsbomb").query("league == @league")
    assert len(aggregate(lineups, TEAM_KEYS)) == player_team_rows
    assert lineups.minutes_played.sum() == minutes


def test_statsbomb_totals():
    lineups = load_lineups("statsbomb")
    assert sum(len(table("statsbomb", lg, "games")) for lg in STATSBOMB) == 1517
    assert len(lineups) == 42096
    assert lineups.minutes_played.sum() == 3145526


def test_statsbomb_la_liga_events_and_actions():
    assert table("statsbomb", "la_liga", "games").n_events.sum() == 1295354
    assert len(table("statsbomb", "la_liga", "actions")) == 757310


def test_statsbomb_no_failures():
    for league in STATSBOMB:
        failures = pd.read_csv(need(INTERIM / "statsbomb" / league / "failures.csv"))
        assert failures.empty


@pytest.mark.parametrize("league", WYSCOUT_GAMES)
def test_wyscout_games(league):
    assert len(table("wyscout", league, "games")) == WYSCOUT_GAMES[league]


def test_wyscout_totals():
    lineups = load_lineups("wyscout")
    assert len(lineups) == 50590
    assert lineups.minutes_played.sum() == 3804657
    per_player = aggregate(lineups, ["player_id"])
    assert len(per_player) == 2571
    assert (per_player.minutes >= 450).sum() == 1974


def test_wyscout_spain_first_ten_games():
    games = table("wyscout", "spain", "games")
    first = games.game_id.head(10).tolist()
    assert first == SPAIN_FIRST_TEN
    events = games[games.game_id.isin(first)].n_events.sum()
    actions = table("wyscout", "spain", "actions").query("game_id in @first")
    assert events == 16697
    assert len(actions) == 12791
