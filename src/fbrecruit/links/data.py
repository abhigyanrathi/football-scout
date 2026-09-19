import pandas as pd

from fbrecruit.minutes import TEAM_KEYS, aggregate, load_lineups
from fbrecruit.paths import INTERIM, PROCESSED

LINKS = PROCESSED / "links"


def tm_table(name, columns=None):
    return pd.read_parquet(INTERIM / "transfermarkt" / f"{name}.parquet", columns=columns)


def tm_appearances():
    games = tm_table("games", ["game_id", "season"])
    return tm_table("appearances").merge(games, on="game_id", how="left")


def minutes_table(name):
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def statsbomb_players():
    """One row per StatsBomb player-team: first non-null names in game order, and minutes."""
    lineups = load_lineups("statsbomb")
    dates = pd.concat(
        pd.read_parquet(p, columns=["game_id", "game_date"])
        for p in sorted((INTERIM / "statsbomb").glob("*/games.parquet"))
    )
    ordered = lineups.merge(dates, on="game_id").sort_values(
        ["game_date", "game_id"], kind="stable"
    )
    keys = ["league", "team_id", "player_id"]
    names = ordered.groupby(keys, as_index=False)[["team_name", "player_name", "nickname"]].first()
    minutes = aggregate(lineups, TEAM_KEYS)[[*keys, "minutes"]]
    return names.merge(minutes, on=keys)
