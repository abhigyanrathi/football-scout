import sys

import pandas as pd

from fbrecruit.paths import INTERIM, PROCESSED

SEASONS = {"statsbomb": "2015/16", "wyscout": "2017/18"}
TEAM_KEYS = ["league", "season", "team_id", "player_id"]
PLAYER_KEYS = ["league", "season", "player_id"]


def load_lineups(provider):
    frames = []
    for path in sorted((INTERIM / provider).glob("*/lineups.parquet")):
        league = path.parent.name
        teams = pd.read_parquet(path.parent / "teams.parquet")[["team_id", "team_name"]]
        df = pd.read_parquet(path).merge(teams, on="team_id", how="left")
        frames.append(df.assign(league=league, season=SEASONS[provider]))
    return pd.concat(frames, ignore_index=True)


def aggregate(lineups, keys):
    """Sum lineup rows per key. A game counts only if the player was on the pitch."""
    rows = lineups.assign(played=lineups.minutes_played > 0)
    return rows.groupby(keys, as_index=False).agg(
        minutes=("minutes_played", "sum"),
        games=("played", "sum"),
        starts=("is_starter", "sum"),
    )


def team_game_minutes(lineups):
    return lineups.groupby(["league", "game_id", "team_id"], as_index=False).minutes_played.sum()


def build(provider):
    lineups = load_lineups(provider)
    labels = lineups.drop_duplicates(TEAM_KEYS)[[*TEAM_KEYS, "team_name", "player_name"]]
    by_team = aggregate(lineups, TEAM_KEYS).merge(labels, on=TEAM_KEYS)
    by_player = aggregate(lineups, PLAYER_KEYS)
    names = lineups.drop_duplicates(PLAYER_KEYS)[[*PLAYER_KEYS, "player_name"]]
    by_player = by_player.merge(names, on=PLAYER_KEYS)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    by_team.to_parquet(PROCESSED / f"minutes_player_team_{provider}.parquet", index=False)
    by_player.to_parquet(PROCESSED / f"minutes_player_{provider}.parquet", index=False)
    return lineups, by_team, by_player


def report(provider, lineups, by_team, by_player):
    print(f"== {provider}")
    print("league, player-team rows, players with minutes, players with 450+ minutes")
    for league, g in by_player.groupby("league"):
        rows = int((by_team.league == league).sum())
        print(league, rows, int((g.minutes > 0).sum()), int((g.minutes >= 450).sum()))
    print("league, min, median, max of team minutes per game")
    tg = team_game_minutes(lineups)
    for league, g in tg.groupby("league"):
        m = g.minutes_played
        print(league, m.min(), m.median(), m.max())
    ordered = tg.sort_values(["minutes_played", "league", "game_id", "team_id"])
    print("10 lowest team-games\n", ordered.head(10).to_string(index=False))
    print("10 highest team-games\n", ordered.tail(10).iloc[::-1].to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for provider in SEASONS:
        report(provider, *build(provider))
