import pandas as pd

from fbrecruit.minutes import PLAYER_KEYS, TEAM_KEYS, aggregate, team_game_minutes


def lineups():
    rows = [
        # game, team, player, minutes, starter
        (1, 1, 10, 90, True),
        (1, 1, 11, 20, False),
        (1, 1, 12, 70, True),
        (1, 2, 20, 90, True),
        (2, 2, 10, 45, True),
        (2, 2, 20, 90, True),
        (2, 1, 11, 0, False),
        (2, 1, 12, 90, True),
    ]
    df = pd.DataFrame(
        rows, columns=["game_id", "team_id", "player_id", "minutes_played", "is_starter"]
    )
    return df.assign(league="liga", season="2015/16")


def test_player_team_rows_keep_teams_apart():
    out = aggregate(lineups(), TEAM_KEYS).set_index(["team_id", "player_id"])
    assert out.loc[(1, 10), "minutes"] == 90
    assert out.loc[(2, 10), "minutes"] == 45
    assert len(out) == 5


def test_substitute_counts_a_game_only_with_minutes():
    out = aggregate(lineups(), TEAM_KEYS).set_index(["team_id", "player_id"])
    sub = out.loc[(1, 11)]
    assert (sub.minutes, sub.games, sub.starts) == (20, 1, 0)


def test_player_who_switched_teams_is_summed():
    out = aggregate(lineups(), PLAYER_KEYS).set_index("player_id")
    mover = out.loc[10]
    assert (mover.minutes, mover.games, mover.starts) == (135, 2, 2)
    assert len(out) == 4


def test_team_game_minutes():
    out = team_game_minutes(lineups()).set_index(["game_id", "team_id"]).minutes_played
    assert out.loc[(1, 1)] == 180
    assert out.loc[(2, 1)] == 90
