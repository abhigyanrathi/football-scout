import re
import sys

import pandas as pd
from socceraction.spadl import config

from fbrecruit.minutes import SEASONS, load_lineups, team_game_minutes
from fbrecruit.paths import INTERIM

NAME_COLUMNS = ["player_name", "nickname", "firstname", "lastname"]
BAD_PATTERN = "|".join(re.escape(c) for c in ("Ã", "Â", "\\"))
RED = config.results.index("red_card")


def duplicate_rows(lineups):
    return lineups[lineups.duplicated(["game_id", "team_id", "player_id"], keep=False)]


def suspicious_names(lineups):
    rows = []
    for col in [c for c in NAME_COLUMNS if c in lineups.columns]:
        bad = lineups[lineups[col].fillna("").str.contains(BAD_PATTERN)]
        rows.append(
            bad[["league", "game_id", "team_id", "player_id"]].assign(column=col, value=bad[col])
        )
    return pd.concat(rows, ignore_index=True)


def actions(provider, league):
    return pd.read_parquet(INTERIM / provider / league / "actions.parquet")


def lowest_with_red_cards(provider, lineups):
    tg = team_game_minutes(lineups)
    low = tg.sort_values(["minutes_played", "league", "game_id", "team_id"]).head(10)
    rows = []
    for r in low.itertuples():
        a = actions(provider, r.league)
        a = a[(a.game_id == r.game_id) & (a.team_id == r.team_id)]
        rows.append(
            (r.league, r.game_id, r.team_id, r.minutes_played, int((a.result_id == RED).sum()))
        )
    return pd.DataFrame(rows, columns=["league", "game_id", "team_id", "minutes", "red_cards"])


def highest_with_last_action(provider, lineups):
    tg = team_game_minutes(lineups)
    high = tg.sort_values(["minutes_played", "league", "game_id", "team_id"]).tail(10).iloc[::-1]
    rows = []
    for r in high.itertuples():
        a = actions(provider, r.league)
        last = a[a.game_id == r.game_id].groupby("period_id").time_seconds.max() / 60
        rows.append(
            (
                r.league,
                r.game_id,
                r.team_id,
                r.minutes_played,
                last.round(1).to_dict(),
                round(11 * last.sum(), 0),
            )
        )
    return pd.DataFrame(
        rows, columns=["league", "game_id", "team_id", "minutes", "last_action_min", "11_x_sum"]
    )


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("red_card result id:", RED, "| SPADL results:", config.results)
    for provider in SEASONS:
        lineups = load_lineups(provider)
        print(f"\n== {provider}: {len(lineups)} lineup rows")
        dup = duplicate_rows(lineups)
        print("duplicate (game_id, team_id, player_id) rows:", len(dup))
        print(dup.head(20).to_string(index=False))
        names = suspicious_names(lineups)
        print("names containing Ã, Â or a backslash, per column:")
        print(names.column.value_counts().to_string())
        print(names.head(20).to_string(index=False))
        print("10 lowest team-games, with the team's red cards in the game:")
        print(lowest_with_red_cards(provider, lineups).to_string(index=False))
        print("10 highest team-games, last action time (minutes) per period:")
        print(highest_with_last_action(provider, lineups).to_string(index=False))


if __name__ == "__main__":
    main()
