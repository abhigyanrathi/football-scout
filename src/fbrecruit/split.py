import math
import sys

import pandas as pd

from fbrecruit.paths import INTERIM
from fbrecruit.sources.statsbomb import LEAGUES

W1_SHARE = 0.6


def games(league):
    return pd.read_parquet(INTERIM / "statsbomb" / league / "games.parquet")


def boundary(max_game_day, share=W1_SHARE):
    return math.ceil(share * max_game_day)


def assign_windows(games):
    """Window 1 is game_day up to ceil(0.6 * max game_day), window 2 the rest.

    Games with a null game_day get no window (NA) and are excluded downstream.
    """
    b = boundary(games.game_day.max())
    window = pd.Series(pd.NA, index=games.index, dtype="Int64")
    known = games.game_day.notna()
    window[known] = (games.game_day[known] > b).astype(int) + 1
    return games.assign(window=window)


def windows():
    """game_id to window for all four leagues, with the league."""
    frames = [assign_windows(games(lg)).assign(league=lg) for lg in LEAGUES]
    return pd.concat(frames, ignore_index=True)[["league", "game_id", "game_day", "window"]]


def date_violations(g):
    w1, w2 = g[g.window == 1], g[g.window == 2]
    first_w2, last_w1 = w2.game_date.min(), w1.game_date.max()
    late_w1 = w1[w1.game_date > first_w2]
    early_w2 = w2[w2.game_date < last_w1]
    largest = max(
        (late_w1.game_date - first_w2).max() if len(late_w1) else pd.Timedelta(0),
        (last_w1 - early_w2.game_date).max() if len(early_w2) else pd.Timedelta(0),
    )
    return {
        "w1_after_first_w2": len(late_w1),
        "w2_before_last_w1": len(early_w2),
        "w1_dates": f"{w1.game_date.min():%Y-%m-%d} to {last_w1:%Y-%m-%d}",
        "w2_dates": f"{first_w2:%Y-%m-%d} to {w2.game_date.max():%Y-%m-%d}",
        "largest_days": round(largest.total_seconds() / 86400, 2),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("1a: league, min game_day, max game_day, distinct game_day, null game_day")
    for lg in LEAGUES:
        d = games(lg).game_day
        print(lg, d.min(), d.max(), d.nunique(), int(d.isna().sum()))

    rows, viol = [], []
    for lg in LEAGUES:
        g = assign_windows(games(lg))
        ids = pd.read_parquet(INTERIM / "statsbomb" / lg / "actions.parquet", columns=["game_id"])
        n = ids.game_id.map(g.set_index("game_id").window)
        gw, aw = g.window.value_counts(), n.value_counts()
        rows.append(
            {
                "league": lg,
                "boundary": boundary(g.game_day.max()),
                "max_day": g.game_day.max(),
                "games_w1": gw.get(1, 0),
                "games_w2": gw.get(2, 0),
                "games_none": int(g.window.isna().sum()),
                "actions_w1": aw.get(1, 0),
                "actions_w2": aw.get(2, 0),
                "actions_none": int(n.isna().sum()),
                "w1_share_games": round(gw.get(1, 0) / gw.sum(), 4),
                "w1_share_actions": round(aw.get(1, 0) / aw.sum(), 4),
            }
        )
        viol.append({"league": lg, **date_violations(g)})
    print("\n1b/1c: windows")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n1d: date-order violations")
    print(pd.DataFrame(viol).to_string(index=False))


if __name__ == "__main__":
    main()
