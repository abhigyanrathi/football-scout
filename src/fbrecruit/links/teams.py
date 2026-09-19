import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from fbrecruit.links import data
from fbrecruit.links.text import norm, norm_club, ratio
from fbrecruit.paths import INTERIM

COMPETITIONS = {"la_liga": "ES1", "premier_league": "GB1", "serie_a": "IT1", "ligue_1": "FR1"}
SEASON = 2015


def assign(sim):
    """Column chosen for each row so that the total similarity is largest and no column repeats."""
    _, cols = linear_sum_assignment(sim, maximize=True)
    return cols


def club_names(games, competition, season):
    g = games[(games.competition_id == competition) & (games.season == season)]
    names = {}
    for club, name in zip(g.home_club_id, g.home_club_name, strict=True):
        names.setdefault(int(club), name)
    for club, name in zip(g.away_club_id, g.away_club_name, strict=True):
        names.setdefault(int(club), name)
    return names


def build_team_map():
    games = data.tm_table("games")
    rows = []
    for league, competition in COMPETITIONS.items():
        teams = pd.read_parquet(INTERIM / "statsbomb" / league / "teams.parquet")
        teams = teams.sort_values("team_id")
        names = club_names(games, competition, SEASON)
        ids = list(names)
        sim = np.array(
            [[ratio(norm_club(t), norm_club(names[c])) for c in ids] for t in teams.team_name]
        )
        for i, j in enumerate(assign(sim)):
            best = int(sim[i].argmax())
            rows.append(
                {
                    "league": league,
                    "team_id": int(teams.team_id.iloc[i]),
                    "sb_team": teams.team_name.iloc[i],
                    "tm_club_id": ids[j],
                    "tm_club": names[ids[j]],
                    "similarity": round(sim[i, j], 3),
                    "best_tm_club_id": ids[best],
                    "best_tm_club": names[ids[best]],
                }
            )
    out = pd.DataFrame(rows)
    out["differs_from_best"] = out.tm_club_id != out.best_tm_club_id
    return out


def name_overlap(team_map, players, appearances):
    """Exact normalised name or nickname matches per StatsBomb team and Transfermarkt club."""
    scope = appearances[appearances.season == SEASON]
    rows = []
    for league, competition in COMPETITIONS.items():
        a = scope[scope.competition_id == competition]
        club_players = {
            int(c): {norm(n) for n in g.player_name.dropna().unique()} - {""}
            for c, g in a.groupby("player_club_id")
        }
        for r in team_map[team_map.league == league].itertuples():
            squad = players[(players.league == league) & (players.team_id == r.team_id)]
            keys = [
                (norm(n), norm(k)) for n, k in zip(squad.player_name, squad.nickname, strict=True)
            ]
            counts = {
                c: sum(1 for n, k in keys if (n and n in names) or (k and k in names))
                for c, names in club_players.items()
            }
            others = {c: v for c, v in counts.items() if c != r.tm_club_id}
            rows.append(
                {
                    "league": league,
                    "sb_team": r.sb_team,
                    "tm_club": r.tm_club,
                    "overlap": counts.get(r.tm_club_id, 0),
                    "max_other_overlap": max(others.values()),
                }
            )
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    team_map = build_team_map()
    data.LINKS.mkdir(parents=True, exist_ok=True)
    team_map.to_parquet(data.LINKS / "team_map.parquet", index=False)
    print("pairs:", len(team_map))
    print(
        "distinct Transfermarkt clubs per league:",
        team_map.groupby("league").tm_club_id.nunique().to_dict(),
    )
    print("pairs with similarity under 0.85:", int((team_map.similarity < 0.85).sum()))
    print("league | StatsBomb team | Transfermarkt club | similarity")
    for r in team_map.itertuples():
        print(f"{r.league} | {r.sb_team} | {r.tm_club} | {r.similarity:.3f}")
    print("\npairs that differ from the independent best match:")
    for r in team_map[team_map.differs_from_best].itertuples():
        print(
            f"{r.league} | {r.sb_team} | assigned {r.tm_club} | independent best {r.best_tm_club}"
        )
    ov = name_overlap(team_map, data.statsbomb_players(), data.tm_appearances())
    print("\nexact-name overlap, assigned club vs largest overlap with another club")
    for r in ov.itertuples():
        print(f"{r.league} | {r.sb_team} | {r.tm_club} | {r.overlap} | {r.max_other_overlap}")
    print(
        "overlap min",
        int(ov.overlap.min()),
        "median",
        float(ov.overlap.median()),
        "max",
        int(ov.overlap.max()),
    )
    print("pairs where another club overlaps more:", int((ov.max_other_overlap > ov.overlap).sum()))


if __name__ == "__main__":
    main()
