import sys

import pandas as pd

from fbrecruit.links import data
from fbrecruit.links.teams import COMPETITIONS, SEASON
from fbrecruit.links.text import norm, ratio, tokens_contained

ACCEPT = 0.85
MARGIN = 0.05


def candidates(appearances):
    """Transfermarkt players per (league, club) with an appearance in the season."""
    scope = appearances[appearances.season == SEASON]
    out = {}
    for league, competition in COMPETITIONS.items():
        a = scope[scope.competition_id == competition].sort_values(["player_id", "date"])
        per = a.groupby(["player_club_id", "player_id"]).player_name.first().reset_index()
        for club, g in per.groupby("player_club_id"):
            out[(league, int(club))] = [
                (int(p), n, norm(n)) for p, n in zip(g.player_id, g.player_name, strict=True)
            ]
    return out


def match_pass1(players, team_map, cands):
    mapped = team_map.set_index(["league", "team_id"])
    rows = []
    for r in players.itertuples():
        m = mapped.loc[(r.league, r.team_id)]
        pool = cands.get((r.league, int(m.tm_club_id)), [])
        a, b = norm(r.player_name), norm(r.nickname)
        scored = []
        for pid, name, nn in pool:
            sa, sn = ratio(a, nn), ratio(b, nn)
            scored.append((max(sa, sn), "name" if sa >= sn else "nickname", pid, name))
        scored.sort(key=lambda x: -x[0])
        best = scored[0] if scored else (0.0, "", None, None)
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best[0] >= ACCEPT and best[0] - second >= MARGIN:
            status = "matched"
        elif best[0] >= ACCEPT:
            status = "ambiguous"
        else:
            status = "unmatched"
        ok = status == "matched"
        rows.append(
            {
                "league": r.league,
                "team_id": r.team_id,
                "team_name": r.team_name,
                "player_id": r.player_id,
                "player_name": r.player_name,
                "nickname": r.nickname,
                "minutes": r.minutes,
                "tm_club_id": int(m.tm_club_id),
                "tm_club": m.tm_club,
                "n_candidates": len(pool),
                "status": status,
                "method": best[1] if ok else "",
                "score": round(best[0], 3),
                "second_score": round(second, 3),
                "tm_player_id": best[2] if ok else pd.NA,
                "tm_name": best[3] if ok else None,
                "best_tm_name": best[3],
                "link_pass": 1 if ok else pd.NA,
            }
        )
    out = pd.DataFrame(rows)
    out["tm_player_id"] = out.tm_player_id.astype("Int64")
    out["link_pass"] = out.link_pass.astype("Int64")
    return out


def match_pass2(links, cands):
    """Token containment within the mapped club, for rows pass 1 left unmatched."""
    out = links.copy()
    for i, r in out[out.status == "unmatched"].iterrows():
        pool = cands.get((r.league, r.tm_club_id), [])
        hits = [
            (pid, name)
            for pid, name, _ in pool
            if tokens_contained(name, r.player_name, r.nickname, min_tokens=2)
        ]
        if len(hits) == 1:
            out.loc[i, ["status", "method", "tm_player_id", "tm_name", "link_pass"]] = [
                "matched",
                "tokens",
                hits[0][0],
                hits[0][1],
                2,
            ]
    out["tm_player_id"] = out.tm_player_id.astype("Int64")
    out["link_pass"] = out.link_pass.astype("Int64")
    return out


def minutes_agreement(links, appearances):
    """Transfermarkt minutes at the mapped club and league divided by StatsBomb minutes."""
    scope = appearances[appearances.season == SEASON]
    tm = scope.groupby(["competition_id", "player_club_id", "player_id"]).minutes_played.sum()
    matched = links[links.status == "matched"].copy()
    keys = zip(
        matched.league.map(COMPETITIONS), matched.tm_club_id, matched.tm_player_id, strict=True
    )
    matched["tm_minutes"] = [tm.get(k, 0) for k in keys]
    matched = matched[matched.minutes > 0]
    matched["ratio"] = matched.tm_minutes / matched.minutes
    return matched


def rate(df):
    return df[df.status == "matched"].minutes.sum() / df.minutes.sum()


def rates_table(links):
    rows = []
    for league in [*sorted(links.league.unique()), "ALL"]:
        d = links if league == "ALL" else links[links.league == league]
        big = d[d.minutes >= 450]
        rows.append(
            {
                "league": league,
                "rows": len(d),
                "matched": int((d.status == "matched").sum()),
                "unmatched": int((d.status == "unmatched").sum()),
                "ambiguous": int((d.status == "ambiguous").sum()),
                "minutes_weighted_rate": round(rate(d), 4),
                "rows_450+": len(big),
                "matched_450+": int((big.status == "matched").sum()),
                "rate_450+_minutes_weighted": round(rate(big), 4),
            }
        )
    return pd.DataFrame(rows)


def report(links, appearances):
    print(rates_table(links).to_string(index=False))
    matched = links[links.status == "matched"]
    print(
        "accepted matches with a score below 1.0 (pass 1):",
        int(((matched.score < 1.0) & (matched.link_pass == 1)).sum()),
    )
    print(
        "unmatched rows with 900 or more minutes:",
        int(((links.status == "unmatched") & (links.minutes >= 900)).sum()),
    )
    print(
        "Transfermarkt players matched to more than one StatsBomb row:",
        int(matched.tm_player_id.duplicated().sum()),
    )
    agree = minutes_agreement(links, appearances)
    for p, g in agree.groupby("link_pass"):
        within = ((g.ratio >= 0.8) & (g.ratio <= 1.2)).mean()
        print(
            f"\nminutes agreement, pass {p}: pairs {len(g)}, median ratio {g.ratio.median():.3f}, "
            f"share within 0.8 to 1.2 {within:.4f}"
        )
        far = g.assign(gap=(g.ratio - 1).abs()).sort_values("gap", ascending=False).head(20)
        print("20 farthest from 1: StatsBomb name | nickname | TM name | StatsBomb min | TM min")
        for r in far.itertuples():
            print(f"{r.player_name} | {r.nickname} | {r.tm_name} | {r.minutes} | {r.tm_minutes}")
    print(
        "\nStatsBomb rows with zero minutes excluded from the agreement:",
        int(((links.status == "matched") & (links.minutes == 0)).sum()),
    )


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    appearances = data.tm_appearances()
    cands = candidates(appearances)
    team_map = pd.read_parquet(data.LINKS / "team_map.parquet")
    pass1 = match_pass1(data.statsbomb_players(), team_map, cands)
    pass1.to_parquet(data.LINKS / "statsbomb_tm_pass1.parquet", index=False)
    print("== after pass 1")
    report(pass1, appearances)
    final = match_pass2(pass1, cands)
    final.to_parquet(data.LINKS / "statsbomb_tm.parquet", index=False)
    print("\n== pass-2 matches: StatsBomb name | nickname | TM name | club | StatsBomb minutes")
    for r in final[final.link_pass == 2].itertuples():
        print(f"{r.player_name} | {r.nickname} | {r.tm_name} | {r.tm_club} | {r.minutes}")
    print("\n== after passes 1 and 2")
    report(final, appearances)


if __name__ == "__main__":
    main()
