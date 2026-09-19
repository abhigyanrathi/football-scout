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


def within_band(ratio):
    return (ratio >= 0.8) & (ratio <= 1.2)


def agreement_summary(agree, groups):
    """Pairs, median ratio and share within 0.8 to 1.2 for pairs with 450 or more minutes."""
    big = agree[agree.minutes >= 450]
    rows = []
    parts = [("ALL", big)] + [(k, big[big[groups] == k]) for k in sorted(big[groups].unique())]
    for label, g in parts:
        rows.append(
            {
                groups: label,
                "pairs": len(g),
                "median_ratio": round(g.ratio.median(), 3),
                "share_within": round(within_band(g.ratio).mean(), 4),
            }
        )
    return pd.DataFrame(rows)


def player_table(links):
    """One row per StatsBomb player over the matched and unmatched rows of all their teams."""
    g = links.assign(is_matched=links.status == "matched").groupby("player_id")
    ids = g.tm_player_id.agg(lambda s: sorted(s.dropna().unique()))
    n_matched = g.is_matched.sum().astype(int)
    return pd.DataFrame(
        {
            "tm_player_id": ids.map(lambda v: v[0] if len(v) == 1 else pd.NA).astype("Int64"),
            "n_tm_ids": ids.map(len),
            "minutes": g.minutes.sum(),
            "n_matched": n_matched,
            "n_unmatched": g.size() - n_matched,
        }
    ).reset_index()


def shared_id_cases(links):
    """Transfermarkt ids linked from more than one row, split by what the rows have in common."""
    matched = links[links.status == "matched"]
    out = {
        "same_player_two_clubs": [],
        "different_players_same_club": [],
        "different_players_different_clubs": [],
    }
    for tm_id, g in matched.groupby("tm_player_id"):
        if len(g) < 2:
            continue
        if g.player_id.nunique() == 1:
            key = "same_player_two_clubs"
        elif g[["league", "team_id"]].drop_duplicates().shape[0] == 1:
            key = "different_players_same_club"
        else:
            key = "different_players_different_clubs"
        out[key].append((tm_id, g))
    return out


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


def report_comparable(links, appearances):
    agree = minutes_agreement(links, appearances)
    print("\n== minutes agreement, pairs with 450 or more StatsBomb minutes")
    print(agreement_summary(agree, "link_pass").to_string(index=False))
    print(agreement_summary(agree, "league").to_string(index=False))
    inside = within_band(agree.ratio)
    big = agree.minutes >= 450
    share_all = agree.minutes[inside].sum() / agree.minutes.sum()
    share_big = agree.minutes[inside & big].sum() / agree.minutes[big].sum()
    print(
        "share of matched StatsBomb minutes in pairs within 0.8 to 1.2: "
        f"all pairs {share_all:.4f}, pairs with 450+ minutes {share_big:.4f}"
    )
    far = agree[big].assign(gap=(agree.ratio - 1).abs()).sort_values("gap", ascending=False)
    print("10 pairs with 450+ minutes farthest from 1: StatsBomb name | TM name | club | ratio")
    for r in far.head(10).itertuples():
        print(
            f"{r.player_name} | {r.tm_name} | {r.tm_club} | "
            f"StatsBomb {r.minutes} TM {r.tm_minutes} ratio {r.ratio:.3f}"
        )


def report_players(links, table):
    print(f"\n== player-level table: {len(table)} StatsBomb players")
    multi = table[table.n_tm_ids > 1]
    print(f"(a) players whose rows link to more than one Transfermarkt id: {len(multi)}")
    print(multi.to_string(index=False))
    split = table[(table.n_matched > 0) & (table.n_unmatched > 0)]
    print(f"(b) players with some rows matched and some unmatched: {len(split)}")
    names = links.drop_duplicates("player_id").set_index("player_id").player_name
    for r in split.itertuples():
        rows = links[links.player_id == r.player_id]
        print(f"{names[r.player_id]} (player_id {r.player_id}), minutes {r.minutes}")
        print(
            rows[["league", "team_name", "minutes", "status", "tm_player_id"]].to_string(
                index=False
            )
        )
    cases = shared_id_cases(links)
    print("(c) Transfermarkt ids linked from more than one row:", sum(map(len, cases.values())))
    for key, found in cases.items():
        print(f"\n{key}: {len(found)}")
        for tm_id, g in found:
            print(f"Transfermarkt id {tm_id}")
            print(
                g[["player_id", "player_name", "league", "team_name", "minutes"]].to_string(
                    index=False
                )
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
    report_comparable(final, appearances)
    table = player_table(final)
    table.to_parquet(data.LINKS / "statsbomb_tm_players.parquet", index=False)
    report_players(final, table)


if __name__ == "__main__":
    main()
