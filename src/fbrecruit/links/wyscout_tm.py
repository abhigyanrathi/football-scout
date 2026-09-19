import json
import sys

import pandas as pd

from fbrecruit.links import data
from fbrecruit.links.text import norm, ratio, tokens_contained
from fbrecruit.paths import RAW
from fbrecruit.sources.wyscout import decode

THRESHOLD = 0.75
LEAGUES = ["GB1", "ES1", "IT1", "L1", "FR1"]
SEASON = 2017


def wyscout_players():
    raw = json.loads((RAW / "wyscout" / "players.json").read_text(encoding="utf-8"))
    wy = pd.DataFrame(raw)
    for col in ["firstName", "lastName", "shortName"]:
        wy[col] = wy[col].map(lambda s: decode(s) if isinstance(s, str) else s)
    wy["dob"] = wy.birthDate.astype(str).str[:10]
    return wy


def tm_by_dob():
    players = data.tm_table("players", ["player_id", "name", "date_of_birth"]).dropna(
        subset=["date_of_birth"]
    )
    players = players.assign(dob=players.date_of_birth.dt.strftime("%Y-%m-%d"))
    return {
        dob: [(int(p), n, norm(n)) for p, n in zip(g.player_id, g.name, strict=True)]
        for dob, g in players.groupby("dob")
    }


def wyscout_minutes():
    """Minutes summed over the five leagues, with the league where the player played most."""
    table = data.minutes_table("minutes_player_wyscout")
    total = table.groupby("player_id").minutes.sum()
    top = table.sort_values(["player_id", "minutes"], ascending=[True, False])
    league = top.groupby("player_id").league.first()
    return pd.DataFrame({"minutes": total, "league": league}).rename_axis("wy_id").reset_index()


def link_pass1(wy, by_dob):
    rows = []
    for r in wy.itertuples():
        pool = by_dob.get(r.dob, [])
        full = norm(f"{r.firstName} {r.lastName}")
        short = norm(r.shortName)
        scored = []
        for pid, name, nn in pool:
            sf, ss = ratio(full, nn), ratio(short, nn)
            scored.append((max(sf, ss), "first+last" if sf >= ss else "shortName", pid, name))
        scored.sort(key=lambda x: -x[0])
        hits = [s for s in scored if s[0] >= THRESHOLD]
        best = scored[0] if scored else (0.0, "", None, None)
        if len(hits) == 1:
            status = "linked"
        elif len(hits) > 1:
            status = "ambiguous"
        elif not pool:
            status = "no_candidate_same_dob"
        else:
            status = "below_threshold"
        ok = status == "linked"
        rows.append(
            {
                "wy_id": int(r.wyId),
                "wy_first": r.firstName,
                "wy_last": r.lastName,
                "wy_short": r.shortName,
                "dob": r.dob,
                "n_candidates": len(pool),
                "n_at_threshold": len(hits),
                "best_score": round(best[0], 3),
                "status": status,
                "method": hits[0][1] if ok else "",
                "tm_player_id": hits[0][2] if ok else pd.NA,
                "tm_name": hits[0][3] if ok else None,
                "best_tm_name": best[3],
                "link_pass": 1 if ok else pd.NA,
            }
        )
    out = pd.DataFrame(rows)
    out["tm_player_id"] = out.tm_player_id.astype("Int64")
    out["link_pass"] = out.link_pass.astype("Int64")
    return out


def link_pass2(links, by_dob):
    """Token containment among Transfermarkt players born the same day, for unlinked players."""
    out = links.copy()
    for i, r in out[out.status != "linked"].iterrows():
        hits = [
            (pid, name)
            for pid, name, _ in by_dob.get(r.dob, [])
            if tokens_contained(name, r.wy_first, r.wy_last, r.wy_short)
        ]
        if len(hits) == 1:
            out.loc[i, ["status", "method", "tm_player_id", "tm_name", "link_pass"]] = [
                "linked",
                "tokens",
                hits[0][0],
                hits[0][1],
                2,
            ]
    out["tm_player_id"] = out.tm_player_id.astype("Int64")
    out["link_pass"] = out.link_pass.astype("Int64")
    return out


def minutes_agreement(links, appearances):
    """Transfermarkt minutes in the five leagues in 2017/18 divided by Wyscout minutes."""
    scope = appearances[(appearances.season == SEASON) & appearances.competition_id.isin(LEAGUES)]
    tm = scope.groupby("player_id").minutes_played.sum()
    linked = links[(links.status == "linked") & (links.minutes >= 450)].copy()
    linked["tm_minutes"] = linked.tm_player_id.map(tm).fillna(0)
    linked["ratio"] = linked.tm_minutes / linked.minutes
    return linked


def rates_table(links):
    big = links[links.minutes >= 450]
    rows = []
    for league in [*sorted(big.league.unique()), "ALL"]:
        d = big if league == "ALL" else big[big.league == league]
        ok = d[d.status == "linked"]
        rows.append(
            {
                "league": league,
                "players": len(d),
                "linked": len(ok),
                "rate": round(len(ok) / len(d), 4),
                "minutes_weighted_rate": round(ok.minutes.sum() / d.minutes.sum(), 4),
            }
        )
    return pd.DataFrame(rows)


def report(links, appearances):
    print("status, all Wyscout players:")
    print(links.status.value_counts().to_string())
    ok = links[links.status == "linked"]
    print("linked with a best score of exactly 1.0:", int((ok.best_score == 1.0).sum()))
    print("Transfermarkt players linked twice:", int(ok.tm_player_id.duplicated().sum()))
    big = links[links.minutes >= 450]
    print(f"\nplayers with 450+ minutes: {len(big)}; status:")
    print(big.status.value_counts().to_string())
    print(rates_table(links).to_string(index=False))
    agree = minutes_agreement(links, appearances)
    for p, g in agree.groupby("link_pass"):
        within = ((g.ratio >= 0.8) & (g.ratio <= 1.2)).mean()
        print(
            f"\nminutes agreement, pass {p}: players {len(g)}, without Transfermarkt minutes "
            f"{int((g.tm_minutes == 0).sum())}, median ratio {g.ratio.median():.3f}, "
            f"share within 0.8 to 1.2 {within:.4f}"
        )
        far = g.assign(gap=(g.ratio - 1).abs()).sort_values("gap", ascending=False).head(20)
        print("20 farthest from 1: Wyscout name | shortName | TM name | Wyscout min | TM min")
        for r in far.itertuples():
            print(
                f"{r.wy_first} {r.wy_last} | {r.wy_short} | {r.tm_name} | "
                f"{r.minutes:.0f} | {r.tm_minutes:.0f}"
            )


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    wy = wyscout_players()
    by_dob = tm_by_dob()
    appearances = data.tm_appearances()
    minutes = wyscout_minutes()
    pass1 = link_pass1(wy, by_dob).merge(minutes, on="wy_id", how="left")
    pass1["minutes"] = pass1.minutes.fillna(0)
    print("Wyscout players:", len(pass1))
    pass1.to_parquet(data.LINKS / "wyscout_tm_pass1.parquet", index=False)
    print("== after pass 1")
    report(pass1, appearances)
    final = link_pass2(pass1, by_dob)
    final.to_parquet(data.LINKS / "wyscout_tm.parquet", index=False)
    print("\n== pass-2 links with 450+ minutes: Wyscout name | shortName | TM name | minutes")
    new = final[(final.link_pass == 2) & (final.minutes >= 450)]
    for r in new.itertuples():
        print(f"{r.wy_first} {r.wy_last} | {r.wy_short} | {r.tm_name} | {r.minutes:.0f}")
    print("pass-2 links in total:", int((final.link_pass == 2).sum()))
    print("\n== after passes 1 and 2")
    report(final, appearances)


if __name__ == "__main__":
    main()
