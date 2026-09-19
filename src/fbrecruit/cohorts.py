import sys

import numpy as np
import pandas as pd

from fbrecruit.links import data
from fbrecruit.paths import PROCESSED

NAMES = {
    "ES1": "La Liga",
    "GB1": "Premier League",
    "IT1": "Serie A",
    "FR1": "Ligue 1",
    "L1": "Bundesliga",
}
LABELS = ["paid", "paid_mirror", "loan_return", "loan_out", "both", "free_other"]
MIN_MINUTES = 450
COHORTS = {
    "A": {
        "season": 2015,
        "scope": ["ES1", "GB1", "IT1", "FR1"],
        "start": "2016-06-01",
        "end": "2016-09-30",
        "links": "statsbomb_tm",
        "linked": "matched",
        "minutes": "sum",
        "link_minutes": 0,
    },
    "B": {
        "season": 2017,
        "scope": ["GB1", "ES1", "IT1", "L1", "FR1"],
        "start": "2018-06-01",
        "end": "2018-09-30",
        "links": "wyscout_tm",
        "linked": "linked",
        "minutes": "max",
        "link_minutes": MIN_MINUTES,
    },
}


def season_players(app, season, scope):
    """Players with 450+ Transfermarkt minutes in the in-scope leagues, with their main club."""
    sc = app[(app.season == season) & app.competition_id.isin(scope)]
    pc = sc.groupby(["player_id", "player_club_id"]).minutes_played.sum().reset_index()
    total = pc.groupby("player_id").minutes_played.sum().rename("min_total")
    pc = pc.sort_values(
        ["player_id", "minutes_played", "player_club_id"], ascending=[True, False, True]
    )
    main = pc.groupby("player_id").first().rename(columns={"player_club_id": "main_club"})
    league = (
        sc.groupby(["player_id", "competition_id"])
        .minutes_played.sum()
        .reset_index()
        .sort_values(
            ["player_id", "minutes_played", "competition_id"], ascending=[True, False, True]
        )
        .groupby("player_id")
        .first()
        .competition_id.rename("league_comp")
    )
    p = main[["main_club"]].join(total).join(league)
    return p[p.min_total >= MIN_MINUTES].copy()


def qualifying_transfers(tr, p, start, end):
    inside = tr[(tr.date >= pd.Timestamp(start)) & (tr.date <= pd.Timestamp(end))]
    q = inside.merge(
        p.reset_index()[["player_id", "main_club"]],
        left_on=["player_id", "from_club_id"],
        right_on=["player_id", "main_club"],
    )
    return q[q.to_club_id != q.from_club_id].sort_values(["player_id", "date"], kind="stable")


def window_movers(tr, p, start, end, legacy=False):
    """The earliest transfer in the window out of the player's main club, as a whole row.

    legacy=True takes the first non-null value of each column instead, which can
    combine the fee of a later transfer with the earliest one.
    """
    q = qualifying_transfers(tr, p, start, end)
    if legacy:
        mv = q.groupby("player_id").first().reset_index()
    else:
        mv = q.drop_duplicates("player_id")
    columns = ["player_id", "date", "from_club_id", "to_club_id", "from_club_name"]
    return mv[[*columns, "to_club_name", "transfer_fee"]]


def mirror_labels(mv, tr):
    """Label each mover by whether the same player has a swapped-clubs move within three years."""
    mv = mv.reset_index(drop=True)
    moves = tr[["player_id", "date", "from_club_id", "to_club_id", "transfer_fee"]]
    mm = (
        mv[["player_id", "date", "from_club_id", "to_club_id"]]
        .reset_index()
        .merge(
            moves,
            left_on=["player_id", "to_club_id", "from_club_id"],
            right_on=["player_id", "from_club_id", "to_club_id"],
            suffixes=("", "_m"),
        )
    )
    near = (mm.date_m >= mm.date - pd.DateOffset(years=3)) & (
        mm.date_m <= mm.date + pd.DateOffset(years=3)
    )
    mm = mm[near]
    free = mm.transfer_fee.isna() | (mm.transfer_fee == 0)
    flags = (
        pd.DataFrame(
            {
                "index": mm["index"],
                "mirror_zero_earlier": free & (mm.date_m < mm.date),
                "mirror_zero_later": free & (mm.date_m > mm.date),
                "mirror_same_day": mm.date_m == mm.date,
            }
        )
        .groupby("index")
        .any()
    )
    mv = mv.join(flags)
    mv["n_mirrors"] = mm.groupby("index").size().reindex(mv.index).fillna(0).astype(int)
    mv["has_mirror"] = mv.n_mirrors > 0
    for col in ["mirror_zero_earlier", "mirror_zero_later", "mirror_same_day"]:
        mv[col] = mv[col].eq(True)
    paid = mv.transfer_fee > 0
    mv["mirror_label"] = np.select(
        [
            paid & mv.has_mirror,
            paid,
            mv.mirror_zero_earlier & mv.mirror_zero_later,
            mv.mirror_zero_earlier,
            mv.mirror_zero_later,
        ],
        ["paid_mirror", "paid", "both", "loan_return", "loan_out"],
        default="free_other",
    )
    return mv


def add_cohort_rules(mv):
    """Permanent moves and the loan-out sensitivity set."""
    june30 = (mv.date.dt.month == 6) & (mv.date.dt.day == 30)
    mv = mv.copy()
    mv["permanent"] = mv.mirror_label.isin(["paid", "paid_mirror"]) | (
        (mv.mirror_label == "free_other") & ~june30
    )
    mv["sensitivity"] = mv.mirror_label == "loan_out"
    return mv


def provider_minutes(links, how, passes):
    """Provider minutes per Transfermarkt player over the links of the given passes."""
    used = links[links.link_pass <= passes]
    return used.groupby("tm_player_id").minutes.agg(how)


def linked_ids(links, cfg, passes):
    minutes = provider_minutes(links, cfg["minutes"], passes)
    return set(minutes[minutes >= cfg["link_minutes"]].index.astype(int))


def levels(mv, linked):
    n3 = mv.player_id.isin(linked)
    n4 = n3 & (mv.minutes_next_season >= MIN_MINUTES)
    n5 = n4 & mv.dest_club_in_scope_leagues
    return n3, n4, n5


def funnel_table(p, mv, stay, scope, linked):
    n3, n4, n5 = levels(mv, linked)
    s3 = stay.index.isin(linked)
    s4 = s3 & (stay.minutes_next >= MIN_MINUTES)
    rows = []
    for name in [NAMES[c] for c in sorted(scope)] + ["ALL"]:
        pm = p.league_comp.map(NAMES) == name if name != "ALL" else pd.Series(True, index=p.index)
        mm = mv.league_comp.map(NAMES) == name if name != "ALL" else pd.Series(True, index=mv.index)
        sm = stay.league_comp.map(NAMES) == name if name != "ALL" else np.ones(len(stay), bool)
        rows.append(
            {
                "league": name,
                "n1": int(pm.sum()),
                "n2": int(mm.sum()),
                "n3": int((n3 & mm).sum()),
                "n4": int((n4 & mm).sum()),
                "n5": int((n5 & mm).sum()),
                "stayers_n1": int(np.sum(sm)),
                "stayers_n3": int(np.sum(s3 & sm)),
                "stayers_n4": int(np.sum(s4 & sm)),
            }
        )
    return pd.DataFrame(rows)


def build(name, app, tr, links, legacy=False):
    cfg = COHORTS[name]
    season, nxt = cfg["season"], cfg["season"] + 1
    minutes = app.groupby(["season", "player_id"]).minutes_played.sum()
    next_minutes = minutes.xs(nxt, level="season")
    games = data.tm_table("games", ["competition_id", "season", "home_club_id", "away_club_id"])
    in_scope = games[(games.season == season) & games.competition_id.isin(cfg["scope"])]
    clubs = set(in_scope.home_club_id) | set(in_scope.away_club_id)

    p = season_players(app, season, cfg["scope"])
    p["minutes_next"] = next_minutes.reindex(p.index).fillna(0)
    mv = window_movers(tr, p, cfg["start"], cfg["end"], legacy)
    mv["league_comp"] = mv.player_id.map(p.league_comp)
    mv["min_total_season"] = mv.player_id.map(p.min_total)
    mv["minutes_next_season"] = mv.player_id.map(p.minutes_next)
    mv["dest_club_in_scope_leagues"] = mv.to_club_id.isin(clubs)
    after = app[app.season == season + 2][["player_id", "player_club_id"]].drop_duplicates()
    pairs = set(zip(after.player_id, after.player_club_id, strict=True))
    mv["returned"] = [(a, b) in pairs for a, b in zip(mv.player_id, mv.from_club_id, strict=True)]
    mv = add_cohort_rules(mirror_labels(mv, tr))

    next_pairs = app[app.season == nxt][["player_id", "player_club_id"]].drop_duplicates()
    next_pairs = set(zip(next_pairs.player_id, next_pairs.player_club_id, strict=True))
    candidates = p[~p.index.isin(mv.player_id)]
    stay = candidates[
        [(i, c) in next_pairs for i, c in zip(candidates.index, candidates.main_club, strict=True)]
    ]

    pass1 = linked_ids(links, cfg, 1)
    both = linked_ids(links, cfg, 2)
    for tag, ids in [("_p1", pass1), ("", both)]:
        n3, n4, n5 = levels(mv, ids)
        mv["n3" + tag], mv["n4" + tag], mv["n5" + tag] = n3, n4, n5
    mv["provider_minutes"] = mv.player_id.map(provider_minutes(links, cfg["minutes"], 2)).fillna(0)
    mv["provider_minutes_450"] = mv.provider_minutes >= MIN_MINUTES
    tables = {
        "pass 1": funnel_table(p, mv, stay, cfg["scope"], pass1),
        "passes 1 and 2": funnel_table(p, mv, stay, cfg["scope"], both),
    }
    return mv, tables


def load_links(name):
    cfg = COHORTS[name]
    links = pd.read_parquet(data.LINKS / f"{cfg['links']}.parquet")
    links = links[links.status == cfg["linked"]]
    return links[["tm_player_id", "minutes", "link_pass"]].astype({"tm_player_id": int})


def level_counts(mv, suffix, mask):
    return {lvl: int((mv[lvl + suffix] & mask).sum()) for lvl in ["n3", "n4", "n5"]}


def report(name, mv, tables):
    print(f"\n==================== cohort {name}")
    for label, table in tables.items():
        print(f"\nfunnel, {label}:")
        print(table.to_string(index=False))
    print("\nlabel counts (n2):")
    print(mv.mirror_label.value_counts().reindex(LABELS).fillna(0).astype(int).to_string())
    print("returned:", mv.returned.value_counts().to_dict())
    day = mv.date.dt.strftime("%m-%d")
    print(
        "movers dated 30 June:",
        int((day == "06-30").sum()),
        "and 1 July:",
        int((day == "07-01").sum()),
    )
    for suffix, label in [("_p1", "pass 1"), ("", "passes 1 and 2")]:
        per = mv.groupby("mirror_label")[[f"n3{suffix}", f"n4{suffix}", f"n5{suffix}"]].sum()
        print(f"\nper label n3, n4, n5, {label}:")
        print(per.reindex(LABELS).astype(int).to_string())
    print("\npermanent movers and the loan-out sensitivity set by level:")
    for suffix, label in [("_p1", "pass 1"), ("", "passes 1 and 2")]:
        for setname, mask in [("permanent", mv.permanent), ("sensitivity", mv.sensitivity)]:
            without = level_counts(mv, suffix, mask)
            flagged = level_counts(mv, suffix, mask & mv.provider_minutes_450)
            print(
                f"{label}, {setname}: without flag {without}, with 450+ provider minutes {flagged}"
            )
    print("\neffect of pass 2 (passes 1 and 2 minus pass 1):")
    for setname, mask in [("all movers", mv.player_id.notna()), ("permanent", mv.permanent)]:
        a, b = level_counts(mv, "_p1", mask), level_counts(mv, "", mask)
        print(f"{setname}: " + ", ".join(f"{k} {a[k]} -> {b[k]} ({b[k] - a[k]:+d})" for k in a))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    app = data.tm_appearances()
    tr = data.tm_table("transfers").rename(columns={"transfer_date": "date"})
    out = PROCESSED / "cohorts"
    out.mkdir(parents=True, exist_ok=True)
    for name in COHORTS:
        mv, tables = build(name, app, tr, load_links(name))
        mv.to_parquet(out / f"movers_{name}.parquet", index=False)
        report(name, mv, tables)


if __name__ == "__main__":
    main()
