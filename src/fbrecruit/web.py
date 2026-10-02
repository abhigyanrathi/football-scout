"""The scouting site's data files, built from the scouting tools' and hidden-gem tables."""

import hashlib
import json
import math
import shutil
import sys
import time

import numpy as np
import pandas as pd

from fbrecruit import gems, scout, squad, style
from fbrecruit import shrinkage as sh
from fbrecruit.links import data
from fbrecruit.logs import Tee, key
from fbrecruit.paths import ROOT
from fbrecruit.sources.statsbomb import LEAGUES

KEYS = sh.KEYS
OUT = ROOT / "web" / "data"
BANDS = [1_000_000, 2_500_000, 5_000_000, 10_000_000, 20_000_000, 40_000_000]
BAND_LABELS = [
    "Under €1m", "€1m to €2.5m", "€2.5m to €5m", "€5m to €10m", "€10m to €20m", "€20m to €40m",
    "€40m and over",
]  # fmt: skip
BUDGETS = [5_000_000, 10_000_000, 20_000_000, 40_000_000, 80_000_000]
CAPS = [1, 2, 3]
Z = 1.645
NOISE = 0.5
CEILING = 10_000
DIMENSION_LABELS = {
    "sb_pressing": "Pressing",
    "possession": "Possession",
    "buildup": "Build-up speed",
    "width": "Width",
    "verticality": "Verticality",
    "depth": "Defensive depth",
    "dribble": "Dribbling",
    "creation": "Shot creation",
}
LESS_RELIABLE = ["depth", "creation"]
GROUP_LABELS = {
    "CB": "Centre-back",
    "FB": "Full-back",
    "MID": "Midfielder",
    "WIDE": "Wide player",
    "FWD": "Forward",
}
LEAGUE_LABELS = {
    "la_liga": "La Liga",
    "premier_league": "Premier League",
    "serie_a": "Serie A",
    "ligue_1": "Ligue 1",
}
# in the order they are tried: a plan gets the first that applies
REASONS = ["more places than signings", "too few candidates", "budget too small"]
# from p9b_season_summary.log and p9b_tools_summary.log
PLAYERS = 1247
GROUPS = {"CB": 257, "FB": 254, "MID": 284, "WIDE": 286, "FWD": 166}
FIT_ROWS = 98450
PAIR_ROWS = 316009
# where a number is a player id, which may be CEILING or more; "*" is any key or position
ID_PATHS = [
    ("players", "*", "id"),
    ("clubs", "*", "players", "*"),
    ("fit", "*", 0),
    ("similar", "*", "*", 0),
    ("baseline", "players", "*"),
    ("plans", "*", "in", "*"),
    ("plans", "*", "out", "*"),
]


def band(price):
    """The index of the price's band: the number of BANDS edges at or below it."""
    return int(np.searchsorted(BANDS, price, side="right"))


def club_key(league, team_id):
    return f"{league}-{int(team_id)}"


def rounded(x, decimals):
    """x as a float rounded to decimals, or None if it is missing."""
    return None if pd.isna(x) else round(float(x), decimals)


def style_scores(values):
    """Centred scores rounded to three decimals, a missing one as 0."""
    return [0.0 if pd.isna(x) else round(float(x), 3) for x in values]


def inputs():
    """The tables the files are built from, each with the columns used."""
    value = pd.read_parquet(gems.value_path())
    cols = [*KEYS, "price", "value_2015", "age", "reason"]
    decision = pd.read_parquet(gems.decision_path(), columns=cols)
    gem = pd.read_parquet(gems.scores_path(), columns=[*KEYS, "gem"])
    later = pd.read_parquet(gems.outcomes_path(), columns=[*KEYS, "value_1", "how_1"])
    season = pd.read_parquet(style.season_path(), columns=[*KEYS, "group", *style.C])
    teams = pd.read_parquet(style.team_season_path(), columns=[*style.TEAM_KEYS, *style.C])
    cols = ["league_k", "team_id_k", "player_id", "score"]
    fit = pd.read_parquet(scout.fit_path(), columns=cols)
    cols = [*KEYS, "player_id_c", "distance", "closeness"]
    pairs = pd.read_parquet(scout.pairs_path(), columns=cols)
    names = data.statsbomb_players()[[*KEYS, "nickname"]]
    return value, decision, gem, later, season, teams, fit, pairs, names


def player_rows(value, decision, gem, later, season, names):
    """The tools' players in gems_value's order with every input of their records."""
    rows = value
    for frame in (decision, gem, later, season.drop(columns="group"), names):
        rows = rows.merge(frame, on=KEYS, how="left", validate="one_to_one")
    return rows


def percentiles(values, groups):
    """Per row, the share of its group's rows with a lower value, times 100, rounded down."""
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    out = np.zeros(len(values), dtype=np.int64)
    for g in np.unique(groups):
        at = groups == g
        lower = np.searchsorted(np.sort(values[at]), values[at], side="left")
        out[at] = 100 * lower // at.sum()
    return out


def gem_ranks(rows):
    """1 for the highest gem score among the priced rows, ties broken by player_id; missing for
    the others."""
    priced = rows[rows.reason == "priced"]
    order = priced.sort_values(["gem", "player_id"], ascending=[False, True]).index
    return pd.Series(np.arange(1, len(order) + 1), index=order).reindex(rows.index)


def player_records(rows):
    """One record per player, in player_id order; a price, and what is computed from it, only
    for the priced players."""
    priced = rows.reason == "priced"
    price = rows.price.astype(float).where(priced)
    named = rows.nickname.map(lambda n: isinstance(n, str) and n != "")
    rows = rows.assign(
        name=rows.nickname.where(named, rows.player_name),
        pct=percentiles(rows.P3_full, rows.group),
        gem_rank=gem_ranks(rows),
        band=[None if pd.isna(p) else band(p) for p in price],
        momentum=price / rows.value_2015.astype(float),
        later=rows.value_1.astype(float) / price,
    )
    out = []
    for r in rows.sort_values("player_id").itertuples():
        centred = [getattr(r, c) for c in style.C]
        out.append(
            {
                "id": int(r.player_id),
                "name": r.name,
                "full": r.player_name,
                "club": club_key(r.league, r.team_id),
                "group": r.group,
                "minutes": int(r.minutes),
                "age": None if pd.isna(r.age) else math.floor(r.age),
                "band": None if pd.isna(r.band) else int(r.band),
                "value": rounded(r.P3_full, 4),
                "low": rounded(r.P3_full - Z * r.P3_sd, 4),
                "high": rounded(r.P3_full + Z * r.P3_sd, 4),
                "pct": int(r.pct),
                "gem": rounded(r.gem, 4),
                "rank": None if pd.isna(r.gem_rank) else int(r.gem_rank),
                "momentum": rounded(r.momentum, 3),
                "later": rounded(r.later, 3),
                "how": None if pd.isna(r.how_1) else r.how_1,
                "style": style_scores(centred),
                "missing": [i for i, x in enumerate(centred) if pd.isna(x)],
            }
        )
    return out


def club_table(teams, rows):
    """The clubs with their key and name, in the order of LEAGUES and then by name."""
    names = rows.groupby(["league", "team_id"]).team_name.first().rename("name").reset_index()
    t = teams.merge(names, on=style.TEAM_KEYS, how="left", validate="one_to_one")
    t["key"] = [club_key(lg, team) for lg, team in zip(t.league, t.team_id, strict=True)]
    t["order"] = t.league.map({lg: i for i, lg in enumerate(LEAGUES)})
    return t.sort_values(["order", "name", "team_id"]).reset_index(drop=True)


def club_records(clubs, rows):
    """One record per club, in the order of clubs."""
    out = []
    for c in clubs.itertuples():
        own = rows.player_id[(rows.league == c.league) & (rows.team_id == c.team_id)]
        out.append(
            {
                "key": c.key,
                "league": c.league,
                "name": c.name,
                "style": style_scores(getattr(c, col) for col in style.C),
                "players": sorted(int(p) for p in own),
            }
        )
    return out


def needs(groups, quotas=squad.QUOTAS):
    """Per group, the places the own players of these groups leave open."""
    groups = np.asarray(groups)
    return {g: max(0, q - int((groups == g).sum())) for g, q in quotas.items()}


def infeasible(need, groups, prices, k, budget):
    """The first reason a plan with these open places and candidates cannot be filled with at
    most k signings within the budget, or None if it can."""
    groups, prices = np.asarray(groups), np.asarray(prices)
    if sum(need.values()) > k:
        return REASONS[0]
    if any((groups == g).sum() < n for g, n in need.items()):
        return REASONS[1]
    if squad.cheapest_cost(prices, groups, need) > budget:
        return REASONS[2]
    return None


def solve(own, cands, budget, k):
    """The plan's rows: the own players at no cost and at most k candidates at their prices."""
    rows = pd.concat([own.assign(price=0), cands], ignore_index=True)
    marked = np.arange(len(rows)) >= len(own)
    args = rows.P3_full, rows.price, rows.group, rows.player_id, budget
    return rows.iloc[squad.pick(*args, marked=marked, most=k)]


def club_plans(own, cands, budgets=BUDGETS, caps=CAPS):
    """A club's baseline, the places its own players leave open and its plans, by budget, then
    cap, then without and with the fit condition. A plan's rows are None if it cannot be
    filled."""
    need = needs(own.group)
    short = {g: n for g, n in need.items() if n}
    base = None
    if not short:
        free = np.zeros(len(own), dtype=np.int64)
        base = own.iloc[squad.pick(own.P3_full, free, own.group, own.player_id, 0)]
    plans = []
    for budget in budgets:
        for k in caps:
            for condition in (False, True):
                c = cands[cands.score > 0] if condition else cands
                reason = infeasible(need, c.group, c.price, k, budget)
                rows = None if reason else solve(own, c, budget, k)
                plans.append(
                    {"budget": budget, "cap": k, "fit": condition, "reason": reason, "rows": rows}
                )
    return base, short, plans


def gain(rows, base):
    """The plan's summed P3_full less the baseline's and the ends of its 90% interval, from the
    P3_sd of the players in one ten but not the other, taken as independent."""
    moved = pd.concat(
        [rows[~rows.player_id.isin(base.player_id)], base[~base.player_id.isin(rows.player_id)]]
    )
    g = rows.P3_full.sum() - base.P3_full.sum()
    half = Z * math.sqrt((moved.P3_sd**2).sum())
    return g, g - half, g + half


def plan_record(plan, own, base):
    """A plan as its club's file holds it, its budget in millions; with a baseline, in and out
    are the players of one ten but not the other, and without one, in is the signings."""
    rows = plan["rows"]
    moved_in, moved_out, total, change = [], [], None, (None, None, None)
    if rows is not None:
        total = rounded(rows.P3_full.sum(), 4)
        if base is None:
            moved_in = rows.player_id[~rows.player_id.isin(own.player_id)]
        else:
            moved_in = rows.player_id[~rows.player_id.isin(base.player_id)]
            moved_out = base.player_id[~base.player_id.isin(rows.player_id)]
            change = tuple(rounded(x, 4) for x in gain(rows, base))
    return {
        "budget": plan["budget"] // 1_000_000,
        "cap": plan["cap"],
        "fit": plan["fit"],
        "status": "infeasible" if rows is None else "ok",
        "reason": plan["reason"],
        "in": sorted(int(p) for p in moved_in),
        "out": sorted(int(p) for p in moved_out),
        "total": total,
        "gain": change[0],
        "low": change[1],
        "high": change[2],
    }


def check_club(own, base, plans, score, prices):
    """HARD STOP unless the baseline is own players only and fills the quotas, and every plan
    filled fills them within its budget with at most its cap of signings, each with a fit row
    for the club, priced and, under the fit condition, with a fit score above 0. score holds the
    club's fit scores and prices the priced players' prices, each by player_id."""
    if base is not None:
        filled = base.group.value_counts().to_dict() == squad.QUOTAS
        assert filled and base.player_id.isin(own.player_id).all(), "HARD STOP: a baseline"
    for p in plans:
        rows = p["rows"]
        if rows is None:
            continue
        signed = rows.player_id[~rows.player_id.isin(own.player_id)]
        ok = (
            rows.group.value_counts().to_dict() == squad.QUOTAS
            and signed.isin(score.index).all()
            and signed.isin(prices.index).all()
            and prices.reindex(signed).sum() <= p["budget"]
            and len(signed) <= p["cap"]
            and (not p["fit"] or (score.reindex(signed) > 0).all())
        )
        assert ok, f"HARD STOP: the plan at {p['budget']}, cap {p['cap']}, fit {p['fit']}"


def fit_rows(fit):
    """A club's candidates in id order with their fit scores."""
    f = fit.sort_values("player_id")
    return [[int(p), round(float(s), 3)] for p, s in zip(f.player_id, f.score, strict=True)]


def similar(own_ids, pairs):
    """Per own player, by his id as a string, his candidates by distance and then id: the
    candidate's id, the distance, the closeness, and 1 if the closeness is at least NOISE."""
    pairs = pairs.sort_values(["player_id", "distance", "player_id_c"])
    at = pairs.groupby("player_id").indices
    out = {}
    for pid in sorted(own_ids):
        p = pairs.iloc[at.get(pid, [])]
        out[str(pid)] = [
            [int(c), round(float(d), 3), round(float(x), 3), int(x >= NOISE)]
            for c, d, x in zip(p.player_id_c, p.distance, p.closeness, strict=True)
        ]
    return out


def meta(counts):
    return {
        "dimensions": [
            {"key": d, "label": DIMENSION_LABELS[d], "less_reliable": d in LESS_RELIABLE}
            for d in style.DIMENSIONS
        ],
        "groups": [{"key": g, "label": GROUP_LABELS[g]} for g in sh.OUTFIELD],
        "leagues": [{"key": lg, "label": LEAGUE_LABELS[lg]} for lg in LEAGUES],
        "bands": BAND_LABELS,
        "budgets_m": [b // 1_000_000 for b in BUDGETS],
        "caps": CAPS,
        "quotas": squad.QUOTAS,
        "interval": 0.9,
        "noise_closeness": NOISE,
        "fit_condition": 0,
        "counts": counts,
    }


def numbers(obj, path=()):
    """Every number in obj, bools aside, with its path of keys and list positions."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from numbers(v, (*path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from numbers(v, (*path, i))
    elif isinstance(obj, int | float) and not isinstance(obj, bool):
        yield path, obj


def is_id(path):
    return any(
        len(p) == len(path) and all(a in ("*", b) for a, b in zip(p, path, strict=True))
        for p in ID_PATHS
    )


def check_numbers(files):
    """The largest number of the files outside an id position; HARD STOP if a number is not
    finite, or one outside an id position is CEILING or more."""
    largest = None
    for obj in files.values():
        for path, x in numbers(obj):
            assert math.isfinite(x), f"HARD STOP: a number at {path} is not finite"
            if not is_id(path):
                assert x < CEILING, f"HARD STOP: {x!r} at {path} is {CEILING} or more"
                largest = x if largest is None else max(largest, x)
    return largest


def dumps(obj):
    """obj as compact JSON in UTF-8 with a final newline; NaN and infinities raise."""
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode("utf-8")


def write(out, payload):
    """Replace the folder out, which must end in web/data, with the files of payload."""
    assert out.parts[-2:] == ("web", "data"), f"HARD STOP: {out} is not a web/data folder"
    if out.exists():
        shutil.rmtree(out)
    for path, content in payload.items():
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def read_files(folder):
    """Every file under folder by its path relative to it, with forward slashes."""
    paths = sorted(p for p in folder.rglob("*") if p.is_file())
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in paths}


def digest(files):
    """The sha256 of the lines '<path> <sha256 of the file>', in path order."""
    lines = "".join(f"{p} {hashlib.sha256(files[p]).hexdigest()}\n" for p in sorted(files))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def build(out=OUT):
    start = time.perf_counter()
    value, decision, gem, later, season, teams, fit, pairs, names = inputs()
    rows = player_rows(value, decision, gem, later, season, names)
    priced = rows[rows.reason == "priced"]
    groups = {g: int((rows.group == g).sum()) for g in sh.OUTFIELD}
    with_2015 = int(priced.value_2015.notna().sum())
    with_later = int((priced.value_1 > 0).sum())
    key(
        f"10a: players {len(rows)}, per group {groups}, priced {len(priced)}, with a 2015 value "
        f"{with_2015}, with a year-later value {with_later}, clubs {len(teams)}"
    )
    same = (
        value[KEYS]
        .sort_values(KEYS)
        .reset_index(drop=True)
        .equals(season[KEYS].sort_values(KEYS).reset_index(drop=True))
    )
    both = value[[*KEYS, "group"]].merge(
        season[[*KEYS, "group"]], on=KEYS, how="left", suffixes=("", "_s"), validate="one_to_one"
    )
    population = (
        len(value) == len(season) == PLAYERS
        and value.player_id.is_unique
        and same
        and (both.group == both.group_s).all()
        and groups == GROUPS
    )
    assert population, "HARD STOP: the players of gems_value and style_player_season"
    ok = (
        len(priced) == gems.PRICED
        and priced.price.notna().all()
        and (priced.price > 0).all()
        and priced.gem.notna().all()
        and with_2015 == gems.WITH_2015
        and with_later == gems.PRICED
    )
    assert ok, "HARD STOP: the priced players"
    clubs = pd.MultiIndex.from_frame(teams[style.TEAM_KEYS])
    inside = pd.MultiIndex.from_frame(rows[["league", "team_id"]]).isin(clubs).all()
    assert len(teams) == style.CLUBS and clubs.is_unique and inside, "HARD STOP: the clubs"
    once = not fit.duplicated(["league_k", "team_id_k", "player_id"]).any()
    tables = len(fit) == FIT_ROWS and once and len(pairs) == PAIR_ROWS
    assert tables, "HARD STOP: the fit or pairs table"

    amounts = [priced.price, priced.value_2015.dropna(), priced.value_1]
    smallest = int(min(a.min() for a in amounts))
    key(f"10a: smallest Transfermarkt amount used {smallest}")
    assert smallest >= CEILING, "HARD STOP: a Transfermarkt amount below the ceiling"

    table = club_table(teams, rows)
    assert table["name"].notna().all(), "HARD STOP: a club without a name"
    prices = priced.set_index("player_id").price.astype("int64")
    fits = dict(tuple(fit.groupby(["league_k", "team_id_k"])))
    near = dict(tuple(pairs.groupby(["league", "team_id"])))
    club_files, short_of, tally, gains = {}, {}, [], []
    for c in table.itertuples():
        at = (rows.league == c.league) & (rows.team_id == c.team_id)
        own = rows.loc[at, ["player_id", "group", "P3_full", "P3_sd"]].sort_values("player_id")
        f = fits[c.league, c.team_id]
        score = f.set_index("player_id").score
        cands = priced.loc[priced.player_id.isin(f.player_id), own.columns]
        cands = cands.assign(price=cands.player_id.map(prices), score=cands.player_id.map(score))
        cands = cands.sort_values("player_id")
        base, short, plans = club_plans(own, cands)
        check_club(own, base, plans, score, prices)
        records = [plan_record(p, own, base) for p in plans]
        for p in plans:
            signed = None
            if p["rows"] is not None:
                signed = int((~p["rows"].player_id.isin(own.player_id)).sum())
            tally.append((p["reason"], signed))
        gains += [r["gain"] for r in records if r["gain"] is not None]
        if short:
            short_of[c.key] = short
        baseline = None
        if base is not None:
            baseline = {
                "players": sorted(int(p) for p in base.player_id),
                "total": rounded(base.P3_full.sum(), 4),
            }
        club_files[f"clubs/{c.key}.json"] = {
            "key": c.key,
            "fit": fit_rows(f),
            "similar": similar(own.player_id, near[c.league, c.team_id]),
            "baseline": baseline,
            "short": short,
            "plans": records,
        }
        print(
            f"club {c.key} {c.name}: own {len(own)}, candidates {len(cands)}, with a fit score "
            f"above 0 {int((cands.score > 0).sum())}, baseline total "
            f"{None if baseline is None else baseline['total']}"
        )

    key(f"10a: clubs with a baseline {len(table) - len(short_of)}")
    for k, short in short_of.items():
        key(f"10a: no baseline {k} short {short}")
    reasons = [r for r, _ in tally]
    listed = ", ".join(f"{r} {reasons.count(r)}" for r in REASONS)
    key(f"10a: plans {len(tally)}: ok {reasons.count(None)}, {listed}")
    made = {k: sum(n == k for _, n in tally) for k in range(max(CAPS) + 1)}
    key(f"10a: ok plans by signings made {made}")
    print(f"10a: gains of the plans with a baseline, smallest {min(gains)}, largest {max(gains)}")

    counts = {"players": len(rows), "priced": len(priced), "clubs": len(table)}
    files = {
        "meta.json": meta(counts),
        "players.json": {"players": player_records(rows)},
        "clubs.json": {"clubs": club_records(table, rows)},
        **club_files,
    }
    key(f"10a: largest number outside id positions {check_numbers(files)}")
    payload = {path: dumps(obj) for path, obj in files.items()}
    identical = payload == {path: dumps(obj) for path, obj in files.items()}
    key(f"10a: second serialization identical {identical}")
    assert identical, "HARD STOP: a second serialization gave other bytes"

    write(out, payload)
    written = read_files(out)
    for path in sorted(written):
        content = written[path]
        key(f"10a: file {path} {len(content)} {hashlib.sha256(content).hexdigest()}")
    size = sum(len(content) for content in written.values())
    key(f"10a: files {len(written)}, bytes {size}, digest {digest(written)}")
    key(f"10a: {time.perf_counter() - start:.1f} s")


def main(argv):
    step = argv[0]
    run = {"build": build}[step]
    sh.LOGS.mkdir(parents=True, exist_ok=True)
    full = open(sh.LOGS / f"p10a_{step}.log", "w", encoding="utf-8", errors="replace")
    brief = open(sh.LOGS / f"p10a_{step}_summary.log", "w", encoding="utf-8", errors="replace")
    sys.stdout = Tee(full, brief)
    try:
        run()
    finally:
        sys.stdout = sys.__stdout__
        full.close()
        brief.close()


if __name__ == "__main__":
    main(sys.argv[1:])
