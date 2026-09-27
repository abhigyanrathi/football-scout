import math
import sys

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from fbrecruit import shrinkage as sh
from fbrecruit import split
from fbrecruit.links import data
from fbrecruit.logs import Tee, key
from fbrecruit.paths import PROCESSED
from fbrecruit.sources.statsbomb import LEAGUES, OUT

BRANCH = "pooled"
KEYS = sh.KEYS
PAIRS = 1144
OUTFIELD_PAIRS = 1067
ROWS = 1401
OUTFIELD_ROWS = 1299
REDRAWS = 2000
TEAMS = 20
VALUATIONS = 656301
VALUATION_DAYS = 365
# in the order they are tried: a row gets the first that applies
REASONS = ["no Transfermarkt id", "shared Transfermarkt id", "no valuation", "priced"]
QUOTAS = {"CB": 2, "FB": 2, "MID": 3, "WIDE": 2, "FWD": 1}
QUANTILES = [0.25, 0.5, 0.75]


def draws_path():
    # written by fbrecruit.ablation, which is not imported here because it imports torch
    return PROCESSED / "ablation_draws.parquet"


def links_path():
    return data.LINKS / "statsbomb_tm_players.parquet"


def pool_path():
    return PROCESSED / "squad_pool.parquet"


def evaluation():
    """The pairs: keys, group, P2 and P3, the only columns of the evaluation table read here."""
    return pd.read_parquet(sh.evaluation_path(BRANCH), columns=[*KEYS, "group", "P2", "P3"])


def estimation():
    cols = [*KEYS, "player_name", "group", "vaep_per90", "theta"]
    return pd.read_parquet(sh.shrinkage_path(BRANCH), columns=cols)


def outfield_values(est):
    """The outfield estimation rows with P2 and P3, as shrinkage.build_evaluation computes them."""
    rows = est[est.group.isin(sh.OUTFIELD)].reset_index(drop=True)
    c1 = rows.league.map(sh.league_levels(BRANCH).xs(1, level="window"))
    return rows.assign(P2=rows.vaep_per90 - c1, P3=rows.theta - c1)


def first_window2():
    """The date and time of each league's first window-2 game."""
    first = {}
    for lg in LEAGUES:
        g = split.assign_windows(split.games(lg))
        first[lg] = g.game_date[g.window == 2].min()
    return pd.Series(first, name="first_window2")


def cutoff(first):
    """The day of the earliest of the leagues' first window-2 games."""
    return first.min().normalize()


def inputs():
    ev = evaluation()
    est = estimation()
    draws = pd.read_parquet(draws_path())
    links = pd.read_parquet(links_path())
    valuations = len(data.tm_table("player_valuations", ["player_id"]))

    pairs = ev[ev.group.isin(sh.OUTFIELD)]
    rows = outfield_values(est)
    duplicated = int(est.duplicated(KEYS).sum())
    missing = int(rows.theta.isna().sum())
    per = draws.groupby(["replicate", "league"])["count"].sum()
    m = pairs.merge(
        rows, on=KEYS, how="left", suffixes=("", "_rows"), indicator=True, validate="one_to_one"
    )
    matched = int((m._merge == "both").sum())
    diff = {p: (m[p] - m[f"{p}_rows"]).abs().max() for p in ("P2", "P3")}

    key(f"1a: evaluation rows {len(ev)}, outfield {len(pairs)}")
    key(
        f"1b: estimation rows {len(est)}, duplicated on {KEYS} {duplicated}, "
        f"outfield {len(rows)}, outfield with a missing theta {missing}"
    )
    key(
        f"1c: ablation_draws rows {len(draws)}, replicates {draws.replicate.nunique()}, "
        f"replicate-league sums {len(per)}, smallest {per.min()}, largest {per.max()}"
    )
    key(
        f"1d: statsbomb_tm_players rows {len(links)}, player_id unique "
        f"{links.player_id.is_unique}; player_valuations rows {valuations}"
    )
    key(
        f"1e: outfield pairs matching one outfield row {matched} of {len(m)}; largest absolute "
        f"difference from the stored P2 {diff['P2']!r}, from the stored P3 {diff['P3']!r}"
    )
    checks = {
        "1a 1,144 pairs, 1,067 outfield": (len(ev), len(pairs)) == (PAIRS, OUTFIELD_PAIRS),
        "1b 1,401 estimation rows, unique, 1,299 outfield, no missing theta": (
            (len(est), duplicated, len(rows), missing) == (ROWS, 0, OUTFIELD_ROWS, 0)
        ),
        "1c 2,000 replicates, each league's counts summing to 20": (
            (draws.replicate.nunique(), len(per)) == (REDRAWS, REDRAWS * len(LEAGUES))
            and (per == TEAMS).all()
        ),
        "1d links unique on player_id, 656,301 valuations": (
            links.player_id.is_unique and valuations == VALUATIONS
        ),
        "1e every outfield pair matched, P2 and P3 equal": (
            (matched, diff["P2"], diff["P3"]) == (OUTFIELD_PAIRS, 0.0, 0.0)
        ),
    }
    for name, passed in checks.items():
        key(f"{name}: {passed}")
    assert all(checks.values()), "HARD STOP: an input check failed"

    first = first_window2()
    key("\n1f: first window-2 game per league")
    key(first.to_string())
    key(f"1f: earliest of the four {first.min()} ({first.idxmin()})")


def team_names():
    frames = [
        pd.read_parquet(OUT / lg / "teams.parquet", columns=["team_id", "team_name"]).assign(
            league=lg
        )
        for lg in LEAGUES
    ]
    return pd.concat(frames, ignore_index=True)


def latest_valuations(ids, cut):
    """Per Transfermarkt id in ids, the latest valuation above zero dated before the cutoff day
    and no more than 365 days before it."""
    v = data.tm_table("player_valuations", ["player_id", "date", "market_value_in_eur"])
    start = cut - pd.Timedelta(days=VALUATION_DAYS)
    v = v[v.player_id.isin(ids) & (v.market_value_in_eur > 0) & (v.date < cut) & (v.date >= start)]
    last = v[v.date == v.groupby("player_id").date.transform("max")]
    values = last.groupby("player_id").market_value_in_eur.nunique()
    key(
        f"2: valuations above zero from {start:%Y-%m-%d} to the day before the cutoff: ids with "
        f"any {len(values)}, with two different values on their latest date {(values > 1).sum()}"
    )
    assert (values == 1).all(), "HARD STOP: two valuations on a player's latest date differ"
    last = last.drop_duplicates("player_id")
    return pd.DataFrame(
        {
            "tm_player_id": last.player_id.astype("Int64"),
            "valuation_date": last.date,
            "price": last.market_value_in_eur.astype("Int64"),
        }
    )


def pool():
    first = first_window2()
    cut = cutoff(first)
    gap = (first.dt.normalize() - cut).dt.days
    key("2: first window-2 game per league")
    key(first.to_string())
    key(f"2: cutoff {cut:%Y-%m-%d}")
    key(f"2: days from the cutoff to each league's first window-2 day {gap.to_dict()}")
    key(f"2: largest gap {gap.max()} days ({gap.idxmax()})")

    ev = evaluation()
    pairs = ev[ev.group.isin(sh.OUTFIELD)]
    links = pd.read_parquet(links_path(), columns=["player_id", "tm_player_id"])
    rows = outfield_values(estimation())
    rows = rows.merge(team_names(), on=["league", "team_id"], how="left", validate="many_to_one")
    rows["pair"] = pd.MultiIndex.from_frame(rows[KEYS]).isin(pd.MultiIndex.from_frame(pairs[KEYS]))
    assert rows.team_name.notna().all(), "a team has no name"
    assert rows.pair.sum() == OUTFIELD_PAIRS, "an outfield pair is not an outfield row"
    rows = rows.merge(links, on="player_id", how="left", validate="many_to_one")

    held = links.tm_player_id.value_counts()
    no_id = rows.tm_player_id.isna().to_numpy()
    shared = rows.tm_player_id.isin(held.index[held > 1]).to_numpy(dtype=bool)
    ids = rows.tm_player_id[~no_id & ~shared]
    rows = rows.merge(
        latest_valuations(ids, cut), on="tm_player_id", how="left", validate="many_to_one"
    )
    no_valuation = rows.price.isna().to_numpy()
    rows["reason"] = np.select([no_id, shared, no_valuation], REASONS[:-1], default=REASONS[-1])

    cols = [*KEYS, "player_name", "team_name", "group", "pair", "P2", "P3"]
    cols += ["tm_player_id", "valuation_date", "price", "reason"]
    out = rows[cols].sort_values(KEYS).reset_index(drop=True)
    out.to_parquet(pool_path(), index=False)

    parts = {"pairs": out[out.pair], "all rows": out}
    counts = {n: p.reason.value_counts().reindex(REASONS, fill_value=0) for n, p in parts.items()}
    key("\n2: rows per reason")
    key(pd.DataFrame(counts).to_string())
    for name, p in parts.items():
        for by, order in (("league", list(LEAGUES)), ("group", sh.OUTFIELD)):
            t = p.assign(priced=p.reason == "priced").groupby(by)
            t = t.agg(rows=("priced", "size"), priced=("priced", "sum")).reindex(order)
            key(f"\n2: {name}, priced by {by}")
            key(t.to_string())

    priced = out[out.pair & (out.reason == "priced")]
    age = (cut - priced.valuation_date).dt.days
    key(
        f"\n2: priced pairs {len(priced)}: price min {priced.price.min()}, median "
        f"{priced.price.median()}, max {priced.price.max()}; valuation age at the cutoff in days, "
        f"median {age.median()}, max {age.max()}"
    )
    dup = links[links.tm_player_id.notna() & links.tm_player_id.duplicated(keep=False)]
    key(f"2: Transfermarkt ids held by more than one StatsBomb player {dup.tm_player_id.nunique()}")
    if len(dup):
        key(dup.sort_values(["tm_player_id", "player_id"]).to_string(index=False))
    key(f"2: wrote {pool_path()} with {len(out)} rows, columns {list(out.columns)}")


def priced_pairs(pool):
    p = pool[pool.pair & (pool.reason == "priced")]
    return p.assign(price=p.price.astype("int64"))


def budgets(prices):
    """Ten times each quantile of the prices, a Series, rounded down to whole euros."""
    return {q: math.floor(10 * prices.quantile(q)) for q in QUANTILES}


def cheapest_cost(prices, groups, quotas=QUOTAS):
    """The cost of the cheapest squad the quotas allow, each group's quota of its lowest prices;
    infinite if a group has fewer rows than its quota."""
    prices, groups = np.asarray(prices), np.asarray(groups)
    cost = 0
    for g, k in quotas.items():
        p = np.sort(prices[groups == g])
        if len(p) < k:
            return math.inf
        cost += p[:k].sum()
    return cost


def costs():
    priced = priced_pairs(pd.read_parquet(pool_path()))
    levels = budgets(priced.price)
    full = cheapest_cost(priced.price, priced.group)
    per_group = priced.group.value_counts().reindex(list(QUOTAS))
    key(f"3: priced pairs {len(priced)}, per group {per_group.to_dict()}")
    for q, b in levels.items():
        key(f"3: q {q}: ten times the quantile {10 * priced.price.quantile(q)!r}, budget {b}")
    key(f"3: cheapest squad from the full pool {full}")

    draws = pd.read_parquet(draws_path(), columns=["replicate", "league", "team_id"])
    rows = draws.merge(priced[["league", "team_id", "group", "price"]], on=["league", "team_id"])
    counts = rows.groupby(["replicate", "group"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=range(REDRAWS), columns=list(QUOTAS), fill_value=0)
    cost = pd.Series({r: cheapest_cost(d.price, d.group) for r, d in rows.groupby("replicate")})
    cost = cost.reindex(range(REDRAWS))
    short = (counts < pd.Series(QUOTAS)).any(axis=1)
    over = ~(cost <= levels[0.25])
    key(f"\n3: redraws {len(counts)}; smallest priced count per group {counts.min().to_dict()}")
    key(f"3: largest cheapest squad over the redraws {cost.max()} (redraw {cost.idxmax()})")
    key(f"3: redraws with a group below its quota {int(short.sum())}")
    key(f"3: redraws whose cheapest squad costs more than {levels[0.25]} {int(over.sum())}")
    key(f"3: full pool's cheapest squad above {levels[0.25]} {full > levels[0.25]}")
    assert not short.any(), "HARD STOP: a redraw has fewer priced pairs in a group than its quota"
    assert full <= levels[0.25] and not over.any(), "HARD STOP: a cheapest squad is over budget"


def pick(values, prices, groups, players, budget, quotas=QUOTAS):
    """The rows of highest summed value that fill each group's quota exactly, with summed price
    at most the budget and at most one row per player; their positions in ascending order."""
    values, prices = np.asarray(values, dtype=float), np.asarray(prices)
    groups, players = np.asarray(groups), np.asarray(players)
    unknown = sorted(set(groups) - set(quotas))
    if unknown:
        raise ValueError(f"groups without a quota: {unknown}")
    ids, rows = np.unique(players, return_counts=True)
    repeated = ids[rows > 1]
    # the solver works in millions of euros; the squad it returns is checked in whole euros
    a = np.array(
        [*(groups == g for g in quotas), prices / 1e6, *(players == p for p in repeated)],
        dtype=float,
    )
    lower = [*quotas.values(), -np.inf, *[-np.inf] * len(repeated)]
    upper = [*quotas.values(), budget / 1e6, *[1] * len(repeated)]
    res = milp(
        -values,
        integrality=np.ones(len(values)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(a, lower, upper),
        options={"mip_rel_gap": 0},
    )
    if res.status != 0:
        raise RuntimeError(f"milp status {res.status}: {res.message}")
    chosen = np.flatnonzero(res.x > 0.5)
    counts = {g: int((groups[chosen] == g).sum()) for g in quotas}
    cost = prices[chosen].sum()
    if counts != quotas or cost > budget or len(set(players[chosen])) < len(chosen):
        raise RuntimeError(
            f"the squad breaks a rule: groups {counts}, cost {cost}, budget {budget}"
        )
    return chosen


def fill(order, prices, groups, players, budget, quotas=QUOTAS):
    """Rows kept in the given order while their group has an open place, their player is not yet
    in the squad and their price leaves enough of the budget to fill every other open place with
    the cheapest rows left of players not yet in the squad; their positions in ascending order."""
    prices, groups, players = np.asarray(prices), np.asarray(groups), np.asarray(players)
    places = dict(quotas)
    squad, spent = [], 0
    for i in order:
        g = groups[i]
        if not places.get(g) or players[i] in players[squad]:
            continue
        rest = places | {g: places[g] - 1}
        free = ~np.isin(players, [*players[squad], players[i]])
        if spent + prices[i] + cheapest_cost(prices[free], groups[free], rest) <= budget:
            squad.append(i)
            spent += prices[i]
            places[g] -= 1
            if not any(places.values()):
                return np.sort(squad)
    raise ValueError("the order ended before every place was filled")


def main(argv):
    step = argv[0]
    sh.LOGS.mkdir(parents=True, exist_ok=True)
    full = open(sh.LOGS / f"p7_{step}.log", "w", encoding="utf-8", errors="replace")
    brief = open(sh.LOGS / f"p7_{step}_summary.log", "w", encoding="utf-8", errors="replace")
    sys.stdout = Tee(full, brief)
    try:
        if step == "inputs":
            inputs()
        elif step == "pool":
            pool()
        elif step == "budgets":
            costs()
    finally:
        sys.stdout = sys.__stdout__
        full.close()
        brief.close()


if __name__ == "__main__":
    main(sys.argv[1:])
