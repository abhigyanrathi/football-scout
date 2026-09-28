import math
import sys

import numpy as np
import pandas as pd

from fbrecruit import shrinkage as sh
from fbrecruit import split, squad
from fbrecruit.links.teams import COMPETITIONS
from fbrecruit.logs import Tee, key
from fbrecruit.paths import INTERIM, PROCESSED
from fbrecruit.sources.statsbomb import LEAGUES

BRANCH = "pooled"
KEYS = sh.KEYS
MINUTES = 900
TOLERANCE = 1e-12
SUMMER_END = pd.Timestamp("2016-08-31")
VALUE_2015 = (pd.Timestamp("2015-05-01"), pd.Timestamp("2015-08-31"))
SEASON = (pd.Timestamp("2015-07-01"), pd.Timestamp("2016-06-30"))
AGE_DAY = pd.Timestamp("2016-07-01")
# in the order they are tried: a row gets the first that applies
REASONS = [
    "no Transfermarkt id",
    "shared Transfermarkt id",
    "no summer valuation",
    "no date of birth",
    "priced",
]
VALUE_COLUMNS = [
    *KEYS,
    "player_name",
    "team_name",
    "group",
    "minutes",
    "P2_full",
    "P3_full",
    "P3_sd",
]
ADDED = [
    "tm_player_id",
    "price",
    "price_date",
    "value_2015",
    "value_2015_date",
    "age",
    "ga90",
    "reason",
]
# la_liga and CB have no indicator
FIT_LEAGUES = ["premier_league", "serie_a", "ligue_1"]
FIT_GROUPS = ["FB", "MID", "WIDE", "FWD"]
CONTROLS = ["constant", "log_price", "age", "age_squared", *FIT_LEAGUES, *FIT_GROUPS]
LISTED = 30
GEM_LIST = [
    "group",
    "player_name",
    "team_name",
    "league",
    "age",
    "price",
    "P3_full",
    "P3_sd",
    "gem",
]
TOP = 100
AGE_EDGES = [-np.inf, 23, 27, 31, np.inf]
AGE_BANDS = ["under 23", "23 to 26", "27 to 30", "31 and over"]
# z of 0.975 plus z of 0.8: a two-sided test at 5% with power 0.8
Z = 2.80158


def value_path():
    return PROCESSED / "gems_value.parquet"


def decision_path():
    return PROCESSED / "gems_decision.parquet"


def scores_path():
    return PROCESSED / "gems_scores.parquet"


def check_levels(c):
    """HARD STOP unless each league's c(league, 1) and c(league, 2) equal the evaluation table's c1
    and c2 to within 1e-12."""
    ev = pd.read_parquet(sh.evaluation_path(BRANCH), columns=["league", "c1", "c2"])
    ev = ev.drop_duplicates()
    key(f"2a: distinct (league, c1, c2) rows of the evaluation table {len(ev)}")
    assert ev.league.is_unique and set(ev.league) == set(LEAGUES), "HARD STOP: c1, c2 per league"
    ev = ev.set_index("league")
    worst = 0.0
    for lg in LEAGUES:
        for w in (1, 2):
            stored = ev.loc[lg, f"c{w}"]
            worst = max(worst, abs(c[lg, w] - stored))
            key(f"2a: c({lg}, {w}) {c[lg, w]!r}, evaluation c{w} {stored!r}")
    key(f"2a: largest absolute difference {worst!r}")
    assert worst <= TOLERANCE, "HARD STOP: a league-window level differs from the evaluation table"


def full_season(t, c):
    """One row per player-team of the window table: its minutes over both windows and y, the
    minutes-weighted mean of its two window rates, each less its league-window level. A missing
    window has no minutes and no value."""
    cols = [*KEYS, "minutes", "vaep_sum"]
    w = t[t.window == 1][cols].merge(
        t[t.window == 2][cols], on=KEYS, how="outer", suffixes=("_1", "_2"), validate="one_to_one"
    )
    m1, m2 = w.minutes_1.fillna(0), w.minutes_2.fillna(0)
    s1, s2 = w.vaep_sum_1.fillna(0.0), w.vaep_sum_2.fillna(0.0)
    c1 = w.league.map(c.xs(1, level="window"))
    c2 = w.league.map(c.xs(2, level="window"))
    minutes = (m1 + m2).astype("int64")
    y = (90 * (s1 + s2) - m1 * c1 - m2 * c2) / minutes
    return w[KEYS].assign(minutes=minutes, y=y).sort_values(KEYS).reset_index(drop=True)


def dispersion(s):
    """phi_g of each outfield group, psi times window-1 minutes over 8100 on phase 4c's estimation
    rows; HARD STOP unless it is the same on every row of the group to within a relative 1e-12."""
    s = s[s.group.isin(sh.OUTFIELD)]
    each = s.psi * s.minutes / sh.PER90
    phi = each.groupby(s.group).mean()
    deviation = ((each - s.group.map(phi)).abs() / s.group.map(phi)).groupby(s.group).max()
    for grp in sh.OUTFIELD:
        key(
            f"2d: {grp}: rows {int((s.group == grp).sum())}, phi {phi[grp]!r}, "
            f"largest relative deviation {deviation[grp]!r}"
        )
    assert (deviation <= TOLERANCE).all(), "HARD STOP: psi x minutes / 8100 varies within a group"
    return phi


def shrink(rows, phi):
    """Each group's Fay-Herriot fit of y on the league indicators and log(minutes / 1000), with
    psi = 8100 phi_g / minutes: P2_full is y, P3_full the posterior mean and P3_sd its standard
    deviation."""
    league = rows.league.to_numpy()
    mins = rows.minutes.to_numpy(dtype=float)
    y = rows.y.to_numpy(dtype=float)
    psi = sh.PER90 * rows.group.map(phi).to_numpy() / mins
    theta, sd, fits = np.full(len(rows), np.nan), np.full(len(rows), np.nan), {}
    for grp, idx in sorted(rows.groupby("group").indices.items()):
        f = sh.fay_herriot(y[idx], sh.design(league[idx], mins[idx], True), psi[idx])
        theta[idx] = f["theta"]
        sd[idx] = np.sqrt(f["tau2"] * psi[idx] / (f["tau2"] + psi[idx]))
        fits[grp] = f
    return rows.drop(columns="y").assign(P2_full=y, P3_full=theta, P3_sd=sd), fits


def one_per_player(rows):
    """Each player's row with the most minutes; of rows with equal minutes, the first in key
    order."""
    order = rows.sort_values(["minutes", *KEYS], ascending=[False, True, True, True], kind="stable")
    return order.drop_duplicates("player_id").sort_values(KEYS).reset_index(drop=True)


def build_value(t, c, g, s, teams):
    """One row per outfield player with at least 900 minutes over the season, with P2_full,
    P3_full and P3_sd, and each group's fit."""
    rows = full_season(t, c)
    kept = rows[rows.minutes >= MINUTES].reset_index(drop=True)
    key(f"2b: player-team rows {len(rows)}, with at least {MINUTES} minutes {len(kept)}")

    direct = kept.merge(g[[*KEYS, "group"]], on=KEYS, how="left", validate="one_to_one").group
    n = g.groupby("player_id").group.nunique()
    single = g[g.player_id.isin(n.index[n == 1])].drop_duplicates("player_id")
    by_player = kept.player_id.map(single.set_index("player_id").group)
    group = direct.fillna(by_player)
    key(
        f"2c: group found on the three keys {int(direct.notna().sum())}, "
        f"on player_id alone {int((direct.isna() & by_player.notna()).sum())}"
    )
    left = {"no group": group.isna(), "GK": group == "GK", "UNKNOWN": group == "UNKNOWN"}
    key("2c: left out: " + ", ".join(f"{k} {int(v.sum())}" for k, v in left.items()))
    rows = kept.assign(group=group)[group.isin(sh.OUTFIELD)].reset_index(drop=True)
    key(f"2c: outfield rows {len(rows)}")

    rows, fits = shrink(rows, dispersion(s))
    for grp, f in fits.items():
        key(f"2e: {grp}: m {f['m']!r}, tau2 {f['tau2']!r}, truncated at zero {f['truncated']}")
        key(f"2e: {grp}: b " + ", ".join(repr(float(v)) for v in f["b"]))
        key(
            f"2e: {grp}: tau2 / (tau2 + psi) smallest {f['shrink'].min()!r}, "
            f"mean {f['shrink'].mean()!r}, largest {f['shrink'].max()!r}"
        )

    per = rows.player_id.value_counts()
    most = rows.minutes == rows.groupby("player_id").minutes.transform("max")
    key(
        f"2f: players {len(per)}, with two rows {int((per == 2).sum())}, with more than two "
        f"{int((per > 2).sum())}, with two rows of the same, largest, minutes "
        f"{int(rows[most].player_id.duplicated().sum())}"
    )
    rows = one_per_player(rows)
    names = t.sort_values([*KEYS, "window"]).drop_duplicates(KEYS)[[*KEYS, "player_name"]]
    rows = rows.merge(names, on=KEYS, how="left", validate="one_to_one")
    rows = rows.merge(teams, on=["league", "team_id"], how="left", validate="many_to_one")
    assert rows.team_name.notna().all(), "a team has no name"
    key(f"2g: rows with no player_name {int(rows.player_name.isna().sum())}")
    return rows[VALUE_COLUMNS], fits


def value_inputs():
    t = sh.v2_table(BRANCH)
    g = pd.read_parquet(sh.group_path(), columns=[*KEYS, "group"])
    s = pd.read_parquet(sh.shrinkage_path(BRANCH), columns=["group", "minutes", "psi"])
    return t, sh.league_levels(BRANCH), g, s, squad.team_names()


def value():
    t, c, g, s, teams = value_inputs()
    check_levels(c)
    out, _ = build_value(t, c, g, s, teams)
    out.to_parquet(value_path(), index=False)
    key(f"\n2h: wrote {value_path()} with {len(out)} rows, columns {list(out.columns)}")
    key(out.league.value_counts().reindex(list(LEAGUES)).to_string())
    key(out.group.value_counts().reindex(sh.OUTFIELD).to_string())
    for col in ["P3_full", "P3_sd"]:
        x = out[col]
        key(f"2h: {col} smallest {x.min()!r}, median {x.median()!r}, largest {x.max()!r}")


def valuations():
    """Transfermarkt valuations dated on or before 2016-08-31, filtered as the file is read."""
    return pd.read_parquet(
        INTERIM / "transfermarkt" / "player_valuations.parquet",
        columns=["player_id", "date", "market_value_in_eur"],
        filters=[("date", "<=", SUMMER_END)],
    )


def appearances():
    """Transfermarkt appearances in the four leagues dated from 2015-07-01 to 2016-06-30, filtered
    as the file is read."""
    return pd.read_parquet(
        INTERIM / "transfermarkt" / "appearances.parquet",
        columns=["player_id", "date", "competition_id", "minutes_played", "goals", "assists"],
        filters=[
            ("date", ">=", SEASON[0]),
            ("date", "<=", SEASON[1]),
            ("competition_id", "in", list(COMPETITIONS.values())),
        ],
    )


def last_games():
    """The date and time of each league's last 2015/16 game."""
    return pd.Series({lg: split.games(lg).game_date.max() for lg in LEAGUES}, name="last_game")


def latest(v, start, end):
    """Per Transfermarkt id, the latest valuation above zero dated from start to end, both days
    included: its value and date, indexed by id."""
    v = v[(v.market_value_in_eur > 0) & (v.date >= start) & (v.date <= end)]
    top = v[v.date == v.groupby("player_id").date.transform("max")]
    values = top.groupby("player_id").market_value_in_eur.nunique()
    assert (values == 1).all(), "HARD STOP: two valuations on a player's latest date differ"
    return top.drop_duplicates("player_id").set_index("player_id")[["market_value_in_eur", "date"]]


def goals_assists(apps):
    """Goals plus assists per 90 minutes per Transfermarkt id, where its minutes are above zero."""
    by = apps.assign(ga=apps.goals + apps.assists).groupby("player_id")
    s = by[["ga", "minutes_played"]].sum()
    s = s[s.minutes_played > 0]
    return 90 * s.ga / s.minutes_played


def build_decision(value, links, v, born, apps, last):
    """value's rows with their Transfermarkt id, price, 2015 value, age and goals and assists per
    90, and the reason each row is priced or not."""
    rows = value.merge(links, on="player_id", how="left", validate="many_to_one")
    held = links.tm_player_id.value_counts()
    no_id = rows.tm_player_id.isna().to_numpy()
    shared = rows.tm_player_id.isin(held.index[held > 1]).to_numpy(dtype=bool)
    rows["tm_player_id"] = rows.tm_player_id.astype("Int64")
    # only an id that is neither missing nor shared is looked up
    ids = rows.tm_player_id[~no_id & ~shared].astype("int64")
    v = v[v.player_id.isin(ids)]
    league = rows.league[ids.index]
    # a price is dated after the last game of the player's league: from the next day
    summer = pd.concat(
        [
            latest(v[v.player_id.isin(ids[league == lg])], day + pd.Timedelta(days=1), SUMMER_END)
            for lg, day in last.dt.normalize().items()
        ]
    )
    before = latest(v, *VALUE_2015)
    dob = born.set_index("player_id").date_of_birth
    found = pd.DataFrame(
        {
            "price": ids.map(summer.market_value_in_eur),
            "price_date": ids.map(summer.date),
            "value_2015": ids.map(before.market_value_in_eur),
            "value_2015_date": ids.map(before.date),
            "age": (AGE_DAY - ids.map(dob)).dt.days / 365.25,
            "ga90": ids.map(goals_assists(apps)),
        }
    )
    rows = rows.join(found)
    rows["reason"] = np.select(
        [no_id, shared, rows.price.isna().to_numpy(), rows.age.isna().to_numpy()],
        REASONS[:-1],
        default=REASONS[-1],
    )
    rows["price"] = rows.price.astype("Int64")
    rows["value_2015"] = rows.value_2015.astype("Int64")
    return rows[[*value.columns, *ADDED]]


def decision_inputs():
    value = pd.read_parquet(value_path())
    links = pd.read_parquet(squad.links_path(), columns=["player_id", "tm_player_id"])
    born = pd.read_parquet(
        INTERIM / "transfermarkt" / "players.parquet", columns=["player_id", "date_of_birth"]
    )
    return value, links, valuations(), born, appearances(), last_games()


def prices():
    value, links, v, born, apps, last = decision_inputs()
    key("3c: last 2015/16 game per league")
    key(last.to_string())
    held = links.tm_player_id.value_counts()
    key(
        f"3a: link rows {len(links)}, player_id unique {links.player_id.is_unique}, Transfermarkt "
        f"ids held by more than one StatsBomb player {int((held > 1).sum())}"
    )
    largest = v.date.max()
    key(
        f"3b: valuations read {len(v)}, above zero {int((v.market_value_in_eur > 0).sum())}, "
        f"dated with a time of day {int((v.date != v.date.dt.normalize()).sum())}"
    )
    key(f"3b: largest date read {largest}, on or before 2016-08-31 {largest <= SUMMER_END}")
    assert largest <= SUMMER_END, "HARD STOP: a valuation dated after 2016-08-31 was read"
    key(
        f"3e: players rows {len(born)}, player_id unique {born.player_id.is_unique}, "
        f"date_of_birth missing {int(born.date_of_birth.isna().sum())}"
    )
    key(
        f"3f: appearances read {len(apps)}, dated {apps.date.min()} to {apps.date.max()}, "
        f"competitions {sorted(apps.competition_id.unique())}"
    )
    assert apps.date.min() >= SEASON[0] and apps.date.max() <= SEASON[1], "HARD STOP: appearances"

    out = build_decision(value, links, v, born, apps, last)
    out.to_parquet(decision_path(), index=False)
    key(f"\n3h: wrote {decision_path()} with {len(out)} rows, columns {list(out.columns)}")
    key("\n3g: rows per reason")
    key(out.reason.value_counts().reindex(REASONS, fill_value=0).to_string())
    for by, order in (("league", list(LEAGUES)), ("group", sh.OUTFIELD)):
        table = pd.crosstab(out[by], out.reason).reindex(index=order, columns=REASONS, fill_value=0)
        key(f"\n3g: rows per reason by {by}")
        key(table.to_string())
    priced = out[out.reason == "priced"]
    key("\n3g: priced rows by the month of price_date")
    key(priced.price_date.dt.to_period("M").value_counts().sort_index().to_string())
    for col in ["price", "age"]:
        x = priced[col]
        key(f"3g: priced {col} smallest {x.min()!r}, median {x.median()!r}, largest {x.max()!r}")
    key(
        f"3g: priced rows {len(priced)}, with value_2015 {int(priced.value_2015.notna().sum())}, "
        f"with ga90 {int(priced.ga90.notna().sum())}"
    )


def controls(rows):
    """The columns of the score fits: a constant, log price, age, age squared and the league and
    group indicators."""
    age = rows.age.to_numpy(dtype=float)
    cols = [np.ones(len(rows)), np.log(rows.price.to_numpy(dtype=float)), age, age**2]
    cols += [(rows.league == lg).to_numpy(dtype=float) for lg in FIT_LEAGUES]
    cols += [(rows.group == g).to_numpy(dtype=float) for g in FIT_GROUPS]
    return np.column_stack(cols)


def ols(y, x):
    """Least-squares coefficients of y on the columns of x, the residuals and R squared."""
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    r = y - x @ b
    return b, r, 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))


def build_scores(rows):
    """rows with gem, gem_P2, gem_stats and momentum, each the residual of its response on the
    controls over the priced rows that have that response, and each fit."""
    priced = rows.reason == "priced"
    responses = {
        "gem": rows.P3_full,
        "gem_P2": rows.P2_full,
        "gem_stats": rows.ga90,
        "momentum": np.log(rows.price.astype(float) / rows.value_2015.astype(float)),
    }
    out, fits = rows.copy(), {}
    for name, y in responses.items():
        use = priced & y.notna()
        b, r, r2 = ols(y[use].to_numpy(dtype=float), controls(rows[use]))
        out[name] = pd.Series(r, index=rows.index[use])
        fits[name] = {"rows": int(use.sum()), "b": b, "r2": r2}
    return out, fits


def scores():
    out, fits = build_scores(pd.read_parquet(decision_path()))
    for name, f in fits.items():
        key(f"\n4c: {name}: rows {f['rows']}, R squared {f['r2']!r}")
        for col, b in zip(CONTROLS, f["b"], strict=True):
            key(f"4c: {name}: {col} {float(b)!r}")

    ranked = out[out.reason == "priced"].sort_values(["gem", "player_id"], ascending=[False, True])
    listed = ranked.head(LISTED)[GEM_LIST]
    key(f"\n4d: the {LISTED} highest gem scores")
    key(listed.to_string(index=False, formatters={"age": "{:.1f}".format}, float_format=repr))
    top = ranked.head(TOP)
    key(f"\n4d: the {TOP} highest gem scores by league")
    key(top.league.value_counts().reindex(list(LEAGUES), fill_value=0).to_string())
    key(f"\n4d: the {TOP} highest gem scores by group")
    key(top.group.value_counts().reindex(sh.OUTFIELD, fill_value=0).to_string())
    key(f"\n4d: the {TOP} highest gem scores by age band")
    band = pd.cut(top.age, AGE_EDGES, right=False, labels=AGE_BANDS)
    key(band.value_counts().reindex(AGE_BANDS).to_string())

    out.to_parquet(scores_path(), index=False)
    key(f"\n4e: wrote {scores_path()} with {len(out)} rows, columns {list(out.columns)}")


def k0(sizes):
    """The average group size of an unbalanced one-way analysis of variance."""
    n = sizes.sum()
    return (n - (sizes**2).sum() / n) / (len(sizes) - 1)


def icc(rows, column):
    """MSB, MSW and k0 of a one-way analysis of variance of column by team, a league and team_id,
    and the intraclass correlation (MSB - MSW) / (MSB + (k0 - 1) MSW), which can be negative."""
    by = rows.groupby(["league", "team_id"])[column]
    n = by.size().to_numpy()
    total, teams = n.sum(), len(n)
    msb = (n * (by.mean().to_numpy() - rows[column].mean()) ** 2).sum() / (teams - 1)
    msw = ((rows[column] - by.transform("mean")) ** 2).sum() / (total - teams)
    k = k0(n)
    return msb, msw, k, (msb - msw) / (msb + (k - 1) * msw)


def power():
    rows = pd.read_parquet(scores_path())
    priced = rows[rows.reason == "priced"]
    pre = priced[priced.value_2015.notna()]
    d = np.log(pre.price.to_numpy(dtype=float) / pre.value_2015.to_numpy(dtype=float))
    _, r, _ = ols(d, controls(pre))
    key(f"5a: priced rows {len(priced)}, with value_2015 {len(pre)}")
    key(
        "5a: d residualized, largest absolute difference from the stored momentum "
        f"{np.abs(r - pre.momentum.to_numpy()).max()!r}"
    )

    sizes = pre.groupby(["league", "team_id"]).size()
    msb, msw, k, raw = icc(pre.assign(d=r), "d")
    rho = max(raw, 0.0)
    key(f"5b: J {len(sizes)!r}, N {int(sizes.sum())!r}")
    key(f"5b: MSB {msb!r}, MSW {msw!r}, k0 {k!r}")
    key(f"5b: ICC raw {raw!r}, ICC {rho!r}")

    test = priced.groupby(["league", "team_id"]).size().to_numpy()
    j_test, n_test, k_test = len(test), int(test.sum()), k0(test)
    de = 1 + (k_test - 1) * rho
    neff = n_test / de
    mde = math.tanh(Z / math.sqrt(neff - 3))
    key(f"5c: J_test {j_test!r}, N_test {n_test!r}, k0_test {k_test!r}")
    key(f"5c: DE {de!r}, NEFF {neff!r}, MDE {mde!r}")


def main(argv):
    step = argv[0]
    run = {"value": value, "prices": prices, "scores": scores, "power": power}[step]
    sh.LOGS.mkdir(parents=True, exist_ok=True)
    full = open(sh.LOGS / f"p8_{step}.log", "w", encoding="utf-8", errors="replace")
    brief = open(sh.LOGS / f"p8_{step}_summary.log", "w", encoding="utf-8", errors="replace")
    sys.stdout = Tee(full, brief)
    try:
        run()
    finally:
        sys.stdout = sys.__stdout__
        full.close()
        brief.close()


if __name__ == "__main__":
    main(sys.argv[1:])
