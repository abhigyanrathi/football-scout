import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from fbrecruit import actionvalue as av
from fbrecruit import calibration as cal
from fbrecruit import minutes
from fbrecruit.paths import INTERIM, PROCESSED
from fbrecruit.sources.statsbomb import LEAGUES
from fbrecruit.split import windows

LOGS = Path("C:/Users/abhig/fb-scratch")
BRANCHES = cal.BRANCHES
# The order the league indicator columns and the per-league arrays are built in.
LEAGUE_ORDER = list(LEAGUES)
GROUPS = ["GK", "CB", "FB", "MID", "WIDE", "FWD", "UNKNOWN"]
OUTFIELD = ["CB", "FB", "MID", "WIDE", "FWD"]
W1_MINUTES = 450
W2_MINUTES = 270
KEYS = ["league", "player_id", "team_id"]
COMPONENTS = {"total": "vaep_value", "offensive": "offensive_value", "defensive": "defensive_value"}
PREDICTORS = ["P0", "P1", "P2", "P3", "P4"]
VARIANTS = {"slope": True, "no_slope": False}
REPLICATES = 2000
SEED = 0
PER90 = 8100.0

WIDE_NAMES = {"Left Midfield", "Right Midfield"}
# StatsBomb starting position names, tried in order; "Substitute" matches none of them.
GROUP_RULES = [
    ("GK", lambda n: n == "Goalkeeper"),
    ("CB", lambda n: "Center Back" in n),
    ("FB", lambda n: "Back" in n),
    ("MID", lambda n: "Defensive Midfield" in n or "Center Midfield" in n),
    ("WIDE", lambda n: "Wing" in n or "Attacking Midfield" in n or n in WIDE_NAMES),
    ("FWD", lambda n: "Forward" in n or "Striker" in n),
]
# Transfermarkt sub_position, used only where a player-team has no positioned window-1 minutes.
TM_GROUPS = {
    "Goalkeeper": "GK",
    "Centre-Back": "CB",
    "Left-Back": "FB",
    "Right-Back": "FB",
    "Defensive Midfield": "MID",
    "Central Midfield": "MID",
    "Attacking Midfield": "WIDE",
    "Left Midfield": "WIDE",
    "Right Midfield": "WIDE",
    "Left Winger": "WIDE",
    "Right Winger": "WIDE",
    "Second Striker": "FWD",
    "Centre-Forward": "FWD",
}


def group_of(name):
    for group, rule in GROUP_RULES:
        if rule(name):
            return group
    return None


def v2_table(branch):
    return pd.read_parquet(PROCESSED / f"player_window_vaep_{branch}_v2.parquet")


def group_path():
    return PROCESSED / "player_group.parquet"


def game_path(branch):
    return PROCESSED / f"player_game_vaep_{branch}.parquet"


def evaluation_path(branch):
    return PROCESSED / f"evaluation_{branch}.parquet"


def shrinkage_path(branch):
    return PROCESSED / f"shrinkage_{branch}.parquet"


def bootstrap_path(branch):
    return PROCESSED / f"bootstrap_{branch}.parquet"


class Tee:
    """Everything to the full log; lines printed through key() reach the summary log as well."""

    def __init__(self, full, brief):
        self.full = full
        self.brief = brief
        self.echo = False

    def write(self, text):
        self.full.write(text)
        if self.echo:
            self.brief.write(text)

    def flush(self):
        self.full.flush()
        self.brief.flush()


def key(*args, **kwargs):
    """Print to the full log and, when one is open, to the summary log as well."""
    tee = sys.stdout if isinstance(sys.stdout, Tee) else None
    if tee is not None:
        tee.echo = True
    print(*args, **kwargs)
    if tee is not None:
        tee.echo = False


def show(frame, decimals=6):
    return frame.to_string(float_format=lambda v: f"{v:.{decimals}f}")


# ---------------------------------------------------------------- step 1: preflight


def refit_lolo_calibrators():
    """Every lolo depth-3 map in calibrators_v2.json, refit on its own rows and settings."""
    stored = cal.load_calibrators_v2()
    fresh = cal.branch_calibrators("lolo", True)
    rows, identical = [], True
    for lg in LEAGUES:
        for lb in cal.LABELS:
            s, f = stored[("lolo", lg, lb)], fresh[(lg, lb)]
            same = s["a"] == f["a"] and s["b"] == f["b"]
            identical = identical and same
            rows.append(
                {"league": lg, "label": lb, "depth": s["depth"], "rows": f["rows"]}
                | {"n_iter": f["n_iter"]}
                | {"max_abs_residual": max(abs(f["resid_level"]), abs(f["resid_slope"]))}
                | {"bit_identical": same}
            )
            key(f"lolo {lg} {lb}: stored a {s['a']!r}, b {s['b']!r}")
            key(f"lolo {lg} {lb}: refit  a {f['a']!r}, b {f['b']!r}")
    key(pd.DataFrame(rows).to_string(index=False))
    worst = max(r["max_abs_residual"] for r in rows)
    key(f"largest absolute residual over the eight lolo maps {worst!r}")
    key(f"every lolo depth-3 refit bit-identical: {identical}")
    assert worst < 1e-6, f"largest absolute residual {worst!r} is not below 1e-6"
    assert identical, "a lolo depth-3 refit is not bit-identical"


def preflight():
    found = sorted(PROCESSED.parent.rglob("calibrators_v2.json"))
    key("== 1a: calibrators_v2.json")
    key("found at:", [str(p) for p in found])
    assert found == [cal.CALIBRATORS_V2_PATH]
    entries = json.loads(found[0].read_text())
    key(f"entries {len(entries)}")
    print(pd.DataFrame(entries).to_string(index=False))
    refit_lolo_calibrators()

    key("\n== 1b: actionvalue.pairs")
    key(inspect.getsource(av.pairs))
    for branch in BRANCHES:
        pr = av.pairs(v2_table(branch), W1_MINUTES, W2_MINUTES)
        per_league = {lg: int((pr.league == lg).sum()) for lg in LEAGUES}
        key(f"{branch}: pairs at {W1_MINUTES}/{W2_MINUTES}: {len(pr)} {per_league}")
    t = v2_table("pooled")
    w1 = t[(t.window == 1) & (t.minutes >= W1_MINUTES)][KEYS]
    w2 = t[(t.window == 2) & (t.minutes >= W2_MINUTES)][KEYS]
    both = w1.merge(w2, on=["league", "player_id"], suffixes=("_w1", "_w2"))
    key(
        f"qualifying rows joined on player alone: {len(both)}, "
        f"different team_id in the two windows: {int((both.team_id_w1 != both.team_id_w2).sum())}"
    )
    multi = t.groupby(["league", "player_id", "window"]).team_id.nunique()
    key(
        f"(league, player_id) with more than one team_id inside one window: "
        f"{int((multi > 1).sum())}"
    )
    key("distinct window-1 team_id per league:")
    key(t[t.window == 1].groupby("league").team_id.nunique().to_string())

    key("\n== 1c: StatsBomb lineup starting_position_name")
    lu = minutes.load_lineups("statsbomb")
    key(
        lu.groupby(["starting_position_name", "is_starter"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
    played = lu.minutes_played > 0
    key(
        f"lineup rows {len(lu)}, null starting_position_name "
        f"{int(lu.starting_position_name.isna().sum())}"
    )
    key(
        f"minutes_played > 0 with no position name: "
        f"{int((played & lu.starting_position_name.isna()).sum())}"
    )
    key(
        f"minutes_played > 0 named Substitute: "
        f"{int((played & (lu.starting_position_name == 'Substitute')).sum())}"
    )

    key("\n== 1d: Transfermarkt sub_position of linked players")
    links = pd.read_parquet(PROCESSED / "links" / "statsbomb_tm_players.parquet")
    tm = pd.read_parquet(INTERIM / "transfermarkt" / "players.parquet")
    key(f"statsbomb_tm_players.parquet rows {len(links)}")
    m = links.merge(
        tm[["player_id", "sub_position"]].rename(columns={"player_id": "tm_player_id"}),
        on="tm_player_id",
        how="left",
        validate="many_to_one",
    )
    key(m.sub_position.value_counts(dropna=False).to_string())


# ---------------------------------------------------------------- step 2: position groups


def window1_lineups():
    lu = minutes.load_lineups("statsbomb")
    w = windows()
    lw = lu.merge(w[["league", "game_id", "window"]], on=["league", "game_id"])
    lw = lw[lw.window == 1].copy()
    lw["team_id"] = lw.team_id.astype("int64")
    return lw


def transfermarkt_groups():
    """StatsBomb player_id to a group, through the link table and Transfermarkt sub_position."""
    links = pd.read_parquet(PROCESSED / "links" / "statsbomb_tm_players.parquet")
    tm = pd.read_parquet(INTERIM / "transfermarkt" / "players.parquet")
    assert not links.player_id.duplicated().any()
    m = links.merge(
        tm[["player_id", "sub_position"]].rename(columns={"player_id": "tm_player_id"}),
        on="tm_player_id",
        how="left",
        validate="many_to_one",
    )
    return dict(zip(m.player_id, m.sub_position.map(TM_GROUPS), strict=True))


def modal_group(lw):
    """The mapped group holding the most window-1 minutes, ties broken on appearances."""
    rows = lw.dropna(subset=["group"]).assign(played=lw.minutes_played > 0)
    per = rows.groupby([*KEYS, "group"], as_index=False).agg(
        group_minutes=("minutes_played", "sum"), apps=("played", "sum")
    )
    per = per[per.group_minutes > 0]
    order = [*KEYS, "group_minutes", "apps"]
    per = per.sort_values(order, ascending=[True] * 3 + [False, False], kind="stable")
    by_key = per.groupby(KEYS, as_index=False)
    first, second = by_key.nth(0), by_key.nth(1)
    best = first.merge(
        second[[*KEYS, "group_minutes", "apps"]], on=KEYS, how="left", suffixes=("", "_next")
    )
    tied = (best.group_minutes == best.group_minutes_next) & (best.apps == best.apps_next)
    key(f"2b: player-teams whose top two groups tie on minutes and appearances {int(tied.sum())}")
    best["group"] = best.group.where(~tied, "UNKNOWN")
    return best[[*KEYS, "group", "group_minutes"]]


def build_groups():
    lw = window1_lineups()
    names = sorted(lw.starting_position_name.unique())
    mapping = {n: group_of(n) for n in names}
    key("2a: position name to group")
    key(pd.Series(mapping, name="group").to_string())
    key("2a: names left unmapped:", [n for n, g in mapping.items() if g is None])
    lw["group"] = lw.starting_position_name.map(mapping)

    totals = lw.groupby(KEYS, as_index=False).agg(
        window1_minutes=("minutes_played", "sum"), player_name=("player_name", "first")
    )
    positioned = (
        lw.dropna(subset=["group"])
        .groupby(KEYS, as_index=False)
        .agg(positioned_minutes=("minutes_played", "sum"))
    )
    out = totals.merge(positioned, on=KEYS, how="left").merge(modal_group(lw), on=KEYS, how="left")
    out["positioned_minutes"] = out.positioned_minutes.fillna(0)

    fallback = transfermarkt_groups()
    missing = out.group.isna()
    out["resolved_by"] = np.where(missing, "transfermarkt", "window-1 lineups")
    out.loc[missing, "group"] = out.loc[missing, "player_id"].map(fallback)
    out["resolved_by"] = out.resolved_by.where(out.group.notna(), "unresolved")
    out["group"] = out.group.fillna("UNKNOWN")
    out["group_minutes"] = out.group_minutes.fillna(0)
    assert out.group.isin(GROUPS).all()
    out = out[[*KEYS, "player_name", "group", "resolved_by", "window1_minutes"]].join(
        out[["positioned_minutes", "group_minutes"]]
    )
    out.to_parquet(group_path(), index=False)

    key(f"\n2d: player-team rows {len(out)}")
    key(out.group.value_counts().reindex(GROUPS).fillna(0).astype(int).to_string())
    key(out.resolved_by.value_counts().to_string())
    unknown = out[out.group == "UNKNOWN"]
    key(f"UNKNOWN rows {len(unknown)}")
    key(unknown[[*KEYS, "player_name", "window1_minutes", "resolved_by"]].to_string(index=False))
    r = out[out.resolved_by == "window-1 lineups"]
    key("median share of window-1 minutes held by the modal group, lineup-resolved rows")
    key(f"against all window-1 minutes: {(r.group_minutes / r.window1_minutes).median()!r}")
    key(f"against positioned minutes:   {(r.group_minutes / r.positioned_minutes).median()!r}")
    print("\ngroup by league")
    print(out.groupby(["league", "group"]).size().unstack(fill_value=0).to_string())


# ---------------------------------------------- step 3: per-game values and dispersion


def calibrated_values(branch):
    """Per-action v2 values of all four leagues, through calibration's shared path."""
    cals = cal.load_calibrators_v2()
    probs = {lg: cal.full_probs_v2(branch, lg, cals, cal.CHOSEN_DEPTH) for lg in LEAGUES}
    return cal.branch_values(branch, probs)


def build_per_game(branch):
    """Window-1 player-game value sums beside that game's lineup minutes."""
    wins = windows()
    values = calibrated_values(branch).merge(
        wins[["league", "game_id", "window"]], on=["league", "game_id"]
    )
    values = values[values.window == 1]
    values["player_id"] = values.player_id.astype("int64")
    values["team_id"] = values.team_id.astype("int64")
    sums = values.groupby([*KEYS, "game_id"], as_index=False).agg(
        **{c: (c, "sum") for c in COMPONENTS.values()}
    )
    lw = window1_lineups()[[*KEYS, "game_id", "minutes_played"]]
    pg = sums.merge(lw, on=[*KEYS, "game_id"], how="outer")
    fill = [*COMPONENTS.values(), "minutes_played"]
    pg[fill] = pg[fill].fillna(0.0)
    pg["minutes_played"] = pg.minutes_played.astype("int64")
    days = wins[wins.window == 1][["league", "game_id", "game_day"]]
    pg = pg.merge(days, on=["league", "game_id"], how="left", validate="many_to_one")
    assert pg.game_day.notna().all()
    print(
        f"{branch}: window-1 player-game rows {len(pg)}, action sums {len(sums)}, "
        f"lineup rows {len(lw)}"
    )
    pg.to_parquet(game_path(branch), index=False)
    return pg


def reconcile(branch, pg):
    """Summing the per-game rows to the window must reproduce the v2 window-1 row exactly."""
    t = v2_table(branch)
    w1 = t[t.window == 1]
    agg = (
        pg.assign(played=pg.minutes_played > 0)
        .groupby(KEYS, as_index=False)
        .agg(
            pg_vaep=("vaep_value", "sum"),
            pg_minutes=("minutes_played", "sum"),
            pg_games=("played", "sum"),
        )
    )
    m = w1.merge(agg, on=KEYS, how="left", validate="one_to_one")
    assert m.pg_vaep.notna().all()
    diff = (m.pg_vaep - m.vaep_sum).abs()
    bad = (diff > 1e-9) | (m.pg_minutes != m.minutes) | (m.pg_games != m.games)
    key(
        f"3c {branch}: window-1 rows {len(m)}, max abs vaep difference {diff.max()!r}, "
        f"mismatched rows {int(bad.sum())}"
    )
    assert not bad.any(), "per-game rows do not reconcile with the v2 window-1 table"


def population(branch):
    t = v2_table(branch)
    w1 = t[(t.window == 1) & (t.minutes >= W1_MINUTES)]
    g = pd.read_parquet(group_path())[[*KEYS, "group"]]
    pop = w1.merge(g, on=KEYS, how="left", validate="one_to_one").reset_index(drop=True)
    assert pop.group.notna().all()
    return pop


def player_stats(pg, pop):
    """Per-player window-1 sufficient statistics over the games the player had minutes in."""
    g = pg[pg.minutes_played > 0].merge(pop[[*KEYS, "group"]], on=KEYS)
    agg = {"n": ("minutes_played", "size"), "m": ("minutes_played", "sum")}
    agg |= {f"sum_{k}": (c, "sum") for k, c in COMPONENTS.items()}
    tot = g.groupby(KEYS, as_index=False).agg(**agg)
    j = g.merge(tot, on=KEYS)
    resid = {
        f"S_{k}": (j[c] - j[f"sum_{k}"] / j.m * j.minutes_played) ** 2 / j.minutes_played
        for k, c in COMPONENTS.items()
    }
    s = j[KEYS].assign(**resid).groupby(KEYS, as_index=False).sum()
    out = (
        pop[[*KEYS, "group", "minutes"]]
        .merge(tot, on=KEYS, how="left", validate="one_to_one")
        .merge(s, on=KEYS, how="left", validate="one_to_one")
    )
    assert len(out) == len(pop) and out.n.notna().all()
    return out


def phi_table(stats):
    """phi per group and component: sum of the within-player sums of squares over sum (n - 1)."""
    d = stats.assign(df=stats.n - 1)
    agg = {"players": ("n", "size"), "df": ("df", "sum")}
    agg |= {f"S_{k}": (f"S_{k}", "sum") for k in COMPONENTS}
    t = d.groupby("group").agg(**agg)
    for k in COMPONENTS:
        t[f"phi_{k}"] = t[f"S_{k}"] / t.df
    return t


def psi_of(rows, phi, component="total"):
    """Sampling variance of a per-90 rate: the group's dispersion over the row's minutes."""
    return PER90 * rows.group.map(phi[f"phi_{component}"]).to_numpy() / rows.minutes.to_numpy()


def split_half(pg, pop, phi):
    """Reported only: odd against even matchdays, against the variance the dispersion implies."""
    g = pg[pg.minutes_played > 0].merge(pop[[*KEYS, "group"]], on=KEYS)
    g = g.assign(half=np.where(g.game_day.astype("int64") % 2 == 1, "odd", "even"))
    a = g.groupby([*KEYS, "group", "half"], as_index=False).agg(
        v=("vaep_value", "sum"), m=("minutes_played", "sum")
    )
    w = a.pivot(index=[*KEYS, "group"], columns="half", values=["v", "m"]).dropna().reset_index()
    w.columns = [*KEYS, "group", "v_even", "v_odd", "m_even", "m_odd"]
    w = w[(w.m_odd > 0) & (w.m_even > 0)]
    gap = (90 * w.v_odd / w.m_odd - 90 * w.v_even / w.m_even) ** 2
    expected = PER90 * w.group.map(phi["phi_total"]) * (1 / w.m_odd + 1 / w.m_even)
    out = (
        w.assign(gap=gap, expected=expected)
        .groupby("group")
        .agg(players=("gap", "size"), observed=("gap", "sum"), expected=("expected", "sum"))
    )
    return out.assign(ratio=out.observed / out.expected)


def dispersion_report(branch, pg, pop):
    stats = player_stats(pg, pop)
    phi = phi_table(stats)
    key(f"\n3d {branch}: estimation population (window-1 minutes >= {W1_MINUTES}) {len(pop)}")
    key(pop.groupby("league").size().to_string())
    key(pop.groupby(["league", "group"]).size().unstack(fill_value=0).to_string())
    key(pop.group.value_counts().reindex(GROUPS).dropna().astype(int).to_string())
    key(f"\n3e {branch}: dispersion of the total, per group")
    t = phi[["players", "S_total", "df", "phi_total"]].assign(
        psi_450=PER90 * phi.phi_total / 450, psi_900=PER90 * phi.phi_total / 900
    )
    key(show(t, 8))
    for g, row in phi.iterrows():
        key(
            f"{branch} {g}: sum S_i {row.S_total!r}, sum (n_i - 1) {row.df!r}, "
            f"phi {row.phi_total!r}"
        )
    key(f"\n3f {branch}: phi of the offensive and defensive components, per group")
    key(show(phi[["phi_offensive", "phi_defensive"]], 8))
    for g, row in phi.iterrows():
        key(f"{branch} {g}: phi offensive {row.phi_offensive!r}, defensive {row.phi_defensive!r}")
    key(f"\n3g {branch}: odd against even matchdays, reported only")
    key(show(split_half(pg, pop, phi), 6))
    return stats, phi


def dispersion():
    for branch in BRANCHES:
        key(f"\n================ {branch}")
        pg = build_per_game(branch)
        reconcile(branch, pg)
        dispersion_report(branch, pg, population(branch))


# ---------------------------------------------------------------- step 4: Fay-Herriot fit


def design(league, minutes_played, with_slope):
    x = np.column_stack([(league == lg).astype(float) for lg in LEAGUES])
    if with_slope:
        x = np.column_stack([x, np.log(minutes_played / 1000.0)])
    return x


def fay_herriot(y, x, psi):
    """Prasad-Rao tau^2, then weighted least squares and the posterior mean."""
    m, p = x.shape
    b_ols = np.linalg.lstsq(x, y, rcond=None)[0]
    r = y - x @ b_ols
    h = np.einsum("ij,jk,ik->i", x, np.linalg.pinv(x.T @ x), x)
    rss, bias = float((r**2).sum()), float(((1.0 - h) * psi).sum())
    moment = (rss - bias) / (m - p) if m > p else 0.0
    tau2 = max(0.0, moment)
    w = np.sqrt(1.0 / (tau2 + psi))
    b = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)[0]
    xb = x @ b
    shrink = tau2 / (tau2 + psi)
    return {
        "m": m,
        "p": p,
        "rss": rss,
        "bias": bias,
        "tau2": tau2,
        "truncated": bool(moment < 0),
        "b": b,
        "xb": xb,
        "shrink": shrink,
        "theta": xb + shrink * (y - xb),
    }


def fit_population(pop, psi):
    """One fit per group and variant. A group with no more rows than columns is left unfitted."""
    out = {v: {} for v in VARIANTS}
    cols = {v: np.full(len(pop), np.nan) for v in ("xb", "theta", "theta_no_slope")}
    league = pop.league.to_numpy()
    mins = pop.minutes.to_numpy(dtype=float)
    y = pop.vaep_per90.to_numpy(dtype=float)
    for g, idx in pop.groupby("group").indices.items():
        for name, with_slope in VARIANTS.items():
            x = design(league[idx], mins[idx], with_slope)
            if len(idx) <= x.shape[1]:
                out[name][g] = None
                continue
            f = fay_herriot(y[idx], x, psi[idx])
            out[name][g] = f
            if name == "slope":
                cols["xb"][idx] = f["xb"]
                cols["theta"][idx] = f["theta"]
            else:
                cols["theta_no_slope"][idx] = f["theta"]
    return out, cols


def report_fits(branch, fits):
    for name in VARIANTS:
        for g, f in sorted(fits[name].items()):
            if f is None:
                key(f"4c {branch} {g} {name}: NOT FITTED, rows do not exceed columns")
                continue
            key(
                f"4c {branch} {g} {name}: m {f['m']!r}, p {f['p']!r}, sum r^2 {f['rss']!r}, "
                f"sum (1 - h) psi {f['bias']!r}, tau2 {f['tau2']!r}, truncated {f['truncated']}"
            )
            key(f"4c {branch} {g} {name}: b " + ", ".join(repr(float(v)) for v in f["b"]))
            key(
                f"4c {branch} {g} {name}: shrinkage min {f['shrink'].min()!r}, "
                f"mean {f['shrink'].mean()!r}, max {f['shrink'].max()!r}"
            )


def reliability(branch, phi, fits):
    """3f, printed here: tau^2 / (tau^2 + psi) at 450 and 900 minutes, tau^2 from 4a."""
    rows = []
    for g, f in sorted(fits["slope"].items()):
        if f is None:
            continue
        for comp in COMPONENTS:
            for mins in (450, 900):
                psi = PER90 * phi.loc[g, f"phi_{comp}"] / mins
                rows.append(
                    {"group": g, "component": comp, "minutes": mins, "psi": psi}
                    | {"tau2": f["tau2"], "reliability": f["tau2"] / (f["tau2"] + psi)}
                )
    t = pd.DataFrame(rows).pivot(
        index=["group", "component"], columns="minutes", values="reliability"
    )
    key(f"\n4d {branch}: reliability tau2 / (tau2 + psi), tau2 from the p = 5 fit")
    key(show(t, 6))
    print(show(pd.DataFrame(rows), 8))


def branch_inputs(branch):
    pg = pd.read_parquet(game_path(branch))
    pop = population(branch)
    stats = player_stats(pg, pop)
    phi = phi_table(stats)
    return pop, stats, phi, psi_of(pop, phi)


def fit():
    for branch in BRANCHES:
        key(f"\n================ {branch}")
        pop, stats, phi, psi = branch_inputs(branch)
        fits, cols = fit_population(pop, psi)
        report_fits(branch, fits)
        reliability(branch, phi, fits)
        out = pop.assign(psi=psi, **cols)
        out.to_parquet(shrinkage_path(branch), index=False)
        key(
            f"\n4e {branch}: shrinkage table rows {len(out)}, "
            f"unfitted rows {int(out.theta.isna().sum())}"
        )


# ------------------------------------------------- steps 5 and 6: predictors and metrics


def league_levels(branch):
    """c(league, window): 90 * summed value over summed minutes across every row of the table."""
    t = v2_table(branch)
    a = t.groupby(["league", "window"]).agg(v=("vaep_sum", "sum"), m=("minutes", "sum"))
    return 90 * a.v / a.m


def build_evaluation(branch):
    t = v2_table(branch)
    pr = av.pairs(t, W1_MINUTES, W2_MINUTES)
    w2 = t[t.window == 2][[*KEYS, "minutes"]].rename(columns={"minutes": "minutes_w2"})
    pr = pr.merge(w2, on=KEYS, how="left", validate="one_to_one")
    sh = pd.read_parquet(shrinkage_path(branch))
    cols = [*KEYS, "group", "minutes", "psi", "xb", "theta", "theta_no_slope"]
    pr = pr.merge(sh[cols].rename(columns={"minutes": "minutes_w1"}), on=KEYS, how="left")
    assert pr.group.notna().all(), "a pair's player is outside the estimation population"

    c = league_levels(branch)
    pr["c1"] = pr.league.map(c.xs(1, level="window"))
    pr["c2"] = pr.league.map(c.xs(2, level="window"))
    pr["target"] = pr.vaep_per90_w2 - pr.c2
    pr["P0"] = 0.0
    pr["P1"] = pr.xb - pr.c1
    pr["P2"] = pr.vaep_per90_w1 - pr.c1
    pr["P3"] = pr.theta - pr.c1
    pr["P4"] = pr.theta_no_slope - pr.c1

    pg = pd.read_parquet(game_path(branch))
    phi = phi_table(player_stats(pg, population(branch)))
    pr["phi"] = pr.group.map(phi.phi_total)
    pr["psi2"] = PER90 * pr.phi / pr.minutes_w2
    pr.to_parquet(evaluation_path(branch), index=False)
    return pr, c


def ols_line(x, y):
    if np.std(x) == 0:
        return float("nan"), float("nan")
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    return float(coef[1]), float(coef[0])


def spearman(x, y):
    if np.std(x) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def subset_metrics(target, preds, psi2):
    """TSE with the target's own sampling variance subtracted, and the shape of the fit."""
    raw = {k: float(((target - v) ** 2).sum()) for k, v in preds.items()}
    correction = float(psi2.sum())
    tse = {k: raw[k] - correction for k in raw}
    rows = []
    for k in PREDICTORS:
        slope, intercept = ols_line(preds[k], target)
        rows.append(
            {"predictor": k, "tse": tse[k], "tse_ratio": tse[k] / tse["P2"]}
            | {"raw_ratio": raw[k] / raw["P2"], "spearman": spearman(preds[k], target)}
            | {"slope": slope, "intercept": intercept}
        )
    return rows, tse["P3"] - tse["P2"]


def subsets(frame):
    """The headline is every outfield pair; each league is outfield too, GK stands alone."""
    out = {"outfield": frame.group.isin(OUTFIELD)}
    out |= {lg: out["outfield"] & (frame.league == lg) for lg in LEAGUES}
    return out | {"GK": frame.group == "GK"}


def metric_table(frame):
    rows, diffs = [], {}
    for name, mask in subsets(frame).items():
        part = frame[mask]
        preds = {k: part[k].to_numpy() for k in PREDICTORS}
        got, diffs[name] = subset_metrics(part.target.to_numpy(), preds, part.psi2.to_numpy())
        rows += [{"subset": name, "pairs": len(part)} | r for r in got]
    return pd.DataFrame(rows), diffs


def evaluate():
    for branch in BRANCHES:
        key(f"\n================ {branch}")
        pr, c = build_evaluation(branch)
        key(f"5a {branch}: league-window levels c")
        for (lg, w), v in c.items():
            key(f"c({lg}, {w}) = {v!r}")
        counts = pr.assign(kind=np.where(pr.group.isin(OUTFIELD), "outfield", pr.group))
        key(f"\n5b {branch}: pairs {len(pr)}")
        key(counts.groupby(["league", "kind"]).size().unstack(fill_value=0).to_string())
        key(counts.kind.value_counts().to_string())

        table, diffs = metric_table(pr)
        key(f"\n6b {branch}: metrics per subset and predictor")
        key(show(table, 6))
        for r in table.itertuples():
            key(
                f"6d {branch} {r.subset} {r.predictor}: tse {r.tse!r}, ratio {r.tse_ratio!r}, "
                f"uncorrected ratio {r.raw_ratio!r}, spearman {r.spearman!r}, "
                f"slope {r.slope!r}, intercept {r.intercept!r}"
            )
        for name, d in diffs.items():
            key(f"6d {branch} {name}: TSE(P3) - TSE(P2) {d!r}")


# ---------------------------------------------------------------- step 7: bootstrap


def cluster_index(t):
    """One cluster per league and window-1 team; window-2 rows join the same team's cluster."""
    cl = t[t.window == 1][["league", "team_id"]].drop_duplicates()
    cl = cl.sort_values(["league", "team_id"]).reset_index(drop=True)
    index = pd.Series(np.arange(len(cl)), index=pd.MultiIndex.from_frame(cl))
    w2 = t[t.window == 2][["league", "team_id"]].drop_duplicates()
    assert pd.MultiIndex.from_frame(w2).isin(index.index).all()
    return cl, index


def team_totals(t, index):
    """Summed value and minutes per cluster and window, the inputs to a replicate's c."""
    v = np.zeros((2, len(index)))
    m = np.zeros((2, len(index)))
    agg = t.groupby(["league", "window", "team_id"]).agg(
        v=("vaep_sum", "sum"), m=("minutes", "sum")
    )
    for (lg, w, tid), row in agg.iterrows():
        v[w - 1, index[(lg, tid)]] = row.v
        m[w - 1, index[(lg, tid)]] = row.m
    return v, m


def bootstrap_inputs(branch):
    """Arrays a replicate needs, so that no replicate regroups the per-game rows."""
    pg = pd.read_parquet(game_path(branch))
    pop = population(branch)
    stats = player_stats(pg, pop)
    ev = pd.read_parquet(evaluation_path(branch))
    t = v2_table(branch)
    cl, index = cluster_index(t)

    groups = [g for g in GROUPS if g != "UNKNOWN"]
    code = {g: i for i, g in enumerate(groups)}
    keep = pop.group.isin(groups).to_numpy()
    pop, stats = pop[keep].reset_index(drop=True), stats[keep].reset_index(drop=True)
    ev = ev[ev.group.isin(groups)].reset_index(drop=True)
    where = pd.Series(np.arange(len(pop)), index=pd.MultiIndex.from_frame(pop[KEYS]))
    league = pop.league.to_numpy()
    mins = pop.minutes.to_numpy(dtype=float)
    return {
        "groups": groups,
        "league_clusters": [np.flatnonzero((cl.league == lg).to_numpy()) for lg in LEAGUES],
        "team_v_m": team_totals(t, index),
        "pop_cluster": index.loc[pd.MultiIndex.from_frame(pop[["league", "team_id"]])].to_numpy(),
        "pop_code": pop.group.map(code).to_numpy(),
        "pop_y": pop.vaep_per90.to_numpy(dtype=float),
        "pop_minutes": mins,
        "pop_s": stats.S_total.to_numpy(dtype=float),
        "pop_df": (stats.n - 1).to_numpy(dtype=float),
        "x": {name: design(league, mins, s) for name, s in VARIANTS.items()},
        "ev_cluster": index.loc[pd.MultiIndex.from_frame(ev[["league", "team_id"]])].to_numpy(),
        "ev_code": ev.group.map(code).to_numpy(),
        "ev_league": np.array([LEAGUE_ORDER.index(lg) for lg in ev.league]),
        "ev_pop": where.loc[pd.MultiIndex.from_frame(ev[KEYS])].to_numpy(),
        "ev_y2": ev.vaep_per90_w2.to_numpy(dtype=float),
        "ev_m2": ev.minutes_w2.to_numpy(dtype=float),
        "masks": subsets(ev),
        "pairs": len(ev),
    }


def replicate(d, mult):
    """One replicate's c, phi, both fits and the drawn pair rows' target and predictors."""
    levels = np.empty((2, len(LEAGUES)))
    v, m = d["team_v_m"]
    for i, cls in enumerate(d["league_clusters"]):
        w = mult[cls]
        levels[:, i] = 90 * (v[:, cls] @ w) / (m[:, cls] @ w)

    code, n_groups = d["pop_code"], len(d["groups"])
    sel = np.repeat(np.arange(len(code)), mult[d["pop_cluster"]])
    s = np.bincount(code[sel], weights=d["pop_s"][sel], minlength=n_groups)
    df = np.bincount(code[sel], weights=d["pop_df"][sel], minlength=n_groups)
    phi = np.divide(s, df, out=np.zeros_like(s), where=df > 0)
    psi = PER90 * phi[code] / d["pop_minutes"]

    y = d["pop_y"]
    out = {name: np.full(len(code), np.nan) for name in ("xb", "slope", "no_slope")}
    for gi in range(n_groups):
        drawn = sel[code[sel] == gi]
        rows = code == gi
        if not len(drawn):
            continue
        for name, x in d["x"].items():
            f = fay_herriot(y[drawn], x[drawn], psi[drawn])
            xb = x[rows] @ f["b"]
            shrink = f["tau2"] / (f["tau2"] + psi[rows])
            out[name][rows] = xb + shrink * (y[rows] - xb)
            if name == "slope":
                out["xb"][rows] = xb

    pick = np.repeat(np.arange(d["pairs"]), mult[d["ev_cluster"]])
    lg, row = d["ev_league"][pick], d["ev_pop"][pick]
    c1, c2 = levels[0][lg], levels[1][lg]
    preds = {
        "P0": np.zeros(len(pick)),
        "P1": out["xb"][row] - c1,
        "P2": y[row] - c1,
        "P3": out["slope"][row] - c1,
        "P4": out["no_slope"][row] - c1,
    }
    target = d["ev_y2"][pick] - c2
    psi2 = PER90 * phi[d["ev_code"][pick]] / d["ev_m2"][pick]
    return pick, target, preds, psi2


def bootstrap_branch(branch):
    d = bootstrap_inputs(branch)
    n_clusters = sum(len(c) for c in d["league_clusters"])
    rng = np.random.default_rng(SEED)
    rows = []
    for rep in range(REPLICATES):
        drawn = np.concatenate([rng.choice(c, size=len(c)) for c in d["league_clusters"]])
        pick, target, preds, psi2 = replicate(d, np.bincount(drawn, minlength=n_clusters))
        for name, mask in d["masks"].items():
            m = mask.to_numpy()[pick]
            got, diff = subset_metrics(target[m], {k: v[m] for k, v in preds.items()}, psi2[m])
            for r in got:
                for metric in ("tse", "tse_ratio", "raw_ratio", "spearman", "slope"):
                    rows.append((rep, name, r["predictor"], metric, r[metric]))
            rows.append((rep, name, "P3-P2", "tse_diff", diff))
    out = pd.DataFrame(rows, columns=["replicate", "subset", "predictor", "metric", "value"])
    out.to_parquet(bootstrap_path(branch), index=False)
    return out


def intervals(out):
    q = out.groupby(["subset", "predictor", "metric"]).value.quantile([0.025, 0.975]).unstack()
    q.columns = ["p2_5", "p97_5"]
    return q.join(out.groupby(["subset", "predictor", "metric"]).value.median().rename("median"))


def bootstrap():
    key(
        f"7c: the action-value classifiers and the calibrators are held fixed across all "
        f"{REPLICATES} replicates; every interval below is conditional on them."
    )
    for branch in BRANCHES:
        key(f"\n================ {branch}")
        out = bootstrap_branch(branch)
        q = intervals(out)
        wanted = q.index.get_level_values("metric").isin(
            ["tse_ratio", "tse_diff", "spearman", "slope"]
        )
        key(f"\n7d {branch}: 2.5 and 97.5 percentiles, {REPLICATES} replicates")
        key(
            "the outfield rows are the headline; each league's rows rest on that league's "
            "20 teams alone"
        )
        key(show(q[wanted], 6))
        head = q.loc[("outfield", "P3", "tse_ratio")]
        key(
            f"\n6c {branch} outfield TSE(P3) / TSE(P2): point estimate and interval "
            f"[{head.p2_5!r}, {head.p97_5!r}]"
        )
        if branch == "pooled":
            key(
                f"6c CLAIM on the pooled branch, outfield: upper end {head.p97_5!r} below 1: "
                f"{'PASS' if head.p97_5 < 1 else 'FAIL'}"
            )
        print(show(q, 8))


def main(argv):
    step = argv[0]
    LOGS.mkdir(parents=True, exist_ok=True)
    full = open(LOGS / f"p4c_{step}.log", "w", encoding="utf-8", errors="replace")
    brief = open(LOGS / f"p4c_{step}_summary.log", "w", encoding="utf-8", errors="replace")
    sys.stdout = Tee(full, brief)
    try:
        if step == "preflight":
            preflight()
        elif step == "groups":
            build_groups()
        elif step == "dispersion":
            dispersion()
        elif step == "fit":
            fit()
        elif step == "evaluate":
            evaluate()
        elif step == "bootstrap":
            bootstrap()
    finally:
        sys.stdout = sys.__stdout__
        full.close()
        brief.close()


if __name__ == "__main__":
    main(sys.argv[1:])
