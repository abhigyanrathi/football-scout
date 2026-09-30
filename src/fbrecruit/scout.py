import re
import sys
import time

import numpy as np
import pandas as pd

from fbrecruit import identify, style
from fbrecruit import shrinkage as sh
from fbrecruit.logs import Tee, key
from fbrecruit.paths import PROCESSED
from fbrecruit.split import windows

KEYS = sh.KEYS
TEAM_KEYS = style.TEAM_KEYS
DIMENSIONS = style.DIMENSIONS
Z, C = style.Z, style.C
P = [f"p_{d}" for d in DIMENSIONS]
NEAREST = 10
WIDE_GAP = 0.5
SCOPES = ["window 1", "window 2", "season"]
# phase 9a's test log, whose odd-even and window correlations the parity halves must reproduce
LOGGED = style.LOGS / "p9a_test_summary.log"
NUMBER = r"([-+0-9.e]+)"


def reference_path():
    return PROCESSED / "scout_reference.parquet"


def pairs_path():
    return PROCESSED / "scout_pairs.parquet"


def fit_path():
    return PROCESSED / "scout_fit.parquet"


def venue_path():
    return PROCESSED / "scout_venue.parquet"


def window_halves_path():
    return PROCESSED / "scout_window_halves.parquet"


def reliability_path():
    return PROCESSED / "scout_reliability.parquet"


def scores(frame, columns):
    """The columns in float64, a missing score counted as 0."""
    return frame[columns].fillna(0.0).to_numpy(np.float64)


def played(t, league, player, team):
    """Whether each player has a row of the v2 table t at the club (league, team)."""
    return pd.MultiIndex.from_arrays([league, player, team]).isin(pd.MultiIndex.from_frame(t[KEYS]))


def reference(halves):
    """Per split, the tools' rows with at least 450 minutes in each half: their minutes, d, the
    distance between their halves' centred scores, and u = d / sqrt(1/m_A + 1/m_B)."""
    a = halves[halves.half == "A"].drop(columns="half")
    b = halves[halves.half == "B"].drop(columns="half")
    m = a.merge(b, on=[*KEYS, "group", "split"], suffixes=("_a", "_b"), validate="one_to_one")
    m = m[(m.minutes_a >= style.HALF_MINUTES) & (m.minutes_b >= style.HALF_MINUTES)]
    gap = scores(m, [f"{c}_a" for c in C]) - scores(m, [f"{c}_b" for c in C])
    d = np.sqrt((gap**2).sum(axis=1))
    ma, mb = m.minutes_a.to_numpy(np.float64), m.minutes_b.to_numpy(np.float64)
    out = m[[*KEYS, "group", "split", "minutes_a", "minutes_b"]].reset_index(drop=True)
    return out.assign(d=d, u=d / np.sqrt(1 / ma + 1 / mb))


def closeness(u, v):
    """For each v, the share of the reference values u at or above it."""
    u = np.sort(np.asarray(u, np.float64))
    return (len(u) - np.searchsorted(u, v, side="left")) / len(u)


def pairs(season, t, ref):
    """Every ordered pair (R, C) of the tools' rows of one group where C has no row of the v2 table
    t at R's club: their distances over the centred and over the standard scores, v, the first over
    sqrt(1/m_R + 1/m_C), and its closeness against the u of ref's rows of the group."""
    out = []
    for group in sh.OUTFIELD:
        s = season[season.group == group].reset_index(drop=True)
        league, player, team = (s[k].to_numpy() for k in KEYS)
        r, c = np.divmod(np.arange(len(s) ** 2), len(s))
        keep = (r != c) & ~played(t, league[r], player[c], team[r])
        r, c = r[keep], c[keep]
        distance = identify.distances(scores(s, C), scores(s, C))[r, c]
        distance_z = identify.distances(scores(s, Z), scores(s, Z))[r, c]
        m = s.minutes.to_numpy(np.float64)
        v = distance / np.sqrt(1 / m[r] + 1 / m[c])
        out.append(
            pd.DataFrame(
                {
                    "league": league[r],
                    "player_id": player[r],
                    "team_id": team[r],
                    "league_c": league[c],
                    "player_id_c": player[c],
                    "team_id_c": team[c],
                    "group": group,
                    "distance": distance,
                    "distance_z": distance_z,
                    "v": v,
                    "closeness": closeness(ref.u[ref.group == group], v),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def nearest(pr, by, n):
    """Each R's n nearest candidates by the column by, ties broken by player_id_c."""
    return pr.sort_values(["player_id", by, "player_id_c"]).groupby("player_id").head(n)


def fit_score(p):
    """The weighted mean of the eight products: weight 1 for pressing, width, depth, dribbling and
    creation, and a third each for possession, build-up speed and verticality."""
    return (
        p.p_sb_pressing
        + p.p_width
        + p.p_depth
        + p.p_dribble
        + p.p_creation
        + (p.p_possession + p.p_buildup + p.p_verticality) / 3
    ) / 6


def fit(season, teams, t):
    """Each club K with each tools' row C who has no row of the v2 table t at K: the eight products
    of C's centred scores and K's, and the fit score."""
    k, c = np.divmod(np.arange(len(teams) * len(season)), len(season))
    league, team = teams.league.to_numpy(), teams.team_id.to_numpy()
    keep = ~played(t, league[k], season.player_id.to_numpy()[c], team[k])
    k, c = k[keep], c[keep]
    out = pd.DataFrame(
        {
            "league_k": league[k],
            "team_id_k": team[k],
            **{col: season[col].to_numpy()[c] for col in [*KEYS, "group"]},
        }
    )
    products = scores(season, C)[c] * teams[C].to_numpy(np.float64)[k]
    out = out.assign(**dict(zip(P, products.T, strict=True)))
    return out.assign(score=fit_score(out))


def tools():
    start = time.perf_counter()
    season = pd.read_parquet(style.season_path())
    teams = pd.read_parquet(style.team_season_path())
    halves = pd.read_parquet(style.halves_path())
    t = identify.window_minutes()
    clubs = t.drop_duplicates(KEYS).groupby("player_id").size()
    key(
        f"5a: tools' rows {len(season)}, clubs {len(teams)}, half rows {len(halves)}; tools' "
        f"players with rows at more than one club in the v2 table "
        f"{int((season.player_id.map(clubs) > 1).sum())}"
    )

    ref = reference(halves)
    for split in style.SPLITS:
        at = ref.split == split
        key(f"\n5b {split}: reference rows {int(at.sum())}")
        for group in sh.OUTFIELD:
            u = ref.u[at & (ref.group == group)]
            key(
                f"5b {split} {group}: rows {len(u)}, u smallest {u.min()!r}, median "
                f"{u.median()!r}, largest {u.max()!r}"
            )
    finite = bool(np.isfinite(ref.u).all())
    key(f"5b: every u finite {finite}")
    assert finite, "HARD STOP: a u is not finite"
    ref.to_parquet(reference_path(), index=False)
    key(f"5b: wrote {reference_path()} with {len(ref)} rows, columns {list(ref.columns)}")

    pr = pairs(season, t, ref[ref.split == "venue"])
    finite = bool(np.isfinite(pr[["distance", "distance_z", "v", "closeness"]].to_numpy()).all())
    fewest = int(pr.groupby("player_id").size().reindex(season.player_id, fill_value=0).min())
    key(f"\n5c: pair rows {len(pr)}")
    key(pr.group.value_counts().reindex(sh.OUTFIELD).to_string())
    key(f"5c: fewest candidates of any R {fewest}; every value finite {finite}")
    near = nearest(pr, "distance", 1).closeness.median()
    key(f"5c: median over R of the closeness of his nearest candidate by distance {near!r}")
    assert finite and fewest >= NEAREST, "HARD STOP: a pair value, or an R with too few candidates"
    pr.to_parquet(pairs_path(), index=False)
    key(f"5c: wrote {pairs_path()} with {len(pr)} rows, columns {list(pr.columns)}")

    fits = fit(season, teams, t)
    finite = bool(np.isfinite(fits[[*P, "score"]].to_numpy(np.float64)).all())
    at = pd.MultiIndex.from_frame(teams[TEAM_KEYS])
    per = fits.groupby(["league_k", "team_id_k"]).size().reindex(at, fill_value=0)
    key(
        f"\n5d: fit rows {len(fits)}; candidates of a club, fewest {int(per.min())}, most "
        f"{int(per.max())}; score smallest {fits.score.min()!r}, median {fits.score.median()!r}, "
        f"largest {fits.score.max()!r}; every value finite {finite}"
    )
    assert finite, "HARD STOP: a fit value is not finite"
    fits.to_parquet(fit_path(), index=False)
    key(f"5d: wrote {fit_path()} with {len(fits)} rows, columns {list(fits.columns)}")
    key(f"5: {time.perf_counter() - start:.1f} s")


def venue_gaps(dealing):
    """Per club and scope, its games and home games on odd and on even matchdays and the gap
    between the shares of each played at home; dealing carries each game's window."""
    parts = (dealing[dealing.window == 1], dealing[dealing.window == 2], dealing)
    out = []
    for scope, part in zip(SCOPES, parts, strict=True):
        odd = part.game_day % 2 == 1
        c = part.assign(g_odd=odd, h_odd=odd & part.home, g_even=~odd, h_even=~odd & part.home)
        s = c.groupby(TEAM_KEYS)[["g_odd", "h_odd", "g_even", "h_even"]].sum().reset_index()
        out.append(s.assign(scope=scope))
    out = pd.concat(out, ignore_index=True)
    out["gap"] = (out.h_odd / out.g_odd - out.h_even / out.g_even).abs()
    return out[[*TEAM_KEYS, "scope", "g_odd", "h_odd", "g_even", "h_even", "gap"]]


def logged():
    """Phase 9a's odd-even r, n and stepped-up value per dimension and window, and its window r and
    n per dimension, from its test log."""
    text = LOGGED.read_text(encoding="utf-8")
    line = rf"^7b (\w+): window (\d) odd-even r {NUMBER}, n (\d+), stepped up {NUMBER}$"
    halves = {
        (d, int(w)): (float(r), int(n), float(s)) for d, w, r, n, s in re.findall(line, text, re.M)
    }
    line = rf"^7b (\w+): windows r {NUMBER}, n (\d+)$"
    across = {d: (float(r), int(n)) for d, r, n in re.findall(line, text, re.M)}
    return halves, across


def window_reliability():
    """For phase 9a's pool, the halves of each window by split from each club's games there,
    standardized with window 1's group moments, their split-half values and the corrected
    stability; HARD STOP unless the parity halves reproduce phase 9a's."""
    t = identify.window_minutes()
    w1 = pd.read_parquet(style.player_path())
    pool = identify.pool(w1, t)
    mom = style.moments(w1, identify.estimation_keys(t))
    games = style.season_games()
    dealing = pd.read_parquet(style.dealing_path())
    key(f"6b: pool rows {len(pool)}")
    out, corr = [], {}
    for w in (1, 2):
        halves, odd, even = style.halves_of(
            games[games.window == w], dealing, pool, mom, "half_window"
        )
        same = style.parity_matches(halves, odd, even)
        key(
            f"6b window {w}: rows split_halves returns {len(odd)}; the parity halves' standard "
            f"scores equal its own exactly: {same}"
        )
        assert same, "HARD STOP: the parity halves differ from split_halves"
        for split in style.SPLITS:
            s = halves[halves.split == split]
            first = pool[KEYS].merge(s[s.half == "A"], on=KEYS, how="left", validate="one_to_one")
            second = pool[KEYS].merge(s[s.half == "B"], on=KEYS, how="left", validate="one_to_one")
            m_a, m_b = first.n_sb_pressing.fillna(0), second.n_sb_pressing.fillna(0)
            both = ((m_a > 0) & (m_b > 0)).to_numpy()
            key(f"6b window {w} {split}: pool rows with minutes in both halves {int(both.sum())}")
            for c in style.half_correlations(first[both], second[both], f"window {w}"):
                corr[split, w, c["dimension"]] = (c["r"], c["n"], 2 * c["r"] / (1 + c["r"]))
            out += [first[m_a > 0].assign(window=w), second[m_b > 0].assign(window=w)]

    w2 = pd.read_parquet(style.window2_path())
    m = pool[KEYS].merge(w2, on=KEYS, how="left", validate="one_to_one")
    across = {c["dimension"]: (c["r"], c["n"]) for c in style.half_correlations(pool, m, "windows")}
    halves_log, across_log = logged()
    key(f"6b: lines read from {LOGGED}: odd-even {len(halves_log)}, windows {len(across_log)}")
    ok = len(halves_log) == 2 * len(DIMENSIONS) and len(across_log) == len(DIMENSIONS)
    for d in DIMENSIONS:
        r, n = across[d]
        lr, ln = across_log[d]
        same = n == ln and style.relative_gap(r, lr) <= style.RELATIVE
        ok &= bool(same)
        key(f"6b {d}: windows r {r!r}, n {n}; logged r {lr!r}, n {ln}; equal {same}")
        for w in (1, 2):
            got, want = corr["parity", w, d], halves_log[d, w]
            same = got[1] == want[1] and all(
                style.relative_gap(x, y) <= style.RELATIVE
                for x, y in ((got[0], want[0]), (got[2], want[2]))
            )
            ok &= bool(same)
            key(
                f"6b {d} parity window {w}: r {got[0]!r}, n {got[1]}, stepped up {got[2]!r}; "
                f"logged {want[0]!r}, {want[1]}, {want[2]!r}; equal {same}"
            )
    assert ok, "HARD STOP: the parity halves do not reproduce phase 9a's logged values"

    rows = []
    for split in style.SPLITS:
        for d in DIMENSIONS:
            (r1, n1, s1), (r2, n2, s2) = corr[split, 1, d], corr[split, 2, d]
            r = across[d][0]
            rows.append(
                {
                    "dimension": d,
                    "split": split,
                    "r_w1": r1,
                    "n_w1": n1,
                    "s_w1": s1,
                    "r_w2": r2,
                    "n_w2": n2,
                    "s_w2": s2,
                    "r_windows": r,
                    "corrected": r / np.sqrt(s1 * s2),
                }
            )
    table = pd.DataFrame(rows)
    halves = pd.concat(out, ignore_index=True)
    halves["minutes"] = halves.n_sb_pressing.astype("int64")
    raw = [*DIMENSIONS, *[f"n_{d}" for d in DIMENSIONS]]
    halves = halves[[*KEYS, "group", "window", "split", "half", "minutes", *raw, *Z]]
    return halves, table


def minutes_thirds(ref, season):
    """Per group, over the reference rows of the venue split: the season minutes at the first and
    second thirds, the rows above the second and at or below the first, and the ratio of their
    median u."""
    minutes = season[[*KEYS, "minutes"]]
    venue = ref[ref.split == "venue"].merge(minutes, on=KEYS, validate="one_to_one")
    out = {}
    for group in sh.OUTFIELD:
        r = venue[venue.group == group]
        lo, hi = np.quantile(r.minutes.to_numpy(np.float64), [1 / 3, 2 / 3])
        top, bottom = r.u[r.minutes > hi], r.u[r.minutes <= lo]
        out[group] = top.median() / bottom.median()
        key(
            f"6c {group}: lo {lo!r}, hi {hi!r}; rows above hi {len(top)}, at or below lo "
            f"{len(bottom)}; median u {top.median()!r} over {bottom.median()!r}, ratio "
            f"{out[group]!r}"
        )
    return out


def league_shares(pr):
    """The means over R of the share of his ten nearest candidates by distance, and by distance_z,
    who play in his league, and of the share of all his candidates who do."""
    pr = pr.assign(same=pr.league_c == pr.league)
    return {
        "centred": nearest(pr, "distance", NEAREST).groupby("player_id").same.mean().mean(),
        "uncentred": nearest(pr, "distance_z", NEAREST).groupby("player_id").same.mean().mean(),
        "all": pr.groupby("player_id").same.mean().mean(),
    }


def club_shares(teams):
    """Per dimension, the between-league sum of squares of the clubs' standard scores over their
    total sum of squares."""
    out = {}
    for d in DIMENSIONS:
        z = teams[f"z_{d}"]
        means = z.groupby(teams.league).transform("mean")
        out[d] = ((means - z.mean()) ** 2).sum() / ((z - z.mean()) ** 2).sum()
    return out


def beside():
    start = time.perf_counter()
    dealing = pd.read_parquet(style.dealing_path())
    wins = windows()[["league", "game_id", "window"]]
    venue = venue_gaps(dealing.merge(wins, on=["league", "game_id"], validate="many_to_one"))
    venue.to_parquet(venue_path(), index=False)
    key(f"6a: wrote {venue_path()} with {len(venue)} rows, columns {list(venue.columns)}")
    gaps = {}
    for scope in SCOPES:
        gap = venue.gap[venue.scope == scope]
        gaps[scope] = (gap.median(), gap.max(), int((gap >= WIDE_GAP).sum()))
        key(
            f"6a {scope}: clubs {len(gap)}, median gap {gaps[scope][0]!r}, largest "
            f"{gaps[scope][1]!r}, clubs with a gap of at least {WIDE_GAP} {gaps[scope][2]}"
        )

    halves, table = window_reliability()
    for path, frame in ((window_halves_path(), halves), (reliability_path(), table)):
        frame.to_parquet(path, index=False)
        key(f"6b: wrote {path} with {len(frame)} rows, columns {list(frame.columns)}")
    for r in table.itertuples():
        key(
            f"6b {r.split} {r.dimension}: r_w1 {r.r_w1!r}, n_w1 {r.n_w1}, s_w1 {r.s_w1!r}; r_w2 "
            f"{r.r_w2!r}, n_w2 {r.n_w2}, s_w2 {r.s_w2!r}; r_windows {r.r_windows!r}; corrected "
            f"{r.corrected!r}"
        )

    ref = pd.read_parquet(reference_path())
    season = pd.read_parquet(style.season_path())
    key("")
    ratios = minutes_thirds(ref, season)
    pr = pd.read_parquet(pairs_path())
    shares = league_shares(pr)
    listed = ", ".join(f"{name} {v!r}" for name, v in shares.items())
    key(f"\n6d: means over R of the share of candidates in his league, {listed}")
    teams = pd.read_parquet(style.team_season_path())
    between = club_shares(teams)
    for d, v in between.items():
        key(f"6e {d}: share of the clubs' variance between leagues {v!r}")

    values = {"P": len(season)}
    values |= {f"P_{g}": int((season.group == g).sum()) for g in sh.OUTFIELD}
    values["K"] = len(teams)
    values["Q"] = int((ref.split == "venue").sum())
    values["QP"] = int((ref.split == "parity").sum())
    for g in sh.OUTFIELD:
        values[f"UV_{g}"] = ref.u[(ref.split == "venue") & (ref.group == g)].median()
        values[f"UP_{g}"] = ref.u[(ref.split == "parity") & (ref.group == g)].median()
        values[f"RT_{g}"] = ratios[g]
    values["NN"] = nearest(pr, "distance", 1).closeness.median()
    values |= {"SL_C": shares["centred"], "SL_U": shares["uncentred"], "SL_0": shares["all"]}
    values |= {f"BL_{i}": between[d] for i, d in enumerate(DIMENSIONS, 1)}
    for name, scope in (("G1", "window 1"), ("G2", "window 2"), ("GS", "season")):
        values[name], values[f"{name}X"], values[f"{name}H"] = gaps[scope]
    venue_rows = table[table.split == "venue"].set_index("dimension")
    for i, d in enumerate(DIMENSIONS, 1):
        values[f"VA_{i}"] = venue_rows.s_w1[d]
        values[f"VB_{i}"] = venue_rows.s_w2[d]
        values[f"VC_{i}"] = venue_rows.corrected[d]
    key("\n6f: the values of the results entries")
    for name, v in values.items():
        key(f"6f [{name}] {v!r}")
    key(f"6: {time.perf_counter() - start:.1f} s")


def main(argv):
    step = argv[0]
    run = {"tools": tools, "beside": beside}[step]
    style.LOGS.mkdir(parents=True, exist_ok=True)
    full = open(style.LOGS / f"p9b_{step}.log", "w", encoding="utf-8", errors="replace")
    brief = open(style.LOGS / f"p9b_{step}_summary.log", "w", encoding="utf-8", errors="replace")
    sys.stdout = Tee(full, brief)
    try:
        run()
    finally:
        sys.stdout = sys.__stdout__
        full.close()
        brief.close()


if __name__ == "__main__":
    main(sys.argv[1:])
