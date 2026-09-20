import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from fbrecruit import actionvalue as av
from fbrecruit import minutes
from fbrecruit.paths import INTERIM, PROCESSED
from fbrecruit.sources.statsbomb import LEAGUES
from fbrecruit.split import games, windows

K = 5
TOL = 1e-10
MAX_ITER = 1000
W1_ACTIONS = 1_832_742
BRANCHES = ["pooled", "lolo"]
LABELS = av.LABELS
FOLDS_PATH = INTERIM / "pooled_folds.parquet"
CALIBRATORS_PATH = INTERIM / "calibrators.json"
CALIBRATORS_V2_PATH = INTERIM / "calibrators_v2.json"
# The only difference between the two candidates: xgboost's default depth is 6.
DEPTH3 = av.PARAMS | {"max_depth": 3}
DEPTHS = {6: False, 3: True}
# The depth the rule chose for each label on window-1 out-of-fold values; see compare_depths.
CHOSEN_DEPTH = {"scores": 3, "concedes": 3}


def tree_params(d3):
    return DEPTH3 if d3 else av.PARAMS


def oof_path(league, d3=False):
    return INTERIM / f"vaep_pred_oof{'_d3' if d3 else ''}_{league}.parquet"


def fold_fit(k, d3=False):
    return f"cf{k}d3" if d3 else f"cf{k}"


def fit_name(branch, league, d3=False):
    return av.rating_fit(branch, league) + ("_d3" if d3 else "")


def assign_folds(g):
    """Within each league, games in matchday, date, game id order take folds 0 to K - 1 in turn."""
    g = g.sort_values(["league", "game_day", "game_date", "game_id"], kind="stable")
    g = g.assign(fold=g.groupby("league").cumcount() % K)
    return g[["league", "game_id", "fold"]].reset_index(drop=True)


def window1_games():
    dates = pd.concat([games(lg)[["game_id", "game_date"]].assign(league=lg) for lg in LEAGUES])
    w = windows()
    return w[w.window == 1].merge(dates, on=["league", "game_id"], validate="one_to_one")


def actions_per_game():
    frames = [
        pd.read_parquet(av.labels_path(lg), columns=["game_id"]).assign(league=lg) for lg in LEAGUES
    ]
    return pd.concat(frames).groupby(["league", "game_id"]).size().rename("actions").reset_index()


def make_folds():
    g = window1_games()
    print("window-1 game_day per league")
    print(g.groupby("league").game_day.agg(["min", "max"]).to_string())
    folds = assign_folds(g)
    t = folds.merge(actions_per_game(), on=["league", "game_id"], how="left", validate="one_to_one")
    table = t.groupby(["league", "fold"]).agg(games=("game_id", "size"), actions=("actions", "sum"))
    print("\n2b: games and actions per league and fold")
    print(table.to_string())

    assert t.actions.notna().all()
    assert len(folds) == len(g) and not folds.duplicated(["league", "game_id"]).any()
    assert set(zip(folds.league, folds.game_id, strict=True)) == set(
        zip(g.league, g.game_id, strict=True)
    )
    assert folds.fold.isin(range(K)).all()
    for lg in LEAGUES:
        w1 = len(av.train_labels([(lg, 1)]))
        print(f"{lg}: fold actions sum {int(table.loc[lg].actions.sum())}, window-1 actions {w1}")
        assert table.loc[lg].actions.sum() == w1
    per_fold = table.groupby("fold").actions.sum()
    print(f"all folds: {int(per_fold.sum())}")
    assert per_fold.sum() == W1_ACTIONS

    training = W1_ACTIONS - per_fold
    print("\n2c: training rows per fold (window-1 actions minus the fold's actions)")
    print(training.rename("training_rows").to_string())
    print(f"sum {int(training.sum())}")
    assert training.sum() == 4 * W1_ACTIONS == 7_330_968
    folds.to_parquet(FOLDS_PATH, index=False)


def fold_parts(k, inside):
    """Window-1 games of each league inside fold k, or outside it."""
    f = pd.read_parquet(FOLDS_PATH)
    f = f[(f.fold == k) == inside]
    return [(lg, set(f[f.league == lg].game_id.tolist())) for lg in LEAGUES]


def fit_fold(k, label, d3=False):
    p = tree_params(d3)
    print("params passed to xgboost.train:", p, "| num_boost_round:", av.ROUNDS)
    parts = fold_parts(k, inside=False)
    y = av.train_labels(parts)[label]
    start = time.perf_counter()
    av.fit(fold_fit(k, d3), labels=[label], parts=parts, params=p)
    secs = time.perf_counter() - start
    print(
        f"fold {k}, {label}: training rows {len(y)}, positives {int(y.sum())}, "
        f"base rate {y.mean():.6f}, wall clock {secs:.1f} s"
    )


def fold_predictions(booster, k):
    parts = fold_parts(k, inside=True)
    return np.concatenate([av.predict(booster, x) for lg, g in parts for x, _ in av.batches(lg, g)])


def refit(d3=False):
    saved = xgb.Booster(model_file=av.model_path(fold_fit(0, d3), "scores"))
    again = av.fit(
        fold_fit(0, d3),
        labels=["scores"],
        save=False,
        parts=fold_parts(0, inside=False),
        params=tree_params(d3),
    )
    first, second = fold_predictions(saved, 0), fold_predictions(again["scores"], 0)
    diff = np.abs(first - second).max()
    print(f"fold 0 rows {len(first)}: predictions identical {np.array_equal(first, second)}")
    print(f"max abs difference {diff!r}")


def predict_oof(d3=False):
    folds = pd.read_parquet(FOLDS_PATH)
    models = {
        k: {lb: xgb.Booster(model_file=av.model_path(fold_fit(k, d3), lb)) for lb in LABELS}
        for k in range(K)
    }
    for lg in LEAGUES:
        fold = folds[folds.league == lg].set_index("game_id").fold
        frames = []
        for x, _ in av.batches(lg, 1):
            ks = x.game_id.map(fold).to_numpy()
            p = {lb: np.full(len(x), np.nan, dtype=np.float32) for lb in LABELS}
            for k in range(K):
                m = ks == k
                if m.any():
                    for lb in LABELS:
                        p[lb][m] = av.predict(models[k][lb], x[m])
            frames.append(x[av.KEYS].assign(**{f"p_{lb}": p[lb] for lb in LABELS}))
        out = pd.concat(frames, ignore_index=True)
        y = av.train_labels([(lg, 1)])
        print(f"{lg}: out-of-fold rows {len(out)}, window-1 actions {len(y)}", flush=True)
        av.check_keys(out, y)
        assert out.notna().all().all() and not out.duplicated(av.KEYS).any()
        out.to_parquet(oof_path(lg, d3), index=False)


def oof_ranges(d3=False):
    """One prediction per window-1 row, and no probability at 0 or 1."""
    bad = []
    for lg in LEAGUES:
        p = pd.read_parquet(oof_path(lg, d3))
        y = av.train_labels([(lg, 1)])
        print(f"{lg}: out-of-fold rows {len(p)}, window-1 actions {len(y)}")
        assert len(p) == len(y)
        bad += ranges(f"out-of-fold depth {3 if d3 else 6} on {lg}", p)
    print("probabilities exactly 0 or 1:", bad or "none")
    assert not bad


def prediction_sets():
    sets = {f"pooled on {lg}": av.pred_path("pooled", lg) for lg in LEAGUES}
    sets |= {f"lolo_{lg} on {lg}": av.pred_path(f"lolo_{lg}", lg) for lg in LEAGUES}
    return sets | {f"out-of-fold on {lg}": oof_path(lg) for lg in LEAGUES}


def ranges(name, p):
    """Print min and max of each probability column; return the columns that reach 0 or 1."""
    bad = []
    for lb in LABELS:
        v = np.asarray(p[f"p_{lb}"])
        print(f"{name} {lb}: rows {len(v)}, min {v.min()!r}, max {v.max()!r}")
        if v.min() <= 0 or v.max() >= 1:
            bad.append(f"{name} {lb}")
    return bad


def metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=np.float64)
    base, mean = y.mean(), p.mean()
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "base_rate": base,
        "mean_predicted": mean,
        "relative_miss": mean / base - 1,
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
    }


def show(frame):
    return frame.to_string(float_format=lambda v: f"{v:.6f}")


def compare():
    bad = []
    for name, path in prediction_sets().items():
        bad += ranges(name, pd.read_parquet(path))
    print("probabilities exactly 0 or 1:", bad or "none")
    assert not bad

    print("\n4d: window 1, all four leagues: pooled model in-sample and out-of-fold")
    y = pd.concat([av.train_labels([(lg, 1)]) for lg in LEAGUES], ignore_index=True)
    for source in ("pooled in-sample", "out-of-fold"):
        frames = []
        for lg in LEAGUES:
            path = oof_path(lg) if source == "out-of-fold" else av.pred_path("pooled", lg)
            p = pd.read_parquet(path)
            frames.append(p[p.game_id.isin(av.game_ids(lg, 1))])
        p = pd.concat(frames, ignore_index=True)
        av.check_keys(p, y)
        for lb in LABELS:
            m = metrics(y[lb], p[f"p_{lb}"])
            print(f"\n{source}, {lb}: " + ", ".join(f"{k} {v!r}" for k, v in m.items()))
            yy, pp = y[lb].to_numpy().astype(int), p[f"p_{lb}"].to_numpy().astype(float)
            print(show(av.reliability(yy, pp)))


def logit(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log(1 - p)


def apply(cal, p):
    return 1 / (1 + np.exp(-(cal["a"] + cal["b"] * logit(p))))


def fit_calibrator(p, y, window):
    """Unpenalized maximum likelihood logistic regression of y on logit(p), window-1 rows only."""
    if (np.asarray(window) != 1).any():
        raise ValueError("calibrators are fit on window-1 rows only")
    x = logit(p).reshape(-1, 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = LogisticRegression(C=np.inf, solver="newton-cholesky", tol=TOL, max_iter=MAX_ITER)
        m.fit(x, np.asarray(y, dtype=int))
    b = float(m.coef_[0, 0])
    if not b > 0:
        raise ValueError(f"calibration slope is not positive: {b!r}")
    return {
        "a": float(m.intercept_[0]),
        "b": b,
        "n_iter": int(m.n_iter_[0]),
        "warnings": [f"{w.category.__name__}: {w.message}" for w in caught],
    }


def raw_rows(branch, league, window, d3=False):
    """Uncalibrated predictions with each row's window, and labels, in file order.

    Pooled window 1 comes from the out-of-fold predictions, since the pooled model trained on it.
    """
    if branch == "pooled" and window == 1:
        p = pd.read_parquet(oof_path(league, d3))
    else:
        p = pd.read_parquet(av.pred_path(fit_name(branch, league, d3), league))
    y = pd.read_parquet(av.labels_path(league))
    ids = av.game_ids(league, window)
    p = p[p.game_id.isin(ids)].reset_index(drop=True)
    y = y[y.game_id.isin(ids)].reset_index(drop=True)
    av.check_keys(p, y)
    w = windows().query("league == @league").set_index("game_id").window
    return p.assign(window=p.game_id.map(w).astype("int64")), y


def fit_all():
    out = []
    for branch in BRANCHES:
        for lg in LEAGUES:
            p, y = raw_rows(branch, lg, 1)
            for lb in LABELS:
                pp, yy = p[f"p_{lb}"].to_numpy(), y[lb].to_numpy().astype(int)
                c = fit_calibrator(pp, yy, p.window)
                x, pc = logit(pp), apply(c, pp)
                out.append(
                    {"branch": branch, "league": lg, "label": lb, "rows": len(yy)}
                    | {"positives": int(yy.sum()), **c}
                    | {"resid_level": (yy - pc).sum() / yy.sum()}
                    | {"resid_slope": ((yy - pc) * x).sum() / np.abs(yy * x).sum()}
                )
    return out


def calibrate():
    first = fit_all()
    cols = ["branch", "league", "label", "rows", "positives", "a", "b", "n_iter"]
    print(pd.DataFrame(first)[[*cols, "resid_level", "resid_slope"]].to_string(index=False))
    print("\nexact a and b:")
    for c in first:
        print(f"{c['branch']} {c['league']} {c['label']}: a {c['a']!r}, b {c['b']!r}")
    warned = [(c["branch"], c["league"], c["label"], c["warnings"]) for c in first if c["warnings"]]
    print("\nwarnings:", warned or "none")
    worst = max(max(abs(c["resid_level"]), abs(c["resid_slope"])) for c in first)
    print(f"largest absolute residual {worst!r}")
    assert worst < 1e-6
    assert all(c["b"] > 0 for c in first)

    second = fit_all()
    pairs = list(zip(first, second, strict=True))
    same = all(c["a"] == d["a"] and c["b"] == d["b"] for c, d in pairs)
    diff = max(max(abs(c["a"] - d["a"]), abs(c["b"] - d["b"])) for c, d in pairs)
    print(f"\n5e: refit a and b bit-identical {same}, largest difference {diff!r}")
    CALIBRATORS_PATH.write_text(json.dumps([{k: c[k] for k in cols} for c in first], indent=1))


def load_calibrators():
    cals = json.loads(CALIBRATORS_PATH.read_text())
    return {(c["branch"], c["league"], c["label"]): c for c in cals}


def same_bins(raw, cal):
    """Bin membership under the binning of actionvalue.reliability."""
    return np.array_equal(av.qbins(raw), av.qbins(cal))


def evaluation(branch, name, window, lb, y, raw, pc):
    r, c = metrics(y, raw), metrics(y, pc)
    note = "calibrator fit rows: mean matches base rate by construction" if window == 1 else ""
    if abs(c["relative_miss"]) > 0.1:
        note += " FLAG: calibrated relative miss over 10%"
    print(f"\n-- reliability, raw and calibrated: {branch} | {name} | window {window} | {lb}")
    table = av.reliability(y, raw).join(av.reliability(y, pc), lsuffix="_raw", rsuffix="_cal")
    print(show(table))
    if name in LEAGUES:
        same = same_bins(raw, pc)
        print(f"same rows in every bin before and after calibration: {same}")
        assert same
    keep = ("mean_predicted", "relative_miss", "brier", "log_loss")
    return (
        {"branch": branch, "leagues": name, "window": window, "label": lb}
        | {k: r[k] for k in ("rows", "positives", "base_rate")}
        | {f"raw_{k}": r[k] for k in keep}
        | {f"cal_{k}": c[k] for k in keep}
        | {"note": note}
    )


def evaluate():
    cals = load_calibrators()
    lines, bad = [], []
    for window in (1, 2):
        for branch in BRANCHES:
            rows = {lg: raw_rows(branch, lg, window) for lg in LEAGUES}
            cal = {
                (lg, lb): apply(cals[(branch, lg, lb)], rows[lg][0][f"p_{lb}"])
                for lg in LEAGUES
                for lb in LABELS
            }
            for lg in LEAGUES:
                name = f"calibrated {branch} {lg} window {window}"
                bad += ranges(name, {f"p_{lb}": cal[(lg, lb)] for lb in LABELS})
            groups = [[lg] for lg in LEAGUES] + ([LEAGUES] if branch == "pooled" else [])
            for group in groups:
                name = group[0] if len(group) == 1 else "all four"
                for lb in LABELS:
                    y = np.concatenate([rows[lg][1][lb].to_numpy().astype(int) for lg in group])
                    raw = np.concatenate([rows[lg][0][f"p_{lb}"].to_numpy() for lg in group])
                    pc = np.concatenate([cal[(lg, lb)] for lg in group])
                    lines.append(evaluation(branch, name, window, lb, y, raw.astype(float), pc))
    print("\ncalibrated probabilities exactly 0 or 1:", bad or "none")
    assert not bad
    out = pd.DataFrame(lines)
    pct = lambda v: f"{v:+.2%}"  # noqa: E731
    fmt = {c: pct for c in ("raw_relative_miss", "cal_relative_miss")}
    print("\n6b: raw and calibrated, one line per evaluation")
    for lb in LABELS:
        for window in (1, 2):
            part = out[(out.label == lb) & (out.window == window)]
            print(f"\n{lb}, window {window}")
            print(part.to_string(index=False, formatters=fmt, float_format=lambda v: f"{v:.6f}"))
    print("\nexact calibrated brier, pooled branch, all four leagues:")
    for r in out[(out.branch == "pooled") & (out.leagues == "all four")].itertuples():
        print(f"window {r.window} {r.label}: {r.cal_brier!r}")


def hl_bins(y, p):
    """Rows, positives and summed predicted probability per bin of actionvalue.reliability."""
    d = pd.DataFrame({"bin": av.qbins(p), "y": np.asarray(y, dtype=int), "p": np.asarray(p, float)})
    return d.groupby("bin").agg(n=("y", "size"), o=("y", "sum"), e=("p", "sum"))


def hl_from_bins(g):
    """Hosmer-Lemeshow: the sum over bins of (O - E)^2 / (E * (1 - E / n))."""
    return float((((g.o - g.e) ** 2) / (g.e * (1 - g.e / g.n))).sum())


def hl(y, p):
    return hl_from_bins(hl_bins(y, p))


def residual_table(y, p):
    g = hl_bins(y, p)
    return g.assign(mean_predicted=g.e / g.n, observed=g.o / g.n, relative_residual=g.o / g.e - 1)


def choose_depth(ll6, hl6, ll3, hl3):
    """Depth 3 replaces depth 6 only when no worse on log loss and strictly better on HL."""
    if ll3 <= ll6 and hl3 < hl6:
        return 3
    if ll6 <= ll3 and hl6 <= hl3:
        return 6
    return "mixed"


def pooled_calibrators(d3):
    return branch_calibrators("pooled", d3)


def branch_calibrators(branch, d3):
    """Maps per league and label, fit on that branch and depth's window-1 rows."""
    out = {}
    for lg in LEAGUES:
        p, y = raw_rows(branch, lg, 1, d3)
        for lb in LABELS:
            pp, yy = p[f"p_{lb}"].to_numpy(), y[lb].to_numpy().astype(int)
            c = fit_calibrator(pp, yy, p.window)
            x, pc = logit(pp), apply(c, pp)
            out[(lg, lb)] = (
                c
                | {"rows": len(yy), "positives": int(yy.sum())}
                | {"resid_level": (yy - pc).sum() / yy.sum()}
                | {"resid_slope": ((yy - pc) * x).sum() / np.abs(yy * x).sum()}
            )
    return out


def show_calibrators(cals):
    cols = ["rows", "positives", "a", "b", "n_iter", "resid_level", "resid_slope"]
    rows = [{"league": lg, "label": lb} | {k: cals[(lg, lb)][k] for k in cols} for lg, lb in cals]
    print(pd.DataFrame(rows).to_string(index=False))
    for lg, lb in cals:
        print(f"{lg} {lb}: a {cals[(lg, lb)]['a']!r}, b {cals[(lg, lb)]['b']!r}")
    print("warnings:", {k: c["warnings"] for k, c in cals.items() if c["warnings"]} or "none")
    worst = max(max(abs(c["resid_level"]), abs(c["resid_slope"])) for c in cals.values())
    print(f"largest absolute residual {worst!r}")
    assert worst < 1e-6
    assert all(c["b"] > 0 for c in cals.values())


def depth_values(d3):
    """Window-1 out-of-fold labels with raw and calibrated probabilities, per league and label."""
    cals = pooled_calibrators(d3)
    out = {}
    for lg in LEAGUES:
        p, y = raw_rows("pooled", lg, 1, d3)
        for lb in LABELS:
            raw = p[f"p_{lb}"].to_numpy().astype(float)
            out[(lg, lb)] = (y[lb].to_numpy().astype(int), raw, apply(cals[(lg, lb)], raw))
    return cals, out


def depth_line(depth, name, lb, y, raw, pc):
    r, c = metrics(y, raw), metrics(y, pc)
    keep = ("mean_predicted", "brier", "log_loss")
    return (
        {"depth": depth, "leagues": name, "label": lb}
        | {k: r[k] for k in ("rows", "positives", "base_rate")}
        | {f"raw_{k}": r[k] for k in keep}
        | {"raw_hl": hl(y, raw)}
        | {f"cal_{k}": c[k] for k in keep}
        | {"cal_hl": hl(y, pc)}
    )


def compare_depths():
    stored = load_calibrators()
    cals, values, lines = {}, {}, []
    for depth, d3 in DEPTHS.items():
        print(f"\n== depth {depth}: pooled-branch calibrators on window-1 out-of-fold rows")
        cals[depth], values[depth] = depth_values(d3)
        show_calibrators(cals[depth])
        again = pooled_calibrators(d3)
        same = all(cals[depth][k]["a"] == again[k]["a"] for k in again)
        same = same and all(cals[depth][k]["b"] == again[k]["b"] for k in again)
        print(f"refit a and b bit-identical: {same}")
        assert same
        if depth == 6:
            match = all(
                cals[6][(lg, lb)]["a"] == stored[("pooled", lg, lb)]["a"]
                and cals[6][(lg, lb)]["b"] == stored[("pooled", lg, lb)]["b"]
                for lg in LEAGUES
                for lb in LABELS
            )
            print(f"4a: refit depth-6 calibrators equal calibrators.json exactly: {match}")
            assert match

    for depth in DEPTHS:
        for lb in LABELS:
            for group in [[lg] for lg in LEAGUES] + [list(LEAGUES)]:
                name = group[0] if len(group) == 1 else "all four"
                y, raw, pc = (
                    np.concatenate([values[depth][(lg, lb)][i] for lg in group]) for i in range(3)
                )
                lines.append(depth_line(depth, name, lb, y, raw, pc))
                if name == "all four":
                    print(f"\n-- depth {depth} | all four | window 1 | {lb} | calibrated bins")
                    print(show(residual_table(y, pc)))
                    print(f"-- depth {depth} | all four | window 1 | {lb} | raw bins")
                    print(show(residual_table(y, raw)))

    out = pd.DataFrame(lines)
    print("\n4c: window 1, raw and calibrated, one line per depth, label and group")
    for lb in LABELS:
        print(f"\n{lb}")
        print(out[out.label == lb].to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print("\n4d: rule inputs at full precision, all four leagues together, calibrated")
    chosen = {}
    for lb in LABELS:
        r = {
            d: out[(out.depth == d) & (out.label == lb) & (out.leagues == "all four")].iloc[0]
            for d in DEPTHS
        }
        ll6, hl6, ll3, hl3 = r[6].cal_log_loss, r[6].cal_hl, r[3].cal_log_loss, r[3].cal_hl
        print(f"{lb}: depth 6 log loss {ll6!r}, HL {hl6!r}")
        print(f"{lb}: depth 3 log loss {ll3!r}, HL {hl3!r}")
        print(f"{lb}: log loss no higher at depth 3 {ll3 <= ll6}, HL lower at depth 3 {hl3 < hl6}")
        chosen[lb] = choose_depth(ll6, hl6, ll3, hl3)
        print(f"{lb}: chosen depth {chosen[lb]}")
    print(f"\n4d outcome: {chosen}")
    return chosen


def full_probs(branch, league, cals=None):
    """Probabilities for every action of a league in file order, calibrated if cals is given.

    Pooled window 1 comes from the out-of-fold predictions.
    """
    p = pd.read_parquet(av.pred_path(av.rating_fit(branch, league), league))
    if branch == "pooled":
        w1 = p.game_id.isin(av.game_ids(league, 1)).to_numpy()
        oof = pd.read_parquet(oof_path(league))
        av.check_keys(p[w1], oof)
        for lb in LABELS:
            p.loc[w1, f"p_{lb}"] = oof[f"p_{lb}"].to_numpy()
    if cals is not None:
        for lb in LABELS:
            p[f"p_{lb}"] = apply(cals[(branch, league, lb)], p[f"p_{lb}"])
    return p


def calibrated_table(name, branch, cals, lineups, wins, mins):
    """Player-window table built from one branch's probabilities, calibrated if cals is given."""
    values = pd.concat(
        [av.action_values(branch, lg, full_probs(branch, lg, cals)) for lg in LEAGUES],
        ignore_index=True,
    )
    return av.player_window_table(name, lineups, wins, mins, values=values)


def check_against(old, new):
    counts = pd.concat(
        [old.groupby(["league", "window"]).size(), new.groupby(["league", "window"]).size()],
        axis=1,
        keys=["existing", "calibrated"],
    )
    print(counts.to_string())
    cols = ["league", "player_id", "team_id", "window", "minutes", "games"]
    same = old[cols].reset_index(drop=True).equals(new[cols].reset_index(drop=True))
    pr = av.pairs(new, 450, 270)
    per_league = {lg: int((pr.league == lg).sum()) for lg in LEAGUES}
    print(f"identical key, minutes and games columns: {same}")
    print(f"pairs at 450 / 270: {len(pr)} {per_league}")
    assert (counts.existing == counts.calibrated).all() and same
    assert len(pr) == 1144
    assert per_league == {"la_liga": 299, "premier_league": 278, "serie_a": 280, "ligue_1": 287}


def spread(table, regular):
    r = table.merge(regular, on=["league", "player_id"])
    return r.groupby(["league", "window"]).vaep_per90.agg(
        n="size",
        mean="mean",
        median="median",
        p10=lambda s: s.quantile(0.1),
        p90=lambda s: s.quantile(0.9),
    )


def tables():
    cals = load_calibrators()
    wins = windows()
    lineups = minutes.load_lineups("statsbomb")
    mins = av.window_minutes(lineups, wins)
    players = pd.read_parquet(PROCESSED / "minutes_player_statsbomb.parquet")
    regular = players[players.minutes >= 450][["league", "player_id"]]
    existing = {b: pd.read_parquet(PROCESSED / f"player_window_vaep_{b}.parquet") for b in BRANCHES}

    new = {}
    for b in BRANCHES:
        new[b] = calibrated_table(f"{b} calibrated", b, cals, lineups, wins, mins)
        print(f"\n7c: {b} rows per league and window")
        check_against(existing[b], new[b])
        new[b].to_parquet(PROCESSED / f"player_window_vaep_{b}_cal.parquet", index=False)
    uncal = calibrated_table("pooled out-of-fold uncalibrated", "pooled", None, lineups, wins, mins)

    sets = {
        "existing pooled": existing["pooled"],
        "pooled window 1 from uncalibrated out-of-fold": uncal[uncal.window == 1],
        "pooled calibrated": new["pooled"],
        "existing lolo": existing["lolo"],
        "lolo calibrated": new["lolo"],
    }
    print("\n7d: vaep_per90 among players with 450+ season minutes")
    stats = {}
    for name, t in sets.items():
        stats[name] = spread(t, regular)
        print(f"\n{name}")
        print(stats[name].to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nexact mean vaep_per90 per league and window:")
    for name in ("pooled calibrated", "lolo calibrated"):
        for (lg, w), m in stats[name]["mean"].items():
            print(f"{name} {lg} window {w}: {m!r}")

    print("\n7e: (lolo mean - pooled mean) / pooled mean, players with 450+ season minutes")
    gap = pd.concat(
        [
            (stats["existing lolo"]["mean"] - stats["existing pooled"]["mean"])
            / stats["existing pooled"]["mean"],
            (stats["lolo calibrated"]["mean"] - stats["pooled calibrated"]["mean"])
            / stats["pooled calibrated"]["mean"],
        ],
        axis=1,
        keys=["existing", "calibrated"],
    )
    print(gap.to_string(float_format=lambda v: f"{v:+.4f}"))

    print("\n7f: NOT A RESULT. How much calibration reorders players: Spearman between existing")
    print("and calibrated vaep_per90, same player-team-window, players with 450+ season minutes")
    keys = ["league", "player_id", "team_id", "window"]
    rows = []
    for b in BRANCHES:
        m = (
            existing[b]
            .merge(new[b], on=keys, suffixes=("_old", "_cal"))
            .merge(regular, on=["league", "player_id"])
        )
        for (lg, w), g in m.groupby(["league", "window"]):
            rho = spearmanr(g.vaep_per90_old, g.vaep_per90_cal).statistic
            rows.append({"branch": b, "league": lg, "window": w, "n": len(g)} | {"spearman": rho})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def full_fit_parts(name):
    """Window-1 parts of a depth-3 full fit, named pooled_d3 or lolo_<league>_d3."""
    return [(lg, 1) for lg in av.FITS[name.removesuffix("_d3")]]


def rated_leagues(name):
    base = name.removesuffix("_d3")
    return LEAGUES if base == "pooled" else [base.removeprefix("lolo_")]


def fit_full_d3(name, label):
    y = av.train_labels(full_fit_parts(name))[label]
    print("params passed to xgboost.train:", DEPTH3, "| num_boost_round:", av.ROUNDS)
    start = time.perf_counter()
    av.fit(name, labels=[label], parts=full_fit_parts(name), params=DEPTH3)
    print(
        f"{name}, {label}: training rows {len(y)}, positives {int(y.sum())}, "
        f"base rate {y.mean():.6f}, wall clock {time.perf_counter() - start:.1f} s"
    )


def predict_full_d3(name):
    for lg in rated_leagues(name):
        av.write_predictions(name, lg)
        print(f"predicted {name} on {lg}", flush=True)


def refit_full_d3(label):
    """pooled_d3 refit once: its window-2 predictions must be bit-identical."""
    saved = xgb.Booster(model_file=av.model_path("pooled_d3", label))
    again = av.fit(
        "pooled_d3",
        labels=[label],
        save=False,
        parts=full_fit_parts("pooled_d3"),
        params=DEPTH3,
    )[label]
    first, second = [], []
    for lg in LEAGUES:
        for x, _ in av.batches(lg, 2):
            first.append(av.predict(saved, x))
            second.append(av.predict(again, x))
    a, b = np.concatenate(first), np.concatenate(second)
    print(f"pooled_d3 {label}: window-2 rows {len(a)}, identical {np.array_equal(a, b)}")
    print(f"max abs difference {np.abs(a - b).max()!r}")


def window1_pooled(d3, in_sample):
    """Pooled window-1 probabilities: the fit's own predictions, or the out-of-fold ones."""
    frames = []
    for lg in LEAGUES:
        if in_sample:
            p = pd.read_parquet(av.pred_path(fit_name("pooled", lg, d3), lg))
            p = p[p.game_id.isin(av.game_ids(lg, 1))]
        else:
            p = pd.read_parquet(oof_path(lg, d3))
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


def insample_against_oof():
    """5c: how far each depth's pooled fit overfits its own window-1 rows."""
    y = pd.concat([av.train_labels([(lg, 1)]) for lg in LEAGUES], ignore_index=True)
    for depth, d3 in DEPTHS.items():
        ins, oof = window1_pooled(d3, True), window1_pooled(d3, False)
        av.check_keys(ins, y)
        av.check_keys(oof, y)
        for lb in LABELS:
            yy = y[lb].to_numpy().astype(int)
            li = log_loss(yy, ins[f"p_{lb}"].to_numpy().astype(float), labels=[0, 1])
            lo = log_loss(yy, oof[f"p_{lb}"].to_numpy().astype(float), labels=[0, 1])
            print(f"depth {depth} {lb}: window-1 log loss in-sample {li!r}, out-of-fold {lo!r}")


def write_calibrators_v2(chosen):
    """5d: sixteen entries, each carrying the depth its label was fit at."""
    stored = load_calibrators()
    fresh = {b: branch_calibrators(b, True) for b in BRANCHES if 3 in chosen.values()}
    cols = ["rows", "positives", "a", "b", "n_iter"]
    out = []
    for branch in BRANCHES:
        for lg in LEAGUES:
            for lb in LABELS:
                depth = chosen[lb]
                c = fresh[branch][(lg, lb)] if depth == 3 else stored[(branch, lg, lb)]
                out.append(
                    {"branch": branch, "league": lg, "label": lb, "depth": depth}
                    | {k: c[k] for k in cols}
                )
    print(pd.DataFrame(out).to_string(index=False))
    for c in out:
        print(f"{c['branch']} {c['league']} {c['label']}: a {c['a']!r}, b {c['b']!r}")
    assert len(out) == 16
    CALIBRATORS_V2_PATH.write_text(json.dumps(out, indent=1))


def load_calibrators_v2():
    cals = json.loads(CALIBRATORS_V2_PATH.read_text())
    return {(c["branch"], c["league"], c["label"]): c for c in cals}


def evaluate_v2(chosen):
    """5e: reported only. Each label at its chosen depth, beside phase 4b's depth-6 values."""
    sets = {6: (load_calibrators(), False), 3: (load_calibrators_v2(), True)}
    lines = []
    for depth, (cals, d3) in sets.items():
        for window in (1, 2):
            for branch in BRANCHES:
                rows = {lg: raw_rows(branch, lg, window, d3) for lg in LEAGUES}
                cal = {
                    (lg, lb): apply(cals[(branch, lg, lb)], rows[lg][0][f"p_{lb}"])
                    for lg in LEAGUES
                    for lb in LABELS
                }
                groups = [[lg] for lg in LEAGUES] + ([LEAGUES] if branch == "pooled" else [])
                for group in groups:
                    name = group[0] if len(group) == 1 else "all four"
                    for lb in LABELS:
                        if depth == 3 and chosen[lb] != 3:
                            continue
                        y = np.concatenate([rows[lg][1][lb].to_numpy().astype(int) for lg in group])
                        raw = np.concatenate([rows[lg][0][f"p_{lb}"].to_numpy() for lg in group])
                        pc = np.concatenate([cal[(lg, lb)] for lg in group])
                        line = evaluation(branch, name, window, lb, y, raw.astype(float), pc)
                        lines.append({"depth": depth} | line)
    out = pd.DataFrame(lines)
    print("\n5e: raw and calibrated, one line per depth, evaluation")
    pct = lambda v: f"{v:+.2%}"  # noqa: E731
    fmt = {c: pct for c in ("raw_relative_miss", "cal_relative_miss")}
    for lb in LABELS:
        for window in (1, 2):
            part = out[(out.label == lb) & (out.window == window)]
            print(f"\n{lb}, window {window}")
            print(part.to_string(index=False, formatters=fmt, float_format=lambda v: f"{v:.6f}"))


def full_probs_v2(branch, league, cals, chosen):
    """Each label from its chosen depth; pooled window 1 from that depth's out-of-fold rows."""
    out = None
    for lb in LABELS:
        d3 = chosen[lb] == 3
        p = pd.read_parquet(av.pred_path(fit_name(branch, league, d3), league))
        if branch == "pooled":
            w1 = p.game_id.isin(av.game_ids(league, 1)).to_numpy()
            oof = pd.read_parquet(oof_path(league, d3))
            av.check_keys(p[w1], oof)
            p.loc[w1, f"p_{lb}"] = oof[f"p_{lb}"].to_numpy()
        if out is None:
            out = p[av.KEYS].copy()
        out[f"p_{lb}"] = apply(cals[(branch, league, lb)], p[f"p_{lb}"])
    return out


def tables_v2(chosen):
    cals = load_calibrators_v2()
    wins = windows()
    lineups = minutes.load_lineups("statsbomb")
    mins = av.window_minutes(lineups, wins)
    players = pd.read_parquet(PROCESSED / "minutes_player_statsbomb.parquet")
    regular = players[players.minutes >= 450][["league", "player_id"]]
    existing = {
        b: pd.read_parquet(PROCESSED / f"player_window_vaep_{b}_cal.parquet") for b in BRANCHES
    }

    new = {}
    for b in BRANCHES:
        values = pd.concat(
            [av.action_values(b, lg, full_probs_v2(b, lg, cals, chosen)) for lg in LEAGUES],
            ignore_index=True,
        )
        new[b] = av.player_window_table(f"{b} v2", lineups, wins, mins, values=values)
        print(f"\n5f: {b} rows per league and window")
        check_against(existing[b], new[b])
        new[b].to_parquet(PROCESSED / f"player_window_vaep_{b}_v2.parquet", index=False)

    sets = {f"{b} {tag}": d[b] for tag, d in (("4b", existing), ("v2", new)) for b in BRANCHES}
    stats = {}
    print("\n5g: vaep_per90 among players with 450+ season minutes")
    for name, t in sets.items():
        stats[name] = spread(t, regular)
        print(f"\n{name}")
        print(stats[name].to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nexact mean vaep_per90 per league and window:")
    for name in ("pooled v2", "lolo v2"):
        for (lg, w), m in stats[name]["mean"].items():
            print(f"{name} {lg} window {w}: {m!r}")

    print("\n5g: (lolo mean - pooled mean) / pooled mean, players with 450+ season minutes")
    gap = pd.concat(
        [
            (stats[f"lolo {t}"]["mean"] - stats[f"pooled {t}"]["mean"])
            / stats[f"pooled {t}"]["mean"]
            for t in ("4b", "v2")
        ],
        axis=1,
        keys=["4b calibrated", "v2"],
    )
    print(gap.to_string(float_format=lambda v: f"{v:+.4f}"))

    print("\n5g: NOT A RESULT. How much the depth change reorders players: Spearman between")
    print("phase 4b's calibrated and the v2 vaep_per90, same player-team-window, 450+ minutes")
    keys = ["league", "player_id", "team_id", "window"]
    rows = []
    for b in BRANCHES:
        m = (
            existing[b]
            .merge(new[b], on=keys, suffixes=("_cal", "_v2"))
            .merge(regular, on=["league", "player_id"])
        )
        for (lg, w), g in m.groupby(["league", "window"]):
            rho = spearmanr(g.vaep_per90_cal, g.vaep_per90_v2).statistic
            rows.append({"branch": b, "league": lg, "window": w, "n": len(g), "spearman": rho})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def adopt():
    print("5c: pooled window-1 log loss, in-sample against out-of-fold")
    insample_against_oof()
    print("\n5d: calibrators_v2")
    write_calibrators_v2(CHOSEN_DEPTH)
    evaluate_v2(CHOSEN_DEPTH)
    tables_v2(CHOSEN_DEPTH)


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    step = argv[0]
    if step == "folds":
        make_folds()
    elif step == "fit":
        fit_fold(int(argv[1]), argv[2])
    elif step == "fit3":
        fit_fold(int(argv[1]), argv[2], d3=True)
    elif step == "refit":
        refit()
    elif step == "refit3":
        refit(d3=True)
    elif step == "predict":
        predict_oof()
    elif step == "predict3":
        predict_oof(d3=True)
        oof_ranges(d3=True)
    elif step == "depths":
        compare_depths()
    elif step == "fit3full":
        fit_full_d3(argv[1], argv[2])
    elif step == "predict3full":
        predict_full_d3(argv[1])
    elif step == "refit3full":
        refit_full_d3(argv[1])
    elif step == "adopt":
        adopt()
    elif step == "compare":
        compare()
    elif step == "calibrate":
        calibrate()
    elif step == "evaluate":
        evaluate()
    elif step == "tables":
        tables()


if __name__ == "__main__":
    main(sys.argv[1:])
