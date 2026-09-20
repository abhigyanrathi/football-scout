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


def oof_path(league):
    return INTERIM / f"vaep_pred_oof_{league}.parquet"


def fold_fit(k):
    return f"cf{k}"


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


def fit_fold(k, label):
    print("params passed to xgboost.train:", av.PARAMS, "| num_boost_round:", av.ROUNDS)
    parts = fold_parts(k, inside=False)
    y = av.train_labels(parts)[label]
    start = time.perf_counter()
    av.fit(fold_fit(k), labels=[label], parts=parts)
    secs = time.perf_counter() - start
    print(
        f"fold {k}, {label}: training rows {len(y)}, positives {int(y.sum())}, "
        f"base rate {y.mean():.6f}, wall clock {secs:.1f} s"
    )


def fold_predictions(booster, k):
    parts = fold_parts(k, inside=True)
    return np.concatenate([av.predict(booster, x) for lg, g in parts for x, _ in av.batches(lg, g)])


def refit():
    saved = xgb.Booster(model_file=av.model_path(fold_fit(0), "scores"))
    again = av.fit(fold_fit(0), labels=["scores"], save=False, parts=fold_parts(0, inside=False))
    first, second = fold_predictions(saved, 0), fold_predictions(again["scores"], 0)
    diff = np.abs(first - second).max()
    print(f"fold 0 rows {len(first)}: predictions identical {np.array_equal(first, second)}")
    print(f"max abs difference {diff!r}")


def predict_oof():
    folds = pd.read_parquet(FOLDS_PATH)
    models = {
        k: {lb: xgb.Booster(model_file=av.model_path(fold_fit(k), lb)) for lb in LABELS}
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
        out.to_parquet(oof_path(lg), index=False)


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


def raw_rows(branch, league, window):
    """Uncalibrated predictions with each row's window, and labels, in file order.

    Pooled window 1 comes from the out-of-fold predictions, since the pooled model trained on it.
    """
    if branch == "pooled" and window == 1:
        p = pd.read_parquet(oof_path(league))
    else:
        p = pd.read_parquet(av.pred_path(av.rating_fit(branch, league), league))
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


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    step = argv[0]
    if step == "folds":
        make_folds()
    elif step == "fit":
        fit_fold(int(argv[1]), argv[2])
    elif step == "refit":
        refit()
    elif step == "predict":
        predict_oof()
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
