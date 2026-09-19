import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import brier_score_loss, log_loss
from socceraction.spadl import add_names
from socceraction.vaep import VAEP, formula
from socceraction.vaep import features as fs
from socceraction.vaep.base import xfns_default

from fbrecruit import minutes
from fbrecruit.paths import INTERIM, PROCESSED
from fbrecruit.sources.statsbomb import LEAGUES
from fbrecruit.split import windows

SEED = 0
PARAMS = {"objective": "binary:logistic", "tree_method": "hist", "random_state": SEED, "n_jobs": 16}
# XGBClassifier's default n_estimators in the installed xgboost.
ROUNDS = 100
NB_PREV = VAEP().nb_prev_actions
COLUMNS = fs.feature_column_names(xfns_default, NB_PREV)
LABELS = ["scores", "concedes"]
KEYS = ["game_id", "action_id"]
BATCH = 100_000
GAMES_PER_WRITE = 20
FITS = {"pooled": list(LEAGUES)} | {f"lolo_{lg}": [o for o in LEAGUES if o != lg] for lg in LEAGUES}
W1_THRESHOLDS = [180, 270, 360, 450]
W2_THRESHOLDS = [180, 270, 360]


def features_path(league):
    return INTERIM / f"vaep_features_{league}.parquet"


def labels_path(league):
    return INTERIM / f"vaep_labels_{league}.parquet"


def model_path(fit, label):
    return INTERIM / f"vaep_model_{fit}_{label}.json"


def pred_path(fit, league):
    return INTERIM / f"vaep_pred_{fit}_{league}.parquet"


def rating_fit(kind, league):
    """The leakage rule: lolo values for a league come from the model that never saw it."""
    return "pooled" if kind == "pooled" else f"lolo_{league}"


def load(league, name):
    return pd.read_parquet(INTERIM / "statsbomb" / league / f"{name}.parquet")


def build_features(league):
    vaep = VAEP()
    games = load(league, "games")
    by_game = dict(tuple(load(league, "actions").groupby("game_id", sort=False)))
    writer, buffer, labels = None, [], []
    for i, (_, game) in enumerate(games.iterrows(), 1):
        a = by_game[game.game_id].reset_index(drop=True)
        x = vaep.compute_features(game, a)
        assert set(x.columns) == set(COLUMNS)
        x = x[COLUMNS].astype("float32")
        x.insert(0, "game_id", a.game_id.to_numpy())
        x.insert(1, "action_id", a.action_id.to_numpy())
        buffer.append(x)
        labels.append(pd.concat([a[KEYS], vaep.compute_labels(game, a)], axis=1))
        if len(buffer) == GAMES_PER_WRITE or i == len(games):
            table = pa.Table.from_pandas(pd.concat(buffer, ignore_index=True), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(features_path(league), table.schema)
            writer.write_table(table)
            buffer = []
            print(f"{league}: {i} of {len(games)} games", flush=True)
    writer.close()
    pd.concat(labels, ignore_index=True).to_parquet(labels_path(league), index=False)


def report_features(league):
    meta = pq.ParquetFile(features_path(league)).metadata
    names = pq.ParquetFile(features_path(league)).schema_arrow.names
    labels = pd.read_parquet(labels_path(league))
    n_actions = len(load(league, "actions"))
    feature_names = [c for c in names if c not in KEYS]
    print(f"== {league}")
    print(
        "feature rows",
        meta.num_rows,
        "| feature columns",
        len(feature_names),
        "(+ game_id, action_id)",
    )
    print("label rows", len(labels), "| action rows", n_actions)
    print("first ten:", feature_names[:10])
    print("last ten:", feature_names[-10:])
    for label in LABELS:
        print(
            f"{label}: positives {int(labels[label].sum())}, base rate {labels[label].mean():.6f}"
        )
    if meta.num_rows != n_actions:
        print(f"MISMATCH: {league} feature rows {meta.num_rows} != action rows {n_actions}")
    assert meta.num_rows == n_actions == len(labels)


def game_ids(league, window):
    w = windows()
    return set(w[(w.league == league) & (w.window == window)].game_id)


def batches(league, window=None):
    """Features and labels of one league, optionally one window, in file order."""
    labels = pd.read_parquet(labels_path(league))
    keep_ids = None if window is None else game_ids(league, window)
    offset = 0
    for batch in pq.ParquetFile(features_path(league)).iter_batches(BATCH):
        x = batch.to_pandas()
        y = labels.iloc[offset : offset + len(x)].reset_index(drop=True)
        offset += len(x)
        assert (x.game_id.to_numpy() == y.game_id.to_numpy()).all()
        assert (x.action_id.to_numpy() == y.action_id.to_numpy()).all()
        if keep_ids is not None:
            keep = x.game_id.isin(keep_ids).to_numpy()
            x, y = x[keep], y[keep]
        if len(x):
            yield x, y


class Rows(xgb.DataIter):
    def __init__(self, parts):
        self.parts = parts
        self.rows = None
        super().__init__()

    def reset(self):
        self.rows = (b for league, window in self.parts for b in batches(league, window))

    def next(self, input_data):
        b = next(self.rows, None)
        if b is None:
            return 0
        x, y = b
        input_data(data=x[COLUMNS], label=y.scores.to_numpy())
        return 1


def train_labels(parts):
    frames = [
        pd.read_parquet(labels_path(lg)).loc[lambda d, lg=lg, w=w: d.game_id.isin(game_ids(lg, w))]
        for lg, w in parts
    ]
    return pd.concat(frames, ignore_index=True)


def fit(name, labels=LABELS, save=True):
    parts = [(lg, 1) for lg in FITS[name]]
    y = train_labels(parts)
    start = time.perf_counter()
    rows = Rows(parts)
    rows.reset()
    dm = xgb.QuantileDMatrix(rows, nthread=PARAMS["n_jobs"])
    built = time.perf_counter() - start
    print(f"== fit {name}: leagues {FITS[name]}, window 1, rows {dm.num_row()} (labels {len(y)})")
    print(f"QuantileDMatrix built in {built:.1f} s")
    boosters = {}
    for label in labels:
        dm.set_label(y[label].to_numpy())
        assert int(dm.get_label().sum()) == int(y[label].sum())
        t = time.perf_counter()
        boosters[label] = xgb.train(PARAMS, dm, ROUNDS, verbose_eval=False)
        secs = time.perf_counter() - t
        print(
            f"{label}: positives {int(y[label].sum())}, base rate {y[label].mean():.6f}, "
            f"train {secs:.1f} s"
        )
        if save:
            boosters[label].save_model(model_path(name, label))
    return boosters


def predict(booster, x):
    return booster.predict(xgb.DMatrix(x[COLUMNS], nthread=PARAMS["n_jobs"]))


def write_predictions(name, league):
    boosters = {label: xgb.Booster(model_file=model_path(name, label)) for label in LABELS}
    frames = []
    for x, _ in batches(league):
        frames.append(x[KEYS].assign(**{f"p_{lb}": predict(b, x) for lb, b in boosters.items()}))
    pd.concat(frames, ignore_index=True).to_parquet(pred_path(name, league), index=False)


def evaluation_rows(name, league, window):
    p = pd.read_parquet(pred_path(name, league))
    y = pd.read_parquet(labels_path(league))
    assert (p[KEYS].to_numpy() == y[KEYS].to_numpy()).all()
    keep = p.game_id.isin(game_ids(league, window)).to_numpy()
    return p[keep].reset_index(drop=True), y[keep].reset_index(drop=True)


def reliability(y, p, bins=10):
    """Equal-count bins of the predicted probability."""
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"bin": q, "y": y, "p": p}).groupby("bin")
    return d.agg(count=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean"))


def calibration(title, p, y):
    print(f"\n-- {title}")
    for label in LABELS:
        yy, pp = y[label].to_numpy().astype(int), p[f"p_{label}"].to_numpy().astype(float)
        base, mean = yy.mean(), pp.mean()
        rel_gap = (mean - base) / base
        flag = "  FLAG: mean predicted differs from base rate by more than 10%" * (
            abs(rel_gap) > 0.1
        )
        print(
            f"{label}: rows {len(yy)}, base rate {base:.6f}, mean predicted {mean:.6f} "
            f"(relative {rel_gap:+.2%}), brier {brier_score_loss(yy, pp):.6f}, "
            f"log loss {log_loss(yy, pp, labels=[0, 1]):.6f}{flag}"
        )
        print(reliability(yy, pp).to_string(float_format=lambda v: f"{v:.6f}"))


def evaluate():
    parts = [evaluation_rows("pooled", lg, 2) for lg in LEAGUES]
    p = pd.concat([a for a, _ in parts], ignore_index=True)
    y = pd.concat([b for _, b in parts], ignore_index=True)
    calibration("pooled model on window 2 of all four leagues", p, y)
    for lg in LEAGUES:
        for w in (1, 2):
            p, y = evaluation_rows(f"lolo_{lg}", lg, w)
            calibration(f"lolo_{lg} on held-out {lg}, window {w}", p, y)


def determinism():
    first = {lb: xgb.Booster(model_file=model_path("pooled", lb)) for lb in LABELS}["scores"]
    second = fit("pooled", labels=["scores"], save=False)["scores"]
    briers, preds = [], []
    for booster in (first, second):
        ps, ys = [], []
        for lg in LEAGUES:
            for x, y in batches(lg, 2):
                ps.append(predict(booster, x))
                ys.append(y.scores.to_numpy().astype(int))
        p, y = np.concatenate(ps), np.concatenate(ys)
        preds.append(p)
        briers.append(brier_score_loss(y, p))
    print(f"brier first {briers[0]!r}, second {briers[1]!r}, identical {briers[0] == briers[1]}")
    diff = np.abs(preds[0] - preds[1]).max()
    print(f"predictions identical {np.array_equal(*preds)}, max abs difference {diff!r}")


def action_values(kind, league):
    """VAEP values of every action in a league, rated per game."""
    actions = add_names(load(league, "actions"))
    p = pd.read_parquet(pred_path(rating_fit(kind, league), league))
    assert (actions[KEYS].to_numpy() == p[KEYS].to_numpy()).all()
    frames = []
    for idx in actions.groupby("game_id", sort=False).indices.values():
        a = actions.iloc[idx].reset_index(drop=True)
        g = p.iloc[idx].reset_index(drop=True)
        v = formula.value(a, g.p_scores.astype(float), g.p_concedes.astype(float))
        frames.append(pd.concat([a[["game_id", "player_id", "team_id"]], v], axis=1))
    return pd.concat(frames, ignore_index=True).assign(league=league)


def window_minutes(lineups, wins):
    """Player-team minutes per window, with the aggregation of the minutes module."""
    lw = lineups.merge(wins[["league", "game_id", "window"]], on=["league", "game_id"])
    return minutes.aggregate(lw, [*minutes.TEAM_KEYS, "window"])


def per90(frame):
    """vaep_per90, dropping rows with no minutes, where it is undefined."""
    out = frame[frame.minutes > 0].copy()
    out["vaep_per90"] = out.vaep_sum / out.minutes * 90
    return out


TABLE_COLUMNS = [
    "league", "season", "player_id", "team_id", "player_name", "window",
    "minutes", "games", "vaep_sum", "offensive_sum", "defensive_sum", "vaep_per90",
]  # fmt: skip


def player_window_table(kind, lineups, wins, mins):
    values = pd.concat([action_values(kind, lg) for lg in LEAGUES], ignore_index=True)
    values = values.merge(wins[["league", "game_id", "window"]], on=["league", "game_id"])
    values["player_id"] = values.player_id.astype("int64")
    keys = ["league", "player_id", "team_id", "window"]
    sums = values.groupby(keys, as_index=False).agg(
        vaep_sum=("vaep_value", "sum"),
        offensive_sum=("offensive_value", "sum"),
        defensive_sum=("defensive_value", "sum"),
    )
    lineup_keys = mins[keys].assign(team_id=mins.team_id.astype("int64"))
    orphan = sums.merge(lineup_keys, on=keys, how="left", indicator=True)
    orphan = orphan[orphan._merge == "left_only"]
    print(f"{kind}: action sums with no lineup row {len(orphan)}, vaep {orphan.vaep_sum.sum():.4f}")
    names = lineups.drop_duplicates(minutes.TEAM_KEYS)[[*minutes.TEAM_KEYS, "player_name"]]
    table = mins.assign(team_id=mins.team_id.astype("int64")).merge(sums, on=keys, how="left")
    table = table.merge(
        names.assign(team_id=names.team_id.astype("int64")),
        on=["league", "season", "team_id", "player_id"],
    )
    table[["vaep_sum", "offensive_sum", "defensive_sum"]] = table[
        ["vaep_sum", "offensive_sum", "defensive_sum"]
    ].fillna(0.0)
    print(f"{kind}: rows with 0 window minutes, excluded {int((table.minutes == 0).sum())}")
    return per90(table)[TABLE_COLUMNS].sort_values(keys).reset_index(drop=True)


def tables():
    wins = windows()
    lineups = minutes.load_lineups("statsbomb")
    mins = window_minutes(lineups, wins)
    season = pd.read_parquet(PROCESSED / "minutes_player_team_statsbomb.parquet")
    total = mins.groupby(minutes.TEAM_KEYS, as_index=False).minutes.sum()
    rec = season[[*minutes.TEAM_KEYS, "minutes"]].merge(
        total, on=minutes.TEAM_KEYS, how="outer", suffixes=("_season", "_windows")
    )
    bad = rec[rec.minutes_season.fillna(-1) != rec.minutes_windows.fillna(-1)]
    print(f"5b: player-team rows {len(rec)}, w1 + w2 minutes != season minutes: {len(bad)}")
    print(bad.to_string(index=False))

    players = pd.read_parquet(PROCESSED / "minutes_player_statsbomb.parquet")
    regular = players[players.minutes >= 450][["league", "player_id"]]
    for kind in ("pooled", "lolo"):
        table = player_window_table(kind, lineups, wins, mins)
        table.to_parquet(PROCESSED / f"player_window_vaep_{kind}.parquet", index=False)
        print(f"\n5d: {kind} rows per league and window")
        print(table.groupby(["league", "window"]).size().unstack().to_string())
        r = table.merge(regular, on=["league", "player_id"])
        stats = r.groupby(["league", "window"]).vaep_per90.agg(
            n="size",
            mean="mean",
            median="median",
            p10=lambda s: s.quantile(0.1),
            p90=lambda s: s.quantile(0.9),
        )
        print(f"{kind}: vaep_per90 among players with 450+ season minutes")
        print(stats.to_string(float_format=lambda v: f"{v:.4f}"))


def pairs(table, t1, t2):
    keys = ["league", "player_id", "team_id"]
    w1 = table[(table.window == 1) & (table.minutes >= t1)][[*keys, "vaep_per90"]]
    w2 = table[(table.window == 2) & (table.minutes >= t2)][[*keys, "vaep_per90"]]
    return w1.merge(w2, on=keys, suffixes=("_w1", "_w2"))


def sensitivity():
    table = pd.read_parquet(PROCESSED / "player_window_vaep_pooled.parquet")
    print("PIPELINE DIAGNOSTIC, NOT A RESULT: no shrinkage, no controls, no held-out structure.")
    print("Pairs are the same player and team in both windows.")
    rows = []
    for t1 in W1_THRESHOLDS:
        for t2 in W2_THRESHOLDS:
            pr = pairs(table, t1, t2)
            row = {"w1_min": t1, "w2_min": t2}
            for lg in LEAGUES:
                row[lg] = int((pr.league == lg).sum())
            a, b = pr.vaep_per90_w1, pr.vaep_per90_w2
            row |= {
                "all": len(pr),
                "pearson": round(pearsonr(a, b).statistic, 4),
                "spearman": round(spearmanr(a, b).statistic, 4),
            }
            rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


def main(argv):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    step = argv[0]
    if step == "features":
        league = argv[1]
        start = time.perf_counter()
        build_features(league)
        print(f"{league}: features and labels in {time.perf_counter() - start:.1f} s")
        report_features(league)
    elif step == "report":
        for league in LEAGUES:
            report_features(league)
    elif step == "fit":
        print("params passed to xgboost.train:", PARAMS, "| num_boost_round:", ROUNDS)
        for name in argv[1:] or FITS:
            fit(name)
    elif step == "predict":
        for name in FITS:
            for league in LEAGUES if name == "pooled" else [name.removeprefix("lolo_")]:
                write_predictions(name, league)
                print(f"predicted {name} on {league}", flush=True)
    elif step == "evaluate":
        evaluate()
    elif step == "determinism":
        determinism()
    elif step == "tables":
        tables()
    elif step == "sensitivity":
        sensitivity()


if __name__ == "__main__":
    main(sys.argv[1:])
