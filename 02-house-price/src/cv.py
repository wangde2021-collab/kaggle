"""Frozen-fold CV utilities + experiment logging. Metric: RMSLE (lower better).

All models are trained on log1p(SalePrice); fold score =
sqrt(mean((log1p(y_true) - clip(pred_log, 0, None))^2)).
"""
import os, csv, json
import numpy as np
import pandas as pd

import config as C

FOLDS_PATH = os.path.join(C.DATA, "folds", "folds.csv")
EXP_PATH = os.path.join(C.BASE, "logs", "experiments.csv")
EXP_FIELDS = ["exp_id", "date", "feature_set", "model", "params_file", "seeds",
              "cv_scheme", "rmsle_mean", "rmsle_std", "rmsle_se",
              "compare_to", "d_mean", "wins", "submitted", "public_lb", "notes"]


def load_data():
    train = pd.read_csv(os.path.join(C.DATA, "train.csv"))
    test = pd.read_csv(os.path.join(C.DATA, "test.csv"))
    return train, test


def load_folds(n_rows):
    f = pd.read_csv(FOLDS_PATH)
    assert len(f) == n_rows
    return f


def fold_matrix(folds_df):
    """Return array shape (n_reps, n_rows) of fold ids."""
    cols = [c for c in folds_df.columns if c.startswith("rep")]
    return folds_df[cols].values.T


def rmsle_fold(y_true_price, pred_log):
    return float(np.sqrt(np.mean((np.log1p(y_true_price)
                                  - np.clip(pred_log, 0, None)) ** 2)))


def dirty_mask(X_raw, y_price):
    return (X_raw["GrLivArea"].values > C.DIRTY["grlivarea"]) & \
           (y_price < C.DIRTY["price"])


def run_cv(make_est, train, folds, reps=None, collect_test=None):
    """make_est() -> fresh estimator with fit(X_raw_df, y_log)/predict(X_raw_df).
    Returns dict with per-fold scores (list over reps*10), oof (n_reps, n) log
    preds, and optionally test log preds averaged over all folds."""
    y_price = train[C.TARGET].values.astype(float)
    y_log = np.log1p(y_price)
    X = train.drop(columns=[C.TARGET])
    fm = fold_matrix(folds)
    n = len(train)
    n_reps = fm.shape[0] if reps is None else reps
    oof = np.full((n_reps, n), np.nan)
    scores = []
    test_pred_sum = None
    for r in range(n_reps):
        for f in range(C.N_SPLITS):
            va = fm[r] == f
            tr = ~va
            Xtr, Xva = X[tr], X[va]
            ytr = y_log[tr]
            # drop dirty rows from training fold only
            keep = ~dirty_mask(Xtr, y_price[tr])
            est = make_est()
            est.fit(Xtr[keep], ytr[keep])
            pred = np.asarray(est.predict(Xva), dtype=float)
            oof[r, va] = pred
            scores.append(rmsle_fold(y_price[va], pred))
            if collect_test is not None:
                tp = np.asarray(est.predict(collect_test), dtype=float)
                test_pred_sum = tp if test_pred_sum is None else test_pred_sum + tp
    out = {"scores": np.array(scores), "oof": oof,
           "mean": float(np.mean(scores)), "std": float(np.std(scores, ddof=1)),
           "se": float(np.std(scores, ddof=1) / np.sqrt(len(scores)))}
    if collect_test is not None:
        out["test_pred"] = test_pred_sum / len(scores)
    return out


def compare(scores_a, scores_b):
    """Paired per-fold diff d = a - b (RMSLE; negative means a better)."""
    d = np.asarray(scores_a) - np.asarray(scores_b)
    return {"d_mean": float(np.mean(d)), "d_std": float(np.std(d, ddof=1)),
            "d_se": float(np.std(d, ddof=1) / np.sqrt(len(d))),
            "wins_a": int(np.sum(d < 0)), "n": len(d)}


def log_experiment(row):
    exists = os.path.exists(EXP_PATH)
    with open(EXP_PATH, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=EXP_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in EXP_FIELDS})


def save_oof(exp_id, oof):
    np.save(os.path.join(C.BASE, "oof", f"{exp_id}.npy"), oof)


def save_test_pred(exp_id, pred_log):
    np.save(os.path.join(C.BASE, "test_preds", f"{exp_id}.npy"), pred_log)


def load_scores(exp_id):
    return np.load(os.path.join(C.BASE, "oof", f"{exp_id}_scores.npy"))


def save_scores(exp_id, scores):
    np.save(os.path.join(C.BASE, "oof", f"{exp_id}_scores.npy"),
            np.asarray(scores))
