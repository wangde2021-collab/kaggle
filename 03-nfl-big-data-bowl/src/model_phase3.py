# NFL Big Data Bowl 2027 — Phase 3c: main modelling.
#
# Scheme §4-§5: M0a (draft capital) / M0b (NGS + stopwatch) baselines vs
# M1 (draft-adjusted track model, two-part Hurdle for Y1 + ridge for Y2),
# M1a (pure-track increment), M1full (full shadow index), M1b (track over
# M0b), M2 (Shadow Agility mediation via game kinematics), M3 (shallow
# LightGBM). Validation: 10x5 repeated stratified nested CV (all imputation
# and position-group z-scoring inside the fold pipeline), forward temporal
# split (train 2023+2024 -> test 2025, Year-1 targets), LOCO, permutation
# tests, stratified bootstrap CIs, BH-FDR on draft-adjusted single-feature
# tests, adversarial validation (features must not predict draft_year).
#
# Feature groups (Round-1 fix: the old CORE_TRACK mixed four static
# stopwatch columns into the track set, confounding M1-vs-M0b):
#   TRACK_PURE : forty-curve (v_max / tau / a0 / speed maintenance) + the
#                nine mass-free skill-drill metrics (drill-normalized,
#                FOLD-CLEAN -- see DrillFoldClean below)
#   STATIC     : speed_score / flying_30 / start_finish_ratio /
#                lower_body_burst (stopwatch-derived; lower_body_burst is a
#                full-sample position z-score, disclosed minor leak)
#   SHADOW     : shadow_agility_pure (skill-only index; headline) and
#                shadow_agility_full (skill + forty + lower-body; contrast)
#
# Fold-clean drill normalization (Round-1 fix): the skill_*_z columns in
# combine_features.parquet were z-scored within drill_name on ALL 510
# players (transductive leak). Phase 3 re-derives them inside each CV fold
# from data/derived/skill_events.parquet: level-1 per-(player, drill) means
# are fixed, but the drill mean/std used for z-scoring are fit on the
# TRAINING fold only (DrillFoldClean). The precomputed columns are never
# read as model inputs. The shadow index itself is a Phase-2 pre-fit
# representation (calibrated on three_cone / short_shuttle, not on Y1/Y2)
# and is used as-is, disclosed.
#
# Other Round-1 fixes:
# - PositionZScore -> FeaturePrep: imputation uses PER-POSITION medians
#   (was global median); pos_group enters as one-hot FIXED EFFECTS; draft
#   capital (log_pick / is_udfa / draft_year) is NOT position-z-scored.
# - Ridge(alpha=10) / LogisticRegression(C=1) -> RidgeCV / LogisticRegressionCV
#   (regularization chosen inside the training fold).
# - Bootstrap CIs use the OOF predictions AVERAGED over repeats (was first
#   repeat only).
# - oof_predictions.parquet is aligned by nfl_id (was a positional paste of
#   a subset OOF onto 510 rows).
# - Permutation null permutes the TRACK feature rows within (pos_group x
#   draft_year) strata (Freedman-Lane style: baseline association kept,
#   only the track increment is broken). The old global-y permutation
#   inflated the null (smoke: null mean +0.38 for log-ASR r).
# - Temporal extrapolation drops numeric draft_year; LOCO reports M0a too.
# - M2 mediation controls pos_group (one-hot) in both stages.
# - KTR reports the WITHIN-POSITION partial correlation (pos dummies
#   partialled out), not the pooled raw r.
# - FDR single-feature tests: features position-z-scored, draft_year as a
#   categorical (C(draft_year)), HC3 robust standard errors.
# - Adversarial validation: macro AUC computed as the mean of per-class
#   one-vs-rest AUCs (version-proof; no multi_class argument).
#
# Usage:
#   python3 src/model_phase3.py [--parquet data/parquet]
#       [--derived data/derived] [--out logs] [--n-perm 1000] [--quick]
# --quick: 2x5 CV, 100 perms, 500 bootstraps (smoke test).
# Writes logs/phase3_results.json + logs/phase3_results.md,
# data/derived/oof_predictions.parquet; appends logs/experiments.csv.
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV,
                                  Ridge, RidgeCV)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42

SKILL_METRICS = ["decel_p95", "accel_p95", "jerk_p95", "lat_accel_p95",
                 "brake_power_p95", "braking_impulse_max",
                 "cut_retention_mean", "cut_reaccel_mean", "s_peak"]
TRACK_PURE = (["v_max_best", "tau_best", "a0_mean", "speed_maintenance_mean"]
              + [f"skill_{m}_z_mean" for m in SKILL_METRICS])
STATIC = ["speed_score", "flying_30", "start_finish_ratio", "lower_body_burst"]
SHADOW_PURE = "shadow_agility_pure"
SHADOW_FULL = "shadow_agility_full"
M0A_COLS = ["log_pick", "is_udfa", "draft_year"]
M0A_TEMPORAL = ["log_pick", "is_udfa"]  # no numeric draft_year in temporal splits
M0B_COLS = ["ngs_athleticism_score", "ngs_final_score", "forty",
            "ten_yd_split", "vertical", "broad_jump"]


class DrillFoldClean(BaseEstimator, TransformerMixin):
    """Fold-clean drill normalization from per-event skill features.

    `level1` is a DataFrame with one row per (nfl_id, drill_name) holding
    the mean of each SKILL_METRICS column across that player's attempts in
    that drill (fixed, non-leaky aggregation). `fit` computes the drill
    mean/std using ONLY the training fold's players; `transform` z-scores
    each player's per-drill values with those stats and aggregates to the
    player level (mean + top-2 across stable drills), producing the same
    column names as the Phase-2 precomputed table.
    """

    def __init__(self, level1, stable_drills, metrics=SKILL_METRICS):
        self.level1 = level1
        self.stable_drills = stable_drills
        self.metrics = metrics

    def fit(self, X, y=None):
        train_ids = set(X["nfl_id"])
        sub = self.level1[
            self.level1["nfl_id"].isin(train_ids)
            & self.level1["drill_name"].isin(self.stable_drills)]
        self.drill_stats_ = {}
        for d, grp in sub.groupby("drill_name"):
            self.drill_stats_[d] = {
                m: (float(grp[m].mean()),
                    float(grp[m].std()) if np.isfinite(grp[m].std())
                    and grp[m].std() > 0 else 1.0)
                for m in self.metrics}
        return self

    def transform(self, X):
        sub = self.level1[
            self.level1["nfl_id"].isin(set(X["nfl_id"]))
            & self.level1["drill_name"].isin(self.stable_drills)].copy()
        for m in self.metrics:
            mu = sub["drill_name"].map(
                lambda d: self.drill_stats_.get(d, {}).get(m, (np.nan, 1.0))[0])
            sd = sub["drill_name"].map(
                lambda d: self.drill_stats_.get(d, {}).get(m, (np.nan, 1.0))[1])
            sub[m + "_z"] = (sub[m] - mu) / sd
        zcols = [m + "_z" for m in self.metrics]
        agg_mean = sub.groupby("nfl_id")[zcols].mean()
        # top-2 mean per player per metric, vectorized: nlargest(2) per
        # group then mean (the old groupby.apply with a nested apply was
        # the main per-fold cost driver)
        top2 = pd.DataFrame({
            zc: sub.groupby("nfl_id")[zc].nlargest(2).groupby(level=0).mean()
            for zc in zcols})
        out = X.copy()
        for m, zc in zip(self.metrics, zcols):
            out[f"skill_{m}_z_mean"] = out["nfl_id"].map(agg_mean[zc])
            out[f"skill_{m}_z_top2"] = out["nfl_id"].map(top2[zc])
        return out


class FeaturePrep(BaseEstimator, TransformerMixin):
    """Fold-clean feature preparation.

    - track_cols: per-POSITION-median imputation, then z-score within
      position (train-fold stats; unseen positions fall back to global).
    - raw_cols: global-median imputation, passed through unchanged
      (draft capital is NOT position-standardized).
    - pos_group appended as one-hot fixed effects (all levels kept).
    """

    def __init__(self, track_cols=(), raw_cols=()):
        self.track_cols = track_cols
        self.raw_cols = raw_cols

    def fit(self, X, y=None):
        self.pos_levels_ = sorted(X["pos_group"].dropna().unique().tolist())
        self.track_med_ = {}
        self.track_global_ = {}
        self.track_by_pos_ = {}
        for c in self.track_cols:
            med = X.groupby("pos_group")[c].median()
            self.track_med_[c] = med.to_dict()
            self.track_med_[c]["__global__"] = float(X[c].median()) \
                if np.isfinite(X[c].median()) else 0.0
            filled = X[c].fillna(X["pos_group"].map(med)).fillna(
                self.track_med_[c]["__global__"])
            self.track_global_[c] = (float(filled.mean()),
                                     float(filled.std()) or 1.0)
            self.track_by_pos_[c] = {}
            for pos, sub in X.assign(_f=filled).groupby("pos_group"):
                sd = sub["_f"].std()
                self.track_by_pos_[c][pos] = (
                    float(sub["_f"].mean()),
                    float(sd) if np.isfinite(sd) and sd > 0 else 1.0)
        self.raw_med_ = {}
        for c in self.raw_cols:
            med = X[c].median()
            self.raw_med_[c] = float(med) if np.isfinite(med) else 0.0
        return self

    def transform(self, X):
        parts = []
        for c in self.track_cols:
            med = X["pos_group"].map(self.track_med_[c]).fillna(
                self.track_med_[c]["__global__"])
            vals = X[c].fillna(med).astype(float)
            gm, gs = self.track_global_[c]
            z = (vals - gm) / gs
            for pos, (mu, sd) in self.track_by_pos_[c].items():
                m = X["pos_group"] == pos
                z.loc[m] = (vals[m] - mu) / sd
            parts.append(np.nan_to_num(z.to_numpy(dtype=float), nan=0.0,
                                       posinf=0.0, neginf=0.0))
        for c in self.raw_cols:
            parts.append(X[c].fillna(self.raw_med_[c]).astype(float)
                         .to_numpy(dtype=float))
        for pos in self.pos_levels_:
            parts.append((X["pos_group"] == pos).astype(float)
                         .to_numpy(dtype=float))
        return np.column_stack(parts) if parts else np.zeros((len(X), 0))


def make_model_pipe(model, track_cols, raw_cols):
    return Pipeline([
        ("prep", FeaturePrep(track_cols=track_cols, raw_cols=raw_cols)),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def model_factory(stratify):
    """Regularization chosen inside the training fold."""
    if stratify:
        return LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=3,
                                    scoring="roc_auc", max_iter=3000)
    return RidgeCV(alphas=np.logspace(-1, 3, 25))


def _permute_track_within_strata(frame, track_cols, strata, rng):
    """Permute track-column rows within (pos_group, draft_year) strata."""
    out = frame.copy()
    for _, idx in strata.groupby(strata).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        perm = rng.permutation(idx)
        out.loc[idx, track_cols] = frame.loc[perm, track_cols].to_numpy()
    return out


def _materialize_folds(base, y, stratify, repeat, drill_cleaner, seed=SEED):
    """One repeat's 5 folds with the drill transform done ONCE per fold.

    The drill features do not depend on the model or on any permutation, so
    the transformed train/validation frames are materialized once and shared
    across all models (repeated CV) and all permutations (permutation test).
    Re-fitting the drill cleaner per model / per permutation was the main
    cost driver: the old code re-ran it ~6x per fold in the main CV and
    1000x per target in the permutation test.
    Returns a list of (tr_idx, va_idx, Xtr, Xva)."""
    cv = (StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + repeat)
          if stratify else KFold(n_splits=5, shuffle=True, random_state=seed + repeat))
    splits = cv.split(base, y) if stratify else cv.split(base)
    folds = []
    for tr, va in splits:
        Xtr, Xva = base.iloc[tr], base.iloc[va]
        if drill_cleaner is not None:
            Xtr = drill_cleaner.fit(Xtr).transform(Xtr)
            Xva = drill_cleaner.transform(Xva)
        folds.append((tr, va, Xtr, Xva))
    return folds


def _oof_from_folds(folds, y, track_cols, raw_cols, stratify, strata,
                    permute_rng=None, model_fn=None):
    """One 5-fold OOF pass over pre-materialized fold frames.

    If permute_rng is given, the track columns are permuted within strata in
    BOTH the training and validation frames of each fold (Freedman-Lane
    style null for the track increment). The cached frames are never
    mutated: _permute_track_within_strata copies before permuting, and no
    pipeline step mutates its input frame."""
    if model_fn is None:
        model_fn = model_factory
    oof = np.full(len(y), np.nan)
    for tr, va, Xtr, Xva in folds:
        if permute_rng is not None and track_cols:
            Xtr = _permute_track_within_strata(
                Xtr, track_cols, strata.iloc[tr], permute_rng)
            Xva = _permute_track_within_strata(
                Xva, track_cols, strata.iloc[va], permute_rng)
        pipe = make_model_pipe(model_fn(stratify), track_cols, raw_cols)
        pipe.fit(Xtr, y.iloc[tr] if hasattr(y, "iloc") else y[tr])
        oof[va] = (pipe.predict_proba(Xva)[:, 1] if stratify
                   else pipe.predict(Xva))
    return oof


def repeated_cv_oof(base, y, model_specs, n_repeats, stratify, drill_cleaner,
                    seed=SEED):
    """Repeated OOF predictions for several models sharing the same folds.

    model_specs: list of (label, track_cols, raw_cols, model_fn or None).
    Returns {label: (oofs, alphas)} with one OOF array per repeat, plus the
    regularization values the inner CV chose. The drill transform is done
    once per (repeat, fold), not once per model."""
    results = {label: ([], []) for label, _, _, _ in model_specs}
    for r in range(n_repeats):
        folds = _materialize_folds(base, y, stratify, r, drill_cleaner, seed)
        for label, track_cols, raw_cols, model_fn in model_specs:
            if model_fn is None:
                model_fn = model_factory
            oof = np.full(len(y), np.nan)
            alphas = results[label][1]
            for tr, va, Xtr, Xva in folds:
                pipe = make_model_pipe(model_fn(stratify), track_cols, raw_cols)
                pipe.fit(Xtr, y.iloc[tr] if hasattr(y, "iloc") else y[tr])
                mdl = pipe.named_steps["model"]
                if hasattr(mdl, "alpha_"):
                    alphas.append(float(mdl.alpha_))
                elif hasattr(mdl, "C_"):
                    alphas.append(float(np.atleast_1d(mdl.C_)[0]))
                oof[va] = (pipe.predict_proba(Xva)[:, 1] if stratify
                           else pipe.predict(Xva))
            results[label][0].append(oof)
    return results


def metric_r(y, pred):
    ok = np.isfinite(y) & np.isfinite(pred)
    return float(np.corrcoef(y[ok], pred[ok])[0, 1]) if ok.sum() > 10 else np.nan


def metric_r2(y, pred):
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() <= 10:
        return np.nan
    ss_res = np.sum((y[ok] - pred[ok]) ** 2)
    ss_tot = np.sum((y[ok] - np.mean(y[ok])) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def metric_auc(y, pred):
    ok = np.isfinite(y) & np.isfinite(pred)
    return float(roc_auc_score(y[ok], pred[ok])) if ok.sum() > 10 else np.nan


def bootstrap_ci(metric_fn, y, pred_a, pred_b, n_boot, rng, strata=None):
    """Percentile CI of (metric(pred_b) - metric(pred_a)) by stratified resample."""
    y = np.asarray(y, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[ok], a[ok], b[ok]
    if strata is not None:
        strata = np.asarray(strata)[ok]
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        if strata is not None:
            idx = np.concatenate([
                rng.choice(np.where(strata == s)[0],
                           size=int((strata == s).sum()), replace=True)
                for s in np.unique(strata)])
        else:
            idx = rng.integers(0, n, size=n)
        try:
            diffs.append(metric_fn(y[idx], b[idx]) - metric_fn(y[idx], a[idx]))
        except Exception:
            continue
    diffs = np.array(diffs)
    return (float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5))) if len(diffs) else (np.nan, np.nan)


def permutation_p(metric_fn, base, y, track_cols, raw_cols, base_track_cols,
                  base_raw_cols, n_perm, stratify, drill_cleaner,
                  fixed_param=None, seed=SEED):
    """Permutation p for the incremental metric of track_cols over baseline.

    Null: track feature rows are permuted within (pos_group x draft_year)
    strata in both train and validation frames, so the baseline association
    is preserved and only the track increment is broken (Freedman-Lane
    style). The fits use a FIXED regularization (the median alpha/C the
    main repeated CV chose, passed as fixed_param) -- re-running the inner
    CV inside 1000 permutations would be prohibitively slow and adds
    nothing to the null.
    """
    rng = np.random.RandomState(seed)

    def fixed_factory(strat):
        if fixed_param is None:
            return model_factory(strat)
        if strat:
            return LogisticRegression(C=float(fixed_param), max_iter=3000)
        return Ridge(alpha=float(fixed_param))

    strata = base["pos_group"].astype(str) + "_" + base["draft_year"].astype(str)
    folds = _materialize_folds(base, y, stratify, 0, drill_cleaner, seed)
    yv = y.to_numpy(dtype=float)
    oof_m1 = _oof_from_folds(folds, y, track_cols, raw_cols, stratify, strata,
                             model_fn=fixed_factory)
    oof_m0 = _oof_from_folds(folds, y, base_track_cols, base_raw_cols,
                             stratify, strata, model_fn=fixed_factory)
    obs = metric_fn(yv, oof_m1) - metric_fn(yv, oof_m0)
    nulls = []
    for _ in range(n_perm):
        oof_p = _oof_from_folds(folds, y, track_cols, raw_cols, stratify,
                                strata, permute_rng=rng,
                                model_fn=fixed_factory)
        nulls.append(metric_fn(yv, oof_p) - metric_fn(yv, oof_m0))
    nulls = np.array(nulls)
    p = float((np.sum(nulls >= obs) + 1) / (len(nulls) + 1))
    return obs, p, float(np.mean(nulls)), float(np.std(nulls))


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def single_feature_tests(df, y_col, feature_cols, mask):
    """Draft-adjusted OLS: y ~ log_pick + is_udfa + C(draft_year) +
    pos_group dummies + one position-z-scored feature; HC3 errors; BH-FDR.

    The feature is position-z-scored (within-position signal only); the
    draft-year effect is categorical, not a numeric trend. NOTE: these are
    descriptive full-sample screening tests on the Phase-2 features (the
    precomputed skill columns); they are not cross-validated estimates and
    do not replace the fold-clean CV results above.
    """
    sub = df[mask].copy()
    y = sub[y_col].astype(float)
    rows = []
    for f in feature_cols:
        ok = sub[[f, "log_pick", "is_udfa", "draft_year", "pos_group"]].notna().all(axis=1) \
            & y.notna()
        if ok.sum() < 30:
            rows.append({"feature": f, "n": int(ok.sum()), "beta": np.nan,
                         "p": np.nan})
            continue
        s2 = sub.loc[ok].copy()
        g = s2.groupby("pos_group")[f]
        s2[f + "_pz"] = (s2[f] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        Xd = pd.get_dummies(s2[["log_pick", "is_udfa", "draft_year",
                                "pos_group"]],
                            columns=["draft_year", "pos_group"],
                            drop_first=True, dtype=float)
        Xd[f + "_pz"] = s2[f + "_pz"]
        Xd = sm.add_constant(Xd.astype(float))
        try:
            res = sm.OLS(y[ok].astype(float), Xd).fit(cov_type="HC3")
            rows.append({"feature": f, "n": int(ok.sum()),
                         "beta": float(res.params[f + "_pz"]),
                         "p": float(res.pvalues[f + "_pz"])})
        except Exception:
            rows.append({"feature": f, "n": int(ok.sum()), "beta": np.nan,
                         "p": np.nan})
    ps = [r["p"] for r in rows]
    qs = bh_fdr([p if np.isfinite(p) else 1.0 for p in ps])
    for r, q in zip(rows, qs):
        r["q_fdr"] = float(q)
    return rows


def _partial_corr(x, y, controls):
    """Partial correlation of x and y after OLS-residualizing both on controls."""
    ok = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    if ok.sum() < 20:
        return np.nan, np.nan
    Xc = sm.add_constant(controls[ok])
    rx = sm.OLS(x[ok], Xc).fit().resid
    ry = sm.OLS(y[ok], Xc).fit().resid
    rho, p = stats.pearsonr(rx, ry)
    return float(rho), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.path.join(BASE, "data", "parquet"))
    ap.add_argument("--derived", default=os.path.join(BASE, "data", "derived"))
    ap.add_argument("--out", default=os.path.join(BASE, "logs"))
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.quick:
        n_rep, n_perm, n_boot = 2, 100, 500
    else:
        n_rep, n_perm, n_boot = 10, args.n_perm, args.n_boot
    rng = np.random.RandomState(SEED)

    feats = pd.read_parquet(os.path.join(args.derived, "combine_features.parquet"))
    tg = pd.read_parquet(os.path.join(args.derived, "targets.parquet"))
    players = pd.read_parquet(os.path.join(args.parquet, "players.parquet"),
                              columns=["nfl_id", "draft_year", "draft_overall_pick"])
    cr = pd.read_parquet(os.path.join(args.parquet, "combine_results.parquet"))
    df = feats.merge(tg, on="nfl_id", how="left", suffixes=("", "_tgt"))
    df = df.merge(players, on="nfl_id", how="left", suffixes=("", "_pl"))
    df = df.merge(cr[["nfl_id", "ngs_athleticism_score", "ngs_final_score",
                      "forty", "ten_yd_split", "vertical", "broad_jump"]],
                  on="nfl_id", how="left")
    df["log_pick"] = np.log(df["draft_overall_pick"].fillna(300.0))
    df["is_udfa"] = df["draft_overall_pick"].isna().astype(int)
    df["pos_group"] = df["combine_position"]
    df["draft_year"] = df["draft_year_pl"] if "draft_year_pl" in df else df["draft_year"]
    # game kinematics for KTR / mediation (optional)
    gk_path = os.path.join(args.derived, "game_kinematics.parquet")
    gk = pd.read_parquet(gk_path) if os.path.exists(gk_path) else None
    if gk is not None:
        df = df.merge(gk, on="nfl_id", how="left")
        df["ktr_speed"] = df["game_s_p99"] / df["v_max_best"]

    # ---- fold-clean drill features: level-1 per-(player, drill) means ----
    drill_cleaner = None
    drill_note = "skill_events.parquet missing; using precomputed skill columns (leaky)"
    sk_path = os.path.join(args.derived, "skill_events.parquet")
    if os.path.exists(sk_path):
        sk = pd.read_parquet(sk_path)
        level1 = sk.groupby(["nfl_id", "drill_name"])[SKILL_METRICS].mean().reset_index()
        dy = players.set_index("nfl_id")["draft_year"]
        level1["draft_year"] = level1["nfl_id"].map(dy)
        cov = level1.groupby(["drill_name", "draft_year"])["nfl_id"].nunique().unstack(fill_value=0)
        stable = [d for d in cov.index
                  if all(cov.loc[d].get(y, 0) >= 20 for y in (2023, 2024, 2025))]
        drill_cleaner = DrillFoldClean(level1, stable)
        drill_note = (f"fold-clean drill normalization from skill_events.parquet "
                      f"({len(stable)} stable drills, {len(level1)} player-drill rows)")
    print("[drill] " + drill_note)

    # model specifications: (label, track_cols, raw_cols)
    MODELS = [
        ("M0a", [], M0A_COLS),
        ("M0b", M0B_COLS, []),
        ("M1a", TRACK_PURE + [SHADOW_PURE], M0A_COLS),
        ("M1", TRACK_PURE + STATIC + [SHADOW_PURE], M0A_COLS),
        ("M1full", TRACK_PURE + STATIC + [SHADOW_FULL], M0A_COLS),
        ("M1b", M0B_COLS + TRACK_PURE + [SHADOW_PURE], []),
    ]
    res = {"seed": SEED, "n_repeats": n_rep, "n_perm": n_perm, "n_boot": n_boot,
           "drill_normalization": drill_note}
    oof_store = {}  # (name, label) -> (nfl_id array, averaged OOF array)

    def run_target(y_col, mask, metric_fn, name, stratify):
        sub = df[mask].copy().reset_index(drop=True)
        y = sub[y_col].astype(float)
        out_d = {}
        specs = [(label, track_cols, raw_cols, None)
                 for label, track_cols, raw_cols in MODELS]
        cv_results = repeated_cv_oof(sub, y, specs, n_rep, stratify,
                                     drill_cleaner)
        for label, _, _, _ in specs:
            oofs, alphas = cv_results[label]
            ms = [metric_fn(y.to_numpy(), o) for o in oofs]
            oof_avg = np.nanmean(np.vstack(oofs), axis=0)
            out_d[label] = {"metric_mean": float(np.mean(ms)),
                            "metric_std": float(np.std(ms)),
                            "alpha_median": float(np.median(alphas))
                            if alphas else np.nan}
            oof_store[(name, label)] = (sub["nfl_id"].to_numpy(), oof_avg)
        # deltas + bootstrap CI on the AVERAGED OOF (Round-1 fix)
        for base, comp in (("M0a", "M1"), ("M0b", "M1b"), ("M0a", "M1a")):
            d = out_d[comp]["metric_mean"] - out_d[base]["metric_mean"]
            lo, hi = bootstrap_ci(metric_fn, y.to_numpy(),
                                  oof_store[(name, base)][1],
                                  oof_store[(name, comp)][1], n_boot, rng,
                                  strata=sub["pos_group"].to_numpy())
            out_d[f"delta_{comp}_vs_{base}"] = {"mean": float(d), "ci95": [lo, hi]}
        # permutation test on the incremental metric (M1 vs M0a), using the
        # median regularization the main CV chose for M1
        _fp = out_d["M1"].get("alpha_median")
        _fp = _fp if (_fp is not None and np.isfinite(_fp)) else None
        obs, p, null_mu, null_sd = permutation_p(
            metric_fn, sub, y, TRACK_PURE + STATIC + [SHADOW_PURE], M0A_COLS,
            [], M0A_COLS, n_perm, stratify, drill_cleaner,
            fixed_param=_fp)
        out_d["permutation"] = {"obs_delta": float(obs), "p_perm": float(p),
                                "null_mean": null_mu, "null_sd": null_sd}
        res[name] = out_d
        print(f"[{name}] " + json.dumps(out_d, default=str))
        return sub

    # ---- Y1 hurdle stage 1: logistic on ASR>=150
    run_target("hurdle_150", df["hurdle_150"].notna(),
               metric_auc, "Y1_hurdle_stage1", stratify=True)
    # ---- Y1 stage 2: ridge on log(ASR) among hurdle passers
    df["log_asr"] = np.log(df["ASR"].replace(0, np.nan))
    run_target("log_asr", df["hurdle_150"] == 1,
               metric_r, "Y1_stage2_logASR", stratify=False)
    # ---- Y2: ridge on EB-shrunk quality (coverage >= 75)
    run_target("z_qual", df["z_qual"].notna(),
               metric_r, "Y2_quality", stratify=False)
    # Y2 R2 as well (from averaged OOF)
    sub2_ids, sub2_oof_m1 = oof_store[("Y2_quality", "M1")]
    sub2 = df[df["nfl_id"].isin(set(sub2_ids))].copy()
    sub2 = sub2.set_index("nfl_id").loc[sub2_ids].reset_index()
    res["Y2_quality"]["r2_M1"] = metric_r2(
        sub2["z_qual"].to_numpy(), sub2_oof_m1)
    res["Y2_quality"]["r2_M0a"] = metric_r2(
        sub2["z_qual"].to_numpy(), oof_store[("Y2_quality", "M0a")][1])
    # per-position Spearman for Y2 M1 (averaged OOF)
    per_pos = {}
    for pos, grp in sub2.groupby("pos_group"):
        yy = grp["z_qual"].to_numpy()
        pp = sub2_oof_m1[sub2["pos_group"].to_numpy() == pos]
        if len(yy) >= 10 and np.isfinite(pp).all():
            rho, p = stats.spearmanr(yy, pp)
            per_pos[pos] = {"n": len(yy), "spearman_rho": float(rho),
                            "p": float(p)}
    res["Y2_quality"]["per_position_spearman"] = per_pos
    # Y2 coverage sensitivity: r at reg_snaps 50 / 150 (single 2-repeat ridge)
    for thr in (50, 150):
        m = (df["reg_snaps"] >= thr) & df["z_qual"].notna()
        s2 = df[m].copy().reset_index(drop=True)
        yv = s2["z_qual"].astype(float)
        oofs, _ = repeated_cv_oof(
            s2, yv, [("M1", TRACK_PURE + STATIC + [SHADOW_PURE], M0A_COLS, None)],
            2, False, drill_cleaner)["M1"]
        res.setdefault("Y2_sensitivity", {})[f"reg_snaps_{thr}"] = {
            "n": int(m.sum()),
            "oof_r_mean": float(np.mean([metric_r(yv.to_numpy(), o)
                                         for o in oofs]))}

    # ---- Year-1 parallel targets (temporal alignment)
    df["log_year1"] = np.log1p(df["year1_reg_snaps"])
    run_target("year1_hurdle_100", df["year1_hurdle_100"].notna(),
               metric_auc, "Y1_year1_hurdle", stratify=True)
    run_target("log_year1", df["year1_reg_snaps"] > 0,
               metric_r, "Y1_year1_logsnaps", stratify=False)

    # ---- Temporal extrapolation: train 2023+2024 -> test 2025 (Year-1).
    #      Numeric draft_year is DROPPED here (Round-1 fix).
    temp = {}
    tr = df[df["draft_year"].isin([2023, 2024])].reset_index(drop=True)
    te = df[df["draft_year"] == 2025].reset_index(drop=True)

    def _fit_predict(tr_f, te_f, ycol, track_cols, raw_cols, stratify):
        Xtr, Xva = tr_f, te_f
        if drill_cleaner is not None:
            Xtr = drill_cleaner.fit(Xtr).transform(Xtr)
            Xva = drill_cleaner.transform(Xva)
        pipe = make_model_pipe(model_factory(stratify), track_cols, raw_cols)
        pipe.fit(Xtr, tr_f[ycol].astype(float))
        return (pipe.predict_proba(Xva)[:, 1] if stratify
                else pipe.predict(Xva))

    for label, ycol, mfn, strat in [
            ("Y1_year1_hurdle", "year1_hurdle_100", metric_auc, True),
            ("Y1_year1_logsnaps", "log_year1", metric_r, False)]:
        for mlabel, track_cols, raw_cols in [
                ("M0a", [], M0A_TEMPORAL),
                ("M1", TRACK_PURE + STATIC + [SHADOW_PURE], M0A_TEMPORAL)]:
            pred = _fit_predict(tr, te, ycol, track_cols, raw_cols, strat)
            temp[f"{label}_{mlabel}"] = {
                "n_train": len(tr), "n_test": len(te),
                "metric": float(mfn(te[ycol].astype(float).to_numpy(), pred))}
    # LOCO: leave-one-cohort-out, Y2 r, M0a AND M1 (Round-1 fix)
    loco = {}
    for yr in (2023, 2024, 2025):
        tr = df[df["draft_year"] != yr].reset_index(drop=True)
        te = df[df["draft_year"] == yr].reset_index(drop=True)
        sub = tr[tr["z_qual"].notna()]
        ste = te[te["z_qual"].notna()]
        if len(ste) < 10:
            continue
        for mlabel, track_cols, raw_cols in [
                ("M0a", [], M0A_TEMPORAL),
                ("M1", TRACK_PURE + STATIC + [SHADOW_PURE], M0A_TEMPORAL)]:
            pred = _fit_predict(sub, ste, "z_qual", track_cols, raw_cols, False)
            loco[f"Y2_test_{yr}_{mlabel}"] = {
                "n_test": len(ste),
                "r": metric_r(ste["z_qual"].to_numpy(), pred)}
    temp["LOCO_Y2"] = loco
    res["temporal"] = temp
    print("[temporal] " + json.dumps(temp, default=str))

    # ---- Adversarial validation: can X_track predict draft_year?
    #      Macro AUC = mean of per-class one-vs-rest AUCs.
    adv = {}
    X = df[TRACK_PURE + STATIC + [SHADOW_PURE] + ["pos_group", "nfl_id",
                                                  "draft_year"]].copy()
    yv = df["draft_year"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    for tr_idx, va_idx in cv.split(X, yv):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        if drill_cleaner is not None:
            Xtr = drill_cleaner.fit(Xtr).transform(Xtr)
            Xva = drill_cleaner.transform(Xva)
        pipe = make_model_pipe(
            LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=3,
                                 scoring="roc_auc_ovr", max_iter=3000),
            TRACK_PURE + STATIC + [SHADOW_PURE], [])
        pipe.fit(Xtr, yv.iloc[tr_idx])
        proba = pipe.predict_proba(Xva)
        classes = list(pipe.named_steps["model"].classes_)
        per_class = []
        for j, cls in enumerate(classes):
            yy = (yv.iloc[va_idx] == cls).astype(int)
            if yy.nunique() < 2:
                continue
            per_class.append(roc_auc_score(yy, proba[:, j]))
        aucs.append(float(np.mean(per_class)) if per_class else np.nan)
    adv["draft_year_macro_auc"] = {"mean": float(np.nanmean(aucs)),
                                   "std": float(np.nanstd(aucs))}
    res["adversarial_validation"] = adv
    print("[adversarial] " + json.dumps(adv))

    # ---- BH-FDR draft-adjusted single-feature tests
    fdr_feats = TRACK_PURE + STATIC + [SHADOW_PURE, SHADOW_FULL]
    res["fdr_Y2"] = single_feature_tests(
        df, "z_qual", fdr_feats, df["z_qual"].notna())
    res["fdr_Y1_stage2"] = single_feature_tests(
        df, "log_asr", fdr_feats, df["hurdle_150"] == 1)
    print("[fdr_Y2] " + json.dumps(res["fdr_Y2"], default=str))

    # ---- M3 shallow LightGBM comparison (same CV folds, own model factory)
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        def lgbm_factory(stratify):
            if stratify:
                return LGBMClassifier(max_depth=2, min_child_samples=25,
                                      n_estimators=200, learning_rate=0.05,
                                      verbose=-1)
            return LGBMRegressor(max_depth=2, min_child_samples=25,
                                 n_estimators=200, learning_rate=0.05,
                                 verbose=-1)

        m3 = {}
        sub = df[df["z_qual"].notna()].reset_index(drop=True)
        yv = sub["z_qual"].astype(float)
        oofs, _ = repeated_cv_oof(
            sub, yv, [("M1", TRACK_PURE + STATIC + [SHADOW_PURE], M0A_COLS,
                       lgbm_factory)],
            n_rep, False, drill_cleaner)["M1"]
        m3["Y2_r_mean"] = float(np.mean([metric_r(yv.to_numpy(), o)
                                         for o in oofs]))
        sub = df.reset_index(drop=True)
        yv = sub["hurdle_150"].astype(int)
        oofs, _ = repeated_cv_oof(
            sub, yv, [("M1", TRACK_PURE + STATIC + [SHADOW_PURE], M0A_COLS,
                       lgbm_factory)],
            n_rep, True, drill_cleaner)["M1"]
        m3["Y1_hurdle_auc_mean"] = float(np.mean([metric_auc(yv.to_numpy(), o)
                                                  for o in oofs]))
        res["M3_lightgbm"] = m3
        print("[M3] " + json.dumps(m3))
    except Exception as e:  # noqa: BLE001
        res["M3_lightgbm"] = {"error": str(e)}
        print("[M3] failed:", e)

    # ---- M2 mediation: shadow_agility -> game kinematics -> Y2
    #      Both stages control pos_group (one-hot) + draft capital.
    med = {}
    if gk is not None:
        mdf = df[df["z_qual"].notna()
                 & df["game_decel_p95"].notna()].copy()
        mdf = mdf.dropna(subset=["shadow_agility_pure", "log_pick", "is_udfa",
                                 "game_decel_p95", "z_qual", "pos_group"])
        if len(mdf) >= 50:
            pos_d = pd.get_dummies(mdf["pos_group"], prefix="pos",
                                   drop_first=True, dtype=float)
            Xa = pd.concat([mdf[["shadow_agility_pure", "log_pick", "is_udfa"]]
                            .astype(float).reset_index(drop=True),
                            pos_d.reset_index(drop=True)], axis=1)
            Xm = pd.concat([mdf[["shadow_agility_pure", "game_decel_p95",
                                 "log_pick", "is_udfa"]].astype(float)
                            .reset_index(drop=True),
                            pos_d.reset_index(drop=True)], axis=1)
            a_res = sm.OLS(mdf["game_decel_p95"].astype(float).reset_index(drop=True),
                           sm.add_constant(Xa)).fit(cov_type="HC3")
            b_res = sm.OLS(mdf["z_qual"].astype(float).reset_index(drop=True),
                           sm.add_constant(Xm)).fit(cov_type="HC3")
            a, b = float(a_res.params["shadow_agility_pure"]), float(
                b_res.params["game_decel_p95"])
            indirect = a * b
            boots = []
            for _ in range(1000):
                s = mdf.sample(len(mdf), replace=True, random_state=rng)
                try:
                    sd = pd.get_dummies(s["pos_group"], prefix="pos",
                                        drop_first=True, dtype=float)
                    sa = pd.concat([s[["shadow_agility_pure", "log_pick",
                                       "is_udfa"]].astype(float)
                                    .reset_index(drop=True),
                                    sd.reset_index(drop=True)], axis=1)
                    sm_ = pd.concat([s[["shadow_agility_pure",
                                         "game_decel_p95", "log_pick",
                                         "is_udfa"]].astype(float)
                                     .reset_index(drop=True),
                                     sd.reset_index(drop=True)], axis=1)
                    ar = sm.OLS(s["game_decel_p95"].astype(float)
                                .reset_index(drop=True),
                                sm.add_constant(sa)).fit()
                    br = sm.OLS(s["z_qual"].astype(float).reset_index(drop=True),
                                sm.add_constant(sm_)).fit()
                    boots.append(float(ar.params["shadow_agility_pure"])
                                 * float(br.params["game_decel_p95"]))
                except Exception:
                    continue
            boots = np.array(boots)
            direct = float(b_res.params["shadow_agility_pure"])
            med = {"n": int(len(mdf)), "a_path": a, "b_path": b,
                   "indirect": float(indirect),
                   "indirect_ci95": [float(np.percentile(boots, 2.5)),
                                     float(np.percentile(boots, 97.5))],
                   "direct_c": direct,
                   "pct_mediated": float(indirect / (indirect + direct))
                   if (indirect + direct) != 0 else np.nan}
            # KTR: within-position partial correlation (Round-1 fix)
            pos_dm = pd.get_dummies(df["pos_group"], drop_first=True,
                                    dtype=float).to_numpy(dtype=float)
            rho_p, p_p = _partial_corr(df["v_max_best"].to_numpy(dtype=float),
                                       df["game_s_p99"].to_numpy(dtype=float),
                                       pos_dm)
            med["ktr_speed_corr_raw"] = metric_r(
                df["v_max_best"].to_numpy(), df["game_s_p99"].to_numpy())
            med["ktr_speed_partial_r"] = rho_p
            med["ktr_speed_partial_p"] = p_p
            med["ktr_speed_mean"] = float(np.nanmean(df["ktr_speed"]))
    res["M2_mediation"] = med
    print("[M2] " + json.dumps(med, default=str))

    # ---- save outputs
    clean = json.loads(json.dumps(res, default=str))
    with open(os.path.join(args.out, "phase3_results.json"), "w") as fh:
        json.dump(clean, fh, indent=2, default=str)
    # OOF predictions aligned by nfl_id (Round-1 fix: no positional paste)
    oof_df = pd.DataFrame({"nfl_id": df["nfl_id"]})
    for (name, label), (ids, oof) in oof_store.items():
        s = pd.Series(oof, index=pd.Index(ids, name="nfl_id"))
        oof_df = oof_df.merge(s.rename(f"{name}_{label}").reset_index(),
                              on="nfl_id", how="left")
    oof_df.to_parquet(os.path.join(args.derived, "oof_predictions.parquet"),
                      engine="pyarrow", compression="zstd", index=False)

    lines = ["# NFL Big Data Bowl 2027 — Phase 3 主建模结果",
             "",
             f"种子 {SEED}；{n_rep}×5 折重复分层 CV；置换 {n_perm} 次；"
             f"自助 {n_boot} 次。",
             "",
             "特征分组：TRACK_PURE（四十曲线 + 9 项无质量技能指标，折内干净 drill 归一化）"
             "；STATIC（静态秒表派生）；SHADOW（纯技能影子指数）。",
             f"Drill 归一化：{drill_note}"]
    for name in ("Y1_hurdle_stage1", "Y1_stage2_logASR", "Y2_quality",
                 "Y1_year1_hurdle", "Y1_year1_logsnaps"):
        d = res.get(name, {})
        lines.append(f"\n## {name}")
        for k in ("M0a", "M0b", "M1a", "M1", "M1full", "M1b"):
            if k in d:
                lines.append(f"- {k}：{d[k]['metric_mean']:.4f} ± {d[k]['metric_std']:.4f}"
                             f"（内层 CV 选定 alpha/C 中位数 {d[k].get('alpha_median', float('nan')):.3g}）")
        for k in ("delta_M1_vs_M0a", "delta_M1b_vs_M0b", "delta_M1a_vs_M0a"):
            if k in d:
                lines.append(f"- {k}：{d[k]['mean']:+.4f}，95% CI "
                             f"[{d[k]['ci95'][0]:+.4f}, {d[k]['ci95'][1]:+.4f}]")
        if "permutation" in d:
            p = d["permutation"]
            lines.append(f"- 置换检验（折内 X 置换）：观测 Δ={p['obs_delta']:+.4f}，"
                         f"p_perm={p['p_perm']:.4f}（零分布均值 "
                         f"{p['null_mean']:+.4f} ± {p['null_sd']:.4f}）")
    lines.append("\n## 时序外推与 LOCO")
    lines.append(json.dumps(res.get("temporal", {}), indent=2, ensure_ascii=False))
    lines.append("\n## 对抗验证（特征预测 draft_year，应 ≤0.55）")
    lines.append(json.dumps(res.get("adversarial_validation", {}), indent=2))
    lines.append("\n## BH-FDR 单特征检验（Y2，顺位校正后，HC3）")
    for r in sorted(res.get("fdr_Y2", []), key=lambda r: (r["p"] if np.isfinite(r["p"]) else 9)):
        lines.append(f"- {r['feature']}：β={r['beta']:+.3f}，p={r['p']:.3g}，"
                     f"q={r['q_fdr']:.3f}，n={r['n']}")
    lines.append("\n## M3 LightGBM / M2 中介 / Y2 敏感性")
    lines.append(json.dumps({k: res.get(k, {}) for k in
                             ("M3_lightgbm", "M2_mediation", "Y2_sensitivity")},
                            indent=2, ensure_ascii=False, default=str))
    with open(os.path.join(args.out, "phase3_results.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", os.path.join(args.out, "phase3_results.json"))
    print("wrote", os.path.join(args.out, "phase3_results.md"))

    # ---- append to experiments.csv
    exp_path = os.path.join(args.out, "experiments.csv")
    rows = []
    for name in ("Y1_hurdle_stage1", "Y1_stage2_logASR", "Y2_quality"):
        d = res.get(name, {})
        if "M1" not in d:
            continue
        rows.append({
            "exp_id": f"phase3_{name}", "date": "2026-11-21", "stage": "phase3",
            "feature_set": "M0a+TRACK_PURE+STATIC+shadow_pure", "model_or_method": "M1",
            "params_file": "src/model_phase3.py", "n_seeds": n_rep,
            "cv_scheme": f"{n_rep}x5 repeated stratified",
            "metric": "AUC" if "hurdle" in name else "r",
            "metric_mean": round(d["M1"]["metric_mean"], 4),
            "metric_std": round(d["M1"]["metric_std"], 4),
            "metric_se": round(d["M1"]["metric_std"] / np.sqrt(n_rep), 4),
            "comparison": "M0a",
            "diff_mean": round(d.get("delta_M1_vs_M0a", {}).get("mean", np.nan), 4),
            "wins": "",
            "notes": f"perm p={d.get('permutation', {}).get('p_perm', np.nan):.4f}",
        })
    if rows:
        new_file = not os.path.exists(exp_path)
        with open(exp_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if new_file:
                w.writeheader()
            w.writerows(rows)
        print(f"appended {len(rows)} rows to {exp_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
