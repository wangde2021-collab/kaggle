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
# Round-2 fixes (Gemini review 2026-11-21, verified against code first):
# - fit_sprint_curve gains a start offset t0 and a physical cap
#   v_max <= s_peak * 1.12; the measured forty_s_peak is now a feature
#   (the old no-offset fit pushed WR v_max_best to 12.20-12.27 yd/s).
# - M0b is split into M0b_stopwatch (forty / ten_yd_split / vertical /
#   broad_jump) and M0c_ngs (NGS scores); M1_over_stopwatch and
#   M1_over_all added (M0all = M0a + stopwatch + NGS combined baseline).
# - M1-family models are two-stage residual / offset stacked fits: the
#   baseline keeps its own regularization, track features fit the residual
#   (regression) or a logit offset (classification), so the baseline is
#   never strangled by the track penalty.
# - permutation_p: baseline and full model each use their OWN median
#   regularization, and 3 repeats are aggregated (was: M1's alpha for
#   both, single repeat=0); the permutation now tests the stacked model.
# - Adversarial validation reports FORTY_ONLY / SKILL_ONLY / STATIC_ONLY
#   block AUCs plus forty_s_peak by draft_year; --within-year enables
#   within-cohort harmonization as a fallback (default off).
# - FDR main family = 9 pure skill features + 5 forty core params
#   (full 19-feature table kept as fdr_Y2_full); draft-pricing regression
#   (-log_pick, Fig 4 data) and the skill_s_peak robustness regression
#   (controls log_pick + forty + ngs_final_score) added.
# - KTR: full 5x4 partial-correlation matrix with n + per-position
#   diagonals; ktr_speed_n recorded. M2 retested with game_snap_burst_1s.
# - Temporal extrapolation: Y1_year1_logsnaps now uses the same
#   year1_reg_snaps > 0 mask as the main CV.
#
# Round-3 fixes (Gemini review #3, 2026-11-22; both bugs VERIFIED against
# code + direct execution before fixing -- see notes/gemini_round3_20261122.md):
# - DrillFoldClean silent-zero bug: fit() used groupby(["drill_name"]),
#   which yields 1-tuple keys in pandas 2.1.4, so transform() lookups
#   missed 100% and all 9 skill features came out all-NaN (then imputed
#   to 0.0); transform() also indexed the per-metric dict by [0]. Fixed:
#   bare-string groupby, .get(k, {}).get(m, ...), plus a hard assert that
#   skill_s_peak_z_mean is <10% NaN after transform.
# - M1_over_stopwatch smuggled draft capital (M0A_COLS) into its baseline
#   while being compared to M0b_stopwatch: the reported +0.1040 was 100%
#   draft pick. Baseline is now M0B_STOPWATCH_COLS only; new M1_track_only
#   (TRACK_PURE only) is the clean head-to-head vs M0b_stopwatch; M0all /
#   M1_over_all now route stopwatch + NGS columns through within-position
#   standardization (only M0A_COLS stay raw).
# - --within-year now ON by decision: unseen (pos, year) groups in LOCO /
#   2025 temporal fall back to the TEST cohort's own unlabeled X
#   distribution (zero target leakage; whole class measured in March,
#   draft in April). Same rule inside DrillFoldClean.
# - permutation_p scale alignment: each null sample is now the mean over
#   the same 3 repeats as obs (was: single-repeat nulls vs 3-repeat-mean
#   obs). Classification stacking offset is now 5-fold cross-fitted
#   inside the training fold (fixed-C offset pipe, to keep the
#   permutation cost unchanged).
# - lower_body_burst is re-derived fold-cleanly inside FeaturePrep from
#   raw vertical / broad_jump (was a full-sample position z-score).
# - New light tables: 5-step skill_s_peak robustness ladder; draft_pricing
#   + fdr_Y2_full gain -forty / vertical / broad_jump; per-position
#   game_s_p99 / forty_s_peak speed-retention table; DL EDGE vs IDL split
#   of the KTR diagonals + Y2 correlation.
#
# Usage:
#   python3 src/model_phase3.py [--parquet data/parquet]
#       [--derived data/derived] [--out logs] [--n-perm 1000] [--quick]
# --quick: 2x5 CV, 100 perms, 500 bootstraps (smoke test).
# Writes logs/phase3_results.json + logs/phase3_results.md,
# data/derived/oof_predictions.parquet; appends logs/experiments.csv.
import argparse
import csv
import datetime
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
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
# Round-2 fallback switch (set from --within-year in main): harmonize drill
# and forty features within (drill_name, draft_year) / (pos_group,
# draft_year) if the adversarial block AUC stays > 0.56 after the t0 fix.
BY_YEAR = False

SKILL_METRICS = ["decel_p95", "accel_p95", "jerk_p95", "lat_accel_p95",
                 "brake_power_p95", "braking_impulse_max",
                 "cut_retention_mean", "cut_reaccel_mean", "s_peak"]
FORTY_COLS = ["v_max_best", "tau_best", "a0_mean", "speed_maintenance_mean",
              "forty_s_peak"]
TRACK_PURE = (FORTY_COLS
              + [f"skill_{m}_z_mean" for m in SKILL_METRICS])
STATIC = ["speed_score", "flying_30", "start_finish_ratio", "lower_body_burst"]
SHADOW_PURE = "shadow_agility_pure"
SHADOW_FULL = "shadow_agility_full"
M0A_COLS = ["log_pick", "is_udfa", "draft_year"]
M0A_TEMPORAL = ["log_pick", "is_udfa"]  # no numeric draft_year in temporal splits
M0B_COLS = ["ngs_athleticism_score", "ngs_final_score", "forty",
            "ten_yd_split", "vertical", "broad_jump"]
# Round-2 fix: M0b is SPLIT -- the old M0b = 0.1959 was almost surely the
# NGS score (university production / film), not the stopwatch.
M0B_STOPWATCH_COLS = ["forty", "ten_yd_split", "vertical", "broad_jump"]
M0C_NGS_COLS = ["ngs_athleticism_score", "ngs_final_score"]
# FDR main family (Round-2 fix): 9 pure skill features + 5 corrected forty
# core params; the 2 redundant Shadow indices and 4 static columns move to
# the full appendix table only.
FDR_MAIN = ([f"skill_{m}_z_mean" for m in SKILL_METRICS] + FORTY_COLS)


# Round-4 fix (2026-11-22, diagnosed on stage2 repeat 1 fold 2): a
# within-group standard deviation can be a FLOATING-POINT RESIDUE instead
# of exactly 0. Tiny (pos, year) groups with 1 observed value + median
# fills (e.g. TE_2025 ten_yd_split [1.56, NaN, NaN] -> filled std
# 2.72e-16) pass the old `sd > 0` guard and explode z to ~1e14, which
# destroyed the pooled OOF r (0.0594) while every per-fold r stayed ~0.4.
# Rule: any group sd below 5% of the column/metric's global sd is
# degenerate -- standardize by the global sd instead (keep the group mean
# for centering). A genuine within-group sd is never that small relative
# to the global spread (observed ratios are 0.19-0.7).
#
# Round-5 fixes (Gemini review #4, 2026-11-22; all bugs VERIFIED against
# code + direct execution before fixing -- see notes/gemini_round4_20261122.md):
# - Bug A (high): FeaturePrep.transform imputed missing track values for
#   UNSEEN (pos, year) groups (LOCO / 2025 temporal) with the
#   cross-position __global__ median -- a 315-lb OL who skipped forty got
#   the 4.62s global median and became a +3 SD phantom. Three-tier
#   fallback now: train (pos, year) median -> TEST cohort's own
#   (pos, year) unlabeled median -> train pos_group median -> __global__
#   (same rule in _lbb_z for vertical / broad_jump).
# - Bug B (high): M0all (repeated_cv_oof, protect=True default) and
#   M1_over_all Stage 1 (protect=False) were fit with different
#   BaselineProtect switches, so delta_M1_over_all_vs_M0all (+0.0194) was
#   mostly a regularization artifact. make_model_pipe now defaults to
#   protect=False; ONLY the classification Stage-2 offset pipe passes
#   protect=True explicitly, so every baseline / Stage 1 is identical.
# - Bug C (low): ktr_speed was game_s_p99 / v_max_best while the md title
#   said / forty_s_peak -- now game_s_p99 / forty_s_peak (the v_max
#   version is kept as ktr_speed_vmax_mean); Y2_sensitivity reg_snaps_50
#   duplicated the main set (z_qual is gated on n_plays >= 75), replaced
#   by n_plays >= 100 / >= 150; experiments.csv date was hardcoded
#   "2026-11-21" -- now datetime.date.today().isoformat().
# - Robustness ladder gains a 5th step: log_pick only on the
#   forty.notna() (n=253) subset, separating the sample-loss effect from
#   the forty-collinearity effect in the skill_s_peak attenuation.
# - Speed: permutation_p caches Stage 1 (base fit + cross-fitted offset)
#   per (repeat, fold) -- the base uses only base_track_cols + raw_cols,
#   which the permutation never touches (it permutes the stacked model's
#   track_cols), so Stage 1 is bit-identical across all 1000 perms and
#   only Stage 2 is refit (asserted: no overlap between base cols and
#   track_cols). FeaturePrep.transform's per-group z loop is vectorized
#   with gkey.map(). Full run drops from ~7h14m to an expected ~30-45m.
def _safe_sd(sd, ref_sd):
    try:
        sd = float(sd)
    except (TypeError, ValueError):
        return 1.0
    ref = float(ref_sd) if ref_sd is not None else 1.0
    if not np.isfinite(ref) or ref <= 0:
        ref = 1.0
    if not np.isfinite(sd) or sd < 0.05 * ref:
        return ref
    return sd


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

    def __init__(self, level1, stable_drills, metrics=SKILL_METRICS,
                 by_year=False):
        self.level1 = level1
        self.stable_drills = stable_drills
        self.metrics = metrics
        # Round-2 fallback: when True, drill stats are computed per
        # (drill_name, draft_year) instead of per drill_name (within-cohort
        # standardization; legal because a whole draft class is measured
        # with one sensor set before the draft).
        self.by_year = by_year

    def fit(self, X, y=None):
        train_ids = set(X["nfl_id"])
        sub = self.level1[
            self.level1["nfl_id"].isin(train_ids)
            & self.level1["drill_name"].isin(self.stable_drills)]
        self.drill_stats_ = {}
        # Round-3 fix (Gemini review #3, Bug 1): NEVER groupby with a
        # single-element LIST -- in pandas 2.1.4, groupby(["drill_name"])
        # yields 1-tuple keys ('DRILL',), so fit() stored (('DRILL',), None)
        # while transform() looked up ('DRILL', None): 0% key match, all 9
        # skill features silently came out all-NaN (then imputed to 0.0).
        # Use a bare string for the single-column case.
        # Round-4 fix: per-metric global sd is the reference for the
        # degenerate-sd floor (see _safe_sd); a drill-year cell with 1
        # observed value must not explode the z-score.
        gstd = {m: (float(sub[m].std()) if np.isfinite(sub[m].std())
                    and sub[m].std() > 0 else 1.0)
                for m in self.metrics}
        if self.by_year:
            for (d, yr), grp in sub.groupby(["drill_name", "draft_year"]):
                self.drill_stats_[(d, yr)] = {
                    m: (float(grp[m].mean()),
                        _safe_sd(grp[m].std(), gstd[m]))
                    for m in self.metrics}
        else:
            for d, grp in sub.groupby("drill_name"):
                self.drill_stats_[(d, None)] = {
                    m: (float(grp[m].mean()),
                        _safe_sd(grp[m].std(), gstd[m]))
                    for m in self.metrics}
        return self

    def transform(self, X):
        sub = self.level1[
            self.level1["nfl_id"].isin(set(X["nfl_id"]))
            & self.level1["drill_name"].isin(self.stable_drills)].copy()
        if self.by_year:
            keys = list(zip(sub["drill_name"], sub["draft_year"]))
        else:
            keys = [(d, None) for d in sub["drill_name"]]
        # Round-3 fix: drill_stats_[key] is a dict {metric: (mu, sd)} --
        # the old code did .get(k, (nan, 1.0))[0], which indexes a DICT by
        # 0 (KeyError as soon as a key matches) and silently returned NaN
        # while no key matched. Index by metric name [m] instead.
        # Round-3 fix (unseen cohort): when by_year=True and a test
        # draft_year never appeared in fit() (only in LOCO / 2025 temporal
        # extrapolation), standardize with that test cohort's OWN unlabeled
        # Combine distribution (the whole class is measured in March, the
        # draft is in April -- zero target / future leakage).
        stats = dict(self.drill_stats_)
        if self.by_year:
            # Round-4 fix: same degenerate-sd floor as fit(); the
            # reference sd comes from this (test) fold's own level1 rows.
            gstd = {m: (float(sub[m].std()) if np.isfinite(sub[m].std())
                        and sub[m].std() > 0 else 1.0)
                    for m in self.metrics}
            for (d, yr), grp in sub.groupby(["drill_name", "draft_year"]):
                if (d, yr) not in stats:
                    stats[(d, yr)] = {
                        m: (float(grp[m].mean()),
                            _safe_sd(grp[m].std(), gstd[m]))
                        for m in self.metrics}
        for m in self.metrics:
            mu = np.array([stats.get(k, {}).get(m, (np.nan, 1.0))[0]
                           for k in keys])
            sd = np.array([stats.get(k, {}).get(m, (np.nan, 1.0))[1]
                           for k in keys])
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
        # Round-3 hard guard: never again silently ship all-NaN skill
        # features (the Bug-1 failure mode). Players with no stable-drill
        # rows at all are genuinely missing, so the bar is 0.90, not 1.00.
        assert out["skill_s_peak_z_mean"].notna().mean() > 0.90, (
            "DrillFoldClean produced >10% NaN skill_s_peak_z_mean -- "
            "drill-key mismatch, refusing to continue silently")
        return out


class FeaturePrep(BaseEstimator, TransformerMixin):
    """Fold-clean feature preparation.

    - track_cols: per-(position, year)-median imputation, then z-score
      within (position, year) (train-fold stats). Unseen (pos, year)
      groups (LOCO / temporal): impute with the TEST cohort's own
      (pos, year) unlabeled median, then the train pos_group median,
      then the global median (Round-5 fix); standardize with the TEST
      cohort's own unlabeled distribution.
    - raw_cols: global-median imputation, passed through unchanged
      (draft capital is NOT position-standardized).
    - pos_group appended as one-hot fixed effects (all levels kept).
    """

    def __init__(self, track_cols=(), raw_cols=(), by_year=False):
        self.track_cols = track_cols
        self.raw_cols = raw_cols
        # Round-2 fallback: when True, track features are z-scored within
        # (pos_group, draft_year) instead of pos_group alone.
        self.by_year = by_year

    def _gkey(self, X):
        g = X["pos_group"].astype(str)
        if self.by_year:
            g = g + "_" + X["draft_year"].astype(str)
        return g

    def fit(self, X, y=None):
        self.pos_levels_ = sorted(X["pos_group"].dropna().unique().tolist())
        self.track_med_ = {}
        self.track_global_ = {}
        self.track_by_pos_ = {}
        # Round-5 fix (Gemini #4, Bug A): per-pos_group (year-free) train
        # medians, the 3rd tier of the unseen-cohort imputation fallback.
        self.track_pos_med_ = {}
        gkey = self._gkey(X)
        pos_only = X["pos_group"].astype(str)
        for c in self.track_cols:
            med = X.groupby(gkey)[c].median()
            self.track_med_[c] = med.to_dict()
            self.track_med_[c]["__global__"] = float(X[c].median()) \
                if np.isfinite(X[c].median()) else 0.0
            self.track_pos_med_[c] = X.groupby(pos_only)[c].median().to_dict()
            filled = X[c].fillna(gkey.map(med)).fillna(
                self.track_med_[c]["__global__"])
            # Round-4 fix: floor the global sd too (a constant column
            # would otherwise divide by a ~1e-16 residue); _safe_sd
            # floors each group sd at 5% of the global sd.
            _gs = float(filled.std())
            _gs = _gs if (np.isfinite(_gs) and _gs > 1e-9) else 1.0
            self.track_global_[c] = (float(filled.mean()), _gs)
            self.track_by_pos_[c] = {}
            for pos, sub in X.assign(_f=filled).groupby(gkey):
                sd = sub["_f"].std()
                self.track_by_pos_[c][pos] = (
                    float(sub["_f"].mean()),
                    _safe_sd(sd, _gs))
        self.raw_med_ = {}
        for c in self.raw_cols:
            med = X[c].median()
            self.raw_med_[c] = float(med) if np.isfinite(med) else 0.0
        # Round-3 fix: lower_body_burst must no longer be a full-sample
        # position z-score (features_combine.py:91, disclosed minor leak).
        # Re-derive it fold-cleanly from the raw vertical / broad_jump
        # columns with TRAIN-fold stats (same grouping as track_cols).
        self.lbb_stats_ = None
        if ("lower_body_burst" in self.track_cols
                and "vertical" in X.columns
                and "broad_jump" in X.columns):
            self.lbb_stats_ = {}
            for c in ("vertical", "broad_jump"):
                med = X.groupby(gkey)[c].median()
                gm = float(X[c].median()) if np.isfinite(X[c].median()) else 0.0
                # Round-5 fix (Gemini #4, Bug A): per-pos_group train
                # median for the unseen-cohort imputation fallback.
                pos_med = X.groupby(pos_only)[c].median().to_dict()
                filled = X[c].fillna(gkey.map(med)).fillna(gm)
                # Round-4 fix: same degenerate-sd floor as the generic
                # track path (see _safe_sd).
                _gs = float(filled.std())
                _gs = _gs if (np.isfinite(_gs) and _gs > 1e-9) else 1.0
                by_pos = {}
                for pos, sub in X.assign(_f=filled).groupby(gkey):
                    sd = sub["_f"].std()
                    by_pos[pos] = (float(sub["_f"].mean()),
                                   _safe_sd(sd, _gs))
                self.lbb_stats_[c] = {
                    "by_pos": by_pos,
                    "global": (float(filled.mean()), _gs),
                    "median": med.to_dict(),
                    "pos_median": pos_med,
                    "global_median": gm}
        return self

    def _lbb_z(self, X, gkey):
        """Fold-clean lower_body_burst: mean of the within-position
        z-scores of raw vertical and broad_jump (train-fold stats; unseen
        (pos, year) groups use the TEST fold's own unlabeled distribution,
        same rule as the generic track path below)."""
        zs = []
        pos_only = X["pos_group"].astype(str)
        for c in ("vertical", "broad_jump"):
            st = self.lbb_stats_[c]
            # Round-5 fix (Gemini #4, Bug A): three-tier imputation --
            # train (pos, year) median -> TEST cohort's own (pos, year)
            # unlabeled median -> train pos_group median -> __global__.
            # The old code fell straight back to the cross-position
            # global median for unseen cohorts.
            med = gkey.map(st["median"])
            med = med.fillna(gkey.map(X.groupby(gkey)[c].median()))
            med = med.fillna(pos_only.map(st["pos_median"]))
            med = med.fillna(st["global_median"])
            vals = X[c].fillna(med).astype(float)
            gm, gs = st["global"]
            z = (vals - gm) / gs
            for pos, (mu, sd) in st["by_pos"].items():
                m = gkey == pos
                z.loc[m] = (vals[m] - mu) / sd
            unseen = ~gkey.isin(set(st["by_pos"].keys()))
            if unseen.any():
                for pos in gkey[unseen].unique():
                    m = gkey == pos
                    mu_u = float(vals[m].mean())
                    # Round-4 fix: floor the test-fold group sd at 5% of
                    # the global sd (see _safe_sd) -- a 1-2 row group with
                    # 1 observed value would otherwise explode z.
                    z.loc[m] = (vals[m] - mu_u) / _safe_sd(
                        float(vals[m].std()), gs)
            zs.append(z)
        return ((zs[0] + zs[1]) / 2.0).to_numpy(dtype=float)

    def transform(self, X):
        parts = []
        gkey = self._gkey(X)
        pos_only = X["pos_group"].astype(str)
        for c in self.track_cols:
            # Round-3 fix: lower_body_burst is re-derived fold-cleanly
            # from raw vertical / broad_jump (see _lbb_z); the precomputed
            # full-sample column is never used as a model input.
            if (c == "lower_body_burst" and self.lbb_stats_ is not None
                    and "vertical" in X.columns
                    and "broad_jump" in X.columns):
                parts.append(np.nan_to_num(
                    self._lbb_z(X, gkey), nan=0.0, posinf=0.0, neginf=0.0))
                continue
            # Round-5 fix (Gemini #4, Bug A): three-tier imputation --
            # train (pos, year) median -> TEST cohort's own (pos, year)
            # unlabeled median -> train pos_group median -> __global__.
            # The old code fell straight back to __global__ for unseen
            # cohorts, turning test-skippers into phantom +3 SD athletes.
            med = gkey.map(self.track_med_[c])
            med = med.fillna(gkey.map(X.groupby(gkey)[c].median()))
            med = med.fillna(pos_only.map(self.track_pos_med_[c]))
            med = med.fillna(self.track_med_[c]["__global__"])
            vals = X[c].fillna(med).astype(float)
            gm, gs = self.track_global_[c]
            # Round-5 speed: vectorize the seen-group z with gkey.map()
            # instead of a per-group .loc loop (identical arithmetic).
            mu_s = gkey.map({p: mu for p, (mu, _sd)
                             in self.track_by_pos_[c].items()})
            sd_s = gkey.map({p: sd for p, (_mu, sd)
                             in self.track_by_pos_[c].items()})
            z = pd.Series(np.nan, index=X.index, dtype=float)
            seen = mu_s.notna()
            z.loc[seen] = (vals[seen] - mu_s[seen]) / sd_s[seen]
            # Round-3 fix: unseen (pos, year) groups (LOCO / 2025 temporal
            # with by_year=True) -- standardize with the TEST fold's own
            # unlabeled X distribution instead of falling back to the
            # cross-cohort global mean.
            unseen = ~gkey.isin(set(self.track_by_pos_[c].keys()))
            if unseen.any():
                for pos in gkey[unseen].unique():
                    m = gkey == pos
                    mu_u = float(vals[m].mean())
                    # Round-4 fix: floor the test-fold group sd at 5% of
                    # the global sd (see _safe_sd).
                    z.loc[m] = (vals[m] - mu_u) / _safe_sd(
                        float(vals[m].std()), gs)
            parts.append(np.nan_to_num(z.to_numpy(dtype=float), nan=0.0,
                                       posinf=0.0, neginf=0.0))
        for c in self.raw_cols:
            parts.append(X[c].fillna(self.raw_med_[c]).astype(float)
                         .to_numpy(dtype=float))
        for pos in self.pos_levels_:
            parts.append((X["pos_group"] == pos).astype(float)
                         .to_numpy(dtype=float))
        return np.column_stack(parts) if parts else np.zeros((len(X), 0))


class BaselineProtect(BaseEstimator, TransformerMixin):
    """Post-standardization rescaling for differential regularization.

    Round-2 fix: a single RidgeCV / LogisticRegressionCV penalty used to
    strangle the strong baseline columns (on Y1 Hurdle, M1's C = 0.072 put
    27x the penalty M0a needed on log_pick) because the same penalty also
    had to hold down 14-18 noisy track columns. After StandardScaler, every
    column past the track block (raw baseline columns + position dummies)
    is scaled x10, so the shared penalty bites ~100x weaker on baseline
    weights than on track weights. Applied ONLY when a model has both a
    track block and a raw block; baseline-only models are untouched.
    """

    def __init__(self, n_track=0):
        self.n_track = n_track

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        if X.shape[1] > self.n_track:
            X[:, self.n_track:] *= 10.0
        return X


def make_model_pipe(model, track_cols, raw_cols, protect=False):
    # Round-5 fix (Gemini #4, Bug B): the default used to be protect=True,
    # so plain baselines (M0all in repeated_cv_oof) were fit WITH
    # BaselineProtect while the stacked Stage-1 base was fit WITHOUT it --
    # a regularization mismatch that fabricated most of
    # delta_M1_over_all_vs_M0all. Default is now False: every baseline and
    # Stage 1 is fit unprotected; only the classification Stage-2 offset
    # pipe passes protect=True explicitly.
    steps = [("prep", FeaturePrep(track_cols=track_cols, raw_cols=raw_cols,
                                  by_year=BY_YEAR)),
             ("scaler", StandardScaler())]
    if protect and track_cols and raw_cols:
        steps.append(("protect", BaselineProtect(n_track=len(track_cols))))
    steps.append(("model", model))
    return Pipeline(steps)


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


def _stacked_stage1(Xtr, ytr, Xva, track_cols, raw_cols, stratify,
                    base_model_fn=None, base_track_cols=()):
    """Stage 1 of the two-stage stack for ONE fold (Round-2 fix).

    Baseline on base_track_cols (position-standardized) + raw_cols
    (draft capital, passed through), with its OWN regularization
    (RidgeCV / LogisticRegressionCV chosen inside the training fold).
    The baseline pipe is built WITHOUT BaselineProtect: the differential
    penalty belongs to the residual stage, not to the baseline fit itself.
    Classification: the baseline's logit is carried as a fixed offset
    column, 5-fold CROSS-FITTED inside the training fold (Round-3 fix;
    was in-sample logit). The offset pipe reuses the baseline's CV-chosen
    C as a FIXED penalty -- refitting a full LogisticRegressionCV inside
    each of the 5 inner folds would ~4x the permutation cost.
    Falls back to the in-sample logit if cross-fitting fails.
    Round-5: factored out of _stacked_fit_predict so permutation_p can
    cache it -- the base uses only base_track_cols + raw_cols, which the
    permutation never touches (it permutes the stacked model's
    track_cols), so Stage 1 is identical across all 1000 permutations.
    Returns (base_pipe, stage1) where stage1 is {"base_va": ..., "resid":
    ...} (regression) or {"off_tr": ..., "off_va": ...} (classification).
    """
    if base_model_fn is None:
        base_model_fn = model_factory
    bpipe = make_model_pipe(base_model_fn(stratify), list(base_track_cols),
                            list(raw_cols), protect=False)
    bpipe.fit(Xtr, ytr)
    if stratify:
        p_tr = bpipe.predict_proba(Xtr)[:, 1]
        p_va = bpipe.predict_proba(Xva)[:, 1]
        eps = 1e-6
        try:
            _mdl = bpipe.named_steps["model"]
            _c = float(np.atleast_1d(
                getattr(_mdl, "C_", getattr(_mdl, "C", 1.0)))[0])
            opipe = Pipeline([
                ("prep", FeaturePrep(track_cols=list(base_track_cols),
                                     raw_cols=list(raw_cols),
                                     by_year=BY_YEAR)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=_c, max_iter=3000))])
            p_tr = cross_val_predict(opipe, Xtr, ytr, cv=5,
                                     method="predict_proba")[:, 1]
        except Exception as e:
            print(f"[stacked-offset] cross-fit failed ({e}); "
                  "using in-sample logit")
        off_tr = np.log(np.clip(p_tr, eps, 1 - eps)
                        / (1 - np.clip(p_tr, eps, 1 - eps)))
        off_va = np.log(np.clip(p_va, eps, 1 - eps)
                        / (1 - np.clip(p_va, eps, 1 - eps)))
        return bpipe, {"off_tr": off_tr, "off_va": off_va}
    base_tr = bpipe.predict(Xtr)
    base_va = bpipe.predict(Xva)
    resid = np.asarray(ytr, dtype=float) - np.asarray(base_tr, dtype=float)
    return bpipe, {"base_va": base_va, "resid": resid}


def _stacked_stage2(Xtr, ytr, Xva, track_cols, stratify, res_model_fn,
                    stage1):
    """Stage 2 only: fit the residual / offset model on the (possibly
    track-permuted) frames, reusing a cached Stage-1 dict from
    _stacked_stage1. Regression -- RidgeCV (alphas up to 1e5, so it can
    shrink to ~0 when the track adds nothing) fits the training-fold
    residuals r = y - y_hat_base; prediction = y_hat_base + r_hat.
    Classification -- the cached baseline logit is carried as a fixed
    offset column (nearly unpenalized via BaselineProtect, protect=True)
    while the track columns take the full L2 penalty.
    Returns (predictions, res_pipe)."""
    if stratify:
        Xtr2 = Xtr.copy()
        Xva2 = Xva.copy()
        Xtr2["__base_logit__"] = stage1["off_tr"]
        Xva2["__base_logit__"] = stage1["off_va"]
        if res_model_fn is None:
            res_model_fn = lambda s: LogisticRegressionCV(
                Cs=np.logspace(-2, 2, 15), cv=3, scoring="roc_auc",
                max_iter=3000)
        # Round-5 fix (Gemini #4, Bug B): this is the ONLY place that
        # still protects -- the offset column must stay nearly
        # unpenalized. make_model_pipe now defaults to protect=False.
        rpipe = make_model_pipe(res_model_fn(stratify), track_cols,
                                ["__base_logit__"], protect=True)
        rpipe.fit(Xtr2, ytr)
        return rpipe.predict_proba(Xva2)[:, 1], rpipe
    if res_model_fn is None:
        res_model_fn = lambda s: RidgeCV(alphas=np.logspace(0, 5, 25))
    rpipe = make_model_pipe(res_model_fn(stratify), track_cols, [])
    rpipe.fit(Xtr, stage1["resid"])
    return stage1["base_va"] + rpipe.predict(Xva), rpipe


def _stacked_fit_predict(Xtr, ytr, Xva, track_cols, raw_cols, stratify,
                         base_model_fn=None, res_model_fn=None,
                         base_track_cols=()):
    """Two-stage residual / offset stacking for ONE fold: Stage 1 +
    Stage 2 (see _stacked_stage1 / _stacked_stage2). Returns
    (predictions, base_pipe, res_pipe)."""
    bpipe, stage1 = _stacked_stage1(
        Xtr, ytr, Xva, track_cols, raw_cols, stratify,
        base_model_fn=base_model_fn, base_track_cols=base_track_cols)
    pred, rpipe = _stacked_stage2(
        Xtr, ytr, Xva, track_cols, stratify, res_model_fn, stage1)
    return pred, bpipe, rpipe


def repeated_cv_oof_stacked(base, y, track_cols, raw_cols, n_repeats,
                            stratify, drill_cleaner, seed=SEED,
                            base_track_cols=()):
    """Repeated OOF predictions for one stacked (baseline + track-residual)
    model, sharing the materialized folds. Returns (oofs, res_alphas,
    base_alphas)."""
    oofs, res_alphas, base_alphas = [], [], []
    for r in range(n_repeats):
        folds = _materialize_folds(base, y, stratify, r, drill_cleaner, seed)
        oof = np.full(len(y), np.nan)
        for tr, va, Xtr, Xva in folds:
            pred, bpipe, rpipe = _stacked_fit_predict(
                Xtr, y.iloc[tr], Xva, track_cols, raw_cols, stratify,
                base_track_cols=base_track_cols)
            oof[va] = pred
            for pipe, store in ((bpipe, base_alphas), (rpipe, res_alphas)):
                mdl = pipe.named_steps["model"]
                if hasattr(mdl, "alpha_"):
                    store.append(float(mdl.alpha_))
                elif hasattr(mdl, "C_"):
                    store.append(float(np.atleast_1d(mdl.C_)[0]))
        oofs.append(oof)
    return oofs, res_alphas, base_alphas


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
                  base_param=None, full_param=None, n_repeats_perm=3,
                  seed=SEED):
    """Permutation p for the incremental metric of a STACKED track model
    over its baseline (Round-2 fix).

    Two Round-2 bugs are fixed here. (1) The baseline and the full model
    used to share ONE fixed regularization -- M1's alpha_median (Y2: 681)
    -- which crushed M0a (own optimum 147) and inflated obs_delta; now
    base_param and full_param are fitted separately (each model's own
    median regularization from the main CV). (2) Only a single repeat
    (repeat=0) was run; now the first n_repeats_perm repeats are
    aggregated (obs = mean over repeats, and -- Round-3 fix -- each NULL
    sample is ALSO the mean over the same n_repeats_perm repeats: the old
    code pooled single-repeat nulls against a 3-repeat-mean obs, whose
    variance is 1/3 of a single repeat's, so the p-value was computed on
    mismatched scales).
    The null permutes the TRACK rows within (pos_group x draft_year)
    strata in both train and validation frames of each fold; the baseline
    fit is never permuted (Freedman-Lane style). The full model is the
    two-stage stacked fit, so the permutation tests the residual increment.
    Round-5: Stage 1 is cached per (repeat, fold) -- the base uses only
    base_track_cols + raw_cols, which the permutation never touches, so
    its fit and cross-fitted offset are identical across all nulls and
    only Stage 2 is refit per permutation (the old code refit Stage 1
    ~5000x per target, ~80% of the 7h14m full run). The cache is only
    valid when the base columns do not overlap the permuted track
    columns; asserted below.
    """
    rng = np.random.RandomState(seed)
    strata = base["pos_group"].astype(str) + "_" + base["draft_year"].astype(str)

    def _fixed_or_cv(param, strat):
        if param is None:
            return model_factory(strat)
        if strat:
            return LogisticRegression(C=float(param), max_iter=3000)
        return Ridge(alpha=float(param))

    base_fn = (lambda s: _fixed_or_cv(base_param, s)) \
        if base_param is not None else None
    res_fn = (lambda s: _fixed_or_cv(full_param, s)) \
        if full_param is not None else None
    yv = y.to_numpy(dtype=float)
    # Round-5 guard: the Stage-1 cache below is only valid when the base
    # columns (base_track_cols + raw_cols) are disjoint from the permuted
    # track columns. True for the only call site (M1 vs M0a:
    # base_track_cols=[], raw_cols=M0A_COLS, track_cols=TRACK_PURE +
    # STATIC + SHADOW_PURE); fail loudly rather than silently caching a
    # permuted base if a future caller changes the column sets.
    assert not (set(base_track_cols) & set(track_cols)) and not (
        set(raw_cols) & set(track_cols)), (
        "permutation_p Stage-1 cache invalid: base columns overlap "
        "permuted track columns")
    # Materialize the repeats ONCE and share them between obs and nulls.
    folds_list = [_materialize_folds(base, y, stratify, r, drill_cleaner,
                                     seed)
                  for r in range(n_repeats_perm)]
    # Baseline OOF per repeat is permutation-invariant: compute once.
    oof_m0_list = [
        _oof_from_folds(folds, y, base_track_cols, base_raw_cols,
                        stratify, strata, model_fn=base_fn)
        for folds in folds_list]
    # Round-5: Stage 1 (base fit + cross-fitted offset) is also
    # permutation-invariant -- compute once per (repeat, fold) and reuse
    # for obs and every null sample; only Stage 2 is refit per perm.
    stage1_cache = [
        [_stacked_stage1(Xtr, y.iloc[tr], Xva, track_cols, raw_cols,
                         stratify, base_model_fn=base_fn,
                         base_track_cols=base_track_cols)[1]
         for tr, va, Xtr, Xva in folds]
        for folds in folds_list]
    obs_list = []
    for folds, oof_m0, s1_list in zip(folds_list, oof_m0_list,
                                     stage1_cache):
        oof_m1 = np.full(len(y), np.nan)
        for (tr, va, Xtr, Xva), s1 in zip(folds, s1_list):
            pred, _ = _stacked_stage2(
                Xtr, y.iloc[tr], Xva, track_cols, stratify, res_fn, s1)
            oof_m1[va] = pred
        obs_list.append(metric_fn(yv, oof_m1) - metric_fn(yv, oof_m0))
    # Round-3 fix: one null sample = mean of the per-repeat permuted
    # deltas over the SAME n_repeats_perm repeats as obs (same cost as
    # before: n_perm // n_repeats_perm null samples x n_repeats_perm
    # repeats = n_perm permuted fits in total).
    n_null = max(1, n_perm // n_repeats_perm)
    nulls = []
    for _ in range(n_null):
        d_list = []
        for folds, oof_m0, s1_list in zip(folds_list, oof_m0_list,
                                         stage1_cache):
            oof_p = np.full(len(y), np.nan)
            for (tr, va, Xtr, Xva), s1 in zip(folds, s1_list):
                if track_cols:
                    Xtr = _permute_track_within_strata(
                        Xtr, track_cols, strata.iloc[tr], rng)
                    Xva = _permute_track_within_strata(
                        Xva, track_cols, strata.iloc[va], rng)
                pred, _ = _stacked_stage2(
                    Xtr, y.iloc[tr], Xva, track_cols, stratify, res_fn, s1)
                oof_p[va] = pred
            d_list.append(metric_fn(yv, oof_p) - metric_fn(yv, oof_m0))
        nulls.append(float(np.mean(d_list)))
    obs = float(np.mean(obs_list))
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


def single_feature_tests(df, y_col, feature_cols, mask,
                         base_controls=("log_pick", "is_udfa")):
    """Draft-adjusted OLS: y ~ base_controls + C(draft_year) +
    pos_group dummies + one position-z-scored feature; HC3 errors; BH-FDR.

    With base_controls=() and y_col = -log_pick this becomes the
    draft-pricing regression (Round-2 fix, Fig 4 forest-plot data).
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
        ok = sub[[f, *base_controls, "draft_year", "pos_group"]].notna().all(axis=1) \
            & y.notna()
        if ok.sum() < 30:
            rows.append({"feature": f, "n": int(ok.sum()), "beta": np.nan,
                         "p": np.nan})
            continue
        s2 = sub.loc[ok].copy()
        g = s2.groupby("pos_group")[f]
        s2[f + "_pz"] = (s2[f] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        Xd = pd.get_dummies(s2[[*base_controls, "draft_year", "pos_group"]],
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


def _robustness_test(df, controls=("log_pick", "forty", "ngs_final_score"),
                     subset_label=None):
    """skill_s_peak_z_mean -> z_qual controlling `controls` (plus is_udfa,
    C(draft_year), pos dummies); HC3.
    Round-2 fix: the headline skill-peak signal must survive the
    stopwatch and the NGS score as joint controls.
    Round-3 fix: run as a 4-step ladder (+pick / +pick+forty /
    +pick+ngs / +pick+forty+ngs) so the attenuation can be decomposed
    instead of reported as a single p=0.089.
    Round-5: `subset_label` tags a run restricted to a pre-filtered df
    (e.g. the forty.notna() subset) so the row stays self-describing.
    """
    f = "skill_s_peak_z_mean"
    sub = df[df["z_qual"].notna()].copy()
    need = [f, "is_udfa", "draft_year", "pos_group"] + list(controls)
    ok = sub[need].notna().all(axis=1) & sub["z_qual"].notna()
    if ok.sum() < 30:
        return {"feature": f, "controls": "+".join(controls),
                "subset": subset_label,
                "n": int(ok.sum()), "beta": np.nan, "p": np.nan}
    s2 = sub.loc[ok].copy()
    g = s2.groupby("pos_group")[f]
    s2[f + "_pz"] = (s2[f] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    Xd = pd.get_dummies(s2[[*controls, "is_udfa",
                            "draft_year", "pos_group"]],
                        columns=["draft_year", "pos_group"],
                        drop_first=True, dtype=float)
    Xd[f + "_pz"] = s2[f + "_pz"]
    Xd = sm.add_constant(Xd.astype(float))
    try:
        r = sm.OLS(s2["z_qual"].astype(float), Xd).fit(cov_type="HC3")
        return {"feature": f, "controls": "+".join(controls),
                "subset": subset_label,
                "n": int(ok.sum()),
                "beta": float(r.params[f + "_pz"]),
                "p": float(r.pvalues[f + "_pz"])}
    except Exception:
        return {"feature": f, "controls": "+".join(controls),
                "subset": subset_label,
                "n": int(ok.sum()), "beta": np.nan, "p": np.nan}


def _robustness_ladder(df):
    """Round-3: 4-step control ladder for skill_s_peak_z_mean.
    Round-5: a 5th step -- log_pick only, restricted to the
    forty.notna() (n=253) subset -- separates the sample-loss effect
    (299 -> 253 when forty is required) from the forty-collinearity
    effect in the attenuation from beta +0.156 to +0.094."""
    steps = [("log_pick",),
             ("log_pick", "forty"),
             ("log_pick", "ngs_final_score"),
             ("log_pick", "forty", "ngs_final_score")]
    out = [_robustness_test(df, controls=s) for s in steps]
    out.append(_robustness_test(df[df["forty"].notna()],
                                controls=("log_pick",),
                                subset_label="forty.notna()"))
    return out


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


def _partial_corr_n(x, y, controls):
    """_partial_corr plus the usable n (Round-2 fix: the KTR matrix must
    record n per cell). Empty controls = plain Pearson r."""
    ok = np.isfinite(x) & np.isfinite(y)
    if controls is not None and np.size(controls):
        ok = ok & np.all(np.isfinite(controls), axis=1)
    n = int(ok.sum())
    if n < 20:
        return np.nan, np.nan, n
    if controls is None or not np.size(controls):
        rho, p = stats.pearsonr(x[ok], y[ok])
        return float(rho), float(p), n
    Xc = sm.add_constant(controls[ok])
    rx = sm.OLS(x[ok], Xc).fit().resid
    ry = sm.OLS(y[ok], Xc).fit().resid
    rho, p = stats.pearsonr(rx, ry)
    return float(rho), float(p), n


def _mediation(df, mediator_col, rng):
    """Single-mediator OLS mediation: shadow_agility_pure -> mediator ->
    z_qual, controlling pos_group (one-hot) + draft capital; 1000
    bootstraps for the indirect effect (HC3). Returns {} when the usable
    sample is < 50. Round-2 fix: the original block only tested
    game_decel_p95 (a chase-contaminated extreme); the retest uses
    game_snap_burst_1s, the first-second post-snap burst that cannot be
    contaminated by late-play pursuit."""
    mdf = df[df["z_qual"].notna() & df[mediator_col].notna()].copy()
    mdf = mdf.dropna(subset=["shadow_agility_pure", "log_pick", "is_udfa",
                             mediator_col, "z_qual", "pos_group"])
    if len(mdf) < 50:
        return {}
    pos_d = pd.get_dummies(mdf["pos_group"], prefix="pos",
                           drop_first=True, dtype=float)
    Xa = pd.concat([mdf[["shadow_agility_pure", "log_pick", "is_udfa"]]
                    .astype(float).reset_index(drop=True),
                    pos_d.reset_index(drop=True)], axis=1)
    Xm = pd.concat([mdf[["shadow_agility_pure", mediator_col,
                         "log_pick", "is_udfa"]].astype(float)
                    .reset_index(drop=True),
                    pos_d.reset_index(drop=True)], axis=1)
    a_res = sm.OLS(mdf[mediator_col].astype(float).reset_index(drop=True),
                   sm.add_constant(Xa)).fit(cov_type="HC3")
    b_res = sm.OLS(mdf["z_qual"].astype(float).reset_index(drop=True),
                   sm.add_constant(Xm)).fit(cov_type="HC3")
    a = float(a_res.params["shadow_agility_pure"])
    b = float(b_res.params[mediator_col])
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
            sm_ = pd.concat([s[["shadow_agility_pure", mediator_col,
                                 "log_pick", "is_udfa"]].astype(float)
                             .reset_index(drop=True),
                             sd.reset_index(drop=True)], axis=1)
            ar = sm.OLS(s[mediator_col].astype(float).reset_index(drop=True),
                        sm.add_constant(sa)).fit()
            br = sm.OLS(s["z_qual"].astype(float).reset_index(drop=True),
                        sm.add_constant(sm_)).fit()
            boots.append(float(ar.params["shadow_agility_pure"])
                         * float(br.params[mediator_col]))
        except Exception:
            continue
    boots = np.array(boots)
    direct = float(b_res.params["shadow_agility_pure"])
    return {"n": int(len(mdf)), "a_path": a, "b_path": b,
            "indirect": float(indirect),
            "indirect_ci95": [float(np.percentile(boots, 2.5)),
                              float(np.percentile(boots, 97.5))]
            if len(boots) else [np.nan, np.nan],
            "direct_c": direct,
            "pct_mediated": float(indirect / (indirect + direct))
            if (indirect + direct) != 0 else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.path.join(BASE, "data", "parquet"))
    ap.add_argument("--derived", default=os.path.join(BASE, "data", "derived"))
    ap.add_argument("--out", default=os.path.join(BASE, "logs"))
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--within-year", action="store_true",
                    help="Round-2 fallback: harmonize drill/forty features "
                         "within (drill_name, draft_year) / (pos_group, "
                         "draft_year) if the adversarial block AUC stays "
                         "> 0.56 after the t0 fix.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global BY_YEAR
    BY_YEAR = args.within_year
    if args.quick:
        n_rep, n_perm, n_boot = 2, 100, 500
    else:
        n_rep, n_perm, n_boot = 10, args.n_perm, args.n_boot
    rng = np.random.RandomState(SEED)

    feats = pd.read_parquet(os.path.join(args.derived, "combine_features.parquet"))
    tg = pd.read_parquet(os.path.join(args.derived, "targets.parquet"))
    players = pd.read_parquet(os.path.join(args.parquet, "players.parquet"),
                              columns=["nfl_id", "draft_year",
                                       "draft_overall_pick", "nfl_position"])
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
        # Round-5 fix (Gemini #4, Bug C): ktr_speed is game_s_p99 /
        # forty_s_peak (two MEASURED peak speeds) to match the md title;
        # the old game_s_p99 / v_max_best (fitted asymptote) is kept
        # separately as ktr_speed_vmax.
        df["ktr_speed"] = df["game_s_p99"] / df["forty_s_peak"]
        df["ktr_speed_vmax"] = df["game_s_p99"] / df["v_max_best"]

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
        drill_cleaner = DrillFoldClean(level1, stable,
                                       by_year=args.within_year)
        drill_note = (f"fold-clean drill normalization from skill_events.parquet "
                      f"({len(stable)} stable drills, {len(level1)} player-drill rows)"
                      + ("; WITHIN-YEAR harmonization ON"
                         if args.within_year else ""))
    print("[drill] " + drill_note)

    # model specifications: (label, track_cols, raw_cols)
    # Round-2 fix: M0b is split (stopwatch vs NGS); M1-family models are
    # two-stage stacked fits (baseline with its own regularization, track
    # features fit on the residual / logit offset), so a strong baseline
    # is never strangled by the track penalty.
    # Round-3 fix (Gemini review #3, Bug 2): M0all used to pass the
    # stopwatch + NGS columns as raw_cols (global-median imputation, no
    # within-position standardization), which made M0all (0.1747) WORSE
    # than M0c_ngs alone (0.1926); they are now track_cols. M1_track_only
    # is the clean pre-draft head-to-head vs M0b_stopwatch (TRACK_PURE
    # only -- no draft capital, no NGS).
    MODELS = [
        ("M0a", [], M0A_COLS),
        ("M0b_stopwatch", M0B_STOPWATCH_COLS, []),
        ("M0c_ngs", M0C_NGS_COLS, []),
        ("M0b", M0B_COLS, []),
        ("M0all", M0B_STOPWATCH_COLS + M0C_NGS_COLS, M0A_COLS),
        ("M1b", M0B_COLS + TRACK_PURE + [SHADOW_PURE], []),
        ("M1_track_only", TRACK_PURE, []),
    ]
    # Stacked models: (label, track_cols, base_track_cols, base_raw_cols).
    # Round-3 fix: M1_over_stopwatch used to carry M0A_COLS in its baseline
    # while being compared against M0b_stopwatch (no draft capital) -- the
    # reported +0.1040 was 100% smuggled draft pick, not track signal.
    # Its baseline is now M0B_STOPWATCH_COLS only.
    STACKED_MODELS = [
        ("M1a", TRACK_PURE + [SHADOW_PURE], [], M0A_COLS),
        ("M1", TRACK_PURE + STATIC + [SHADOW_PURE], [], M0A_COLS),
        ("M1full", TRACK_PURE + STATIC + [SHADOW_FULL], [], M0A_COLS),
        ("M1_over_stopwatch", TRACK_PURE + [SHADOW_PURE],
         M0B_STOPWATCH_COLS, []),
        ("M1_over_all", TRACK_PURE + STATIC + [SHADOW_PURE],
         M0B_STOPWATCH_COLS + M0C_NGS_COLS, M0A_COLS),
    ]
    res = {"seed": SEED, "n_repeats": n_rep, "n_perm": n_perm, "n_boot": n_boot,
           "within_year_harmonization": bool(args.within_year),
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
        stack_base_alphas = {}
        for label, track_cols, base_track_cols, base_raw_cols in STACKED_MODELS:
            oofs, res_alphas, base_alphas = repeated_cv_oof_stacked(
                sub, y, track_cols, base_raw_cols, n_rep, stratify,
                drill_cleaner, base_track_cols=base_track_cols)
            cv_results[label] = (oofs, res_alphas)
            stack_base_alphas[label] = base_alphas
        for label, _, _, *_ in MODELS + STACKED_MODELS:
            oofs, alphas = cv_results[label]
            ms = [metric_fn(y.to_numpy(), o) for o in oofs]
            oof_avg = np.nanmean(np.vstack(oofs), axis=0)
            out_d[label] = {"metric_mean": float(np.mean(ms)),
                            "metric_std": float(np.std(ms)),
                            "alpha_median": float(np.median(alphas))
                            if alphas else np.nan}
            if label in stack_base_alphas:
                out_d[label]["base_alpha_median"] = float(
                    np.median(stack_base_alphas[label]))
            oof_store[(name, label)] = (sub["nfl_id"].to_numpy(), oof_avg)
        # deltas + bootstrap CI on the AVERAGED OOF (Round-1 fix)
        for base, comp in (("M0a", "M1"), ("M0b", "M1b"), ("M0a", "M1a"),
                           ("M0b_stopwatch", "M1_over_stopwatch"),
                           ("M0b_stopwatch", "M1_track_only"),
                           ("M0all", "M1_over_all")):
            d = out_d[comp]["metric_mean"] - out_d[base]["metric_mean"]
            lo, hi = bootstrap_ci(metric_fn, y.to_numpy(),
                                  oof_store[(name, base)][1],
                                  oof_store[(name, comp)][1], n_boot, rng,
                                  strata=sub["pos_group"].to_numpy())
            out_d[f"delta_{comp}_vs_{base}"] = {"mean": float(d), "ci95": [lo, hi]}
        # permutation test on the incremental metric (stacked M1 vs M0a),
        # each model with its OWN median regularization (Round-2 fix: the
        # old code passed M1's alpha for both, crushing M0a) and aggregated
        # over 3 repeats (old code: single repeat=0).
        _bp = out_d["M0a"].get("alpha_median")
        _fp = out_d["M1"].get("alpha_median")
        obs, p, null_mu, null_sd = permutation_p(
            metric_fn, sub, y, TRACK_PURE + STATIC + [SHADOW_PURE], M0A_COLS,
            [], M0A_COLS, n_perm, stratify, drill_cleaner,
            base_param=_bp if (_bp is not None and np.isfinite(_bp)) else None,
            full_param=_fp if (_fp is not None and np.isfinite(_fp)) else None)
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
    # Y2 coverage sensitivity: r at n_plays 100 / 150 (single 2-repeat
    # ridge). Round-5 fix (Gemini #4, Bug C): the old reg_snaps >= 50
    # subset was identical to the main set (z_qual is already gated on
    # n_plays >= 75 in targets.py), so it tested nothing.
    for thr in (100, 150):
        m = (df["n_plays"] >= thr) & df["z_qual"].notna()
        s2 = df[m].copy().reset_index(drop=True)
        yv = s2["z_qual"].astype(float)
        oofs, _, _ = repeated_cv_oof_stacked(
            s2, yv, TRACK_PURE + STATIC + [SHADOW_PURE], M0A_COLS,
            2, False, drill_cleaner)
        res.setdefault("Y2_sensitivity", {})[f"n_plays_{thr}"] = {
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

    def _fit_predict(tr_f, te_f, ycol, track_cols, raw_cols, stratify,
                     stacked=False):
        Xtr, Xva = tr_f, te_f
        if drill_cleaner is not None:
            Xtr = drill_cleaner.fit(Xtr).transform(Xtr)
            Xva = drill_cleaner.transform(Xva)
        if stacked:
            pred, _, _ = _stacked_fit_predict(
                Xtr, tr_f[ycol].astype(float), Xva, track_cols, raw_cols,
                stratify)
            return pred
        pipe = make_model_pipe(model_factory(stratify), track_cols, raw_cols)
        pipe.fit(Xtr, tr_f[ycol].astype(float))
        return (pipe.predict_proba(Xva)[:, 1] if stratify
                else pipe.predict(Xva))

    for label, ycol, mfn, strat in [
            ("Y1_year1_hurdle", "year1_hurdle_100", metric_auc, True),
            ("Y1_year1_logsnaps", "log_year1", metric_r, False)]:
        tr_l, te_l = tr, te
        if label == "Y1_year1_logsnaps":
            # Round-2 fix: match the main-CV mask (year1_reg_snaps > 0);
            # the old code scored 2025 players with zero Year-1 snaps too.
            tr_l = tr[tr["year1_reg_snaps"] > 0].reset_index(drop=True)
            te_l = te[te["year1_reg_snaps"] > 0].reset_index(drop=True)
        for mlabel, track_cols, raw_cols in [
                ("M0a", [], M0A_TEMPORAL),
                ("M1", TRACK_PURE + STATIC + [SHADOW_PURE], M0A_TEMPORAL)]:
            pred = _fit_predict(tr_l, te_l, ycol, track_cols, raw_cols, strat,
                                stacked=(mlabel == "M1"))
            temp[f"{label}_{mlabel}"] = {
                "n_train": len(tr_l), "n_test": len(te_l),
                "metric": float(mfn(te_l[ycol].astype(float).to_numpy(), pred))}
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
            pred = _fit_predict(sub, ste, "z_qual", track_cols, raw_cols, False,
                                stacked=(mlabel == "M1"))
            loco[f"Y2_test_{yr}_{mlabel}"] = {
                "n_test": len(ste),
                "r": metric_r(ste["z_qual"].to_numpy(), pred)}
    temp["LOCO_Y2"] = loco
    res["temporal"] = temp
    print("[temporal] " + json.dumps(temp, default=str))

    # ---- Adversarial validation: can X_track predict draft_year?
    #      Macro AUC = mean of per-class one-vs-rest AUCs. Round-2 fix:
    #      block diagnostics (FORTY / SKILL / STATIC) localize which
    #      feature family carries the cohort signal, so the within-year
    #      harmonization fallback can be targeted instead of blind.
    adv = {}
    adv_blocks = {
        "ALL": TRACK_PURE + STATIC + [SHADOW_PURE],
        "FORTY_ONLY": FORTY_COLS,
        "SKILL_ONLY": [f"skill_{m}_z_mean" for m in SKILL_METRICS]
        + [SHADOW_PURE],
        "STATIC_ONLY": STATIC + M0B_COLS,
    }
    yv = df["draft_year"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for block_name, block_cols in adv_blocks.items():
        Xb = df[block_cols + ["pos_group", "nfl_id",
                              "draft_year"]].copy()
        aucs = []
        for tr_idx, va_idx in cv.split(Xb, yv):
            Xtr, Xva = Xb.iloc[tr_idx], Xb.iloc[va_idx]
            if drill_cleaner is not None:
                Xtr = drill_cleaner.fit(Xtr).transform(Xtr)
                Xva = drill_cleaner.transform(Xva)
            pipe = make_model_pipe(
                LogisticRegressionCV(Cs=np.logspace(-2, 2, 15), cv=3,
                                     scoring="roc_auc_ovr", max_iter=3000),
                block_cols, [])
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
        adv[f"draft_year_macro_auc_{block_name}"] = {
            "mean": float(np.nanmean(aucs)), "std": float(np.nanstd(aucs))}
    adv["draft_year_macro_auc"] = adv["draft_year_macro_auc_ALL"]
    # Round-2 diagnostic: is the MEASURED peak speed drifting by year?
    # (forty_s_peak is the raw observed max; v_max_best is the fitted
    # asymptote, which was inflated to 12.20-12.27 yd/s for WRs.)
    _sp = df.dropna(subset=["forty_s_peak"])
    adv["forty_s_peak_by_year"] = {
        str(yr): {"mean_all": float(g["forty_s_peak"].mean()),
                  "n_all": int(len(g)),
                  "mean_WR": float(g[g["pos_group"] == "WR"]["forty_s_peak"].mean())
                  if (g["pos_group"] == "WR").any() else np.nan,
                  "n_WR": int((g["pos_group"] == "WR").sum())}
        for yr, g in _sp.groupby("draft_year")}
    res["adversarial_validation"] = adv
    print("[adversarial] " + json.dumps(adv))

    # ---- BH-FDR draft-adjusted single-feature tests
    # Round-2 fix: main FDR family = 9 pure skill features + 5 corrected
    # forty core params; the 2 redundant Shadow indices and 4 static
    # columns move to the full appendix table only.
    fdr_feats_full = TRACK_PURE + STATIC + [SHADOW_PURE, SHADOW_FULL]
    # Round-3 fix (Fig 4 contrast rows): the traditional stopwatch
    # columns belong in the pricing / full-FDR tables too. neg_forty is
    # -forty so that "higher = faster", matching the other speed features.
    df["neg_forty"] = -df["forty"]
    pricing_extra = ["neg_forty", "vertical", "broad_jump"]
    fdr_feats_full = fdr_feats_full + pricing_extra
    res["fdr_Y2"] = single_feature_tests(
        df, "z_qual", FDR_MAIN, df["z_qual"].notna())
    res["fdr_Y2_full"] = single_feature_tests(
        df, "z_qual", fdr_feats_full, df["z_qual"].notna())
    res["fdr_Y1_stage2"] = single_feature_tests(
        df, "log_asr", FDR_MAIN, df["hurdle_150"] == 1)
    print("[fdr_Y2] " + json.dumps(res["fdr_Y2"], default=str))
    # ---- Draft-pricing regression (Fig 4 forest-plot data, Round-2 fix):
    #      same features, outcome -log_pick (no draft control -- the market
    #      price IS the outcome), controls pos_group + C(draft_year) only.
    df["neg_log_pick"] = -df["log_pick"]
    res["draft_pricing"] = single_feature_tests(
        df, "neg_log_pick", FDR_MAIN + pricing_extra,
        df["neg_log_pick"].notna(),
        base_controls=())
    print("[draft_pricing] " + json.dumps(res["draft_pricing"], default=str))
    # ---- Robustness: skill_s_peak_z_mean, 4-step control ladder
    #      (Round-3 fix; the last step is the old single number).
    res["robustness_skill_s_peak_ladder"] = _robustness_ladder(df)
    res["robustness_skill_s_peak"] = res["robustness_skill_s_peak_ladder"][3]
    # Round-5: index 3 is the full 4th step (+pick+forty+ngs); index 4 is
    # the n=253 subset control and must NOT become the headline row.
    print("[robustness-ladder] " + json.dumps(
        res["robustness_skill_s_peak_ladder"], default=str))
    print("[robustness] " + json.dumps(res["robustness_skill_s_peak"],
                                        default=str))

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
    #      Round-2 fix: retest with game_snap_burst_1s (first-second
    #      post-snap burst, free of late-play pursuit contamination), and
    #      complete the 5x4 Combine-to-Pads transfer matrix (with n).
    med = {}
    if gk is not None:
        med = _mediation(df, "game_decel_p95", rng)
        burst = _mediation(df, "game_snap_burst_1s", rng)
        if burst:
            med["burst_1s"] = burst
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
        med["ktr_speed_n"] = int(
            (np.isfinite(df["v_max_best"].to_numpy(dtype=float))
             & np.isfinite(df["game_s_p99"].to_numpy(dtype=float))).sum())
        med["ktr_speed_mean"] = float(np.nanmean(df["ktr_speed"]))
        # Round-2 fix: full 5x4 Combine-to-Pads transfer matrix
        # (position-controlled partial r, with n per cell) + per-position
        # diagonal values.
        combine_vars = ["v_max_best", "forty_s_peak",
                        "skill_accel_p95_z_mean", "skill_decel_p95_z_mean",
                        "skill_lat_accel_p95_z_mean"]
        game_vars = ["game_s_p99", "game_accel_p95", "game_decel_p95",
                     "game_lat_accel_p95"]
        ktr_rows = []
        for cv_ in combine_vars:
            for gv in game_vars:
                rho, p, n = _partial_corr_n(
                    df[cv_].to_numpy(dtype=float),
                    df[gv].to_numpy(dtype=float), pos_dm)
                ktr_rows.append({"combine": cv_, "game": gv,
                                 "partial_r": rho, "p": p, "n": n})
        med["ktr_matrix"] = ktr_rows
        med["ktr_by_position"] = {}
        for cv_, gv in [("v_max_best", "game_s_p99"),
                        ("skill_accel_p95_z_mean", "game_accel_p95"),
                        ("skill_decel_p95_z_mean", "game_decel_p95"),
                        ("skill_lat_accel_p95_z_mean", "game_lat_accel_p95")]:
            med["ktr_by_position"][f"{cv_}~{gv}"] = {}
            for pos, grp in df.groupby("pos_group"):
                rho, p, n = _partial_corr_n(
                    grp[cv_].to_numpy(dtype=float),
                    grp[gv].to_numpy(dtype=float),
                    np.empty((len(grp), 0)))
                med["ktr_by_position"][f"{cv_}~{gv}"][str(pos)] = {
                    "partial_r": rho, "p": p, "n": n}
        # Round-3 fix: never report the mixed 0.65 ktr_speed_mean alone --
        # linemen play in a 2-5 yard trench (game_s_p99 ~4.5-6.0 yd/s) while
        # WR/DB are ~0.80+. Per-position speed-retention table.
        med["ktr_speed_by_position"] = {}
        for pos, grp in df.groupby("pos_group"):
            ok = grp.dropna(subset=["game_s_p99", "forty_s_peak"])
            if len(ok) == 0:
                continue
            med["ktr_speed_by_position"][str(pos)] = {
                "n": int(len(ok)),
                "game_s_p99_mean": float(ok["game_s_p99"].mean()),
                "game_s_p99_median": float(ok["game_s_p99"].median()),
                "forty_s_peak_mean": float(ok["forty_s_peak"].mean()),
                "forty_s_peak_median": float(ok["forty_s_peak"].median()),
                "ktr_speed_mean": float(ok["ktr_speed"].mean()),
                "ktr_speed_median": float(ok["ktr_speed"].median()),
                # Round-5 (Gemini #4, Bug C): keep the old v_max-based
                # retention alongside the forty_s_peak-based headline.
                "ktr_speed_vmax_mean": float(ok["ktr_speed_vmax"].mean()),
                "ktr_speed_vmax_median": float(ok["ktr_speed_vmax"].median())}
        # Round-3 fix: DL mixes 250 lb EDGE rushers (DE 36 + OLB 30) with
        # 310+ lb interior linemen (DT 47 + NT 5) -- the DL-only KTR/Y2
        # signal may be a between-subgroup body-type artifact. Recompute
        # the four KTR diagonals and the Y2-vs-M1-OOF Spearman inside each
        # subgroup before anyone cites the DL numbers.
        med["dl_edge_idl_split"] = {}
        if "nfl_position" in df.columns:
            dl = df[df["pos_group"] == "DL"].copy()
            dl["dl_sub"] = np.where(
                dl["nfl_position"].isin(["DE", "OLB"]), "EDGE",
                np.where(dl["nfl_position"].isin(["DT", "NT"]), "IDL",
                         "other"))
            _y2_ids, _y2_oof = oof_store.get(("Y2_quality", "M1"),
                                            (np.array([]), np.array([])))
            _oof_map = dict(zip(_y2_ids.tolist(), _y2_oof.tolist())) \
                if len(_y2_ids) else {}
            for sub_name, sgrp in dl.groupby("dl_sub"):
                entry = {"n": int(len(sgrp))}
                for cv_, gv in [("v_max_best", "game_s_p99"),
                                ("skill_accel_p95_z_mean", "game_accel_p95"),
                                ("skill_decel_p95_z_mean", "game_decel_p95"),
                                ("skill_lat_accel_p95_z_mean",
                                 "game_lat_accel_p95")]:
                    rho, p, n = _partial_corr_n(
                        sgrp[cv_].to_numpy(dtype=float),
                        sgrp[gv].to_numpy(dtype=float),
                        np.empty((len(sgrp), 0)))
                    entry[f"{cv_}~{gv}"] = {"partial_r": rho, "p": p,
                                            "n": n}
                yy = sgrp["z_qual"].to_numpy(dtype=float)
                pp = np.array([_oof_map.get(i, np.nan)
                               for i in sgrp["nfl_id"].to_numpy()])
                okm = np.isfinite(yy) & np.isfinite(pp)
                if okm.sum() >= 10:
                    rho, p = stats.spearmanr(yy[okm], pp[okm])
                    entry["Y2_M1_spearman"] = {"rho": float(rho),
                                               "p": float(p),
                                               "n": int(okm.sum())}
                med["dl_edge_idl_split"][str(sub_name)] = entry
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
             "Round-2 改动：M0b 已拆为 M0b_stopwatch（纯秒表/双跳）与 M0c_ngs"
             "（NGS 评分）；M1 系列为两阶段残差/偏置堆叠（基线用自身最优惩罚）；"
             "置换检验基线与全模型各自用正确正则化、聚合 3 次重复。"
             "\nRound-3 改动（第 4 轮修复，Gemini 第 3 轮核对）：DrillFoldClean 静默清零"
             "修复（单列 groupby 不用列表、transform 补 [m]、加非空断言）；M1_over_stopwatch"
             "基线去掉顺位、新增 M1_track_only 纯追踪对照；M0all/M1_over_all 的秒表与 NGS 列"
             "改走位置内标准化；--within-year 同届内调和开启（未见届用测试届自身无标签分布）；"
             "permutation_p 每个 null 样本也跨 3 次重复取均值（尺度对齐）；分类偏置列改折内"
             "5 折交叉拟合；lower_body_burst 在 FeaturePrep 内由原始 vertical/broad_jump 折内重算。",
             f"Drill 归一化：{drill_note}"]
    for name in ("Y1_hurdle_stage1", "Y1_stage2_logASR", "Y2_quality",
                 "Y1_year1_hurdle", "Y1_year1_logsnaps"):
        d = res.get(name, {})
        lines.append(f"\n## {name}")
        for k in ("M0a", "M0b_stopwatch", "M0c_ngs", "M0b", "M0all", "M1a",
                  "M1", "M1full", "M1b", "M1_track_only",
                  "M1_over_stopwatch", "M1_over_all"):
            if k in d:
                lines.append(f"- {k}：{d[k]['metric_mean']:.4f} ± {d[k]['metric_std']:.4f}"
                             f"（内层 CV 选定 alpha/C 中位数 {d[k].get('alpha_median', float('nan')):.3g}）")
        for k in ("delta_M1_vs_M0a", "delta_M1b_vs_M0b", "delta_M1a_vs_M0a",
                  "delta_M1_over_stopwatch_vs_M0b_stopwatch",
                  "delta_M1_track_only_vs_M0b_stopwatch",
                  "delta_M1_over_all_vs_M0all"):
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
    lines.append("\n## Y2 分位置 Spearman（M1 折外预测，第 5 轮新增进 md）")
    for pos, v in res.get("Y2_quality", {}).get(
            "per_position_spearman", {}).items():
        lines.append(f"- {pos}：n={v['n']}，Spearman ρ={v['spearman_rho']:+.3f}，"
                     f"p={v['p']:.3g}")
    lines.append("\n## BH-FDR 单特征检验（Y2，顺位校正后，HC3）")
    for r in sorted(res.get("fdr_Y2", []), key=lambda r: (r["p"] if np.isfinite(r["p"]) else 9)):
        lines.append(f"- {r['feature']}：β={r['beta']:+.3f}，p={r['p']:.3g}，"
                     f"q={r['q_fdr']:.3f}，n={r['n']}")
    lines.append("\n## 选秀定价回归（-log_pick，Fig 4 森林图数据；只控制位置 + 届别）")
    for r in sorted(res.get("draft_pricing", []), key=lambda r: (r["p"] if np.isfinite(r["p"]) else 9)):
        lines.append(f"- {r['feature']}：β={r['beta']:+.3f}，p={r['p']:.3g}，"
                     f"q={r['q_fdr']:.3f}，n={r['n']}")
    lines.append("\n## 稳健性：skill_s_peak_z_mean 控制 log_pick + forty + ngs_final_score")
    lines.append(json.dumps(res.get("robustness_skill_s_peak", {}),
                            indent=2, ensure_ascii=False))
    lines.append("\n## 稳健性阶梯：skill_s_peak_z_mean（五步控制，第 5 轮新增第 5 步）")
    for r in res.get("robustness_skill_s_peak_ladder", []):
        _sub = f"（子集 {r.get('subset')}）" if r.get("subset") else ""
        lines.append(f"- {r['controls']}{_sub}：β={r['beta']:+.3f}，p={r['p']:.3g}，"
                     f"n={r['n']}")
    lines.append("\n## KTR 5x4 偏相关矩阵（位置内，附 n）")
    for r in res.get("M2_mediation", {}).get("ktr_matrix", []):
        lines.append(f"- {r['combine']} ~ {r['game']}：r={r['partial_r']:+.3f}，"
                     f"p={r['p']:.3g}，n={r['n']}")
    lines.append("\n## M3 LightGBM / M2 中介 / Y2 敏感性")
    lines.append(json.dumps({k: res.get(k, {}) for k in
                             ("M3_lightgbm", "M2_mediation", "Y2_sensitivity")},
                            indent=2, ensure_ascii=False, default=str))
    lines.append("\n## 分位置实战速度保持率 game_s_p99 / forty_s_peak（第 4 轮新增）")
    for pos, v in res.get("M2_mediation", {}).get(
            "ktr_speed_by_position", {}).items():
        lines.append(f"- {pos}：n={v['n']}，game_s_p99 均值 {v['game_s_p99_mean']:.2f} / "
                     f"中位数 {v['game_s_p99_median']:.2f}，forty_s_peak 均值 "
                     f"{v['forty_s_peak_mean']:.2f} / 中位数 {v['forty_s_peak_median']:.2f}，"
                     f"保持率均值 {v['ktr_speed_mean']:.3f} / 中位数 {v['ktr_speed_median']:.3f}")
    lines.append("\n## DL 亚位置拆分 EDGE vs IDL（第 4 轮新增）")
    lines.append(json.dumps(res.get("M2_mediation", {}).get(
        "dl_edge_idl_split", {}), indent=2, ensure_ascii=False, default=str))
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
            "exp_id": f"phase3_{name}", "date": datetime.date.today().isoformat(), "stage": "phase3",
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
