# NFL Big Data Bowl 2027 — Phase 2: Combine feature engineering.
#
# Builds the player-level Combine feature table from:
#   (a) combine_results.csv  -> static / derived body + speed features
#       (position-group z-scores; opt-out columns kept only as diagnostics,
#       NEVER as model inputs -- see scheme §3a)
#   (b) combine_tracking.csv -> 10 Hz kinematics per scheme §3b:
#       Step 0 physical cleaning (PLAYER only; drop sensor jumps dis>1.3 yd
#       or s>12.5 yd/s; drop attempts with >5% bad frames or <10 valid frames)
#       Step 1 forty-yard Furusawa-Hill curve decomposition
#       Step 2 skill-drill aggregates (deceleration / cut efficiency /
#       burst kinetics) + drill-normalized z-scores + Shadow Agility
#       calibration on non-opt-out players (ridge, OOF).
#
# Round-1 fixes (Gemini review 2026-11-21):
# - forty_features: `dist` is sliced after truncating `sub` (was indexing a
#   truncated array with the full-length mask); split times use np.interp
#   instead of searchsorted stair-steps; attempts with < 38 yd are flagged.
# - skill_drill_features: brake "power" is now true power (m*s*|a_t|) with a
#   mass-free power proxy alongside; braking impulse integrates the actual dt;
#   jerk_p95 is restricted to the first 1.0 s (scheme §3b definition);
#   cut_events uses the smoothed-direction cut-arc definition.
# - drill_normalize: only the 31 three-year-stable drills (G4) are used;
#   two-level aggregation (per (player, drill) mean first, then across
#   drills) removes attempt-count weighting bias; a COD-heavy variant
#   (top-50% drills by median lat_accel_p95 within drill_type) is reported.
# - calibrate_shadow_agility: targets are POSITION-Z-SCORED inside each CV
#   fold (the old pooled raw-seconds target vs position-zeroed features was
#   the main cause of the r=0.10/0.18 "failure"); mass-weighted features are
#   dropped from calibration; RidgeCV picks alpha inside folds; Pure
#   (skill-only) and Full (skill + forty + lower-body) shadow indices are
#   both reported; a P0 gold-standard check runs the same operator on the
#   THREE_CONE_DRILL / SHORT_SHUTTLE trajectories themselves.
# - lower_body_burst in static_features is a full-sample position z-score
#   (minor unsupervised leakage, kept for compatibility); the calibration
#   computes its own fold-clean version from raw vertical / broad_jump.
#
# Output: <out>/combine_features.parquet (one row per player, 510 rows)
#         + <out>/combine_features_summary.json (counts, fit quality, OOF r)
#         + <out>/forty_attempts.parquet + <out>/skill_events.parquet
#
# Usage:
#   python3 src/features_combine.py [--parquet data/parquet] [--out data/derived]
#       [--limit-events N]   # smoke test: cap events processed per drill type
#       [--p0]               # run the P0 gold-standard diagnostic only
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinematics import compute_kinematics, cut_events, fit_sprint_curve  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LB_TO_KG = 0.45359237

# Mass-free skill metrics used for Shadow calibration (Round-1 P1: the
# mass-weighted peak_mech_power / high_speed_brake_power are excluded --
# within a position, heavier players are slower yet score higher power).
SKILL_METRICS = ["decel_p95", "accel_p95", "jerk_p95", "lat_accel_p95",
                 "brake_power_p95", "braking_impulse_max",
                 "cut_retention_mean", "cut_reaccel_mean", "s_peak"]
SKILL_METRICS_MASS = ["peak_mech_power", "high_speed_brake_power"]


def zscore_within(df, cols, group):
    out = df.copy()
    for c in cols:
        g = df.groupby(group)[c]
        out[c + "_z"] = (df[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return out


def static_features(cr):
    df = cr.copy()
    df["bmi"] = 703.0 * df["combine_weight"] / df["combine_height"] ** 2
    df["arm_to_height"] = df["arm_length"] / df["combine_height"]
    df["ape_index"] = df["wing_span"] - df["combine_height"]
    df["hand_to_height"] = df["hand_size"] / df["combine_height"]
    df["speed_score"] = df["combine_weight"] * 200.0 / df["forty"] ** 4
    df["flying_30"] = df["forty"] - df["ten_yd_split"]
    df["start_finish_ratio"] = df["ten_yd_split"] / df["forty"]
    # lower-body burst composite: mean of position-group z(vertical), z(broad_jump)
    df = zscore_within(df, ["vertical", "broad_jump"], "combine_position")
    df["lower_body_burst"] = df[["vertical_z", "broad_jump_z"]].mean(axis=1)
    df["opted_out_agility"] = (df["three_cone"].isna() & df["short_shuttle"].isna()).astype(int)
    df["opted_out_forty"] = df["forty"].isna().astype(int)
    df["opt_out_count"] = df[["three_cone", "short_shuttle", "bench_reps", "forty"]].isna().sum(axis=1)
    keep = ["nfl_id", "combine_position", "bmi", "arm_to_height", "ape_index",
            "hand_to_height", "speed_score", "flying_30", "start_finish_ratio",
            "lower_body_burst", "opted_out_agility", "opted_out_forty", "opt_out_count"]
    return df[keep]


def _clean_event(traj):
    """Physical cleaning for one attempt; returns cleaned df or None."""
    df = traj.sort_values("time").copy()
    bad = (df["dis"] > 1.30) | (df["s"] > 12.5)
    if bad.mean() > 0.05 or len(df) - int(bad.sum()) < 10:
        return None
    df = df[~bad]
    if len(df) < 10:
        return None
    return df


def forty_features(events):
    """events: dict event_id -> cleaned trajectory df (PLAYER, FORTY_YARD_DASH)."""
    rows = []
    for eid, df in events.items():
        df = compute_kinematics(df)
        s = df["s_smooth"].to_numpy()
        # sprint start: first frame of 3 consecutive frames with s > 0.5
        start = None
        for i in range(len(s) - 2):
            if (s[i:i + 3] > 0.5).all():
                start = i
                break
        if start is None:
            continue
        sub = df.iloc[start:].copy().reset_index(drop=True)
        dist = np.cumsum(sub["dis"].to_numpy())
        end = int(np.searchsorted(dist, 40.0))
        if end < 10:
            continue
        sub = sub.iloc[: end + 1]
        dist = dist[: len(sub)]  # Round-1 fix: slice dist AFTER truncating sub
        t = sub["t_sec"].to_numpy() - sub["t_sec"].iloc[0]
        fit = fit_sprint_curve(t, sub["s_smooth"].to_numpy())

        def split_time(d0, d1):
            # Round-1 fix: linear interpolation instead of searchsorted steps
            if dist[-1] < d1:
                return np.nan
            return float(np.interp(d1, dist, t) - np.interp(d0, dist, t))

        s_end = sub["s_smooth"].to_numpy()
        tail = s_end[dist >= 30.0]
        speed_maint = float(np.mean(tail) / fit["v_max"]) if len(tail) and np.isfinite(fit["v_max"]) else np.nan
        rows.append({
            "event_id": eid, "nfl_id": int(df["nfl_id"].iloc[0]),
            "v_max": fit["v_max"], "tau": fit["tau"], "a0": fit["a0"], "r2": fit["r2"],
            "split_0_5yd": split_time(0, 5), "split_5_10yd": split_time(5, 10),
            "split_10_20yd": split_time(10, 20), "speed_maintenance": speed_maint,
            "s_peak": float(np.max(s_end)),
            "complete_40yd": bool(dist[-1] >= 38.0),
        })
    return pd.DataFrame(rows)


def skill_drill_features(events, weight_kg_by_player, pos_by_player):
    """events: dict (event_id) -> cleaned trajectory df (PLAYER, SKILL_DRILLS_*)."""
    rows = []
    for eid, df in events.items():
        nfl_id = int(df["nfl_id"].iloc[0])
        df = compute_kinematics(df)
        s = df["s_smooth"].to_numpy()
        # trim idle head/tail (keep s >= 0.5 window)
        act = np.where(s >= 0.5)[0]
        if len(act) < 10:
            continue
        df = df.iloc[act[0]: act[-1] + 1].copy().reset_index(drop=True)
        s = df["s_smooth"].to_numpy()
        a_t = df["a_t"].to_numpy()
        a_n = df["a_n"].to_numpy()
        jerk = df["jerk"].to_numpy()
        t_sec = df["t_sec"].to_numpy()
        dt = df["dt"].to_numpy(dtype=float)
        m = weight_kg_by_player.get(nfl_id, np.nan)
        neg = a_t[a_t < 0]
        pos_a = a_t[a_t > 0]
        cuts = cut_events(df)
        # braking impulse: max single continuous deceleration drop,
        # integrating the ACTUAL dt (Round-1 fix; was hardcoded 0.1)
        brake_impulse = 0.0
        run = 0.0
        for v, dti in zip(a_t, dt):
            if v < 0 and np.isfinite(dti) and dti > 0:
                run += -v * dti
                brake_impulse = max(brake_impulse, run)
            else:
                run = 0.0
        hi = s > 3.5
        # mass-free braking power proxy (W/kg): s * |a_t| on braking frames
        brake_power = s * np.abs(np.minimum(a_t, 0))
        # jerk restricted to the first 1.0 s after movement onset (scheme §3b)
        jerk_1s = np.abs(jerk[t_sec <= 1.0])
        rows.append({
            "event_id": eid, "nfl_id": nfl_id,
            "drill_name": df["drill_name"].iloc[0],
            "drill_type": df["drill_type"].iloc[0],
            "decel_p95": float(np.percentile(np.abs(neg), 95)) if len(neg) else np.nan,
            "accel_p95": float(np.percentile(pos_a, 95)) if len(pos_a) else np.nan,
            "jerk_p95": float(np.percentile(jerk_1s, 95)) if len(jerk_1s) else np.nan,
            "lat_accel_p95": float(np.percentile(a_n, 95)),
            # mass-weighted versions (kept for reporting, NOT for calibration)
            "peak_mech_power": float(np.max(m * s * np.maximum(a_t, 0))) if np.isfinite(m) else np.nan,
            "high_speed_brake_power": float(np.max(m * s[hi] * np.abs(np.minimum(a_t[hi], 0)))) if hi.any() and np.isfinite(m) else np.nan,
            "brake_power_p95": float(np.percentile(brake_power[brake_power > 0], 95)) if (brake_power > 0).any() else np.nan,
            "braking_impulse_max": float(brake_impulse),
            "n_cuts": len(cuts),
            "cut_retention_mean": float(np.nanmean([c["retention"] for c in cuts])) if cuts else np.nan,
            "cut_reaccel_mean": float(np.nanmean([c["reaccel_slope"] for c in cuts])) if cuts else np.nan,
            "cut_turn_angle_mean": float(np.nanmean([c["turn_angle_deg"] for c in cuts])) if cuts else np.nan,
            "s_peak": float(np.max(s)),
        })
    return pd.DataFrame(rows)


def drill_normalize(per_event, min_players=20, stable_drills=None):
    """Z-score each metric within drill_name, then per-player aggregates.

    Round-1 fixes: (1) only `stable_drills` (three-year >= min_players) are
    used, so single-year drills cannot leak across cohorts; (2) two-level
    aggregation -- mean per (player, drill) first, then mean / top-2 across
    drills, so drills with many attempts no longer dominate.
    Returns (player-level DataFrame, n_drills_used, cod_player DataFrame or
    None). The COD-heavy variant restricts to drills whose median
    lat_accel_p95 is in the top 50% within their drill_type group.
    """
    metrics = SKILL_METRICS + SKILL_METRICS_MASS
    counts = per_event.groupby("drill_name")["nfl_id"].nunique()
    keep_drills = counts[counts >= min_players].index
    if stable_drills is not None:
        keep_drills = [d for d in keep_drills if d in set(stable_drills)]
    sub = per_event[per_event["drill_name"].isin(keep_drills)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["nfl_id"]), 0, None

    def _aggregate(frame):
        # level 1: per (player, drill) mean
        per_pd = frame.groupby(["nfl_id", "drill_name"])[metrics].mean().reset_index()
        # level 2: z-score within drill (stats from the frame itself)
        for c in metrics:
            g = per_pd.groupby("drill_name")[c]
            per_pd[c + "_z"] = (per_pd[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        zcols = [c + "_z" for c in metrics]
        agg_mean = per_pd.groupby("nfl_id")[zcols].mean().add_suffix("_mean")
        top2_parts = []
        for zc in zcols:
            t = per_pd.groupby("nfl_id")[zc].apply(lambda s: s.nlargest(2).mean())
            top2_parts.append(t.rename(zc + "_top2"))
        top2 = pd.concat(top2_parts, axis=1)
        out = agg_mean.join(top2, how="outer").reset_index()
        out = out.rename(columns={c: "skill_" + c for c in out.columns if c != "nfl_id"})
        return out

    player = _aggregate(sub)
    # COD-heavy variant: drills with above-median turning load within type
    cod = None
    if "drill_type" in per_event.columns:
        med = sub.groupby("drill_name")["lat_accel_p95"].median()
        dtype_of = per_event.drop_duplicates("drill_name").set_index("drill_name")["drill_type"]
        med_df = med.rename("med_lat").reset_index()
        med_df["dtype"] = med_df["drill_name"].map(dtype_of)
        keep_cod = set()
        for _, grp in med_df.groupby("dtype"):
            thr = grp["med_lat"].median()
            keep_cod |= set(grp.loc[grp["med_lat"] >= thr, "drill_name"])
        cod_sub = sub[sub["drill_name"].isin(keep_cod)]
        if not cod_sub.empty and cod_sub["drill_name"].nunique() >= 3:
            cod = _aggregate(cod_sub)
            cod = cod.rename(columns={c: "skill_cod_" + c[6:] if c.startswith("skill_") else c
                                      for c in cod.columns if c != "nfl_id"})
    return player, len(keep_drills), cod


def _position_z_fit(df, cols, group):
    """Fit per-group mean/std; returns (transformed df, stats dict)."""
    stats = {}
    out = df.copy()
    for c in cols:
        g = df.groupby(group)[c]
        mu = g.transform("mean")
        sd = g.transform("std").replace(0, np.nan)
        stats[c] = (df.groupby(group)[c].mean().to_dict(),
                    df.groupby(group)[c].std().replace(0, np.nan).to_dict())
        out[c] = (df[c] - mu) / sd
    return out, stats


def _position_z_apply(df, cols, group, stats, global_stats):
    out = df.copy()
    for c in cols:
        mu_map, sd_map = stats[c]
        mu = df[group].map(mu_map)
        sd = df[group].map(sd_map)
        gmu, gsd = global_stats[c]
        mu = mu.fillna(gmu)
        sd = sd.fillna(gsd).replace(0, np.nan)
        out[c] = (df[c] - mu) / sd
    return out


def calibrate_position_z(X, y_raw, pos, label="", n_splits=5, seed=42):
    """Fold-clean position-z calibration.

    The target (raw -seconds, higher = more agile) and every feature are
    z-scored WITHIN position using TRAIN-fold statistics only; RidgeCV picks
    alpha inside the fold. Returns (oof_r_within, oof_r2_within, oof_r_pooled,
    per-fold r list, fitted full model + stats for scoring all players).
    """
    X = X.copy()
    y_raw = pd.Series(y_raw, dtype=float)
    pos = pd.Series(pos)
    feat_cols = [c for c in X.columns if not X[c].isna().all()]
    X = X[feat_cols]
    if not feat_cols:
        return {"label": label, "n": 0, "oof_r_within": np.nan,
                "oof_r2_within": np.nan, "oof_r_pooled_raw": np.nan,
                "fold_r": [], "alpha": np.nan}, None
    ok = y_raw.notna() & X[feat_cols].notna().any(axis=1)
    X, y_raw, pos = X[ok], y_raw[ok], pos[ok]
    if len(X) < 20:
        return {"label": label, "n": int(len(X)), "oof_r_within": np.nan,
                "oof_r2_within": np.nan, "oof_r_pooled_raw": np.nan,
                "fold_r": [], "alpha": np.nan}, None
    X = X.fillna(X.median(numeric_only=True))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(X), np.nan)
    fold_rs = []
    for tr, va in kf.split(X):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y_raw.iloc[tr], y_raw.iloc[va]
        ptr, pva = pos.iloc[tr], pos.iloc[va]
        # position-z target on train stats
        ytr_z = (ytr - ytr.groupby(ptr).transform("mean")) / ytr.groupby(ptr).transform("std").replace(0, np.nan)
        ytr_z = ytr_z.fillna(0.0)
        mu_map = ytr.groupby(ptr).mean().to_dict()
        sd_map = ytr.groupby(ptr).std().replace(0, np.nan).to_dict()
        pv = pos.iloc[va]
        yva_z = (yva - pv.map(mu_map)) / pv.map(sd_map)
        yva_z = pd.Series(np.asarray(yva_z, dtype=float), index=yva.index)
        # position-z features on train stats
        Xtr_z, stats = _position_z_fit(Xtr.assign(_pos=ptr), feat_cols, "_pos")
        gstats = {c: (float(Xtr[c].mean()), float(Xtr[c].std() or 1.0)) for c in feat_cols}
        Xva_z = _position_z_apply(Xva.assign(_pos=pva), feat_cols, "_pos", stats, gstats)
        Xtr_z = Xtr_z[feat_cols].fillna(0.0).to_numpy(dtype=float)
        Xva_z = Xva_z[feat_cols].fillna(0.0).to_numpy(dtype=float)
        model = RidgeCV(alphas=np.logspace(0, 4, 25))
        model.fit(Xtr_z, ytr_z.to_numpy(dtype=float))
        oof[va] = model.predict(Xva_z)
        if np.isfinite(oof[va]).all() and np.std(oof[va]) > 0:
            fold_rs.append(float(np.corrcoef(oof[va], yva_z.fillna(0.0))[0, 1]))
    y_z_all = (y_raw - y_raw.groupby(pos).transform("mean")) / y_raw.groupby(pos).transform("std").replace(0, np.nan)
    okm = np.isfinite(oof) & np.isfinite(y_z_all.to_numpy(dtype=float))
    r_within = float(np.corrcoef(oof[okm], y_z_all.to_numpy(dtype=float)[okm])[0, 1]) if okm.sum() > 10 else np.nan
    r2_within = float(1 - np.sum((y_z_all.to_numpy(dtype=float)[okm] - oof[okm]) ** 2)
                      / np.sum((y_z_all.to_numpy(dtype=float)[okm] - np.mean(y_z_all.to_numpy(dtype=float)[okm])) ** 2)) if okm.sum() > 10 else np.nan
    # pooled raw-seconds r (the OLD metric, for comparison)
    r_pooled = float(np.corrcoef(oof[okm], y_raw.to_numpy(dtype=float)[okm])[0, 1]) if okm.sum() > 10 else np.nan
    # full fit for scoring all players (position stats on the calibration set)
    X_z, stats = _position_z_fit(X.assign(_pos=pos), feat_cols, "_pos")
    gstats = {c: (float(X[c].mean()), float(X[c].std() or 1.0)) for c in feat_cols}
    y_z = (y_raw - y_raw.groupby(pos).transform("mean")) / y_raw.groupby(pos).transform("std").replace(0, np.nan)
    full = RidgeCV(alphas=np.logspace(0, 4, 25))
    full.fit(X_z[feat_cols].fillna(0.0).to_numpy(dtype=float), y_z.fillna(0.0).to_numpy(dtype=float))
    return {"label": label, "n": int(len(X)), "oof_r_within": r_within,
            "oof_r2_within": r2_within, "oof_r_pooled_raw": r_pooled,
            "fold_r": [round(v, 3) for v in fold_rs],
            "alpha": float(full.alpha_)}, (full, stats, gstats, feat_cols)


def _score_shadow(model_pack, X_all, pos_all):
    if model_pack is None:
        return np.full(len(X_all), np.nan)
    full, stats, gstats, feat_cols = model_pack
    X = X_all.reindex(columns=feat_cols)
    X = X.fillna(X.median(numeric_only=True))
    Xz = _position_z_apply(X.assign(_pos=pos_all), feat_cols, "_pos", stats, gstats)
    return full.predict(Xz[feat_cols].fillna(0.0).to_numpy(dtype=float))


def calibrate_shadow_agility(skill_player, cr, forty_agg):
    """Ridge calibration of Shadow Agility on non-opt-out players.

    Returns (shadow index DataFrame with pure + full versions, report dict).
    Feature sets (mass-free only for Pure): skill _mean + _top2 columns.
    Full adds forty-curve (v_max_best, tau_best) + fold-clean lower-body
    burst. Targets are position-z-scored inside folds (see
    calibrate_position_z).
    """
    feat_pure = [c for c in skill_player.columns
                 if c.startswith("skill_") and
                 (c.endswith("_mean") or c.endswith("_top2")) and
                 not any(m in c for m in ("peak_mech_power", "high_speed_brake_power"))]
    df = skill_player.merge(
        cr[["nfl_id", "combine_position", "three_cone", "short_shuttle",
            "vertical", "broad_jump"]], on="nfl_id", how="left")
    df = df.merge(forty_agg[["nfl_id", "v_max_best", "tau_best"]],
                  on="nfl_id", how="left")
    # fold-clean lower-body burst (raw vertical / broad_jump -> position z)
    lb = zscore_within(df[["nfl_id", "combine_position", "vertical", "broad_jump"]],
                       ["vertical", "broad_jump"], "combine_position")
    df["lower_body_burst_fc"] = lb[["vertical_z", "broad_jump_z"]].mean(axis=1)
    feat_full = feat_pure + ["v_max_best", "tau_best", "lower_body_burst_fc"]
    report = {}
    preds = {}
    for target, col in [("three_cone", "three_cone"), ("short_shuttle", "short_shuttle")]:
        y_raw = -df[col]
        for version, feats in [("pure", feat_pure), ("full", feat_full)]:
            res, pack = calibrate_position_z(
                df[feats], y_raw, df["combine_position"],
                label=f"shadow_{version}_vs_{target}")
            report[f"shadow_{version}_vs_{target}"] = res
            preds[f"{version}_{target}"] = _score_shadow(
                pack, df[feats], df["combine_position"])
    # combine the two targets into one index per version (mean of the two
    # standardized predictions), then standardize the index itself
    out = df[["nfl_id"]].copy()
    for version in ("pure", "full"):
        a = pd.Series(preds[f"{version}_three_cone"], index=df.index)
        b = pd.Series(preds[f"{version}_short_shuttle"], index=df.index)
        idx = 0.5 * (a / (np.std(a) or 1.0) + b / (np.std(b) or 1.0))
        out[f"shadow_agility_{version}"] = idx
    out["shadow_agility"] = out["shadow_agility_pure"]  # default = Pure
    return out, report


def p0_gold_standard(ct, cr, weight, pos):
    """P0: same operator on THREE_CONE_DRILL / SHORT_SHUTTLE trajectories.

    Each target is predicted ONLY from its own drill's trajectories
    (three_cone from THREE_CONE_DRILL, short_shuttle from SHORT_SHUTTLE):
    mixing the two drills' kinematics would dilute the drill-specific
    signature. If the kinematic operator can predict position-z agility
    from the ground-truth drills' own trajectories (r > 0.65), the
    bottleneck is cross-drill aggregation; if r < 0.55, the operator
    itself is broken.
    Returns a report dict.
    """
    gt = ct[ct["drill_type"].isin(["THREE_CONE_DRILL", "SHORT_SHUTTLE"])].copy()
    gt = gt[gt["entity_type"] == "PLAYER"]
    events = {}
    for eid, traj in gt.groupby("event_id"):
        clean = _clean_event(traj)
        if clean is not None:
            events[eid] = clean
    feats = skill_drill_features(events, weight, pos)
    if feats.empty:
        return {"p0": {"error": "no ground-truth drill events kept"}}
    report = {}
    for target, col, dtype in [
            ("three_cone", "three_cone", "THREE_CONE_DRILL"),
            ("short_shuttle", "short_shuttle", "SHORT_SHUTTLE")]:
        # per-player mean of raw metrics from THIS drill type only
        sub_f = feats[feats["drill_type"] == dtype]
        per_player = sub_f.groupby("nfl_id")[SKILL_METRICS].mean().reset_index()
        df = per_player.merge(
            cr[["nfl_id", "combine_position", col]],
            on="nfl_id", how="left")
        res, _ = calibrate_position_z(
            df[SKILL_METRICS], -df[col], df["combine_position"],
            label=f"p0_vs_{target}")
        res["n_events"] = int(len(sub_f))
        report[f"p0_vs_{target}"] = res
    return {"p0": report}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.path.join(BASE, "data", "parquet"))
    ap.add_argument("--out", default=os.path.join(BASE, "data", "derived"))
    ap.add_argument("--limit-events", type=int, default=None,
                    help="smoke test: cap events processed per drill_type group")
    ap.add_argument("--min-drill-players", type=int, default=20,
                    help="min players per drill_name for normalization (lower for smoke tests)")
    ap.add_argument("--p0", action="store_true",
                    help="run the P0 gold-standard diagnostic and exit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cr = pd.read_parquet(os.path.join(args.parquet, "combine_results.parquet"))
    players = pd.read_parquet(os.path.join(args.parquet, "players.parquet"),
                              columns=["nfl_id"])
    ct = pd.read_parquet(os.path.join(args.parquet, "combine_tracking.parquet"))
    ct = ct[ct["entity_type"] == "PLAYER"].copy()
    ct["time"] = pd.to_datetime(ct["time"])
    weight = dict(zip(cr["nfl_id"], cr["combine_weight"] * LB_TO_KG))
    pos = dict(zip(cr["nfl_id"], cr["combine_position"]))

    # three-year-stable drills (G4): >= min_players in EACH of 2023/2024/2025
    cov = ct.groupby(["drill_name", "draft_year"])["nfl_id"].nunique().unstack(fill_value=0)
    stable = [d for d in cov.index
              if all(cov.loc[d].get(y, 0) >= args.min_drill_players for y in (2023, 2024, 2025))]
    print(f"stable drills (>= {args.min_drill_players} players each year): {len(stable)}")

    if args.p0:
        report = p0_gold_standard(ct, cr, weight, pos)
        print(json.dumps(report, indent=2))
        with open(os.path.join(args.out, "p0_gold_standard.json"), "w") as fh:
            json.dump(report, fh, indent=2)
        return 0

    # ---- group events ----
    ct = ct.sort_values(["event_id", "time"])
    grouped = ct.groupby("event_id")
    forty_events, skill_events = {}, {}
    dropped = {"bad_frames": 0, "too_short": 0}
    n_seen = {"FORTY_YARD_DASH": 0}
    skill_seen = 0
    for eid, traj in grouped:
        dtype = traj["drill_type"].iloc[0]
        if dtype == "FORTY_YARD_DASH":
            if args.limit_events and n_seen["FORTY_YARD_DASH"] >= args.limit_events:
                continue
            n_seen["FORTY_YARD_DASH"] += 1
            clean = _clean_event(traj)
            if clean is None:
                dropped["bad_frames"] += 1
                continue
            forty_events[eid] = clean
        elif dtype.startswith("SKILL_DRILLS"):
            if args.limit_events and skill_seen >= args.limit_events:
                continue
            skill_seen += 1
            clean = _clean_event(traj)
            if clean is None:
                dropped["bad_frames"] += 1
                continue
            skill_events[eid] = clean
    print(f"forty events kept: {len(forty_events)}; skill events kept: {len(skill_events)}; "
          f"dropped: {dropped}")

    # ---- forty curve features ----
    forty = forty_features(forty_events)
    forty.to_parquet(os.path.join(args.out, "forty_attempts.parquet"),
                     engine="pyarrow", compression="zstd", index=False)
    forty_agg = (forty.groupby("nfl_id")
                 .agg(v_max_best=("v_max", "max"), tau_best=("tau", "min"),
                      v_max_mean=("v_max", "mean"), tau_mean=("tau", "mean"),
                      a0_mean=("a0", "mean"),
                      split_0_5yd_mean=("split_0_5yd", "mean"),
                      split_5_10yd_mean=("split_5_10yd", "mean"),
                      split_10_20yd_mean=("split_10_20yd", "mean"),
                      speed_maintenance_mean=("speed_maintenance", "mean"),
                      sprint_r2_mean=("r2", "mean"),
                      sprint_r2_min=("r2", "min"),
                      complete_share=("complete_40yd", "mean"))
                 .reset_index())
    print(f"forty fitted: {len(forty)} attempts, R2 mean {forty['r2'].mean():.3f}, "
          f"R2>0.95 share {(forty['r2'] > 0.95).mean():.1%}")

    # ---- skill drill features + drill normalization ----
    skill = skill_drill_features(skill_events, weight, pos)
    # keep the per-event table: later review rounds may need fold-clean
    # drill normalization (re-derive z-scores inside CV folds) instead of
    # the full-sample normalization in drill_normalize() below.
    skill.to_parquet(os.path.join(args.out, "skill_events.parquet"),
                     engine="pyarrow", compression="zstd", index=False)
    skill_player, n_drills, skill_cod = drill_normalize(
        skill, min_players=args.min_drill_players, stable_drills=stable)
    print(f"skill events: {len(skill)}; stable drills used for normalization: {n_drills}")
    if skill_cod is not None:
        print(f"COD-heavy drills variant: {skill_cod.shape[1] - 1} columns")

    # ---- shadow agility calibration ----
    if skill_player.empty:
        shadow = pd.DataFrame({"nfl_id": cr["nfl_id"]})
        for c in ("shadow_agility_pure", "shadow_agility_full", "shadow_agility"):
            shadow[c] = np.nan
        cal_report = {"error": "no skill-drill players; calibration skipped"}
    else:
        shadow, cal_report = calibrate_shadow_agility(skill_player, cr, forty_agg)
    print("shadow calibration:", json.dumps(cal_report))

    # ---- assemble ----
    feats = static_features(cr)
    feats = feats.merge(forty_agg, on="nfl_id", how="left")
    feats = feats.merge(skill_player, on="nfl_id", how="left")
    if skill_cod is not None:
        feats = feats.merge(skill_cod, on="nfl_id", how="left")
    feats = feats.merge(shadow, on="nfl_id", how="left")
    assert len(feats) == 510, f"expected 510 players, got {len(feats)}"
    dest = os.path.join(args.out, "combine_features.parquet")
    feats.to_parquet(dest, engine="pyarrow", compression="zstd", index=False)
    summary = {
        "players": len(feats),
        "forty_attempts": len(forty),
        "forty_r2_mean": float(forty["r2"].mean()),
        "forty_r2_gt95_share": float((forty["r2"] > 0.95).mean()),
        "skill_events": len(skill),
        "drills_normalized": n_drills,
        "stable_drills": len(stable),
        "shadow_calibration": cal_report,
        "dropped_events": dropped,
    }
    with open(os.path.join(args.out, "combine_features_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("wrote", dest, "| columns:", len(feats.columns))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
