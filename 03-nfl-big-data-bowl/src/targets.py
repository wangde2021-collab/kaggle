# NFL Big Data Bowl 2027 — Phase 3a: target variables Y1 / Y2 / Y3.
#
# Builds the player-level target table from data/parquet (Phase 1 ETL output).
# Scheme notes/gemini_scheme.md §2 (dual-core targets):
#   Y1  exposure-adjusted REG scrimmage snaps: ASR = REG_snaps / E,
#       E = 2026 - draft_year in {1,2,3}; Hurdle flag ASR >= 150 / season,
#       plus Year-1-only parallel target (rookie-season REG snaps, matched
#       on the games.season column -- NOT game_id//1e6, which drops
#       January regular-season games into the wrong season).
#   Y2  EB-shrunk position-specific single-play quality score:
#       position-specific metrics from player_play (REG, non-nullified only),
#       context-residualized where the scheme requires it, EB-shrunk toward
#       the position mean with a per-metric ANOVA variance decomposition,
#       then z-scored within position and averaged into one composite.
#       Coverage gate: POSITION-SPECIFIC n_plays >= 75 (routes for WR/TE,
#       pass-rush snaps for DL, dropbacks for OL, coverage snaps for DB),
#       with z-scoring done AFTER gating on the gated set.
#   Y3  REG start rate = career_games_started / (17 * E) (robustness only).
#
# Hard assertions (scheme §6 leak checklist item 5):
#   player_play rows used are ALL season_type == 'REG' and
#   play_nullified_by_penalty == 'N' (ETL already keeps REG only in the
#   game_tracking parquet; player_play is re-filtered here and asserted).
#
# Round-1 fixes (Gemini review 2026-11-21):
# - Year-1 matched on games.season (87 games where calendar year != season).
# - DB coverage filter excludes coverage_assignment == 'NO_ASSIGNMENT'
#   (29,442 rows) in addition to NaN.
# - _residualize no longer backfills raw values where covariates are missing
#   (those rows stay NaN and are excluded from the metric mean).
# - EB shrinkage rewritten as a per-metric ANOVA variance decomposition
#   (within-player vs between-player variance -> kappa = sigma_w^2 /
#   sigma_b^2, applied to the RAW player means, not to position-z-scored
#   values whose group means are ~0 and whose kappa was clipped to 200).
# - Explicit bool -> float mapping for object-dtype True/False/NaN columns
#   (pressure_allowed / quick_pressure / unblocked_pressure): the old
#   `unblocked != 1` comparison passed NaN rows through as "not unblocked".
# - OL restricted to plays with pass_rushers_encountered >= 1.
# - Y3 sanity: first REG season must be draft_year or draft_year + 1.
#
# Usage:
#   python3 src/targets.py [--parquet data/parquet] [--out data/derived]
# Writes <out>/targets.parquet (510 rows) + <out>/targets_summary.json.
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POS_GROUP = {"WR": "WR", "TE": "TE", "OL": "OL", "DL": "DL", "DB": "DB"}
# Position-specific coverage thresholds (denominator = the position's own
# play count, e.g. routes for WR/TE). All 75; kept as a dict so a position
# can be tightened later without touching the gate logic.
POS_MIN_PLAYS = {"WR": 75, "TE": 75, "OL": 75, "DL": 75, "DB": 75}


def _bool01(s):
    """Explicit True -> 1.0, False -> 0.0, anything else (NaN) -> NaN.

    Required because the pressure columns are object dtype; `x != 1` is
    True for NaN, so NaN rows used to leak through boolean filters.
    """
    return s.map({True: 1.0, False: 0.0}).astype(float)


def _residualize(df, y_col, x_cols, group_col="combine_position"):
    """OLS-residualize y on context columns within each position group.

    Play-level: fits one OLS per position on rows with all values present,
    returns the residual series (NaN where inputs are missing -- missing
    covariates are NOT backfilled with the raw value). Falls back to the raw
    value only when a whole group has too few rows or no variance (a
    group-level fallback, disclosed in the summary).
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby(group_col).groups.items():
        sub = df.loc[idx]
        ok = sub[[y_col] + x_cols].notna().all(axis=1)
        if ok.sum() < 30 or sub.loc[ok, y_col].nunique() < 2:
            out.loc[idx] = sub[y_col]
            continue
        X = sm.add_constant(pd.get_dummies(sub.loc[ok, x_cols],
                                            columns=[c for c in x_cols
                                                     if sub[c].dtype == object],
                                            drop_first=True, dtype=float))
        X = X.astype(float)
        try:
            res = sm.OLS(sub.loc[ok, y_col].astype(float), X).fit().resid
        except Exception:
            out.loc[idx] = sub[y_col]
            continue
        out.loc[res.index] = res
    return out


def _eb_shrink_player_means(means, vars_, ns, group):
    """Empirical-Bayes shrinkage of per-player metric means.

    Per position group, method-of-moments variance decomposition on the
    PLAY-level metric:
      sigma_w^2 = weighted mean of within-player variances
      sigma_b^2 = Var(player means) - sigma_w^2 / n_bar   (floored)
      kappa     = sigma_w^2 / sigma_b^2  (clipped to [0, 200])
      shrunk_i  = (n_i * mean_i + kappa * mu_group) / (n_i + kappa)
    Players with few plays are shrunk hard toward the group mean; players
    with many plays keep their observed mean. Applied to RAW means (before
    any z-scoring).
    """
    out = means.copy()
    for grp, idx in group.groupby(group).groups.items():
        m = means.loc[idx]
        if m.notna().sum() < 2:
            continue  # metric not defined for this position group
        v = vars_.loc[idx].fillna(0.0)
        n = ns.loc[idx].astype(float)
        w = (n - 1).clip(lower=0)
        sw2 = float(np.sum(w * v) / max(np.sum(w), 1.0))
        var_means = float(np.var(m, ddof=1)) if len(m) > 1 else 0.0
        nbar = float(np.mean(n)) if len(n) else 1.0
        sb2 = var_means - sw2 / max(nbar, 1.0)
        if not np.isfinite(sb2) or sb2 <= 0:
            sb2 = max(0.05 * sw2, 1e-12)  # no detectable between-player signal
        kappa = float(np.clip(sw2 / sb2, 0.0, 200.0))
        mu = float(np.mean(m))
        out.loc[idx] = (n * m + kappa * mu) / (n + kappa)
    return out


def _build_composite(table, metrics, group_col="pos_group"):
    """EB-shrink each metric, z-score within position, average the z's.

    `table` is indexed by nfl_id with columns pos_group and, per metric m,
    f"{m}_mean" / f"{m}_var" / f"{m}_n". Returns (composite Series, z DataFrame).
    """
    zcols = []
    for m in metrics:
        means = table[f"{m}_mean"]
        vars_ = table[f"{m}_var"]
        ns = table[f"{m}_n"]
        shrunk = _eb_shrink_player_means(means, vars_, ns, table[group_col])
        g = shrunk.groupby(table[group_col])
        z = (shrunk - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        zcols.append(z.rename(m + "_z"))
    zdf = pd.concat(zcols, axis=1)
    return zdf.mean(axis=1, skipna=True), zdf


def _player_stats(play_df, metric_cols):
    """Per-player mean / var / n for each play-level metric column."""
    g = play_df.groupby("nfl_id")
    out = {}
    for m in metric_cols:
        out[f"{m}_mean"] = g[m].mean()
        out[f"{m}_var"] = g[m].var(ddof=1)
        out[f"{m}_n"] = g[m].count()
    return pd.DataFrame(out)


def build_y1(pp, games, players, career):
    reg_ids = set(games.loc[games["season_type"] == "REG", "game_id"])
    season_of = games.set_index("game_id")["season"]
    reg = pp[pp["game_id"].isin(reg_ids)
             & (pp["play_nullified_by_penalty"] == "N")].copy()
    assert (reg["game_id"].isin(reg_ids)).all()
    assert (reg["play_nullified_by_penalty"] == "N").all()
    reg["season"] = reg["game_id"].map(season_of)
    per_player = reg.groupby("nfl_id").size().rename("reg_snaps")
    t = players[["nfl_id", "draft_year"]].copy()
    t["E"] = 2026 - t["draft_year"]
    t = t.set_index("nfl_id").join(per_player).fillna({"reg_snaps": 0})
    t["ASR"] = t["reg_snaps"] / t["E"]
    t["hurdle_150"] = (t["ASR"] >= 150).astype(int)
    # Year-1 only: rookie-season REG snaps, matched on games.season.
    # (game_id//1e6 is the CALENDAR year: 87 games have calendar year !=
    # season, e.g. game 2024010707 is season 2023, REG, Week 18.)
    y1 = reg[reg["season"] == reg["nfl_id"].map(
        players.set_index("nfl_id")["draft_year"])]
    y1c = y1.groupby("nfl_id").size().rename("year1_reg_snaps")
    t = t.join(y1c).fillna({"year1_reg_snaps": 0})
    t["year1_hurdle_100"] = (t["year1_reg_snaps"] >= 100).astype(int)
    # first REG season sanity: draft_year, draft_year + 1, or (rare late
    # debut) draft_year + 2 -- e.g. Daniel Scott (2023 5th round), Jovaughn
    # Gwyn (2023 7th round) and Dalton Wagner (UDFA 2023) first played REG
    # snaps in 2025 (13 / 11 / 2 snaps). Full data: {0: 378, 1: 20, 2: 3}.
    first_season = reg.groupby("nfl_id")["season"].min()
    dy = players.set_index("nfl_id")["draft_year"]
    gap = first_season - dy.reindex(first_season.index)
    gap_counts = gap.value_counts(dropna=False).to_dict()
    assert gap.isin([0, 1, 2]).all(), f"unexpected first-season gap: {gap_counts}"
    # cross-check vs official career totals
    cc = career.set_index("nfl_id")
    t["career_snaps"] = cc["career_offensive_snaps"] + cc["career_defensive_snaps"]
    t["career_snaps_per_E"] = t["career_snaps"] / t["E"]
    corr = float(np.corrcoef(t["ASR"], t["career_snaps_per_E"])[0, 1])
    # Y3 start rate
    t["games_started"] = cc["career_games_started"]
    t["start_rate"] = t["games_started"] / (17.0 * t["E"])
    return t.reset_index(), len(reg), corr, gap_counts


def build_y2(pp, games, players, combine_results):
    """Position-specific play-quality composite + EB shrinkage."""
    reg_ids = set(games.loc[games["season_type"] == "REG", "game_id"])
    reg = pp[pp["game_id"].isin(reg_ids)
             & (pp["play_nullified_by_penalty"] == "N")].copy()
    pos = combine_results.set_index("nfl_id")["combine_position"]
    reg["combine_position"] = reg["nfl_id"].map(pos)
    reg["pos_group"] = reg["combine_position"].map(POS_GROUP)
    tables = []
    # ---- WR / TE: separation (residualized on cushion + route), YACOE,
    #      target rate per route, receiving yards per route
    wr = reg[reg["pos_group"].isin(["WR", "TE"])].copy()
    wr["sep_resid"] = _residualize(
        wr, "separation_at_pass_forward", ["cushion", "route_ran"])
    wr["yacoe"] = wr["yards_after_catch"] - wr["expected_yards_after_catch"]
    wr["is_route"] = wr["route_ran"].notna()
    wr_r = wr[wr["is_route"]].copy()
    wr_r["m_tprr"] = _bool01(wr_r["target"])
    wr_r["m_yprr"] = wr_r["rec_yards"].fillna(0.0)
    wr_r["m_sep"] = wr_r["sep_resid"]
    wr_r["m_yacoe"] = wr_r["yacoe"]
    st = _player_stats(wr_r, ["m_sep", "m_yacoe", "m_tprr", "m_yprr"])
    st["pos_group"] = wr_r.drop_duplicates("nfl_id").set_index("nfl_id")["pos_group"]
    st["n_plays"] = st["m_tprr_n"]
    tables.append(st)
    # ---- DL / EDGE: get-off, quick pressure (excl. unblocked), pressure rate
    dl = reg[reg["pos_group"] == "DL"].copy()
    dl = dl[dl["player_get_off"].notna()].copy()  # pass-rush snaps only
    dl["m_getoff"] = -dl["player_get_off"]
    dl["m_quick_press"] = ((_bool01(dl["quick_pressure"]) == 1.0)
                           & (_bool01(dl["unblocked_pressure"]) == 0.0)).astype(float)
    dl["m_press_rate"] = dl["time_to_pressure"].notna().astype(float)
    st = _player_stats(dl, ["m_getoff", "m_quick_press", "m_press_rate"])
    st["pos_group"] = "DL"
    st["n_plays"] = st["m_getoff_n"]
    tables.append(st)
    # ---- OL: peak pressure probability allowed (residualized), pressure
    #      rate, quick-collapse rate; dropbacks only, >=1 pass rusher faced
    ol = reg[reg["pos_group"] == "OL"].copy()
    ol = ol[ol["dropback_duration"].notna()].copy()
    ol = ol[ol["pass_rushers_encountered"].fillna(0) >= 1].copy()
    ol["ppa_resid"] = _residualize(
        ol, "peak_pressure_probability_allowed",
        ["dropback_duration", "pass_rushers_encountered"])
    ol["m_ppa"] = -ol["ppa_resid"]
    ol["m_press_allowed_inv"] = 1.0 - _bool01(ol["pressure_allowed"])
    ol["m_quick_collapse_inv"] = 1.0 - (
        ol["time_to_pressure_allowed"] < 2.5).astype(float)
    st = _player_stats(ol, ["m_ppa", "m_press_allowed_inv", "m_quick_collapse_inv"])
    st["pos_group"] = "OL"
    st["n_plays"] = st["m_ppa_n"]
    tables.append(st)
    # ---- DB: coverage EPA (residualized on coverage type), tackle rate.
    #      Only plays with a REAL coverage assignment (excludes NaN and
    #      'NO_ASSIGNMENT', 29,442 rows).
    db = reg[reg["pos_group"] == "DB"].copy()
    db = db[db["coverage_assignment"].notna()
            & (db["coverage_assignment"] != "NO_ASSIGNMENT")].copy()
    db["epa_resid"] = _residualize(
        db, "expected_points_added", ["team_coverage_man_zone"])
    db["m_epa"] = -db["epa_resid"]
    db["m_tackle_rate"] = db["tackle"].fillna(0.0) + 0.5 * db["assist"].fillna(0.0)
    st = _player_stats(db, ["m_epa", "m_tackle_rate"])
    st["pos_group"] = "DB"
    st["n_plays"] = st["m_epa_n"]
    tables.append(st)

    y2 = pd.concat(tables)
    y2.index.name = "nfl_id"
    metric_cols = sorted({c[:-5] for c in y2.columns if c.endswith("_mean")})
    # diagnostic composite on ALL players (pre-gate)
    z_qual_raw, _ = _build_composite(y2, metric_cols)
    y2["z_qual_raw"] = z_qual_raw
    # primary composite: gate first (position-specific n_plays), then
    # re-run EB shrinkage + z-scoring on the gated set only
    thr = y2["pos_group"].map(POS_MIN_PLAYS)
    gated = y2[y2["n_plays"] >= thr].copy()
    z_qual_gated, _ = _build_composite(gated, metric_cols)
    y2["z_qual"] = np.nan
    y2.loc[gated.index, "z_qual"] = z_qual_gated
    y2["y2_covered"] = y2["n_plays"] >= thr
    return y2.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.path.join(BASE, "data", "parquet"))
    ap.add_argument("--out", default=os.path.join(BASE, "data", "derived"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    players = pd.read_parquet(os.path.join(args.parquet, "players.parquet"))
    games = pd.read_parquet(os.path.join(args.parquet, "games.parquet"))
    pp = pd.read_parquet(os.path.join(args.parquet, "player_play.parquet"))
    career = pd.read_parquet(os.path.join(args.parquet, "player_career_successes.parquet"))
    cr = pd.read_parquet(os.path.join(args.parquet, "combine_results.parquet"))

    y1, n_reg, corr, gap_counts = build_y1(pp, games, players, career)
    y2 = build_y2(pp, games, players, cr)

    t = y1.merge(y2[["nfl_id", "pos_group", "n_plays", "z_qual_raw", "z_qual",
                     "y2_covered"]], on="nfl_id", how="left")
    # players outside the four Y2 position groups have no Y2 row at all
    t["y2_covered"] = t["y2_covered"].fillna(False)
    t["pos_group"] = t["nfl_id"].map(
        cr.set_index("nfl_id")["combine_position"]).map(POS_GROUP)
    # sensitivity flags on career REG snaps (old definition, for comparison)
    for thr in (50, 150):
        t[f"y2_covered_reg_{thr}"] = (t["reg_snaps"] >= thr)
    assert len(t) == 510, f"expected 510 players, got {len(t)}"
    dest = os.path.join(args.out, "targets.parquet")
    t.to_parquet(dest, engine="pyarrow", compression="zstd", index=False)

    summary = {
        "players": len(t),
        "reg_plays_used": n_reg,
        "asr_vs_career_snaps_corr": round(corr, 4),
        "first_season_gap_counts": {str(k): int(v) for k, v in gap_counts.items()},
        "y1": {
            "zero_reg_snaps": int((t["reg_snaps"] == 0).sum()),
            "hurdle_150_n": int(t["hurdle_150"].sum()),
            "year1_hurdle_100_n": int(t["year1_hurdle_100"].sum()),
            "asr_median": float(t["ASR"].median()),
            "asr_mean": float(t["ASR"].mean()),
        },
        "y2": {
            "covered": int(t["y2_covered"].sum()),
            "covered_reg_50": int(t["y2_covered_reg_50"].sum()),
            "covered_reg_150": int(t["y2_covered_reg_150"].sum()),
            "by_position": {k: int(v) for k, v in
                            t[t["y2_covered"]].groupby("pos_group").size().items()},
            "z_qual_mean": float(np.nanmean(t["z_qual"])),
            "z_qual_std": float(np.nanstd(t["z_qual"])),
            "z_qual_raw_mean": float(np.nanmean(t["z_qual_raw"])),
            "z_qual_raw_std": float(np.nanstd(t["z_qual_raw"])),
        },
        "y3_start_rate_mean": float(t["start_rate"].mean()),
    }
    with open(os.path.join(args.out, "targets_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
