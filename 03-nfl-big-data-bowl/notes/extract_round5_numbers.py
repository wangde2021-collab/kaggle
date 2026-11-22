#!/usr/bin/env python3
"""Extract round-5 server numbers from phase3_results.json (NOT hand-copied).

Usage: python3 extract_round5_numbers.py <path-to-phase3_results.json>
Prints every number the Gemini round-5 draft needs, straight from the JSON.
KTR / DL-split blocks are found recursively (they live under M2_mediation
in the JSON dump embedded in phase3_results.md as well).
"""
import json
import sys


def find(d, key):
    """Recursively find first dict/list value stored under `key`."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = find(v, key)
            if r is not None:
                return r
    return None


def fmt(x, nd=4):
    if x is None:
        return "None"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def ci(d):
    if not d:
        return "None"
    return f"[{fmt(d[0])}, {fmt(d[1])}]"


def section(d, name, models):
    print(f"== {name} ==")
    for m in models:
        b = d.get(m, {})
        print(f"  {m}: mean={fmt(b.get('metric_mean'))} std={fmt(b.get('metric_std'))}"
              f" alpha_med={fmt(b.get('alpha_median'))}"
              + (f" base_alpha={fmt(b.get('base_alpha_median'))}" if b.get('base_alpha_median') is not None else ""))
    for k, v in d.items():
        if k.startswith("delta_"):
            print(f"  {k}: mean={fmt(v.get('mean'))} CI95={ci(v.get('ci95'))}")
    p = d.get("permutation", {})
    if p:
        print(f"  permutation: obs={fmt(p.get('obs_delta'))} p_perm={fmt(p.get('p_perm'))}"
              f" null_mean={fmt(p.get('null_mean'))} null_sd={fmt(p.get('null_sd'))}")
    print()


def main(path):
    d = json.load(open(path))
    print(f"seed={d.get('seed')} n_repeats={d.get('n_repeats')} n_perm={d.get('n_perm')}"
          f" n_boot={d.get('n_boot')} within_year={d.get('within_year_harmonization')}")
    print(f"drill: {d.get('drill_normalization')}")
    print()
    models = ["M0a", "M0b_stopwatch", "M0c_ngs", "M0b", "M0all", "M1b",
              "M1_track_only", "M1a", "M1", "M1full", "M1_over_stopwatch", "M1_over_all"]
    section(d["Y1_hurdle_stage1"], "Y1_hurdle_stage1", models)
    section(d["Y1_stage2_logASR"], "Y1_stage2_logASR", models)
    section(d["Y2_quality"], "Y2_quality", models)
    print(f"Y2 r2_M1={fmt(d['Y2_quality'].get('r2_M1'))} r2_M0a={fmt(d['Y2_quality'].get('r2_M0a'))}")
    print()

    adv = d.get("adversarial_validation", {})
    print("== adversarial_validation ==")
    for k in ["draft_year_macro_auc_ALL", "draft_year_macro_auc_FORTY_ONLY",
              "draft_year_macro_auc_SKILL_ONLY", "draft_year_macro_auc_STATIC_ONLY"]:
        b = adv.get(k, {})
        print(f"  {k}: mean={fmt(b.get('mean'))} std={fmt(b.get('std'))}")
    fsp = adv.get("forty_s_peak_by_year", {})
    for y in sorted(fsp):
        b = fsp[y]
        print(f"  forty_s_peak {y}: all={fmt(b.get('mean_all'))} (n={b.get('n_all')})"
              f" WR={fmt(b.get('mean_WR'))} (n={b.get('n_WR')})")
    print()

    print("== temporal ==")
    for k in ["Y1_year1_hurdle", "Y1_year1_logsnaps"]:
        b = d.get(k, {})
        print(f"  {k}: {b}")
    t = d.get("temporal", {})
    for k in ["Y1_year1_hurdle_M0a", "Y1_year1_hurdle_M1",
              "Y1_year1_logsnaps_M0a", "Y1_year1_logsnaps_M1"]:
        print(f"  temporal/{k}: {t.get(k)}")
    loco = t.get("LOCO_Y2", {})
    for k in sorted(loco):
        print(f"  temporal/LOCO_Y2/{k}: {loco[k]}")
    print()

    print("== fdr_Y2 (q<0.05) ==")
    for x in d.get("fdr_Y2", []):
        q = x.get("q_fdr", x.get("q", 1))
        if q < 0.05:
            print(f"  {x.get('feature')}: beta={fmt(x.get('beta'))} p={x.get('p')} q={fmt(q)}")
    print("  fdr_Y2_full rows:", len(d.get("fdr_Y2_full", [])))
    print()

    print("== draft_pricing (key rows) ==")
    for x in d.get("draft_pricing", []):
        if x.get("feature") in ("forty_s_peak", "v_max_best", "skill_s_peak_z_mean",
                                 "neg_forty", "vertical", "broad_jump"):
            print(f"  {x.get('feature')}: beta={fmt(x.get('beta'))} p={x.get('p')} q={fmt(x.get('q_fdr', x.get('q')))}")
    print()

    print("== robustness_skill_s_peak_ladder ==")
    for i, x in enumerate(d.get("robustness_skill_s_peak_ladder", []), 1):
        print(f"  step{i}: controls={x.get('controls')} subset={x.get('subset')}"
              f" n={x.get('n')} beta={fmt(x.get('beta'))} p={x.get('p')}")
    print()

    print("== Y2_sensitivity ==")
    for k, v in d.get("Y2_sensitivity", {}).items():
        print(f"  {k}: n={v.get('n')} oof_r_mean={fmt(v.get('oof_r_mean'))}")
    print()

    print("== M3_lightgbm ==", d.get("M3_lightgbm"))
    m2 = d.get("M2_mediation", {})
    print("== M2_mediation ==")
    print(f"  decel: n={m2.get('n')} indirect={fmt(m2.get('indirect'))}"
          f" CI95={ci(m2.get('indirect_ci95'))} pct={fmt(m2.get('pct_mediated'))}")
    b1 = m2.get("burst_1s", {})
    print(f"  burst_1s: n={b1.get('n')} indirect={fmt(b1.get('indirect'))}"
          f" CI95={ci(b1.get('indirect_ci95'))} pct={fmt(b1.get('pct_mediated'))}")
    print(f"  ktr_speed_corr_raw={fmt(m2.get('ktr_speed_corr_raw'))}"
          f" partial_r={fmt(m2.get('ktr_speed_partial_r'))} p={m2.get('ktr_speed_partial_p')}"
          f" n={m2.get('ktr_speed_n')} mean={fmt(m2.get('ktr_speed_mean'))}"
          f" vmax_mean={fmt(m2.get('ktr_speed_vmax_mean'))}")
    print()

    ksp = find(d, "ktr_speed_by_position")
    print("== ktr_speed_by_position ==")
    if isinstance(ksp, dict):
        for pos in ["WR", "TE", "DB", "DL", "OL"]:
            b = ksp.get(pos, {})
            print(f"  {pos}: mean={fmt(b.get('ktr_speed_mean'))} median={fmt(b.get('ktr_speed_median'))}"
                  f" vmax_mean={fmt(b.get('ktr_speed_vmax_mean'))}")
    else:
        print("  NOT IN JSON (see phase3_results.md)")
    print()

    print("== per_position_spearman (Y2_quality) ==")
    pps = d.get("Y2_quality", {}).get("per_position_spearman", {})
    for pos in ["DB", "DL", "OL", "TE", "WR"]:
        b = pps.get(pos, {})
        print(f"  {pos}: n={b.get('n')} rho={fmt(b.get('spearman_rho'))} p={b.get('p')}")
    print()

    dl = find(d, "dl_edge_idl_split")
    if dl is None:
        dl = find(d, "dl_split")
    print("== DL split (EDGE vs IDL) ==")
    print(" ", json.dumps(dl, indent=1)[:2000] if dl else "NOT IN JSON (see phase3_results.md)")


if __name__ == "__main__":
    main(sys.argv[1])
