#!/usr/bin/env python3
"""Extract round-4 draft numbers from a frozen phase3_results.json.

Iron rule (REFLECTIONS.md): numbers in a Gemini draft must be generated
by script from the frozen results JSON of the current round -- never
hand-copied, never reused from another round (smoke / earlier server run).

Usage:
    python3 extract_round4_numbers.py <phase3_results.json> [targets_summary.json]

Prints a fixed-format block to paste into notes/gemini_round4_draft.md.
Verifies the within_year flag is True (round 4 runs with --within-year).
"""
import json
import sys


def m(d, *keys):
    cur = d
    for k in keys:
        cur = cur[k]
    return cur


def r4(x):
    return round(float(x), 4)


def main(path, targets_path=None):
    d = json.load(open(path))
    assert d.get("within_year_harmonization") is True, (
        "within_year_harmonization is not True -- wrong file/round")
    print(f"# Round-4 numbers (from {path})")
    print(f"seed={d['seed']} n_repeats={d['n_repeats']} "
          f"n_perm={d['n_perm']} n_boot={d['n_boot']}")
    print()
    for tag in ("Y1_hurdle_stage1", "Y1_stage2_logASR", "Y2_quality"):
        blk = d[tag]
        keys = ["M0a", "M0b_stopwatch", "M0c_ngs", "M0b", "M0all",
                "M1", "M1a", "M1b", "M1_track_only",
                "M1_over_stopwatch", "M1_over_all"]
        print(f"[{tag}]")
        for k in keys:
            if k in blk:
                print(f"  {k}: {r4(blk[k]['metric_mean'])}")
        for dk, label in (
                ("delta_M1_over_stopwatch_vs_M0b_stopwatch",
                 "delta M1_over_stopwatch vs M0b_stopwatch"),
                ("delta_M1_track_only_vs_M0b_stopwatch",
                 "delta M1_track_only vs M0b_stopwatch"),
                ("delta_M1_vs_M0a", "delta M1 vs M0a")):
            if dk in blk:
                dd = blk[dk]
                print(f"  {label}: {r4(dd['mean'])} "
                      f"CI [{r4(dd['ci95'][0])}, {r4(dd['ci95'][1])}]")
        if "permutation" in blk:
            p = blk["permutation"]
            print(f"  permutation: obs {r4(p['obs_delta'])} "
                  f"p={r4(p['p_perm'])} null_sd={r4(p['null_sd'])}")
        print()
    print("[temporal / LOCO]")
    t = d["temporal"]
    for k in ("Y1_year1_hurdle_M0a", "Y1_year1_hurdle_M1",
              "Y1_year1_logsnaps_M0a", "Y1_year1_logsnaps_M1"):
        print(f"  {k}: {r4(t[k]['metric'])}")
    for k, v in t["LOCO_Y2"].items():
        print(f"  LOCO {k}: r={r4(v['r'])} n={v['n_test']}")
    print()
    print("[adversarial_validation]")
    a = d["adversarial_validation"]
    for k in ("draft_year_macro_auc_ALL", "draft_year_macro_auc_FORTY_ONLY",
              "draft_year_macro_auc_SKILL_ONLY",
              "draft_year_macro_auc_STATIC_ONLY"):
        print(f"  {k}: {r4(a[k]['mean'])} (std {r4(a[k]['std'])})")
    for yr, v in sorted(a["forty_s_peak_by_year"].items()):
        print(f"  forty_s_peak {yr}: all {r4(v['mean_all'])} (n={v['n_all']}) "
              f"WR {r4(v['mean_WR'])} (n={v['n_WR']})")
    print()
    print("[robustness_skill_s_peak_ladder]")
    for s in d["robustness_skill_s_peak_ladder"]:
        print(f"  {s['controls']}: beta={r4(s['beta'])} "
              f"p={r4(s['p'])} n={s['n']}")
    print()
    print("[draft_pricing key betas]")
    for row in d["draft_pricing"]:
        name = row.get("feature") or row.get("name")
        print(f"  {name}: beta={r4(row['beta'])} p={row.get('p')}")
    print()
    print("[M3_lightgbm]", {k: r4(v) for k, v in d["M3_lightgbm"].items()})
    print("[M2_mediation] indirect=",
          r4(d["M2_mediation"]["indirect"]),
          "CI", [r4(x) for x in d["M2_mediation"]["indirect_ci95"]],
          "burst_1s indirect=",
          r4(d["M2_mediation"]["burst_1s"]["indirect"]))
    print("[ktr_speed_mean]", r4(d["M2_mediation"]["ktr_speed_mean"]))
    dl = d["M2_mediation"]["dl_edge_idl_split"]
    for grp in ("EDGE", "IDL"):
        print(f"[DL {grp}] n={dl[grp]['n']} "
              f"decel r={r4(dl[grp]['skill_decel_p95_z_mean~game_decel_p95']['partial_r'])} "
              f"p={dl[grp]['skill_decel_p95_z_mean~game_decel_p95']['p']:.3g} "
              f"Y2_spearman rho={r4(dl[grp]['Y2_M1_spearman']['rho'])} "
              f"p={dl[grp]['Y2_M1_spearman']['p']:.3g} "
              f"n={dl[grp]['Y2_M1_spearman']['n']}")
    if targets_path:
        tg = json.load(open(targets_path))
        print()
        print("[targets]", json.dumps(tg)[:2000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
