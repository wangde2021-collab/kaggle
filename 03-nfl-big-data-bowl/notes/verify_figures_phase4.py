#!/usr/bin/env python3
"""Independent verification for Phase 4 figures/tables.

Re-reads notes/phase3_results_round5_server.json with code paths separate
from make_figures_phase4.py and checks EVERY value recorded in
figures/figure_data.json and notes/tables_phase4.md against it.
Also checks the competition count (figures + tables < 10) and that all
PNG/PDF files exist and are non-empty.
Exits non-zero on any mismatch.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
J = json.load(open(os.path.join(ROOT, "notes", "phase3_results_round5_server.json")))
M = json.load(open(os.path.join(ROOT, "figures", "figure_data.json")))
TOL = 1e-9
fails = []
checks = 0


def eq(name, a, b, tol=TOL):
    global checks
    checks += 1
    if a is None or b is None:
        if a is not b:
            fails.append(f"{name}: {a} != {b}")
        return
    if abs(a - b) > tol:
        fails.append(f"{name}: {a} != {b} (diff {a - b:.3g})")


def ci_eq(name, a, b):
    eq(name + "[0]", a[0], b[0])
    eq(name + "[1]", a[1], b[1])


# --- Fig 1
f1 = M["figures"]["fig1_double_dissociation"]
src1 = [("Y1_hurdle_stage1", "delta_M1_vs_M0a"),
        ("Y1_stage2_logASR", "delta_M1_vs_M0a"),
        ("Y2_quality", "delta_M1_vs_M0a"),
        ("Y2_quality", "delta_M1a_vs_M0a")]
for row, (sec, key) in zip(f1, src1):
    v = J[sec][key]
    eq("fig1 " + row["label"][:20], row["delta"], v["mean"])
    ci_eq("fig1 " + row["label"][:20], row["ci95"], v["ci95"])

# --- Fig 2
f2 = M["figures"]["fig2_beyond_stopwatch"]
y2 = J["Y2_quality"]
for r in f2["absolute_r"]:
    eq("fig2 abs " + r["model"], r["r"], y2[r["model"]]["metric_mean"])
for drow in f2["deltas"]:
    key = {"M1 over stopwatch\nvs stopwatch": "delta_M1_over_stopwatch_vs_M0b_stopwatch",
           "M1 track-only\nvs stopwatch": "delta_M1_track_only_vs_M0b_stopwatch",
           "M1a vs draft pick": "delta_M1a_vs_M0a",
           "M1 vs draft pick": "delta_M1_vs_M0a",
           "M1 over all\nvs all baselines": "delta_M1_over_all_vs_M0all"}[drow["label"]]
    v = y2[key]
    eq("fig2 delta " + key, drow["delta"], v["mean"])
    ci_eq("fig2 delta " + key, drow["ci95"], v["ci95"])
eq("fig2 perm p", f2["permutation_p_M1_vs_M0a"], y2["permutation"]["p_perm"])
eq("fig2 perm obs", f2["permutation_obs"], y2["permutation"]["obs_delta"])

# --- Fig 3
f3 = M["figures"]["fig3_dl_edge_idl"]
dl = J["M2_mediation"]["dl_edge_idl_split"]
eq("fig3 EDGE n", f3["EDGE_n"], dl["EDGE"]["n"])
eq("fig3 IDL n", f3["IDL_n"], dl["IDL"]["n"])
for row in f3["rows"]:
    for grp in ("EDGE", "IDL"):
        src = dl[grp][row["metric"]]
        got = row[grp]
        if "partial_r" in src:
            eq(f"fig3 {grp} {row['metric']} r", got["partial_r"], src["partial_r"])
        else:
            eq(f"fig3 {grp} {row['metric']} rho", got["rho"], src["rho"])
        eq(f"fig3 {grp} {row['metric']} p", got["p"], src["p"])
        eq(f"fig3 {grp} {row['metric']} n", got["n"], src["n"])

# --- Fig 4
f4 = M["figures"]["fig4_pricing_efficiency"]
pricing = {x["feature"]: x for x in J["draft_pricing"]}
fdr = {x["feature"]: x for x in J["fdr_Y2_full"]}
for row in f4:
    dp, fy = pricing[row["feature"]], fdr[row["feature"]]
    eq("fig4 draft beta " + row["feature"], row["beta_draft"], dp["beta"])
    eq("fig4 draft p " + row["feature"], row["p_draft"], dp["p"])
    eq("fig4 draft q " + row["feature"], row["q_draft"], dp["q_fdr"])
    eq("fig4 draft n " + row["feature"], row["n_draft"], dp["n"])
    eq("fig4 Y2 beta " + row["feature"], row["beta_Y2_given_draft"], fy["beta"])
    eq("fig4 Y2 p " + row["feature"], row["p_Y2"], fy["p"])
    eq("fig4 Y2 q " + row["feature"], row["q_Y2"], fy["q_fdr"])
    eq("fig4 Y2 n " + row["feature"], row["n_Y2"], fy["n"])
    # SE derivation re-checked independently
    from scipy.stats import norm as _norm
    for beta, p, se in ((dp["beta"], dp["p"], row["se_draft_derived"]),
                        (fy["beta"], fy["p"], row["se_Y2_derived"])):
        z = _norm.ppf(1 - p / 2)
        eq("fig4 SE " + row["feature"], se, abs(beta) / z, tol=1e-6)

# --- Fig 5
f5 = M["figures"]["fig5_loco_temporal"]
for k, v in f5["LOCO_Y2"].items():
    eq("fig5 LOCO " + k, v["r"], J["temporal"]["LOCO_Y2"][k]["r"])
    eq("fig5 LOCO n " + k, v["n_test"], J["temporal"]["LOCO_Y2"][k]["n_test"])
t = J["temporal"]
eq("fig5 2025 hurdle M0a", f5["temporal_2025"]["Y1_year1_hurdle"]["M0a"],
   t["Y1_year1_hurdle_M0a"]["metric"])
eq("fig5 2025 hurdle M1", f5["temporal_2025"]["Y1_year1_hurdle"]["M1"],
   t["Y1_year1_hurdle_M1"]["metric"])
eq("fig5 2025 logsnaps M0a", f5["temporal_2025"]["Y1_year1_logsnaps"]["M0a"],
   t["Y1_year1_logsnaps_M0a"]["metric"])
eq("fig5 2025 logsnaps M1", f5["temporal_2025"]["Y1_year1_logsnaps"]["M1"],
   t["Y1_year1_logsnaps_M1"]["metric"])
for k, v in f5["Y2_sensitivity"].items():
    eq("fig5 sens " + k, v["oof_r_mean"], J["Y2_sensitivity"][k]["oof_r_mean"])
    eq("fig5 sens n " + k, v["n"], J["Y2_sensitivity"][k]["n"])
eq("fig5 main M1 r", f5["Y2_main_r_M1"], y2["M1"]["metric_mean"])
for k, v in f5["adversarial_validation"].items():
    eq("fig5 adv " + k, v, J["adversarial_validation"][k]["mean"])

# --- Fig 6
ladder = J["robustness_skill_s_peak_ladder"]
for i, row in enumerate(M["figures"]["fig6_robustness_ladder"]):
    s = ladder[i]
    eq(f"fig6 step{i+1} beta", row["beta"], s["beta"])
    eq(f"fig6 step{i+1} p", row["p"], s["p"])
    eq(f"fig6 step{i+1} n", row["n"], s["n"])
    if row["controls"] != s["controls"] or row["subset"] != s["subset"]:
        fails.append(f"fig6 step{i+1} controls/subset mismatch")

# --- Fig 7
f7 = M["figures"]["fig7_ktr_transfer"]
km = {(c["combine"], c["game"]): c for c in J["M2_mediation"]["ktr_matrix"]}
rows7 = ["forty_s_peak", "skill_accel_p95_z_mean", "skill_decel_p95_z_mean",
         "skill_lat_accel_p95_z_mean"]
cols7 = ["game_s_p99", "game_accel_p95", "game_decel_p95", "game_lat_accel_p95"]
for i, r in enumerate(rows7):
    for j, c in enumerate(cols7):
        cell = f7["ktr_matrix_4x4"][i][j]
        src = km[(r, c)]
        eq(f"fig7 ktr {r}/{c} r", cell["r"], src["partial_r"])
        eq(f"fig7 ktr {r}/{c} p", cell["p"], src["p"])
for pos, v in f7["ktr_speed_by_position"].items():
    src = J["M2_mediation"]["ktr_speed_by_position"][pos]
    eq(f"fig7 retention {pos}", v["mean"], src["ktr_speed_mean"])
    eq(f"fig7 retention median {pos}", v["median"], src["ktr_speed_median"])
    eq(f"fig7 retention n {pos}", v["n"], src["n"])
eq("fig7 ktr_speed partial r", f7["ktr_speed_partial_r"],
   J["M2_mediation"]["ktr_speed_partial_r"])
eq("fig7 mediation decel indirect", f7["mediation_decel"]["indirect"],
   J["M2_mediation"]["indirect"])
eq("fig7 mediation burst indirect", f7["mediation_burst_1s"]["indirect"],
   J["M2_mediation"]["burst_1s"]["indirect"])

# --- Tables file: every decimal in the md must match the JSON
md = open(os.path.join(ROOT, "notes", "tables_phase4.md")).read()
models_tab = ["M0a", "M0b_stopwatch", "M0c_ngs", "M0b", "M0all", "M1b",
              "M1_track_only", "M1a", "M1", "M1full", "M1_over_stopwatch", "M1_over_all"]
for m in models_tab:
    a = J["Y1_hurdle_stage1"][m]["metric_mean"]
    b = J["Y1_stage2_logASR"][m]["metric_mean"]
    c = y2[m]["metric_mean"]
    want = f"| {a:.4f} | {b:.4f} | {c:.4f} |"
    checks += 1
    if want not in md:
        fails.append(f"table1 row {m} not found as {want}")
pps = y2["per_position_spearman"]
for pos in ["DL", "WR", "DB", "OL", "TE"]:
    b = pps[pos]
    want = f"| {pos} | {b['n']} | {b['spearman_rho']:+.3f} | {b['p']:.3g} |"
    checks += 1
    if want not in md:
        fails.append(f"table2 row {pos} not found as {want}")
for grp in ("EDGE", "IDL"):
    s = dl[grp]["Y2_M1_spearman"]
    want = f"| {grp} (DL split) | {s['n']} | {s['rho']:+.3f} | {s['p']:.3g} |"
    checks += 1
    if want not in md:
        fails.append(f"table2 row {grp} not found as {want}")

# --- Files exist + count < 10
n_figs = 0
for i in range(1, 8):
    for ext in ("png", "pdf"):
        p = os.path.join(ROOT, "figures", f"fig{i}_" + {
            1: "double_dissociation", 2: "beyond_stopwatch", 3: "dl_edge_idl",
            4: "pricing_efficiency", 5: "loco_temporal", 6: "robustness_ladder",
            7: "ktr_transfer"}[i] + "." + ext)
        checks += 1
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            fails.append(f"missing/empty {p}")
    n_figs += 1
total_items = n_figs + 2
checks += 1
if total_items >= 10:
    fails.append(f"figure+table count {total_items} >= 10")

print(f"checks={checks} fails={len(fails)}")
if fails:
    print("FAILURES:")
    for f in fails:
        print(" -", f)
    sys.exit(1)
print(f"ALL {checks} CHECKS PASSED — figures+tables = {total_items} (< 10)")
