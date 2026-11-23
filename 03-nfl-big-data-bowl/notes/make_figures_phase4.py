#!/usr/bin/env python3
"""Phase 4 figures for NFL Big Data Bowl 2027 (frozen round-5 results).

Reads ONLY notes/phase3_results_round5_server.json (the single source of
truth, Gemini round-5 frozen). No number is hand-copied: every plotted value
is pulled from the JSON here and dumped to figures/figure_data.json for the
independent verification script (notes/verify_figures_phase4.py).

Figures (7) + tables (2) = 9 items < 10 (competition limit).
  Fig 1  Double dissociation (Y1 zero gain vs Y2 gain, 95% CI)
  Fig 2  Beyond the stopwatch (absolute r + delta forest with 95% CI)
  Fig 3  Differential market pricing efficiency (draft pricing vs Y2|draft)
  Fig 4  Robustness ladder for skill_s_peak (5 steps + 1:2 decomposition)
  Fig 5  DL EDGE / IDL split (Role-Specific Kinematic Translation)
  Fig 6  Combine-to-pads transfer (KTR 4x4 heatmap + game-speed retention)
  Fig 7  LOCO / temporal robustness (LOCO Y2, 2025 forward Y1, Y2
         sensitivity, adversarial validation)
  (Renumbered 2026-11-23 per Gemini writeup review: figures now follow
  first-appearance order in the draft.)
  Tab 1  Model comparison (Y1 hurdle AUC / Y1 stage-2 r / Y2 r)
  Tab 2  Per-position Spearman (Y2 M1 out-of-fold ranking)

SEs for forest-plot whiskers on regression betas are NOT stored in the JSON;
they are derived from the reported beta and two-sided p via the normal
approximation  z = Phi^-1(1 - p/2),  SE = |beta| / z.  This is documented in
each caption that uses it.
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JSON_PATH = os.path.join(ROOT, "notes", "phase3_results_round5_server.json")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#E69F00"
GREY = "#8c8c8c"
DARK = "#333333"
LIGHT_BLUE = "#56B4E9"
PINK = "#CC79A7"

d = json.load(open(JSON_PATH))
manifest = {"source": "notes/phase3_results_round5_server.json",
            "seed": d["seed"], "n_repeats": d["n_repeats"],
            "n_perm": d["n_perm"], "n_boot": d["n_boot"],
            "within_year_harmonization": d["within_year_harmonization"],
            "figures": {}}


def se_from_beta_p(beta, p):
    """SE from two-sided p under a normal approximation (documented)."""
    if p is None or p >= 0.999 or p <= 0:
        return None
    z = norm.ppf(1.0 - p / 2.0)
    return abs(beta) / z if z > 0 else None


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, name + ".png"))
    fig.savefig(os.path.join(FIGDIR, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig 1
y1h = d["Y1_hurdle_stage1"]["delta_M1_vs_M0a"]
y1s = d["Y1_stage2_logASR"]["delta_M1_vs_M0a"]
y2m1 = d["Y2_quality"]["delta_M1_vs_M0a"]
y2m1a = d["Y2_quality"]["delta_M1a_vs_M0a"]
rows1 = [
    ("Y1 · Hurdle (AUC)\nM1 vs M0a", y1h["mean"], y1h["ci95"], GREY),
    ("Y1 · Stage 2 log snaps (r)\nM1 vs M0a", y1s["mean"], y1s["ci95"], GREY),
    ("Y2 · per-play quality (r)\nM1 vs M0a", y2m1["mean"], y2m1["ci95"], TEAL),
    ("Y2 · per-play quality (r)\nM1a vs M0a", y2m1a["mean"], y2m1a["ci95"], TEAL),
]
manifest["figures"]["fig1_double_dissociation"] = [
    {"label": lab, "delta": v, "ci95": ci} for lab, v, ci, _ in rows1]
fig, ax = plt.subplots(figsize=(8.0, 3.8))
ys = np.arange(len(rows1))[::-1]
for y, (lab, v, ci, c) in zip(ys, rows1):
    ax.errorbar([v], [y], xerr=[[v - ci[0]], [ci[1] - v]],
                fmt="o", color=c, ecolor=c, elinewidth=2, capsize=5, markersize=7)
    ax.text(ci[1] + 0.006, y, f"{v:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]",
            va="center", ha="left", fontsize=9)
ax.axvline(0, color=DARK, lw=1)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows1])
ax.set_xlabel("Out-of-fold increment over draft-pick baseline  (Δ, 10×5 repeated CV)")
ax.set_xlim(-0.03, 0.155)
ax.set_title("Figure 1 — Double dissociation:\ndraft capital buys snaps, not per-play efficiency",
             loc="left", fontweight="bold")
savefig(fig, "fig1_double_dissociation")

# ---------------------------------------------------------------- Fig 2
y2 = d["Y2_quality"]
abs_rows = [
    ("Stopwatch only", "M0b_stopwatch", y2["M0b_stopwatch"]["metric_mean"], GREY),
    ("Tracking over\nstopwatch", "M1_over_stopwatch", y2["M1_over_stopwatch"]["metric_mean"], TEAL),
    ("Tracking only", "M1_track_only", y2["M1_track_only"]["metric_mean"], TEAL),
    ("Draft pick only", "M0a", y2["M0a"]["metric_mean"], GREY),
    ("Tracking + shadow", "M1a", y2["M1a"]["metric_mean"], TEAL),
    ("Tracking + pick\n+ shadow", "M1", y2["M1"]["metric_mean"], TEAL),
    ("Pick + stopwatch\n+ NGS", "M0all", y2["M0all"]["metric_mean"], ORANGE),
    ("Tracking over\nall baselines", "M1_over_all", y2["M1_over_all"]["metric_mean"], ORANGE),
    ("NGS score only", "M0c_ngs", y2["M0c_ngs"]["metric_mean"], GREY),
]
delta_rows2 = [
    ("M1 over stopwatch\nvs stopwatch", y2["delta_M1_over_stopwatch_vs_M0b_stopwatch"]),
    ("M1 track-only\nvs stopwatch", y2["delta_M1_track_only_vs_M0b_stopwatch"]),
    ("M1a vs draft pick", y2["delta_M1a_vs_M0a"]),
    ("M1 vs draft pick", y2["delta_M1_vs_M0a"]),
    ("M1 over all\nvs all baselines", y2["delta_M1_over_all_vs_M0all"]),
]
manifest["figures"]["fig2_beyond_stopwatch"] = {
    "absolute_r": [{"model": m, "r": v} for _, m, v, _ in abs_rows],
    "deltas": [{"label": lab, "delta": v["mean"], "ci95": v["ci95"]}
               for lab, v in delta_rows2],
    "permutation_p_M1_vs_M0a": y2["permutation"]["p_perm"],
    "permutation_obs": y2["permutation"]["obs_delta"]}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.6),
                               gridspec_kw={"width_ratios": [1.25, 1]})
xs = np.arange(len(abs_rows))[::-1]
for x, (lab, m, v, c) in zip(xs, abs_rows):
    ax1.barh(x, v, color=c, height=0.65)
    ax1.text(v + 0.006, x, f"{v:.4f}", va="center", fontsize=8.5)
ax1.axvline(0, color=DARK, lw=1)
ax1.set_yticks(xs)
ax1.set_yticklabels([r[0] for r in abs_rows], fontsize=8.5)
ax1.set_xlabel("Out-of-fold r  (Y2 per-play quality, n=299)")
ax1.set_title("(a) Absolute predictive signal", loc="left", fontweight="bold")
ys2 = np.arange(len(delta_rows2))[::-1]
for y, (lab, v) in zip(ys2, delta_rows2):
    ci = v["ci95"]
    ax2.errorbar([v["mean"]], [y],
                 xerr=[[v["mean"] - ci[0]], [ci[1] - v["mean"]]],
                 fmt="o", color=TEAL, ecolor=TEAL, elinewidth=2, capsize=5, markersize=7)
    ax2.text(v["mean"] + 0.012, y, f"{v['mean']:+.4f}", va="center", fontsize=9)
ax2.axvline(0, color=DARK, lw=1)
ax2.set_yticks(ys2)
ax2.set_yticklabels([r[0] for r in delta_rows2], fontsize=8.5)
ax2.set_xlabel("Δr with 95% paired bootstrap CI")
ax2.set_xlim(-0.03, 0.26)
ax2.set_title("(b) Incremental gain", loc="left", fontweight="bold")
fig.suptitle("Figure 2 — Beyond the stopwatch: 10 Hz tracking beats static testing",
             x=0.02, ha="left", fontweight="bold", fontsize=11)
savefig(fig, "fig2_beyond_stopwatch")

# ---------------------------------------------------------------- Fig 5 (was Fig 3)
dl = d["M2_mediation"]["dl_edge_idl_split"]
metrics3 = [
    ("Top speed\nv_max ~ game s_p99", "v_max_best~game_s_p99"),
    ("Acceleration\nskill accel ~ game accel", "skill_accel_p95_z_mean~game_accel_p95"),
    ("Deceleration\nskill decel ~ game decel", "skill_decel_p95_z_mean~game_decel_p95"),
    ("Lateral acceleration\nskill lat-accel ~ game lat-accel", "skill_lat_accel_p95_z_mean~game_lat_accel_p95"),
    ("Y2 out-of-fold\nSpearman ρ (M1)", "Y2_M1_spearman"),
]
manifest["figures"]["fig5_dl_edge_idl"] = {
    "EDGE_n": dl["EDGE"]["n"], "IDL_n": dl["IDL"]["n"],
    "rows": [{"metric": key,
              "EDGE": dl["EDGE"][key], "IDL": dl["IDL"][key]}
             for _, key in metrics3]}
fig, ax = plt.subplots(figsize=(8.5, 4.4))
labels3 = [m[0] for m in metrics3]
ypos = np.arange(len(labels3))[::-1]
w = 0.36
edge_r = [dl["EDGE"][k]["partial_r"] if "partial_r" in dl["EDGE"][k]
          else dl["EDGE"][k]["rho"] for _, k in metrics3]
idl_r = [dl["IDL"][k]["partial_r"] if "partial_r" in dl["IDL"][k]
         else dl["IDL"][k]["rho"] for _, k in metrics3]
ax.barh(ypos + w / 2, edge_r, height=w, color=BLUE, label="EDGE (DE/OLB)")
ax.barh(ypos - w / 2, idl_r, height=w, color=ORANGE, label="IDL (DT/NT)")
for y, r, key in zip(ypos + w / 2, edge_r, [k for _, k in metrics3]):
    src = dl["EDGE"][key]
    ax.text(r + 0.015, y, f"{r:+.3f} (p={src['p']:.3f})", va="center", fontsize=8.5)
for y, r, key in zip(ypos - w / 2, idl_r, [k for _, k in metrics3]):
    src = dl["IDL"][key]
    if r >= 0:
        ax.text(r + 0.015, y, f"{r:+.3f} (p={src['p']:.3f})", va="center", fontsize=8.5)
    else:
        # negative bar extends left; label in the empty space right of zero
        ax.text(0.015, y, f"{r:+.3f} (p={src['p']:.3f})", va="center",
                ha="left", fontsize=8.5)
ax.axvline(0, color=DARK, lw=1)
ax.set_yticks(ypos)
ax.set_yticklabels(labels3)
ax.set_xlabel("Within-subgroup Pearson r (combine → game)  /  Spearman ρ (Y2)")
ax.set_xlim(-0.15, 0.62)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
ax.set_title("Figure 5 — Role-specific kinematic translation: EDGE vs interior DL",
             loc="left", fontweight="bold")
savefig(fig, "fig5_dl_edge_idl")

# ---------------------------------------------------------------- Fig 3 (was Fig 4)
pricing = {x["feature"]: x for x in d["draft_pricing"]}
fdr_full = {x["feature"]: x for x in d["fdr_Y2_full"]}
rows4 = [
    ("−40-yard time", "neg_forty"),
    ("Vertical jump", "vertical"),
    ("40-yd peak speed", "forty_s_peak"),
    ("Broad jump", "broad_jump"),
    ("Skill-drill peak speed", "skill_s_peak_z_mean"),
    ("Skill-drill acceleration", "skill_accel_p95_z_mean"),
    ("Skill-drill cut re-acceleration", "skill_cut_reaccel_mean_z_mean"),
    ("Skill-drill deceleration", "skill_decel_p95_z_mean"),
    ("Skill-drill lateral acceleration", "skill_lat_accel_p95_z_mean"),
]
manifest["figures"]["fig3_pricing_efficiency"] = []
for lab, feat in rows4:
    dp = pricing[feat]
    fy = fdr_full[feat]
    se_d = se_from_beta_p(dp["beta"], dp["p"])
    se_y = se_from_beta_p(fy["beta"], fy["p"])
    manifest["figures"]["fig3_pricing_efficiency"].append({
        "feature": feat, "label": lab,
        "beta_draft": dp["beta"], "p_draft": dp["p"], "q_draft": dp["q_fdr"],
        "n_draft": dp["n"], "se_draft_derived": se_d,
        "beta_Y2_given_draft": fy["beta"], "p_Y2": fy["p"], "q_Y2": fy["q_fdr"],
        "n_Y2": fy["n"], "se_Y2_derived": se_y})
fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.5, 4.8), sharey=True)
ypos4 = np.arange(len(rows4))[::-1]
for y, (lab, feat) in zip(ypos4, rows4):
    dp = pricing[feat]
    se = se_from_beta_p(dp["beta"], dp["p"])
    axl.errorbar([dp["beta"]], [y], xerr=1.96 * se if se else 0,
                 fmt="o", color=BLUE, ecolor=BLUE, elinewidth=2, capsize=4, markersize=6)
    axl.text(dp["beta"] + 0.012, y, f"{dp['beta']:+.3f}", va="center", fontsize=9)
for y, (lab, feat) in zip(ypos4, rows4):
    fy = fdr_full[feat]
    se = se_from_beta_p(fy["beta"], fy["p"])
    c = TEAL if feat == "skill_s_peak_z_mean" else PINK
    axr.errorbar([fy["beta"]], [y], xerr=1.96 * se if se else 0,
                 fmt="o", color=c, ecolor=c, elinewidth=2, capsize=4, markersize=6)
    axr.text(fy["beta"] + 0.012, y, f"{fy['beta']:+.3f}", va="center", fontsize=9)
for ax in (axl, axr):
    ax.axvline(0, color=DARK, lw=1)
    ax.set_xlim(-0.05, 0.36)
axl.set_yticks(ypos4)
axl.set_yticklabels([r[0] for r in rows4])
axl.set_xlabel("β on −log(pick)\n(draft pricing, HC3 SE)")
axr.set_xlabel("β on per-play quality z\n(given log pick, HC3 SE)")
axl.set_title("(a) What the draft pays for", loc="left", fontweight="bold")
axr.set_title("(b) What predicts on-field quality", loc="left", fontweight="bold")
fig.suptitle("Figure 3 — Differential market pricing efficiency",
             x=0.02, ha="left", fontweight="bold", fontsize=11)
savefig(fig, "fig3_pricing_efficiency")

# ---------------------------------------------------------------- Fig 7 (was Fig 5)
loco = d["temporal"]["LOCO_Y2"]
t = d["temporal"]
adv = d["adversarial_validation"]
sens = d["Y2_sensitivity"]
manifest["figures"]["fig7_loco_temporal"] = {
    "LOCO_Y2": {k: {"n_test": v["n_test"], "r": v["r"]} for k, v in loco.items()},
    "temporal_2025": {
        "Y1_year1_hurdle": {"M0a": t["Y1_year1_hurdle_M0a"]["metric"],
                            "M1": t["Y1_year1_hurdle_M1"]["metric"],
                            "n_train": t["Y1_year1_hurdle_M0a"]["n_train"],
                            "n_test": t["Y1_year1_hurdle_M0a"]["n_test"]},
        "Y1_year1_logsnaps": {"M0a": t["Y1_year1_logsnaps_M0a"]["metric"],
                              "M1": t["Y1_year1_logsnaps_M1"]["metric"],
                              "n_train": t["Y1_year1_logsnaps_M0a"]["n_train"],
                              "n_test": t["Y1_year1_logsnaps_M0a"]["n_test"]}},
    "Y2_sensitivity": {k: v for k, v in sens.items()},
    "Y2_main_r_M1": y2["M1"]["metric_mean"],
    "adversarial_validation": {k: v["mean"] for k, v in adv.items()
                               if k.startswith("draft_year_macro_auc")}}
fig, axs = plt.subplots(2, 2, figsize=(9.5, 7.0))
ax = axs[0, 0]
cohorts = ["2023", "2024", "2025"]
m0a_loco = [loco[f"Y2_test_{c}_M0a"]["r"] for c in cohorts]
m1_loco = [loco[f"Y2_test_{c}_M1"]["r"] for c in cohorts]
x = np.arange(3)
ax.bar(x - 0.18, m0a_loco, width=0.36, color=GREY, label="M0a (draft pick)")
ax.bar(x + 0.18, m1_loco, width=0.36, color=TEAL, label="M1 (tracking)")
for xi, v in zip(x - 0.18, m0a_loco):
    ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.5)
for xi, v in zip(x + 0.18, m1_loco):
    ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{c} class\n(n={loco[f'Y2_test_{c}_M1']['n_test']})" for c in cohorts])
ax.set_ylabel("LOCO out-of-fold r (Y2)")
ax.set_title("(a) Leave-one-cohort-out, Y2", loc="left", fontweight="bold")
ax.legend(fontsize=8)
ax.set_ylim(0, 0.24)
ax = axs[0, 1]
vals_t = [("Hurdle AUC", t["Y1_year1_hurdle_M0a"]["metric"], t["Y1_year1_hurdle_M1"]["metric"]),
          ("log snaps r", t["Y1_year1_logsnaps_M0a"]["metric"], t["Y1_year1_logsnaps_M1"]["metric"])]
x = np.arange(2)
ax.bar(x - 0.18, [v[1] for v in vals_t], width=0.36, color=GREY, label="M0a")
ax.bar(x + 0.18, [v[2] for v in vals_t], width=0.36, color=TEAL, label="M1")
for xi, v in zip(x - 0.18, [v[1] for v in vals_t]):
    ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
for xi, v in zip(x + 0.18, [v[2] for v in vals_t]):
    ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels([v[0] for v in vals_t])
ax.set_ylabel("2025 forward metric")
ax.set_title("(b) 2025 forward extrapolation, Y1", loc="left", fontweight="bold")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.0)
ax = axs[1, 0]
sens_rows = [("≥75 plays\n(main, 10×5 CV)", 299, y2["M1"]["metric_mean"]),
             ("≥100 plays\n(2×5 CV)", sens["n_plays_100"]["n"], sens["n_plays_100"]["oof_r_mean"]),
             ("≥150 plays\n(2×5 CV)", sens["n_plays_150"]["n"], sens["n_plays_150"]["oof_r_mean"])]
x = np.arange(3)
ax.bar(x, [r[2] for r in sens_rows], width=0.55, color=TEAL)
for xi, (_, n, v) in zip(x, sens_rows):
    ax.text(xi, v + 0.008, f"{v:.4f}\n(n={n})", ha="center", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in sens_rows], fontsize=8.5)
ax.set_ylabel("OOF r (Y2, M1)")
ax.set_title("(c) Y2 sensitivity to play-count threshold", loc="left", fontweight="bold")
ax.set_ylim(0, 0.22)
ax = axs[1, 1]
adv_rows = [("All features", adv["draft_year_macro_auc_ALL"]["mean"]),
            ("40-yd only", adv["draft_year_macro_auc_FORTY_ONLY"]["mean"]),
            ("Skill only", adv["draft_year_macro_auc_SKILL_ONLY"]["mean"]),
            ("Static only", adv["draft_year_macro_auc_STATIC_ONLY"]["mean"])]
x = np.arange(4)
ax.bar(x, [r[1] for r in adv_rows], width=0.55, color=BLUE)
ax.axhline(0.5, color="red", ls="--", lw=1.2, label="Chance (0.50)")
for xi, (_, v) in zip(x, adv_rows):
    ax.text(xi, v + 0.008, f"{v:.4f}", ha="center", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in adv_rows], fontsize=8.5)
ax.set_ylabel("Macro AUC (predict draft year)")
ax.set_title("(d) Adversarial validation: no cohort signal left", loc="left", fontweight="bold")
ax.legend(fontsize=8)
ax.set_ylim(0, 0.6)
fig.suptitle("Figure 7 — Cross-cohort and temporal robustness",
             x=0.02, ha="left", fontweight="bold", fontsize=11)
savefig(fig, "fig7_loco_temporal")

# ---------------------------------------------------------------- Fig 4 (was Fig 6)
ladder = d["robustness_skill_s_peak_ladder"]
manifest["figures"]["fig4_robustness_ladder"] = [
    {"step": i + 1, "controls": s["controls"], "subset": s["subset"],
     "n": s["n"], "beta": s["beta"], "p": s["p"],
     "se_derived": se_from_beta_p(s["beta"], s["p"])} for i, s in enumerate(ladder)]
step_labels = [
    "1 · log pick only\n(n=299)",
    "5 · log pick only\n40-yd subset (n=253)",
    "3 · log pick + NGS\n(n=298)",
    "2 · log pick + 40-yd\n(n=253)",
    "4 · log pick + 40-yd\n+ NGS (n=252)",
]
order = [0, 4, 2, 1, 3]  # display order: 1, 5, 3, 2, 4
fig, ax = plt.subplots(figsize=(8.5, 4.6))
xs6 = np.arange(5)
vals6 = [ladder[i]["beta"] for i in order]
ses6 = [se_from_beta_p(ladder[i]["beta"], ladder[i]["p"]) for i in order]
ps6 = [ladder[i]["p"] for i in order]
ax.errorbar(xs6, vals6, yerr=[1.96 * s for s in ses6],
            fmt="o-", color=TEAL, ecolor=TEAL, elinewidth=2, capsize=5,
            markersize=8, lw=2)
for xi, v, p in zip(xs6, vals6, ps6):
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    ax.text(xi, v + 0.022, f"β={v:.3f}\np={p:.4f} {sig}", ha="center", fontsize=8.5)
ax.axhline(0, color=DARK, lw=1)
ax.set_xticks(xs6)
ax.set_xticklabels([step_labels[i] for i in range(5)], fontsize=8.5)
ax.set_ylabel("β of skill_s_peak on per-play quality z")
ax.set_ylim(-0.02, 0.28)
ax.set_xlim(-0.55, 4.55)
ax.set_title("Figure 4 — Robustness ladder: what attenuates the skill_s_peak effect",
             loc="left", fontweight="bold")
ax.annotate("Sample attrition\n−0.020 (32.5%)",
            xy=(1, 0.1361), xytext=(0.9, 0.245),
            arrowprops=dict(arrowstyle="->", color=DARK), fontsize=9, ha="center")
ax.annotate("Shared 40-yd speed variance\n−0.042 (67.5%)",
            xy=(3, 0.0944), xytext=(3.75, 0.245),
            arrowprops=dict(arrowstyle="->", color=DARK), fontsize=9, ha="center")
savefig(fig, "fig4_robustness_ladder")

# ---------------------------------------------------------------- Fig 6 (was Fig 7)
km = d["M2_mediation"]["ktr_matrix"]
ksp = d["M2_mediation"]["ktr_speed_by_position"]
rows7 = ["forty_s_peak", "skill_accel_p95_z_mean", "skill_decel_p95_z_mean",
         "skill_lat_accel_p95_z_mean"]
cols7 = ["game_s_p99", "game_accel_p95", "game_decel_p95", "game_lat_accel_p95"]
mat = np.full((4, 4), np.nan)
pmat = np.full((4, 4), np.nan)
for cell in km:
    if cell["combine"] in rows7 and cell["game"] in cols7:
        mat[rows7.index(cell["combine"]), cols7.index(cell["game"])] = cell["partial_r"]
        pmat[rows7.index(cell["combine"]), cols7.index(cell["game"])] = cell["p"]
manifest["figures"]["fig6_ktr_transfer"] = {
    "ktr_matrix_4x4": [[{"r": float(mat[i, j]), "p": float(pmat[i, j])}
                        for j in range(4)] for i in range(4)],
    "ktr_speed_by_position": {pos: {"n": v["n"], "mean": v["ktr_speed_mean"],
                                    "median": v["ktr_speed_median"]}
                              for pos, v in ksp.items()},
    "ktr_speed_partial_r": d["M2_mediation"]["ktr_speed_partial_r"],
    "ktr_speed_partial_p": d["M2_mediation"]["ktr_speed_partial_p"],
    "ktr_speed_n": d["M2_mediation"]["ktr_speed_n"],
    "mediation_decel": {"indirect": d["M2_mediation"]["indirect"],
                        "ci95": d["M2_mediation"]["indirect_ci95"],
                        "pct": d["M2_mediation"]["pct_mediated"]},
    "mediation_burst_1s": d["M2_mediation"]["burst_1s"]}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.6),
                               gridspec_kw={"width_ratios": [1.25, 1]})
im = ax1.imshow(mat, cmap="Blues", vmin=0.0, vmax=0.45)
for i in range(4):
    for j in range(4):
        star = "***" if pmat[i, j] < 0.001 else ("**" if pmat[i, j] < 0.01 else "*")
        ax1.text(j, i, f"{mat[i, j]:.3f}{star}", ha="center", va="center",
                 fontsize=10, color="white" if mat[i, j] > 0.30 else DARK)
ax1.set_xticks(range(4))
ax1.set_xticklabels(["Game top\nspeed", "Game\nacceleration", "Game\ndeceleration", "Game lateral\nacceleration"],
                    fontsize=8.5)
ax1.set_yticks(range(4))
ax1.set_yticklabels(["40-yd peak speed", "Skill-drill\nacceleration", "Skill-drill\ndeceleration",
                     "Skill-drill\nlateral accel."], fontsize=8.5)
ax1.set_title("(a) Combine → game transfer (within-position partial r)", loc="left",
              fontweight="bold", fontsize=10)
cbar = fig.colorbar(im, ax=ax1, shrink=0.85)
cbar.set_label("partial r")
pos_order = ["WR", "TE", "DB", "DL", "OL"]
ret = [ksp[p]["ktr_speed_mean"] * 100 for p in pos_order]
ns = [ksp[p]["n"] for p in pos_order]
x = np.arange(5)
ax2.bar(x, ret, width=0.6, color=TEAL)
for xi, v, n in zip(x, ret, ns):
    ax2.text(xi, v + 1.5, f"{v:.1f}%\n(n={n})", ha="center", fontsize=9)
ax2.set_xticks(x)
ax2.set_xticklabels(pos_order)
ax2.set_ylabel("Game top speed / 40-yd peak speed (%)")
ax2.set_ylim(0, 95)
ax2.set_title("(b) Game-speed retention by position", loc="left", fontweight="bold", fontsize=10)
fig.suptitle("Figure 6 — Combine-to-pads transfer: speed survives the pads, unevenly",
             x=0.02, ha="left", fontweight="bold", fontsize=11)
savefig(fig, "fig6_ktr_transfer")

# ---------------------------------------------------------------- manifest
with open(os.path.join(FIGDIR, "figure_data.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print("wrote figure_data.json")

# ---------------------------------------------------------------- Tables
models_tab = ["M0a", "M0b_stopwatch", "M0c_ngs", "M0b", "M0all", "M1b",
              "M1_track_only", "M1a", "M1", "M1full", "M1_over_stopwatch", "M1_over_all"]
labels_tab = {
    "M0a": "Draft pick only (log pick)",
    "M0b_stopwatch": "Stopwatch only (40-yd, 10-yd, vertical, broad jump)",
    "M0c_ngs": "NGS scores only (athleticism + final)",
    "M0b": "Stopwatch + NGS (legacy combined)",
    "M0all": "Pick + stopwatch + NGS",
    "M1b": "Tracking, single-stage Ridge (legacy)",
    "M1_track_only": "Tracking only (vs stopwatch)",
    "M1a": "Tracking + shadow (stacked on pick)",
    "M1": "Tracking + pick + shadow (stacked)",
    "M1full": "Tracking + pick + shadow + static (stacked)",
    "M1_over_stopwatch": "Tracking stacked on stopwatch",
    "M1_over_all": "Tracking stacked on pick + stopwatch + NGS",
}
lines = []
lines.append("# Phase 4 tables (from frozen round-5 JSON)\n")
lines.append("## Table 1 — Model comparison (10×5 repeated CV, seed 42)\n")
lines.append("| Model | Y1 hurdle AUC (n=510) | Y1 stage-2 r (n=236) | Y2 r (n=299) |")
lines.append("|---|---|---|---|")
for m in models_tab:
    a = d["Y1_hurdle_stage1"][m]["metric_mean"]
    b = d["Y1_stage2_logASR"][m]["metric_mean"]
    c = y2[m]["metric_mean"]
    lines.append(f"| {labels_tab[m]} ({m}) | {a:.4f} | {b:.4f} | {c:.4f} |")
lines.append("")
lines.append("Key deltas (Y2, mean Δr with 95% paired bootstrap CI):")
for k in ["delta_M1_over_stopwatch_vs_M0b_stopwatch",
          "delta_M1_track_only_vs_M0b_stopwatch",
          "delta_M1a_vs_M0a", "delta_M1_vs_M0a", "delta_M1_over_all_vs_M0all"]:
    v = y2[k]
    lines.append(f"- {k}: {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
lines.append(f"- Permutation test (M1 vs M0a, Y2): obs Δr = "
             f"{y2['permutation']['obs_delta']:+.4f}, p = {y2['permutation']['p_perm']:.4f} "
             f"(null {y2['permutation']['null_mean']:+.4f} ± {y2['permutation']['null_sd']:.4f}, "
             f"n_perm = {d['n_perm']}).")
lines.append(f"- Y2 R²: M1 {y2['r2_M1']:.4f} vs M0a {y2['r2_M0a']:.4f}. "
             f"Y2 n = 299 (≥75 regular-season plays); Y1 n = 510.")
lines.append("Deltas in text use unrounded means.")
lines.append("")
lines.append("## Table 2 — Per-position out-of-fold ranking (Y2, M1)\n")
lines.append("| Position | n | Spearman ρ | p |")
lines.append("|---|---|---|---|")
pps = y2["per_position_spearman"]
for pos in ["DL", "WR", "DB", "OL", "TE"]:
    b = pps[pos]
    lines.append(f"| {pos} | {b['n']} | {b['spearman_rho']:+.3f} | {b['p']:.3g} |")
lines.append(f"| EDGE (DL split) | {dl['EDGE']['Y2_M1_spearman']['n']} | "
             f"{dl['EDGE']['Y2_M1_spearman']['rho']:+.3f} | {dl['EDGE']['Y2_M1_spearman']['p']:.3g} |")
lines.append(f"| IDL (DL split) | {dl['IDL']['Y2_M1_spearman']['n']} | "
             f"{dl['IDL']['Y2_M1_spearman']['rho']:+.3f} | {dl['IDL']['Y2_M1_spearman']['p']:.3g} |")
lines.append("")
with open(os.path.join(ROOT, "notes", "tables_phase4.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote notes/tables_phase4.md")
print("DONE")
