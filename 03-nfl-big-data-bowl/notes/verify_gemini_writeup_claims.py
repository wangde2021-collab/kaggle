#!/usr/bin/env python3
"""Verify Gemini writeup-review claims against the frozen JSON + code.

Run: python3 notes/verify_gemini_writeup_claims.py
Prints each claim with the frozen evidence (JSON value / code line).
Does not modify anything.
"""
import json
import re

J = json.load(open("notes/phase3_results_round5_server.json"))
MD = open("notes/phase3_results_round5_server.md").read()


def show(title, value):
    print(f"\n### {title}\n  {value}")


# 1. Y1 ASR definition
tgt = open("src/targets.py").read()
m = re.search(r".{0,120}ASR.{0,200}", tgt, re.S)
show("Y1 ASR (targets.py)", re.sub(r"\s+", " ", m.group(0))[:400] if m else "NOT FOUND")
for pat in ["150", "n_pass", "236"]:
    hits = [l.strip() for l in tgt.splitlines() if pat in l][:3]
    print(f"  targets.py '{pat}':", hits)

# 2. Combine frames: raw vs PLAYER-only
etl = open("src/etl_parquet.py").read()
print("\n### combine_tracking rows (etl EXPECTED_ROWS)")
print(" ", re.search(r'"combine_tracking\.csv":\s*\d+', etl).group(0))
print("  PLAYER filter in etl:", "entity_type" in etl,
      [l.strip() for l in etl.splitlines() if "entity_type" in l][:3])
import os
if os.path.exists("data/parquet/manifest.json"):
    man = json.load(open("data/parquet/manifest.json"))
    print("  manifest combine_tracking rows:", man.get("combine_tracking.csv"))
else:
    print("  local data/parquet/manifest.json absent (parquet lives on server);",
          "EDA audit says PLAYER frames = 431,094 (notes/eda_audit.md G2)")

# 3. Permutation mechanics
mod = open("src/model_phase3.py").read()
for pat in ["n_repeats_perm", "n_perm"]:
    hits = [l.strip() for l in mod.splitlines() if pat in l and ("=" in l or "range" in l)][:6]
    print(f"\n### model_phase3.py '{pat}'")
    for h in hits:
        print("  ", h)
print("  permutation p (JSON):", J["Y2_quality"]["permutation"])

# 4. Y1 M1a deltas (Abstract mix check)
y1h, y1s = J["Y1_hurdle_stage1"], J["Y1_stage2_logASR"]
show("Y1 hurdle delta M1 vs M0a", y1h["delta_M1_vs_M0a"])
show("Y1 hurdle delta M1a vs M0a", y1h["delta_M1a_vs_M0a"])
show("Y1 stage2 delta M1 vs M0a", y1s["delta_M1_vs_M0a"])
show("Y1 stage2 delta M1a vs M0a", y1s["delta_M1a_vs_M0a"])

# 5. DL EDGE/IDL split
print("\n### DL EDGE/IDL split (JSON)")
print(json.dumps(J.get("dl_edge_idl_split", "MISSING"), indent=2)[:1500])
pp = J["Y2_quality"]["per_position_spearman"]
print("  per-position DL n:", pp.get("DL"), "| EDGE/IDL in per-position?",
      [k for k in pp.keys()])

# 6. Shadow calibration n
print("\n### Shadow block")
sh = J.get("shadow_agility_proxy", J.get("shadow", {}))
print(json.dumps(sh, indent=2)[:1200])

# 7. Missing forty counts
print("\n### missing forty")
for k in ["missing_forty_n", "n_missing_forty", "forty_missing"]:
    if k in J:
        print(" ", k, "=", J[k])
print("  MD mentions 86 missing forty:", MD.count("86"))
print("  MD 'missing a 40':", [l.strip() for l in MD.splitlines() if "missing a 40" in l][:3])

# 8. fdr_Y2_full pricing betas
f = J["fdr_Y2_full"]
print("\n### fdr_Y2_full (pricing-relevant)")
for feat in ["neg_forty", "vertical", "broad_jump", "forty_s_peak", "skill_s_peak",
             "skill_accel_p95", "skill_lat_accel_p95"]:
    print(" ", feat, "->", f.get(feat))
print("  draft_pricing skill_accel_p95:", J["draft_pricing"].get("skill_accel_p95"))
print("  draft_pricing skill_lat_accel_p95:", J["draft_pricing"].get("skill_lat_accel_p95"))

# 9. M3 LightGBM
print("\n### M3 LightGBM")
print(json.dumps(J["M3_lightgbm"], indent=2)[:800])

# 10. M1b vs M0b Y2
show("Y2 delta M1b vs M0b", J["Y2_quality"]["delta_M1b_vs_M0b"])

# 11. Temporal / LOCO / sensitivity n
print("\n### temporal")
print(json.dumps(J["temporal"], indent=2)[:1500])
print("\n### Y2 sensitivity")
print(json.dumps(J["Y2_sensitivity"], indent=2)[:600])

# 12. Fig 4 n ranges: pricing + quality sample sizes
dp = J["draft_pricing"]
ns = sorted({v.get("n") for v in dp.values() if isinstance(v, dict) and "n" in v})
print("\n### draft_pricing n range:", ns)
fy = J["fdr_Y2_full"]
ns2 = sorted({v.get("n") for v in fy.values() if isinstance(v, dict) and "n" in v})
print("### fdr_Y2_full n range:", ns2)

# 13. KTR n
print("\n### KTR")
ktr = J.get("ktr_transfer", J.get("ktr", {}))
print(json.dumps(ktr, indent=2)[:1200])

# 14. EB shrinkage kappa cap
print("\n### kappa cap in MD:")
for l in MD.splitlines():
    if "kappa" in l.lower() or "200" in l and "cap" in l.lower():
        print("  ", l.strip()[:200])
