#!/usr/bin/env python3
"""Audit every number in notes/writeup_draft.md against frozen sources.

Allowed values come from:
  (a) notes/phase3_results_round5_server.json — every float, rounded to
      2/3/4 decimals (with sign), and every int;
  (b) the frozen server md's shadow-calibration block (Pure/Full r and R2);
  (c) an explicit allowlist for derived/structural values, each with its
      source noted (differences of JSON values, percentages, years, counts,
      thresholds, section/figure numbers, runtime).
Any draft number not in the allowed set is printed and the script exits 1.
"""
import json
import math
import re
import sys

ROOT = "."
J = json.load(open("notes/phase3_results_round5_server.json"))
DRAFT = open("notes/writeup_draft.md").read()

allowed = set()

def add_num(x):
    if isinstance(x, bool):
        return
    if isinstance(x, int):
        allowed.add(str(x))
        allowed.add(f"{x:,}")
    elif isinstance(x, float):
        for nd in (2, 3, 4):
            allowed.add(f"{x:.{nd}f}")
            allowed.add(f"{x:+.{nd}f}")
        if x != 0 and (abs(x) < 0.001 or abs(x) >= 10000):
            allowed.add(f"{x:.2g}")
            allowed.add(f"{x:.1e}")

def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
    elif isinstance(o, (int, float)):
        add_num(o)

walk(J)

# (b) shadow calibration (frozen server md, combine_features_summary.json block)
for v in (0.42068022525442056, 0.17652730996847277, 0.4936569840534179,
          0.24359559685812104, 0.28359515507775596, 0.07941512297855102,
          0.4460276880310409, 0.19805768302552773):
    add_num(v)

# (c) explicit allowlist: (value, source)
ALLOW = {
    # derived differences / percentages (from JSON values)
    "0.020": "0.1562-0.1361 sample-attrition step",
    "0.042": "0.1361-0.0944 collinearity step",
    "32.5": "0.0201/0.0618 decomposition share",
    "67.5": "0.0417/0.0618 decomposition share",
    "3.3": "mediation pct -0.0332 x100",
    "7.8": "mediation pct 0.0779 x100",
    "81.5": "ktr WR 0.8151 x100",
    "75.5": "ktr TE 0.7547 x100",
    "73.3": "ktr DB 0.7329 x100",
    "62.4": "ktr DL 0.6243 x100",
    "51.7": "ktr OL 0.5170 x100",
    "100": "skill_s_peak 100% coverage; also 100% in text",
    "46": "299-253 players missing a 40-yd time",
    "0.11": "1/sqrt(80) sampling error, Gemini round-5 wording",
    "5.8": "5h46m22s full-pipeline runtime",
    "431,094": "combine_tracking PLAYER frames (eda_audit.md G2; features_combine filters entity_type==PLAYER)",
    "31": "31 three-year-stable drills (features_combine G4; eda_audit.md G4)",
    "333": "1000//3 permutation null draws (model_phase3 n_perm//n_repeats_perm=3)",
    "180": "shadow 3-cone n (frozen MD shadow block)",
    "201": "shadow short-shuttle n (frozen MD shadow block)",
    "200": "EB shrinkage kappa cap (frozen MD eb_kappa: DB m_epa, DL quick-press)",
    "17": "86/510 players missing a forty = 16.9% (combine_results.csv)",
    "60": "short-shuttle missing 309/510=60.6%, three-cone 330/510=64.7% (combine_results.csv)",
    # structural / narrative numbers
    "2023": "draft class year", "2024": "draft class year",
    "2025": "draft class year", "2027": "competition year",
    "2000": "word limit", "150": "Y1 hurdle snap threshold",
    "75": "Y2 play-count threshold",
    "463,189": "combine_tracking rows (data_profile/final_report)",
    "10": "10 Hz tracking; 10x5 CV",
    "14": "pre-registered kinematic features",
    "8": "Python modules",
    "2": "2-vCPU machine; 2,000 bootstraps written as 2,000",
    "1": "Figure/Table/section numbers; 1,000 permutations written as 1,000",
    "3": "section/figure numbers; 3-cone drill",
    "4": "section/figure numbers; four static tests; 4x4 matrix",
    "5": "section/figure numbers; five-step ladder",
    "6": "section/figure numbers",
    "7": "section/figure numbers; Figure 7",
    "9": "nine items total (figures+tables)",
    "28": "stable drills (JSON drill_normalization string)",
    "40": "40-yard dash distance",
    "3.1": "section number", "3.2": "section number", "3.3": "section number",
    "3.4": "section number", "3.5": "section number",
}
for v in ALLOW:
    allowed.add(v)
    allowed.add(f"+{v}" if not v.startswith(("+", "-")) else v)
    allowed.add(f"-{v}" if not v.startswith(("+", "-")) else v)

SUPER = {"⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
         "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}

# extra structural values
for v in ("40", "3.1", "3.2", "3.3", "3.4", "3.5", "0.5", "0.05", "0.01", "0.001"):
    allowed.add(v)

def normalise_sci(tok):
    m = re.fullmatch(r"(\d+\.?\d*)×10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", tok)
    if not m:
        return None
    mant, exp = m.group(1), "".join(SUPER[c] for c in m.group(2))
    return f"{float(mant):.1f}e{int(exp)}"

problems = []
checked = 0
# strip HTML comments and URLs (usernames/paths inside URLs are not values)
text = re.sub(r"<!--.*?-->", " ", DRAFT, flags=re.S)
text = re.sub(r"https?://\S+", " ", text)
# find number-like tokens: optional sign, digits with commas/decimals,
# optional % or scientific-notation suffix
for m in re.finditer(r"[−\-+]?\d[\d,]*(?:\.\d+)?(?:\s*×\s*10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)?%?", text):
    tok = m.group(0).strip()
    # skip numbers glued to a preceding letter via hyphen (stage-2, step-1, ...)
    if m.start() > 0 and text[m.start() - 1].isalpha():
        continue
    # skip numbers immediately followed by a letter (e.g. 10Hz handled below)
    if m.end() < len(text) and text[m.end()].isalpha() and not tok.endswith("%"):
        # '10Hz' style: the digits are a unit label, not a value
        continue
    tok = tok.replace("−", "-")
    if "×" in tok:
        sci = normalise_sci(tok.replace(" ", ""))
        if sci is None:
            problems.append((tok, "unparsable sci notation"))
            continue
        val = sci
        # compare against JSON floats in sci form
        checked += 1
        ok = False
        for a in allowed:
            if "e" in a:
                try:
                    if abs(float(a) - float(val)) <= abs(float(val)) * 0.06:
                        ok = True
                        break
                except ValueError:
                    pass
        if not ok:
            problems.append((tok, "sci value not in frozen JSON"))
        continue
    tok_clean = tok.rstrip("%")
    is_pct = tok.endswith("%")
    checked += 1
    if tok_clean in allowed:
        continue
    # plain integer with comma already in allowed; try float forms
    try:
        f = float(tok_clean.replace(",", ""))
        cands = {f"{f:.2f}", f"{f:.3f}", f"{f:.4f}", f"{f:+.2f}",
                 f"{f:+.3f}", f"{f:+.4f}", str(int(f)) if f == int(f) else None}
        cands.discard(None)
        if cands & allowed:
            continue
    except ValueError:
        pass
    # context for the report
    start = max(0, m.start() - 60)
    problems.append((tok, "..." + text[start:m.end() + 40].replace("\n", " ") + "..."))

print(f"numbers checked: {checked}, problems: {len(problems)}")
for tok, ctx in problems:
    print(f"  NOT ALLOWED: {tok}  :: {ctx}")
sys.exit(1 if problems else 0)
