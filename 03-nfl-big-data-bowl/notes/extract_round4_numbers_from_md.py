#!/usr/bin/env python3
"""Extract round-4 draft numbers from the frozen server results MD file.

Iron rule (REFLECTIONS.md): numbers in a Gemini draft must be generated
by script from the frozen results of the current round -- never
hand-copied, never reused from another round (smoke / earlier server run).

The round-4 browser task returned no downloadable phase3_results.json,
so the frozen source is notes/phase3_results_round4_server.md (the
server's phase3_results.md text transcribed verbatim from the browser
report). This script parses THAT file only.

Usage:
    python3 extract_round4_numbers_from_md.py <phase3_results_round4_server.md>
"""
import json
import re
import sys

MODELS = ["M0a", "M0b_stopwatch", "M0c_ngs", "M0b", "M0all", "M1", "M1a",
          "M1b", "M1_track_only", "M1_over_stopwatch", "M1_over_all",
          "M1full"]


def parse_model_block(text, tag):
    sec = re.search(r"## " + re.escape(tag) + r"\n(.*?)(?=\n## |\Z)",
                    text, re.S)
    assert sec, f"section {tag} not found"
    body = sec.group(1)
    out = {}
    for k in MODELS:
        m = re.search(r"^- " + re.escape(k) + r"：([\-0-9.]+) ± ([0-9.]+)",
                      body, re.M)
        if m:
            out[k] = (float(m.group(1)), float(m.group(2)))
    for dk in ("delta_M1_vs_M0a", "delta_M1b_vs_M0b", "delta_M1a_vs_M0a",
               "delta_M1_over_stopwatch_vs_M0b_stopwatch",
               "delta_M1_track_only_vs_M0b_stopwatch",
               "delta_M1_over_all_vs_M0all"):
        m = re.search(r"^- " + re.escape(dk) + r"：([+\-0-9.]+)，95% CI "
                      r"\[([+\-0-9.]+), ([+\-0-9.]+)\]", body, re.M)
        if m:
            out[dk] = (float(m.group(1)), float(m.group(2)),
                       float(m.group(3)))
    m = re.search(r"观测 Δ=([+\-0-9.]+)，p_perm=([0-9.]+)", body)
    if m:
        out["permutation"] = (float(m.group(1)), float(m.group(2)))
    return out


def parse_json_block(text, header):
    """Extract the JSON object following `header` by brace counting
    (the md's pasted JSON has extra trailing braces, so a non-greedy
    regex stops too early)."""
    start = text.find(header)
    assert start >= 0, f"header {header} not found"
    i = text.find("{", start)
    assert i >= 0, f"no JSON object after {header}"
    depth = 0
    in_str = False
    esc = False
    j = i
    while j < len(text):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[i:j + 1])
        j += 1
    raise ValueError(f"unbalanced JSON after {header}")


def r4(x):
    return round(float(x), 4)


def main(path):
    text = open(path).read()
    assert "WITHIN-YEAR harmonization ON" in text, (
        "within-year flag not found in file -- wrong round?")
    print(f"# Round-4 numbers (parsed from {path})")
    for tag in ("Y1_hurdle_stage1", "Y1_stage2_logASR", "Y2_quality",
                "Y1_year1_hurdle", "Y1_year1_logsnaps"):
        blk = parse_model_block(text, tag)
        print(f"[{tag}]")
        for k in MODELS:
            if k in blk:
                print(f"  {k}: {r4(blk[k][0])} +- {r4(blk[k][1])}")
        for dk in ("delta_M1_vs_M0a", "delta_M1b_vs_M0b",
                   "delta_M1a_vs_M0a",
                   "delta_M1_over_stopwatch_vs_M0b_stopwatch",
                   "delta_M1_track_only_vs_M0b_stopwatch",
                   "delta_M1_over_all_vs_M0all"):
            if dk in blk:
                v = blk[dk]
                print(f"  {dk}: {r4(v[0])} CI [{r4(v[1])}, {r4(v[2])}]")
        if "permutation" in blk:
            print(f"  permutation: obs {r4(blk['permutation'][0])} "
                  f"p={r4(blk['permutation'][1])}")
        print()
    t = parse_json_block(text, "## 时序外推与 LOCO")
    print("[temporal / LOCO]")
    for k in ("Y1_year1_hurdle_M0a", "Y1_year1_hurdle_M1",
              "Y1_year1_logsnaps_M0a", "Y1_year1_logsnaps_M1"):
        print(f"  {k}: {r4(t[k]['metric'])} "
              f"(n_train {t[k]['n_train']}, n_test {t[k]['n_test']})")
    for k, v in t["LOCO_Y2"].items():
        print(f"  LOCO {k}: r={r4(v['r'])} n={v['n_test']}")
    print()
    a = parse_json_block(text, "## 对抗验证（特征预测 draft_year，应 ≤0.55）")
    print("[adversarial_validation]")
    for k in ("draft_year_macro_auc_ALL", "draft_year_macro_auc_FORTY_ONLY",
              "draft_year_macro_auc_SKILL_ONLY",
              "draft_year_macro_auc_STATIC_ONLY"):
        print(f"  {k}: {r4(a[k]['mean'])} (std {r4(a[k]['std'])})")
    for yr, v in sorted(a["forty_s_peak_by_year"].items()):
        print(f"  forty_s_peak {yr}: all {r4(v['mean_all'])} "
              f"(n={v['n_all']}) WR {r4(v['mean_WR'])} (n={v['n_WR']})")
    print()
    print("[robustness_skill_s_peak_ladder]")
    sec = re.search(r"## 稳健性阶梯：skill_s_peak_z_mean.*?\n(.*?)(?=\n## |\Z)",
                    text, re.S)
    for line in sec.group(1).strip().splitlines():
        m = re.match(r"- (.+)：β=([+\-0-9.]+)，p=([0-9.]+)，n=(\d+)", line)
        if m:
            print(f"  {m.group(1)}: beta={r4(m.group(2))} "
                  f"p={r4(m.group(3))} n={m.group(4)}")
    print()
    print("[draft_pricing key betas]")
    sec = re.search(r"## 选秀定价回归.*?\n(.*?)(?=\n## |\Z)", text, re.S)
    for line in sec.group(1).strip().splitlines():
        m = re.match(r"- ([^：]+)：β=([+\-0-9.]+)，p=([0-9.e\-+]+)，q=([0-9.]+)，n=(\d+)",
                     line)
        if m:
            print(f"  {m.group(1)}: beta={r4(m.group(2))} "
                  f"p={m.group(3)} q={m.group(4)} n={m.group(5)}")
    print()
    print("[fdr_Y2 main family]")
    sec = re.search(r"## BH-FDR 单特征检验.*?\n(.*?)(?=\n## |\Z)", text, re.S)
    for line in sec.group(1).strip().splitlines():
        m = re.match(r"- ([^：]+)：β=([+\-0-9.]+)，p=([0-9.e\-+]+)，q=([0-9.]+)，n=(\d+)",
                     line)
        if m:
            print(f"  {m.group(1)}: beta={r4(m.group(2))} "
                  f"p={m.group(3)} q={m.group(4)} n={m.group(5)}")
    print()
    m3 = re.search(r'"Y2_r_mean": ([0-9.]+),\s*"Y1_hurdle_auc_mean": ([0-9.]+)',
                   text)
    print(f"[M3_lightgbm] Y2_r={r4(m3.group(1))} "
          f"Y1_hurdle_auc={r4(m3.group(2))}")
    m2 = re.search(r'"indirect": ([+\-0-9.]+),\s*"indirect_ci95": \[\s*'
                  r'([+\-0-9.]+),\s*([+\-0-9.]+)', text)
    print(f"[M2_mediation] indirect={r4(m2.group(1))} "
          f"CI [{r4(m2.group(2))}, {r4(m2.group(3))}]")
    mb = re.search(r'"burst_1s": \{\s*"n": \d+,\s*"a_path": [0-9.]+,'
                  r'\s*"b_path": [0-9.]+,\s*"indirect": ([+\-0-9.]+),'
                  r'\s*"indirect_ci95": \[\s*([+\-0-9.]+),\s*([+\-0-9.]+)',
                  text)
    print(f"  burst_1s indirect={r4(mb.group(1))} "
          f"CI [{r4(mb.group(2))}, {r4(mb.group(3))}]")
    print(f"[ktr_speed_mean] {r4(re.search(chr(34) + 'ktr_speed_mean' + chr(34) + r': ([0-9.]+)', text).group(1))}")
    print("[DL EDGE/IDL]")
    m2block = parse_json_block(text, "## M3 LightGBM / M2 中介 / Y2 敏感性")
    dl = m2block["dl_edge_idl_split"]
    for grp in ("EDGE", "IDL"):
        g = dl[grp]
        print(f"  {grp} n={g['n']} decel r="
              f"{r4(g['skill_decel_p95_z_mean~game_decel_p95']['partial_r'])} "
              f"p={g['skill_decel_p95_z_mean~game_decel_p95']['p']:.3g} "
              f"Y2_spearman rho={r4(g['Y2_M1_spearman']['rho'])} "
              f"p={g['Y2_M1_spearman']['p']:.3g} "
              f"n={g['Y2_M1_spearman']['n']}")


if __name__ == "__main__":
    main(sys.argv[1])
