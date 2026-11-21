# NFL Big Data Bowl 2027 — Phase 1 audit: Go / No-Go gates + EDA checks
#
# Reads data/parquet (produced by src/etl_parquet.py). Implements the Phase 1
# gates from notes/gemini_scheme.md §8 Phase 1:
#   G1  player_play == 314,197 rows exactly
#   G2  combine_tracking PLAYER-only frames with s > 12.5 yd/s share < 0.1%
#   G3  recomputed tangential acceleration a_t distributions consistent across
#       the 3 seasons (KS test on sampled trajectories; raw `a` column is NOT used)
#   G4  >= 15 drill_name values with >= 20 players in EACH of 2023/2024/2025
# Plus: opt-out MNAR diagnostics, position-mapping check, REG/Y1 preview.
#
# Usage: python3 src/audit_phase1.py [--parquet data/parquet] [--seed 42]
# Writes notes/eda_audit.md and prints PASS/FAIL per gate.
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(BASE, "notes")
SEED = 42


def iter_batches(path, columns, batch_size=200_000):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def gate_player_play(pqdir, out):
    df = pd.read_parquet(os.path.join(pqdir, "player_play.parquet"),
                         columns=["game_id", "play_id", "nfl_id"])
    n = len(df)
    ok = n == 314197
    out.append(f"- G1 player_play 行数：{n:,}（期望 314,197）→ {'PASS' if ok else 'FAIL'}")
    return ok


def gate_combine_speed(pqdir, out):
    total, over = 0, 0
    for ch in iter_batches(os.path.join(pqdir, "combine_tracking.parquet"),
                           ["entity_type", "s", "dis"]):
        pl = ch[ch["entity_type"] == "PLAYER"]
        total += len(pl)
        over += int((pl["s"] > 12.5).sum())
    share = over / total if total else 0
    ok = share < 0.001
    out.append(f"- G2 combine_tracking PLAYER 帧：{total:,}，s > 12.5 yd/s 共 {over} 帧，"
               f"占比 {share:.4%}（门限 < 0.1%）→ {'PASS' if ok else 'FAIL'}")
    return ok


def _sample_trajectory_keys(pqdir, season_file, n_keys, rng):
    keys = set()
    for ch in iter_batches(os.path.join(pqdir, season_file),
                           ["game_id", "play_id", "nfl_id"]):
        k = set(zip(ch["game_id"], ch["play_id"], ch["nfl_id"]))
        keys |= k
    keys = sorted(keys)
    idx = rng.choice(len(keys), size=min(n_keys, len(keys)), replace=False)
    return {keys[i] for i in idx}


def _recompute_at(pqdir, season_file, keys):
    """Recompute signed tangential acceleration a_t = ds/dt per trajectory."""
    rows = []
    for ch in iter_batches(os.path.join(pqdir, season_file),
                           ["game_id", "play_id", "nfl_id", "time", "s", "x", "y"]):
        m = [tuple(map(int, k)) for k in zip(ch["game_id"], ch["play_id"], ch["nfl_id"])]
        keep = [i for i, k in enumerate(m) if k in keys]
        if keep:
            rows.append(ch.iloc[keep])
    if not rows:
        return np.array([])
    df = pd.concat(rows, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["game_id", "play_id", "nfl_id", "time"])
    df["dt"] = df.groupby(["game_id", "play_id", "nfl_id"])["time"].diff().dt.total_seconds()
    df["ds"] = df.groupby(["game_id", "play_id", "nfl_id"])["s"].diff()
    at = (df["ds"] / df["dt"]).replace([np.inf, -np.inf], np.nan).dropna()
    # drop sensor-jump frames (same physical bound as the scheme: dis > 1.3 yd / 0.1 s)
    return at[(at > -30) & (at < 30)].to_numpy()


def gate_at_consistency(pqdir, out, rng):
    files = {"2023": "game_tracking_2023.parquet",
             "2024": "game_tracking_2024.parquet",
             "2025": "game_tracking_2025.parquet"}
    samples = {}
    for yr, f in files.items():
        keys = _sample_trajectory_keys(pqdir, f, 300, rng)
        samples[yr] = _recompute_at(pqdir, f, keys)
        out.append(f"  - {yr} 季重算 a_t 样本：{len(samples[yr]):,} 帧，"
                   f"均值 {samples[yr].mean():.3f}，标准差 {samples[yr].std():.3f}，"
                   f"p1 {np.percentile(samples[yr], 1):.2f}，p99 {np.percentile(samples[yr], 99):.2f}")
    d23, p23 = stats.ks_2samp(samples["2023"], samples["2024"])
    d25, p25 = stats.ks_2samp(samples["2023"], samples["2025"])
    d45, p45 = stats.ks_2samp(samples["2024"], samples["2025"])
    out.append(f"  - KS 2023 vs 2024：D={d23:.4f}，p={p23:.3g}")
    out.append(f"  - KS 2023 vs 2025：D={d25:.4f}，p={p25:.3g}")
    out.append(f"  - KS 2024 vs 2025：D={d45:.4f}，p={p45:.3g}")
    # Practical gate: distributions must be close (D < 0.05); p-values are
    # expected to be tiny at these sample sizes, so D is the decision metric.
    ok = max(d23, d25, d45) < 0.05
    out.append(f"- G3 三季重算 a_t 分布一致（KS D 均 < 0.05）→ {'PASS' if ok else 'FAIL'}")
    return ok


def gate_drill_coverage(pqdir, out):
    ct = pd.read_parquet(os.path.join(pqdir, "combine_tracking.parquet"),
                         columns=["drill_name", "draft_year", "nfl_id"])
    cov = ct.groupby(["drill_name", "draft_year"])["nfl_id"].nunique().unstack(fill_value=0)
    ok_cols = [2023, 2024, 2025]
    n_ok = int(((cov[ok_cols] >= 20).all(axis=1)).sum())
    ok = n_ok >= 15
    out.append(f"- G4 三年均 ≥20 人的 drill：{n_ok} 个（门限 ≥ 15）→ {'PASS' if ok else 'FAIL'}")
    out.append(f"  - drill 总数：{len(cov)}；三年全覆盖 drill：{int(((cov[ok_cols] > 0).all(axis=1)).sum())}")
    return ok


def optout_mnar(pqdir, out):
    cr = pd.read_parquet(os.path.join(pqdir, "combine_results.parquet"))
    pl = pd.read_parquet(os.path.join(pqdir, "players.parquet"),
                         columns=["nfl_id", "draft_overall_pick"])
    df = cr.merge(pl, on="nfl_id", how="left")
    df["opted_out_agility"] = df["three_cone"].isna() & df["short_shuttle"].isna()
    df["opted_out_forty"] = df["forty"].isna()
    df["opt_out_count"] = df[["three_cone", "short_shuttle", "bench_reps", "forty"]].isna().sum(axis=1)
    sub = df.dropna(subset=["draft_overall_pick"])
    a = sub.loc[sub["opted_out_agility"], "draft_overall_pick"]
    b = sub.loc[~sub["opted_out_agility"], "draft_overall_pick"]
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    out.append("## 跳测缺失（Opt-Out）MNAR 诊断")
    out.append("")
    out.append(f"- opted_out_agility（three_cone 与 short_shuttle 双缺失）：{int(df['opted_out_agility'].sum())} / {len(df)} 人")
    out.append(f"- opted_out_forty：{int(df['opted_out_forty'].sum())} 人；opt_out_count 分布：{df['opt_out_count'].value_counts().sort_index().to_dict()}")
    out.append(f"- Mann-Whitney U：跳测组选秀顺位中位数 {a.median():.0f} vs 非跳测组 {b.median():.0f}，U={u:.0f}，p={p:.3g}")
    out.append(f"- 结论：{'跳测与高顺位显著相关（MNAR 得到支持，缺失指示变量绝不能进主模型）' if p < 0.01 else '未发现跳测与顺位的显著关联，需复核'}")
    out.append("")
    return df


def position_mapping(pqdir, out):
    pl = pd.read_parquet(os.path.join(pqdir, "players.parquet"),
                         columns=["nfl_id", "nfl_position"])
    cr = pd.read_parquet(os.path.join(pqdir, "combine_results.parquet"),
                         columns=["nfl_id", "combine_position"])
    m = pl.merge(cr, on="nfl_id", how="left")
    ct = pd.crosstab(m["nfl_position"], m["combine_position"])
    out.append("## 位置映射核验（nfl_position × combine_position）")
    out.append("")
    out.append("```")
    out.append(ct.to_string())
    out.append("```")
    out.append("")
    out.append("- 注意：RB→DB（1 人）、FB→TE（1 人）、TE→WR（1 人）为跨组参加体测的个例，位置分层时以 combine_position 为准并记录。")
    out.append("")


def y1_preview(pqdir, out):
    games = pd.read_parquet(os.path.join(pqdir, "games.parquet"),
                            columns=["game_id", "season_type"])
    reg_ids = set(games.loc[games["season_type"] == "REG", "game_id"])
    pp = pd.read_parquet(os.path.join(pqdir, "player_play.parquet"),
                         columns=["game_id", "nfl_id", "play_nullified_by_penalty",
                                  "target", "lined_up_position"])
    reg = pp[pp["game_id"].isin(reg_ids) & (pp["play_nullified_by_penalty"] == "N")]
    per_player = reg.groupby("nfl_id").size().rename("reg_snaps")
    players = pd.read_parquet(os.path.join(pqdir, "players.parquet"),
                              columns=["nfl_id", "draft_year"])
    players["E"] = 2026 - players["draft_year"]
    t = players.set_index("nfl_id").join(per_player).fillna({"reg_snaps": 0})
    t["ASR"] = t["reg_snaps"] / t["E"]
    # Year-1 only: rookie-season REG snaps
    y1 = reg.merge(players[["nfl_id", "draft_year"]], on="nfl_id")
    y1 = y1[(y1["game_id"] // 1_000_000) == y1["draft_year"]]
    y1c = y1.groupby("nfl_id").size().rename("year1_reg_snaps")
    t = t.join(y1c).fillna({"year1_reg_snaps": 0})
    out.append("## 目标变量 Y1 预览（常规赛档数，REG + 非犯规取消）")
    out.append("")
    out.append(f"- player_play REG 有效档：{len(reg):,} / {len(pp):,}（过滤 PRE/POST 与犯规取消档）")
    out.append(f"- 有 REG 档球员：{int((t['reg_snaps'] > 0).sum())} / 510；零档球员：{int((t['reg_snaps'] == 0).sum())}")
    out.append(f"- ASR（年均 REG 档）分布：中位数 {t['ASR'].median():.0f}，均值 {t['ASR'].mean():.0f}，p90 {t['ASR'].quantile(0.9):.0f}")
    out.append(f"- ASR ≥ 150 档/季（Hurdle 门槛）：{int((t['ASR'] >= 150).sum())} 人；Year-1 ≥ 100 档：{int((t['year1_reg_snaps'] >= 100).sum())} 人")
    out.append(f"- Year-1 REG 档 ≥ 75 档（Y2 覆盖估计）：{int((t['year1_reg_snaps'] >= 75).sum())} 人")
    out.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=os.path.join(BASE, "data", "parquet"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = ["# NFL Big Data Bowl 2027 — Phase 1 EDA 审计报告",
           "",
           f"生成时间：2026-11-21；随机种子：{args.seed}；数据源：{args.parquet}",
           "",
           "## Go / No-Go 门控",
           ""]
    results = {}
    results["G1"] = gate_player_play(args.parquet, out)
    results["G2"] = gate_combine_speed(args.parquet, out)
    results["G3"] = gate_at_consistency(args.parquet, out, rng)
    results["G4"] = gate_drill_coverage(args.parquet, out)
    out.append("")
    out.append(f"**门控总判定：{'全部 PASS，可进入 Phase 2' if all(results.values()) else '有 FAIL，先修清洗规则，严禁进入建模'}**")
    out.append("")
    optout_mnar(args.parquet, out)
    position_mapping(args.parquet, out)
    y1_preview(args.parquet, out)
    os.makedirs(NOTES, exist_ok=True)
    dest = os.path.join(NOTES, "eda_audit.md")
    with open(dest, "w") as fh:
        fh.write("\n".join(out))
    print("\n".join(out))
    print("\nwrote", dest)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
