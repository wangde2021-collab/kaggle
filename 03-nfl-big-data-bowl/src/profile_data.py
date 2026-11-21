"""NFL Big Data Bowl 2027 - data profiling (local, chunked for big files).

Writes notes/data_profile.md. Small files are read fully; tracking files
(3.6M-11.2M rows) are read in chunks to fit the 6GB cloud VM.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "notes", "data_profile.md")

SMALL = ["players.csv", "combine_results.csv", "games.csv",
         "player_career_successes.csv"]
MEDIUM = ["player_play.csv", "combine_tracking.csv"]
BIG = ["game_tracking_2023.csv", "game_tracking_2024.csv", "game_tracking_2025.csv"]

lines = []
def w(s=""):
    lines.append(s)

def fmt_dtype(s):
    return str(s.dtype)

def profile_small(path, name):
    df = pd.read_csv(path)
    w(f"## {name}")
    w("")
    w(f"- 行数：{len(df):,}（不含表头）；列数：{df.shape[1]}；文件大小：{os.path.getsize(path)/1024/1024:.2f} MB")
    w("")
    w("| 列名 | 类型 | 缺失数 | 缺失率 | 唯一值数 | 样例/备注 |")
    w("|---|---|---|---|---|---|")
    for c in df.columns:
        miss = int(df[c].isna().sum())
        rate = miss / len(df)
        nun = int(df[c].nunique(dropna=True))
        sample = df[c].dropna().iloc[0] if len(df[c].dropna()) else None
        w(f"| {c} | {fmt_dtype(df[c])} | {miss:,} | {rate:.1%} | {nun:,} | {sample} |")
    w("")
    w("样例行（前 3 行）：")
    w("```")
    w(df.head(3).to_string())
    w("```")
    w("")
    return df

def profile_chunked(path, name, chunksize=500_000, extra_cols=()):
    reader = pd.read_csv(path, chunksize=chunksize)
    total = 0
    miss = None
    cols = None
    dtypes = None
    head = None
    num_sum = {}
    num_min = {}
    num_max = {}
    num_n = {}
    cat_counts = {}
    distinct = {c: set() for c in extra_cols}
    for i, ch in enumerate(reader):
        if i == 0:
            cols = list(ch.columns)
            dtypes = {c: str(ch[c].dtype) for c in cols}
            head = ch.head(5)
            miss = ch.isna().sum()
            for c in extra_cols:
                distinct[c].update(ch[c].dropna().unique().tolist())
        else:
            miss = miss.add(ch.isna().sum(), fill_value=0)
            for c in extra_cols:
                distinct[c].update(ch[c].dropna().unique().tolist())
        total += len(ch)
        for c in ch.columns:
            if pd.api.types.is_numeric_dtype(ch[c]):
                s = ch[c].dropna()
                if len(s):
                    num_sum[c] = num_sum.get(c, 0.0) + float(s.sum())
                    num_n[c] = num_n.get(c, 0) + len(s)
                    num_min[c] = min(num_min.get(c, np.inf), float(s.min()))
                    num_max[c] = max(num_max.get(c, -np.inf), float(s.max()))
        for c in ("entity_type", "drill_type", "event"):
            if c in ch.columns:
                vc = ch[c].value_counts(dropna=False)
                cat_counts[c] = cat_counts.get(c, pd.Series(dtype=int)).add(vc, fill_value=0)
    w(f"## {name}")
    w("")
    w(f"- 行数：{total:,}（不含表头）；列数：{len(cols)}；文件大小：{os.path.getsize(path)/1024/1024:.2f} MB（分块读取，每块 {chunksize:,} 行）")
    w("")
    w("| 列名 | 类型 | 缺失数 | 缺失率 | 均值 | 最小值 | 最大值 |")
    w("|---|---|---|---|---|---|---|")
    for c in cols:
        m = int(miss[c])
        rate = m / total
        if c in num_sum and num_n.get(c, 0):
            mean = num_sum[c] / num_n[c]
            w(f"| {c} | {dtypes[c]} | {m:,} | {rate:.1%} | {mean:.4f} | {num_min[c]:.4f} | {num_max[c]:.4f} |")
        else:
            w(f"| {c} | {dtypes[c]} | {m:,} | {rate:.1%} | — | — | — |")
    w("")
    for c, vc in cat_counts.items():
        w(f"{c} 分布（Top 10）：")
        w("```")
        w(vc.sort_values(ascending=False).head(10).to_string())
        w("```")
        w("")
    for c, s in distinct.items():
        w(f"- {c} 唯一值数：{len(s):,}")
    w("")
    w("样例行（前 5 行）：")
    w("```")
    w(head.to_string())
    w("```")
    w("")
    return total

w("# NFL Big Data Bowl 2027 — 数据画像")
w("")
w("生成时间：2026-11-21（北京时间）；工具：本地云端 VM（pandas 2.1.4 / numpy 1.26.4），大文件分块读取。")
w("数据来源：Kaggle CLI 下载 nfl-big-data-bowl-2027，9 个 CSV，共 2.31 GB（压缩包 489 MB）。")
w("")

dfs = {}
for f in SMALL:
    dfs[f] = profile_small(os.path.join(DATA, f), f)
for f in MEDIUM:
    dfs[f] = pd.read_csv(os.path.join(DATA, f))
    df = dfs[f]
    w(f"## {f}")
    w("")
    w(f"- 行数：{len(df):,}（不含表头）；列数：{df.shape[1]}；文件大小：{os.path.getsize(os.path.join(DATA, f))/1024/1024:.2f} MB")
    w("")
    w("| 列名 | 类型 | 缺失数 | 缺失率 | 唯一值数 | 均值 | 最小值 | 最大值 |")
    w("|---|---|---|---|---|---|---|---|")
    for c in df.columns:
        miss = int(df[c].isna().sum())
        rate = miss / len(df)
        nun = int(df[c].nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            w(f"| {c} | {fmt_dtype(df[c])} | {miss:,} | {rate:.1%} | {nun:,} | {df[c].mean():.4f} | {df[c].min():.4f} | {df[c].max():.4f} |")
        else:
            w(f"| {c} | {fmt_dtype(df[c])} | {miss:,} | {rate:.1%} | {nun:,} | — | — | — |")
    w("")
    w("样例行（前 3 行）：")
    w("```")
    w(df.head(3).to_string())
    w("```")
    w("")
    del df

for f in BIG:
    profile_chunked(os.path.join(DATA, f), f, extra_cols=["nfl_id", "game_id", "play_id"])

# ---- 关键分布与跨表链接检查 ----
w("## 关键分布与跨表链接检查")
w("")
players = dfs["players.csv"]
cr = dfs["combine_results.csv"]
cs = dfs["player_career_successes.csv"]
games = dfs["games.csv"]
w("### players.csv")
w("```")
w(f"draft_year 分布：\n{players['draft_year'].value_counts().sort_index().to_string()}")
w("")
w(f"nfl_position 分布：\n{players['nfl_position'].value_counts().to_string()}")
w("")
w(f"draft_round 缺失（UDFA）数：{int(players['draft_round'].isna().sum())} / {len(players)}")
w(f"college 前 10：\n{players['college_name'].value_counts().head(10).to_string()}")
w("```")
w("")
w("### combine_results.csv")
w("```")
w(f"三项 drill 缺失（opt-out）数：three_cone={int(cr['three_cone'].isna().sum())}, short_shuttle={int(cr['short_shuttle'].isna().sum())}, bench_reps={int(cr['bench_reps'].isna().sum())} / {len(cr)}")
w(f"forty 描述统计：\n{cr['forty'].describe().to_string()}")
w(f"ngs_final_score 描述统计：\n{cr['ngs_final_score'].describe().to_string()}")
w("```")
w("")
w("### player_career_successes.csv（目标变量候选）")
w("```")
w(f"{cs.describe().to_string()}")
w("")
w(f"pro_bowl_original_ballot > 0 人数：{int((cs['pro_bowl_original_ballot'] > 0).sum())}")
w(f"ap_all_pro_1st_team > 0 人数：{int((cs['ap_all_pro_1st_team'] > 0).sum())}")
w(f"career_games_started == 0 人数：{int((cs['career_games_started'] == 0).sum())}")
w(f"career_offensive_snaps + 防守 snaps 均为 0 人数：{int(((cs['career_offensive_snaps'] == 0) & (cs['career_defensive_snaps'] == 0)).sum())}")
w("```")
w("")
w("### games.csv")
w("```")
w(f"season 分布：\n{games['season'].value_counts().sort_index().to_string()}")
w(f"season_type 分布：\n{games['season_type'].value_counts().to_string()}")
w("```")
w("")
w("### 跨表 nfl_id 链接检查")
w("```")
p_ids = set(players["nfl_id"])
cr_ids = set(cr["nfl_id"])
cs_ids = set(cs["nfl_id"])
pp = dfs["player_play.csv"]
ct = dfs["combine_tracking.csv"]
w(f"players nfl_id：{len(p_ids)}；combine_results nfl_id：{len(cr_ids)}，与 players 交集：{len(p_ids & cr_ids)}，combine_results 多出：{len(cr_ids - p_ids)}，players 缺失：{len(p_ids - cr_ids)}")
w(f"career_successes nfl_id：{len(cs_ids)}，与 players 交集：{len(p_ids & cs_ids)}")
w(f"player_play nfl_id 唯一值：{pp['nfl_id'].nunique()}，与 players 交集：{len(p_ids & set(pp['nfl_id'].unique()))}")
w(f"combine_tracking nfl_id 唯一值（不含空）：{ct['nfl_id'].nunique()}，与 players 交集：{len(p_ids & set(ct['nfl_id'].dropna().unique()))}")
w(f"player_play 行数：{len(pp):,}；game_id 唯一值：{pp['game_id'].nunique()}；play_id 唯一值：{pp['play_id'].nunique()}")
w(f"combine_tracking event_id 唯一值：{ct['event_id'].nunique()}；drill_type 分布：\n{ct['drill_type'].value_counts().to_string()}")
w(f"player_play 中 target=True 行数：{int(pp['target'].sum())} / {len(pp)}")
w("```")
w("")

with open(OUT, "w") as fh:
    fh.write("\n".join(lines))
print("wrote", OUT, len(lines), "lines")
