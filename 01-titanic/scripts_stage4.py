"""第 4 阶段：特征工程消融 G1-G7+G9（LR 与 LGBM 双参考模型，25 折 compare），
G8 单独决策（普通 CV + 分组 CV），反向消融，确定 F_best 写进 config.py。"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "workspace",
                                "kaggle-data", "titanic", "src"))
from datetime import date
import numpy as np
import pandas as pd

from config import BASE, F0_CAT, F0_NUM, GROUPS, GROUP_ORDER
from cv import compare, load_folds, log_experiment, run_cv, run_group_cv
from features import add_nonlabel_features
from models import make_model

train = pd.read_csv(os.path.join(BASE, "data", "raw", "train.csv"))
test = pd.read_csv(os.path.join(BASE, "data", "raw", "test.csv"))
y = train["Survived"]
folds = load_folds()
today = str(date.today())
tr, te = add_nonlabel_features(train, test)

REFS = ["lr", "lgbm"]


def run(exp_id, num_cols, cat_cols, model, use_g8=False, use_group_age=False):
    s = run_cv(exp_id, tr, y, te, folds, num_cols, cat_cols,
               lambda seed: make_model(model, seed=seed),
               use_g8=use_g8, use_group_age=use_group_age)
    log_experiment({"exp_id": exp_id, "date": today,
                    "feature_set": "+".join(num_cols + cat_cols),
                    "model": model, "params_file": "", "n_seeds": 1,
                    "cv_scheme": "RSKF 5x5 seed42",
                    "acc_mean": round(s["acc_mean"], 4),
                    "acc_std": round(s["acc_std"], 4),
                    "acc_se": round(s["acc_se"], 4),
                    "logloss_mean": round(s["logloss_mean"], 4),
                    "auc_mean": round(s["auc_mean"], 4), "group_cv_acc": "",
                    "vs": "", "d_mean": "", "wins": "", "submitted": "no",
                    "public_lb": "",
                    "note": f"ablation use_g8={use_g8} use_group_age={use_group_age}"})
    return s


def keep_decision(cmps):
    """保留规则：至少一个参考模型 mean(d)>0 且胜折>=13/25，
    另一个不明显下降 mean(d)>-0.002。cmps: {model: compare dict}"""
    good = [m for m, c in cmps.items()
            if c["d_mean"] > 0 and c["wins_a"] >= 13]
    ok_other = all(c["d_mean"] > -0.002 for c in cmps.values())
    return bool(good) and ok_other


results = []
num_cols, cat_cols = list(F0_NUM), list(F0_CAT)
use_group_age = False
# 起点 F0 的双参考分数（B2/B5 已跑过，复用其 fold_metrics 做 compare 基线）
base_ids = {"lr": "B2_lr_F0", "lgbm": "B5_lgbm_F0"}
prev_ids = dict(base_ids)
retained = []
for g in GROUP_ORDER:
    add_num, add_cat = GROUPS[g]
    new_num = num_cols + add_num
    new_cat = cat_cols + add_cat
    new_age = use_group_age or (g == "G6")
    tag = f"F0+{'+'.join(retained + [g])}" if retained else f"F0+{g}"
    cmps, sums = {}, {}
    for m in REFS:
        exp_id = f"abl_{tag.replace('+', '_')}_{m}"
        sums[m] = run(exp_id, new_num, new_cat, m, use_group_age=new_age)
        cmps[m] = compare(exp_id, prev_ids[m])
    keep = keep_decision(cmps)
    results.append((g, tag, keep, cmps, sums))
    print(f"{g} {tag}: keep={keep} | " +
          " | ".join(f"{m}: acc={sums[m]['acc_mean']:.4f} "
                     f"d={cmps[m]['d_mean']:+.4f} wins={cmps[m]['wins_a']}/25 "
                     f"p={cmps[m]['p']:.3f}" for m in REFS), flush=True)
    if keep:
        num_cols, cat_cols = new_num, new_cat
        use_group_age = new_age
        retained.append(g)
        prev_ids = {m: f"abl_{tag.replace('+', '_')}_{m}" for m in REFS}
    # 不保留时 prev_ids 不变（仍与上一步保留的集合比）

print("正向消融保留:", retained, "use_group_age:", use_group_age, flush=True)

# ---- 反向消融：从完整保留集每次去掉一组 ----
full_num, full_cat = list(num_cols), list(cat_cols)
full_age = use_group_age
full_ids = {}
for m in REFS:
    exp_id = f"abl_full_{m}"
    s = run(exp_id, full_num, full_cat, m, use_group_age=full_age)
    full_ids[m] = exp_id
    print(f"full {m}: acc={s['acc_mean']:.4f}", flush=True)
dropped = []
for g in list(retained):
    add_num, add_cat = GROUPS[g]
    rn = [c for c in full_num if c not in add_num]
    rc = [c for c in full_cat if c not in add_cat]
    rage = full_age and g != "G6"
    cmps, sums = {}, {}
    for m in REFS:
        exp_id = f"abl_no{g}_{m}"
        sums[m] = run(exp_id, rn, rc, m, use_group_age=rage)
        cmps[m] = compare(exp_id, full_ids[m])
    # 去掉后不降甚至涨 -> 删：至少一个模型 d>=0 且另一个 d>-0.002
    drop = (any(c["d_mean"] >= 0 for c in cmps.values())
            and all(c["d_mean"] > -0.002 for c in cmps.values()))
    print(f"去掉 {g}: drop={drop} | " +
          " | ".join(f"{m}: acc={sums[m]['acc_mean']:.4f} "
                     f"d={cmps[m]['d_mean']:+.4f} wins={cmps[m]['wins_a']}/25"
                     for m in REFS), flush=True)
    if drop:
        dropped.append(g)
        full_num, full_cat, full_age = rn, rc, rage
        retained.remove(g)
        # full_ids 不变：后续仍与原始完整集比（近似处理，记录在案）
print("反向消融删除:", dropped, "最终保留:", retained, flush=True)

# ---- F_best 参照：最终保留集重跑一次 ----
final_ids = {}
for m in REFS:
    exp_id = f"Fbest_{m}"
    s = run(exp_id, full_num, full_cat, m, use_group_age=full_age)
    final_ids[m] = exp_id
    print(f"F_best {m}: acc={s['acc_mean']:.4f}±{s['acc_std']:.4f}", flush=True)

# ---- Pclass 当类别 vs 当数值（在 F_best 上，LR+LGBM）----
for m in REFS:
    exp_id = f"abl_pclasscat_{m}"
    s = run(exp_id, [c for c in full_num if c != "Pclass"],
            full_cat + ["Pclass"], m, use_group_age=full_age)
    c = compare(exp_id, final_ids[m])
    print(f"Pclass当类别 {m}: acc={s['acc_mean']:.4f} d={c['d_mean']:+.4f} "
          f"wins={c['wins_a']}/25", flush=True)

# ---- G8 单独决策：普通 CV + 分组 CV（Ticket）----
g8_res = {}
for m in REFS:
    exp_id = f"abl_G8_{m}"
    s = run(exp_id, full_num + ["GroupSurv"], full_cat, m,
            use_g8=True, use_group_age=full_age)
    c = compare(exp_id, final_ids[m])
    g_base = run_group_cv(tr, y, te, tr["Ticket"], full_num, full_cat,
                          lambda seed: make_model(m, seed=seed),
                          use_group_age=full_age)
    g_g8 = run_group_cv(tr, y, te, tr["Ticket"], full_num + ["GroupSurv"],
                        full_cat, lambda seed: make_model(m, seed=seed),
                        use_g8=True, use_group_age=full_age)
    g8_res[m] = {"cv_d": c["d_mean"], "cv_wins": c["wins_a"],
                 "group_base": g_base["acc_mean"], "group_g8": g_g8["acc_mean"],
                 "acc_g8": s["acc_mean"]}
    print(f"G8 {m}: 普通CV acc={s['acc_mean']:.4f} d={c['d_mean']:+.4f} "
          f"wins={c['wins_a']}/25 | 分组CV 无G8={g_base['acc_mean']:.4f} "
          f"有G8={g_g8['acc_mean']:.4f}", flush=True)
g8_keep = all(v["cv_d"] > 0 and v["group_g8"] > v["group_base"]
              for v in g8_res.values())
print("G8 保留进主线:", g8_keep, flush=True)

# ---- 写 F_best 进 config.py ----
cfg_path = os.path.join(BASE, "src", "config.py")
with open(cfg_path) as f:
    cfg = f.read()
cfg = re.sub(r"F_BEST_NUM = .*", f"F_BEST_NUM = {full_num!r}", cfg)
cfg = re.sub(r"F_BEST_CAT = .*", f"F_BEST_CAT = {full_cat!r}", cfg)
cfg = re.sub(r"F_BEST_USE_G8 = .*", f"F_BEST_USE_G8 = {g8_keep!r}", cfg)
cfg = re.sub(r"F_BEST_USE_GROUP_AGE = .*",
             f"F_BEST_USE_GROUP_AGE = {full_age!r}", cfg)
with open(cfg_path, "w") as f:
    f.write(cfg)
print("F_best:", full_num, full_cat, "G8:", g8_keep, "group_age:", full_age)

# ---- 写 notes/feature_ablation.md ----
md = ["# 特征消融（2026-11-19）\n",
      "参考模型：LR（对特征敏感）与 LGBM（默认中等参数：1000 棵、lr 0.05、num_leaves 15、早停 100）。",
      "保留规则：至少一个模型 mean(d)>0 且胜折≥13/25，另一个 mean(d)>-0.002。",
      "起点 F0：Pclass/Sex/Age/SibSp/Parch/Fare/Embarked。",
      "TicketFreq 在 train+test 合并后计算（不含标签，比赛允许）；Fare 缺失按 Pclass 中位数填。\n",
      "## 正向消融\n",
      "| 组 | 加入后 | 保留 | LR acc / d / 胜折 | LGBM acc / d / 胜折 |",
      "|---|---|---|---|---|"]
for g, tag, keep, cmps, sums in results:
    md.append(f"| {g} | {tag} | {'是' if keep else '否'} | "
              f"{sums['lr']['acc_mean']:.4f} / {cmps['lr']['d_mean']:+.4f} / {cmps['lr']['wins_a']} | "
              f"{sums['lgbm']['acc_mean']:.4f} / {cmps['lgbm']['d_mean']:+.4f} / {cmps['lgbm']['wins_a']} |")
md.append(f"\n正向保留：{retained}（反向删除 {dropped} 后最终保留）")
md.append(f"\n## 反向消融\n\n从完整保留集每次去掉一组，去掉后不降甚至涨（至少一个模型 d≥0 且另一个 >-0.002）则删。删除：{dropped}。")
md.append("\n## Pclass 当类别 vs 当数值\n\n在最终特征集上测试，见 logs/experiments.csv（abl_pclasscat_*）。"
          "未采用则保持数值。")
md.append("\n## G8 组生还特征单独决策\n")
for m, v in g8_res.items():
    md.append(f"- {m}：普通 CV 有 G8 acc={v['acc_g8']:.4f}，d={v['cv_d']:+.4f}，胜折 {v['cv_wins']}/25；"
              f"分组 CV 无 G8={v['group_base']:.4f}，有 G8={v['group_g8']:.4f}。")
md.append(f"\n结论：G8 {'保留进主线' if g8_keep else '不进主线，只作激进版备选'}"
          "（两种 CV 都提升才保留）。")
md.append(f"\n## F_best\n\n数值列：{full_num}\n\n类别列：{full_cat}\n\n"
          f"分组 Age 填补：{full_age}；G8：{g8_keep}。")
with open(os.path.join(BASE, "notes", "feature_ablation.md"), "w") as f:
    f.write("\n".join(md))
print("STAGE4 DONE")
