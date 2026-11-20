"""第 7 阶段：决策门。候选 C 替换当前提交 S=B6（新 25 折 CV 分数）须同时满足：
(1) mean(d) >= max(0.003, 1*se(d))；(2) 胜折 >= 15/25；
(3) std(C) <= std(S)+0.005；(4) 分组 CV（Ticket）上 C 不低于 S。
S 指 B6 在新 25 折下的分数（0.8305），不是旧单次 5 折的 0.8160。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "workspace",
                                "kaggle-data", "titanic", "src"))
from datetime import date
import numpy as np
import pandas as pd

from config import BASE, F0_CAT, F0_NUM, F_BEST_CAT, F_BEST_NUM, F_BEST_USE_GROUP_AGE
from cv import compare, load_folds, run_group_cv
from features import add_nonlabel_features
from models import load_best_params, make_model

train = pd.read_csv(os.path.join(BASE, "data", "raw", "train.csv"))
test = pd.read_csv(os.path.join(BASE, "data", "raw", "test.csv"))
y = train["Survived"]
tr, te = add_nonlabel_features(train, test)
today = str(date.today())
S = "B6_softvote_F0"

fm_s = pd.read_csv(os.path.join(BASE, "logs", f"fold_metrics_{S}.csv"))
std_s = float(fm_s["acc"].std(ddof=1))
mean_s = float(fm_s["acc"].mean())
print(f"S=B6 新25折: acc={mean_s:.4f}±{std_s:.4f}")

CANDIDATES = ["E3_weighted", "E6_greedy", "E2_hardvote5", "tuned_lgbm",
              "tuned_xgb", "E5_stacking", "E1_softvote"]
results = {}
for c in CANDIDATES:
    fm = pd.read_csv(os.path.join(BASE, "logs", f"fold_metrics_{c}.csv"))
    cmp = compare(c, S)
    thr = max(0.003, cmp["d_se"])
    g1 = cmp["d_mean"] >= thr
    g2 = cmp["wins_a"] >= 15
    g3 = float(fm["acc"].std(ddof=1)) <= std_s + 0.005
    results[c] = {"acc_mean": float(fm["acc"].mean()),
                  "acc_std": float(fm["acc"].std(ddof=1)),
                  "cmp": cmp, "g1": g1, "g2": g2, "g3": g3,
                  "thr": thr, "g4": None}
    print(f"{c}: acc={results[c]['acc_mean']:.4f}±{results[c]['acc_std']:.4f} "
          f"d={cmp['d_mean']:+.4f} se={cmp['d_se']:.4f} wins={cmp['wins_a']}/25 "
          f"p={cmp['p']:.4f} | 门1 {g1} 门2 {g2} 门3 {g3}", flush=True)

# ---- 门 4：分组 CV（Ticket）。只对通过前三门的候选做 ----
# S 的分组 CV：B6 成员（B2-B5，F0 旧参数）在分组折上的 OOF 等权平均
passers = [c for c in CANDIDATES
           if results[c]["g1"] and results[c]["g2"] and results[c]["g3"]]
print("通过前三门:", passers)


def group_oof(model, num_cols, cat_cols, params, use_group_age=False):
    return run_group_cv(tr, y, te, tr["Ticket"], num_cols, cat_cols,
                        lambda seed: make_model(model, params=params, seed=seed),
                        use_group_age=use_group_age)["oof"]


if passers:
    # B6 分组 OOF
    old_params = {
        "lr": {},
        "rf": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        "xgb": {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6,
                "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 1,
                "reg_lambda": 1.0, "early_stopping_rounds": None},
        "lgbm": {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 31,
                 "min_child_samples": 20, "subsample": 1.0, "subsample_freq": 0,
                 "colsample_bytree": 1.0, "reg_lambda": 0.0},
    }
    b6_g = np.mean([group_oof(m, F0_NUM, F0_CAT, old_params[m])
                    for m in ["lr", "rf", "xgb", "lgbm"]], axis=0)
    b6_gacc = float(((b6_g >= 0.5).astype(int) == y.values).mean())
    print(f"分组CV S=B6: acc={b6_gacc:.4f}", flush=True)

    for c in passers:
        if c == "E3_weighted":
            w = {"lgbm": 0.816, "svc": 0.155, "knn": 0.029}
            oofs = [group_oof(m, F_BEST_NUM, F_BEST_CAT, load_best_params(m),
                              use_group_age=F_BEST_USE_GROUP_AGE)
                    for m in ["lgbm", "svc", "knn"]]
            g = sum(wi * o for wi, o in zip([0.816, 0.155, 0.029], oofs))
        elif c == "E6_greedy":
            oofs = [group_oof(m, F_BEST_NUM, F_BEST_CAT, load_best_params(m),
                              use_group_age=F_BEST_USE_GROUP_AGE)
                    for m in ["lgbm", "rf"]]
            g = np.mean(oofs, axis=0)
        elif c in ("tuned_lgbm", "tuned_xgb"):
            m = c.replace("tuned_", "")
            g = group_oof(m, F_BEST_NUM, F_BEST_CAT, load_best_params(m),
                          use_group_age=F_BEST_USE_GROUP_AGE)
        elif c == "E2_hardvote5":
            oofs = [group_oof(m, F_BEST_NUM, F_BEST_CAT, load_best_params(m),
                              use_group_age=F_BEST_USE_GROUP_AGE)
                    for m in ["lgbm", "xgb", "rf", "svc", "cat"]]
            g = (np.mean([(o >= 0.5).astype(int) for o in oofs], axis=0) >= 0.5).astype(float)
        else:
            g = None
        if g is not None:
            gacc = float(((g >= 0.5).astype(int) == y.values).mean())
            results[c]["g4"] = gacc >= b6_gacc
            results[c]["group_cv_acc"] = gacc
            print(f"分组CV {c}: acc={gacc:.4f} vs S={b6_gacc:.4f} 门4 {results[c]['g4']}",
                  flush=True)
        else:
            results[c]["g4"] = None

winner = None
for c in passers:
    if results[c]["g4"]:
        if winner is None or results[c]["cmp"]["d_mean"] > results[winner]["cmp"]["d_mean"]:
            winner = c
print("胜出方案:", winner)

# ---- 写 notes/decision_20261119.md ----
md = [f"# 决策（{today}）\n",
      f"当前提交 S = B6（上轮 7 特征软投票）在新 25 折 CV 下的分数："
      f"acc={mean_s:.4f}±{std_s:.4f}。注意：不是旧单次 5 折的 0.8160，"
      "旧分数绝不参与比较。\n",
      "决策门（须同时满足）：(1) mean(d)≥max(0.003, 1×se(d))；"
      "(2) 胜折≥15/25；(3) std(C)≤std(S)+0.005；(4) 分组 CV（Ticket）不低于 S。\n",
      "## 候选比较（vs S=B6）\n",
      "| 候选 | CV acc | d 均值 | d 标准误 | 胜折 | p | 门1 | 门2 | 门3 | 分组CV | 门4 |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for c in CANDIDATES:
    r = results[c]
    cmp = r["cmp"]
    g4 = ("-" if r["g4"] is None else
          (f"{r.get('group_cv_acc', float('nan')):.4f} 通过" if r["g4"]
           else f"{r.get('group_cv_acc', float('nan')):.4f} 未通过"))
    md.append(f"| {c} | {r['acc_mean']:.4f}±{r['acc_std']:.4f} | {cmp['d_mean']:+.4f} | "
              f"{cmp['d_se']:.4f} | {cmp['wins_a']}/25 | {cmp['p']:.4f} | "
              f"{'✓' if r['g1'] else '✗'} | {'✓' if r['g2'] else '✗'} | "
              f"{'✓' if r['g3'] else '✗'} | {g4} | "
              f"{'✓' if r['g4'] else ('✗' if r['g4'] is False else '-')} |")
md.append(f"\n## 结论\n\n{'胜出方案：' + winner + '，通过全部四条决策门，替换当前提交。' if winner else '没有候选通过全部四条决策门，不替换当前提交。'}")
if winner:
    r = results[winner]
    md.append(f"\n{winner}：CV {r['acc_mean']:.4f}±{r['acc_std']:.4f}，"
              f"vs S d={r['cmp']['d_mean']:+.4f}（se={r['cmp']['d_se']:.4f}），"
              f"胜折 {r['cmp']['wins_a']}/25，分组 CV {r.get('group_cv_acc', float('nan')):.4f}。")
md.append("\n## 平局与选择偏差说明\n\n"
          "候选越多，CV 最高分越偏乐观；最终入选方案的 CV 分数应打折看待。"
          "E3 与 E6 的 CV 分数接近（见上表），若两者都通过门，选 d 均值更大的；"
          "打平则选更简单的（模型更少优先）。本轮未用 LB 验证（未提交，无 Public LB）。")
with open(os.path.join(BASE, "notes", "decision_20261119.md"), "w") as f:
    f.write("\n".join(md))
print("STAGE7 DONE winner=", winner)
