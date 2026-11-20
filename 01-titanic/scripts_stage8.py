"""第 8 阶段：胜出方案 E3 全量重训 + 提交前检查 + 版本化存档。
E3 = 加权软投票，成员 tuned lgbm / svc / knn（F_best），全局权重
lgbm 0.816 / svc 0.155 / knn 0.029（全量 OOF 优化 LogLoss 得到）。
lgbm 全量迭代数 = CV 平均最佳迭代数 115.44 × 1.15 ≈ 133，5 种子平均；
svc/knn 单种子。不提交到 Kaggle。"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "workspace",
                                "kaggle-data", "titanic", "src"))
from datetime import date
import numpy as np
import pandas as pd

from config import BASE, F_BEST_CAT, F_BEST_NUM
from features import add_nonlabel_features
from models import build_preprocessor, load_best_params, make_model

train = pd.read_csv(os.path.join(BASE, "data", "raw", "train.csv"))
test = pd.read_csv(os.path.join(BASE, "data", "raw", "test.csv"))
y = train["Survived"]
tr, te = add_nonlabel_features(train, test)
X = tr[F_BEST_NUM + F_BEST_CAT]
X_test = te[F_BEST_NUM + F_BEST_CAT]
today = str(date.today())

WEIGHTS = {"lgbm": 0.816, "svc": 0.155, "knn": 0.029}
LGBM_N = int(round(115.44 * 1.15))  # 133
print("lgbm 全量 n_estimators:", LGBM_N)

test_probs = {}
for member in ["lgbm", "svc", "knn"]:
    params = load_best_params(member)
    seeds = (42, 43, 44, 45, 46) if member == "lgbm" else (42,)
    ps = []
    for seed in seeds:
        kind, model = make_model(member, params=params, seed=seed,
                                 full_train=True,
                                 n_estimators=LGBM_N if member == "lgbm" else None)
        pre, cat_idx = build_preprocessor(kind, F_BEST_NUM, F_BEST_CAT)
        X_t = pre.fit_transform(X)
        Xte_t = pre.transform(X_test)
        model.fit(X_t, y)
        ps.append(model.predict_proba(Xte_t)[:, 1])
    test_probs[member] = np.mean(ps, axis=0)
    print(f"{member}: 全量重训完成（{len(seeds)} 种子），"
          f"test 概率均值 {test_probs[member].mean():.4f}", flush=True)

p_final = sum(WEIGHTS[m] * test_probs[m] for m in WEIGHTS)
pred = (p_final >= 0.5).astype(int)
sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred})

# ---- 提交前检查 ----
checks = []
checks.append(("行数=418", len(sub) == 418))
checks.append(("列名 PassengerId/Survived", list(sub.columns) == ["PassengerId", "Survived"]))
checks.append(("PassengerId 与 test.csv 完全一致",
               (sub["PassengerId"].values == test["PassengerId"].values).all()))
checks.append(("PassengerId 无重复无缺失",
               sub["PassengerId"].is_unique and sub["PassengerId"].notna().all()))
checks.append(("Survived 为整数 0/1 无 NaN",
               set(sub["Survived"].unique()) <= {0, 1} and sub["Survived"].notna().all()
               and str(sub["Survived"].dtype).startswith("int")))
rate = float(sub["Survived"].mean())
checks.append((f"预测生还率 {rate:.4f} 在 0.33-0.42", 0.33 <= rate <= 0.42))
old = pd.read_csv(os.path.join(BASE, "submission.csv"))
diff_old = int((sub["Survived"].values != old["Survived"].values).sum())
checks.append((f"与当前提交差异 {diff_old} 人（>60 需人工复查）", diff_old <= 60))
b1 = (test["Sex"] == "female").astype(int)
diff_b1 = int((sub["Survived"].values != b1.values).sum())
checks.append((f"与 B1 性别规则差异 {diff_b1} 人（合理范围一般 20-60）",
               20 <= diff_b1 <= 60))
for name, ok in checks:
    print(("PASS " if ok else "FAIL ") + name)
assert all(ok for _, ok in checks), "提交前检查未全部通过"

# ---- 版本化存档 ----
os.makedirs(os.path.join(BASE, "submissions"), exist_ok=True)
idx_path = os.path.join(BASE, "submissions", "index.csv")
if not os.path.exists(idx_path):
    # 先把根目录旧提交归档为 sub_000（旧单次 5 折软投票版）
    shutil.copyfile(os.path.join(BASE, "submission.csv"),
                    os.path.join(BASE, "submissions",
                                 "sub_000_prev_softvote_cv0.8160old.csv"))
    pd.DataFrame([{"seq": 0, "exp_id": "prev_softvote_old",
                   "cv_mean": 0.8160, "cv_std": "",
                   "submitted": "no", "public_lb": "",
                   "submit_time": "", "note": "旧单次5折软投票版，替换前存档；0.8160为旧单次5折分数，不与新CV比较"}],
                 columns=["seq", "exp_id", "cv_mean", "cv_std", "submitted",
                          "public_lb", "submit_time", "note"]).to_csv(idx_path, index=False)
    print("旧提交已归档为 sub_000_prev_softvote_cv0.8160old.csv")

fname = "sub_001_E3_weighted_cv0.8413.csv"
out_path = os.path.join(BASE, "submissions", fname)
sub.to_csv(out_path, index=False)
# 根目录 submission.csv 只作当前最优副本
shutil.copyfile(out_path, os.path.join(BASE, "submission.csv"))
row = {"seq": 1, "exp_id": "E3_weighted", "cv_mean": 0.8413, "cv_std": 0.0235,
       "submitted": "no", "public_lb": "", "submit_time": "",
       "note": "加权软投票 lgbm0.816/svc0.155/knn0.029，F_best，lgbm 5种子全量重训 n=133；"
               "已生成文件，未提交到 Kaggle（待用户点头）"}
pd.DataFrame([row]).to_csv(idx_path, mode="a", header=False, index=False)
print("写出:", out_path, "行数:", len(sub), "预测生还:", int(sub["Survived"].sum()), "/", len(sub))
print("STAGE8 DONE")
