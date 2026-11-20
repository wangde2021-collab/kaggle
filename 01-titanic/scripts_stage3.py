"""第 3 阶段：EDA + 基线 B0-B6（全部 25 折）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "workspace",
                                "kaggle-data", "titanic", "src"))
from datetime import date
import numpy as np
import pandas as pd

from config import BASE, F0_CAT, F0_NUM
from cv import fold_lists_from_df, load_folds, log_experiment, run_cv
from features import add_nonlabel_features
from models import make_model

train = pd.read_csv(os.path.join(BASE, "data", "raw", "train.csv"))
test = pd.read_csv(os.path.join(BASE, "data", "raw", "test.csv"))
y = train["Survived"]
folds = load_folds()
fold_list = fold_lists_from_df(folds)
today = str(date.today())

# ---------------- EDA ----------------
L = []
L.append("# EDA 结论（2026-11-19）\n")
L.append(f"train {len(train)} 行，test {len(test)} 行；目标生还率 {y.mean():.4f}（{int(y.sum())}/{len(y)}）。\n")
L.append("## 缺失率")
miss = pd.DataFrame({"train缺失": train.isna().mean().round(4),
                     "test缺失": test.isna().mean().round(4)})
L.append(miss.to_string())
L.append("\n\n## 分组生还率（train）")
for col in ["Sex", "Pclass", "Embarked", "SibSp", "Parch"]:
    g = train.groupby(col)["Survived"].agg(["mean", "count"]).round(4)
    L.append(f"\n### {col}\n{g.to_string()}")
tr, te = add_nonlabel_features(train, test)
for col in ["Title", "FamilyBucket", "Deck", "TicketPrefix"]:
    g = tr.groupby(col)["Survived"].agg(["mean", "count"]).round(4)
    L.append(f"\n### {col}\n{g.to_string()}")
L.append("\n\n## Age / Fare 分箱生还率")
tr2 = tr.copy()
tr2["AgeBin"] = pd.cut(tr2["Age"], bins=[0, 10, 18, 30, 45, 60, 100])
L.append("\nAge:\n" + tr2.groupby("AgeBin", observed=True)["Survived"].agg(["mean", "count"]).round(4).to_string())
tr2["FareBin"] = pd.cut(tr2["Fare"], bins=[-1, 7.9, 14.5, 31, 100, 600])
L.append("\nFare:\n" + tr2.groupby("FareBin", observed=True)["Survived"].agg(["mean", "count"]).round(4).to_string())
L.append("\n\n## Ticket 重复次数分布（train+test 合并）")
L.append("\n" + tr["TicketFreq"].value_counts().sort_index().to_string())
L.append("\n\n## train vs test 分布对比")
comp = pd.DataFrame({
    "train均值": tr[["Age", "Fare", "SibSp", "Parch", "FamilySize"]].mean().round(2),
    "test均值": te[["Age", "Fare", "SibSp", "Parch", "FamilySize"]].mean().round(2),
    "train女比例": [None] * 5,
})
comp["train女比例"] = [np.nan] * 5
L.append("\n" + comp.to_string())
L.append(f"\n\ntrain 女性比例 {(tr['Sex'] == 'female').mean():.4f}，test 女性比例 {(te['Sex'] == 'female').mean():.4f}")
L.append(f"\ntrain Pclass 分布 {tr['Pclass'].value_counts(normalize=True).round(3).to_dict()}")
L.append(f"\ntest Pclass 分布 {te['Pclass'].value_counts(normalize=True).round(3).to_dict()}")
L.append("\n\n## 对抗验证（第 2 阶段）")
L.append("\nLightGBM 5 折 AUC 均值 0.5017 ≈ 0.5，train/test 分布一致，CV 可信。"
        "最重要特征为 Fare/Age（均为数值本身，非分布偏移信号）。")
L.append("\n\n## 分组泄漏检查（第 2 阶段）")
L.append("\n分组 CV（Ticket，LR，F0）均值 0.7957，与普通 25 折 LR 0.7926 接近，"
        "F0 层面无明显组内泄漏。G8 仍需单独做分组 CV 决策。")
with open(os.path.join(BASE, "notes", "eda.md"), "w") as f:
    f.write("\n".join(L))
print("eda.md written")

# ---------------- 基线 ----------------
X = tr[F0_NUM + F0_CAT]
X_test = te[F0_NUM + F0_CAT]

def acc_from_oof(oof, y):
    """oof (5,891) -> 25 折逐折 accuracy 列表（与 run_cv 的折顺序一致）。"""
    accs = []
    for i, (trn, va) in enumerate(fold_list):
        rep, fold = divmod(i, 5)
        p = oof[rep, va]
        accs.append(float(((p >= 0.5).astype(int) == y.iloc[va].values).mean()))
    return accs

def save_fold_metrics(exp_id, accs, oof):
    from sklearn.metrics import log_loss, roc_auc_score
    rows = []
    for i, (trn, va) in enumerate(fold_list):
        rep, fold = divmod(i, 5)
        p = oof[rep, va]
        rows.append({"rep": rep, "fold": fold,
                     "acc": float(((p >= 0.5).astype(int) == y.iloc[va].values).mean()),
                     "logloss": float(log_loss(y.iloc[va], p, labels=[0, 1])),
                     "auc": float(roc_auc_score(y.iloc[va], p))})
    pd.DataFrame(rows).to_csv(os.path.join(BASE, "logs", f"fold_metrics_{exp_id}.csv"), index=False)

def log_row(exp_id, s, model, feature_set="F0", vs="", d_mean="", wins="",
            note="", group_cv=""):
    log_experiment({"exp_id": exp_id, "date": today, "feature_set": feature_set,
                    "model": model, "params_file": "", "n_seeds": 1,
                    "cv_scheme": "RSKF 5x5 seed42", "acc_mean": round(s["acc_mean"], 4),
                    "acc_std": round(s["acc_std"], 4), "acc_se": round(s["acc_se"], 4),
                    "logloss_mean": round(s["logloss_mean"], 4),
                    "auc_mean": round(s["auc_mean"], 4), "group_cv_acc": group_cv,
                    "vs": vs, "d_mean": d_mean, "wins": wins,
                    "submitted": "no", "public_lb": "", "note": note})

# B0 全猜 0
oof0 = np.zeros((5, len(y)))
accs0 = acc_from_oof(oof0, y)
s0 = {"acc_mean": float(np.mean(accs0)), "acc_std": float(np.std(accs0, ddof=1)),
      "acc_se": float(np.std(accs0, ddof=1) / 5), "logloss_mean": float("nan"),
      "auc_mean": float("nan")}
save_fold_metrics("B0", accs0, oof0)
np.save(os.path.join(BASE, "oof", "B0.npy"), oof0)
print("B0 全猜0: acc=%.4f" % s0["acc_mean"])
log_row("B0", s0, "majority-0", note="全猜 0，多数类下限")

# B1 性别规则
oof1 = np.zeros((5, len(y)))
pred1 = (tr["Sex"] == "female").astype(float).values
for rep in range(5):
    oof1[rep, :] = pred1
accs1 = acc_from_oof(oof1, y)
s1 = {"acc_mean": float(np.mean(accs1)), "acc_std": float(np.std(accs1, ddof=1)),
      "acc_se": float(np.std(accs1, ddof=1) / 5), "logloss_mean": float("nan"),
      "auc_mean": float("nan")}
save_fold_metrics("B1", accs1, oof1)
np.save(os.path.join(BASE, "oof", "B1.npy"), oof1)
print("B1 性别规则: acc=%.4f" % s1["acc_mean"])
log_row("B1", s1, "gender-rule", note="女=1 男=0")

# B2-B5：上轮 7 特征 + 旧参数（XGB/LGBM 早停 100 轮、上限 300 棵，见备注）
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
names = {"lr": "B2_lr_F0", "rf": "B3_rf_F0", "xgb": "B4_xgb_F0", "lgbm": "B5_lgbm_F0"}
for m, exp_id in names.items():
    s = run_cv(exp_id, X, y, X_test, folds, F0_NUM, F0_CAT,
               lambda seed, m=m: make_model(m, params=old_params[m], seed=seed))
    print(f"{exp_id}: acc={s['acc_mean']:.4f}±{s['acc_std']:.4f} "
          f"logloss={s['logloss_mean']:.4f} auc={s['auc_mean']:.4f} "
          f"best_iter_mean={s['best_iter_mean']}")
    log_row(exp_id, s, m, note="上轮7特征+旧参数，新25折CV；xgb/lgbm早停100轮")

# B6：B2-B5 OOF 等权软投票（不重新训练）
oofs = [np.load(os.path.join(BASE, "oof", f"{e}.npy")) for e in names.values()]
oof6 = np.mean(oofs, axis=0)
test6 = np.mean([np.load(os.path.join(BASE, "test_preds", f"{e}.npy"))
                 for e in names.values()], axis=0)
accs6 = acc_from_oof(oof6, y)
s6 = {"acc_mean": float(np.mean(accs6)), "acc_std": float(np.std(accs6, ddof=1)),
      "acc_se": float(np.std(accs6, ddof=1) / 5),
      "logloss_mean": 0.0, "auc_mean": 0.0}
save_fold_metrics("B6_softvote_F0", accs6, oof6)
np.save(os.path.join(BASE, "oof", "B6_softvote_F0.npy"), oof6)
np.save(os.path.join(BASE, "test_preds", "B6_softvote_F0.npy"), test6)
from sklearn.metrics import log_loss, roc_auc_score
lls, aucs = [], []
for i, (trn, va) in enumerate(fold_list):
    rep, fold = divmod(i, 5)
    p = oof6[rep, va]
    lls.append(log_loss(y.iloc[va], p, labels=[0, 1]))
    aucs.append(roc_auc_score(y.iloc[va], p))
s6["logloss_mean"] = float(np.mean(lls))
s6["auc_mean"] = float(np.mean(aucs))
print("B6 软投票(F0): acc=%.4f±%.4f logloss=%.4f auc=%.4f"
      % (s6["acc_mean"], s6["acc_std"], s6["logloss_mean"], s6["auc_mean"]))
log_row("B6_softvote_F0", s6, "soft-vote(lr+rf+xgb+lgbm)",
        note="B2-B5 OOF等权平均；当前提交 S 的新CV参照")
print("STAGE3 DONE")
