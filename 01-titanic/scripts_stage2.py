"""第 2 阶段：冻结 25 折 + 对抗验证 + 分组泄漏检查 + run_cv 烟囱测试。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "workspace",
                                "kaggle-data", "titanic", "src"))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from config import BASE, F0_NUM, F0_CAT
from cv import (fold_lists_from_df, load_folds, make_folds, run_cv,
                run_group_cv)
from features import add_nonlabel_features
from models import make_model

train = pd.read_csv(os.path.join(BASE, "data", "raw", "train.csv"))
test = pd.read_csv(os.path.join(BASE, "data", "raw", "test.csv"))
y = train["Survived"]

# 1) 冻结折
folds = make_folds(y)
print("folds.csv:", folds.shape)
print(folds.head(3).to_string(index=False))
# 分层检查：每折正例率
for rep in range(5):
    rates = []
    for fold in range(5):
        va = folds[f"rep{rep}_fold"].values == fold
        rates.append(round(float(y[va].mean()), 4))
    print(f"rep{rep} 各折正例率:", rates, " 全体:", round(float(y.mean()), 4))

# 2) 特征（非标签部分）
tr, te = add_nonlabel_features(train, test)
X = tr[F0_NUM + F0_CAT]
X_test = te[F0_NUM + F0_CAT]

# 3) 对抗验证：train=0, test=1，LightGBM 5 折 AUC
adv_X = pd.concat([tr[F0_NUM + F0_CAT], te[F0_NUM + F0_CAT]], ignore_index=True)
adv_y = np.array([0] * len(tr) + [1] * len(te))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), F0_NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), F0_CAT)])
aucs, imps = [], None
for trn, van in skf.split(adv_X, adv_y):
    m = Pipeline([("pre", pre), ("clf", LGBMClassifier(
        n_estimators=200, random_state=42, n_jobs=2, verbose=-1))])
    m.fit(adv_X.iloc[trn], adv_y[trn])
    p = m.predict_proba(adv_X.iloc[van])[:, 1]
    aucs.append(roc_auc_score(adv_y[van], p))
    imps = m.named_steps["clf"].feature_importances_
print("对抗验证 AUC: mean=%.4f folds=%s" % (np.mean(aucs), [round(a, 4) for a in aucs]))
names = (pre.fit(adv_X).get_feature_names_out())
top = sorted(zip(names, imps), key=lambda t: -t[1])[:5]
print("对抗验证最重要特征:", [(n, int(v)) for n, v in top])

# 4) 分组 CV 检查（LR, F0, 分组=Ticket）
g = run_group_cv(X, y, X_test, tr["Ticket"], F0_NUM, F0_CAT,
                 lambda s: make_model("lr", seed=s))
print("分组 CV (Ticket, LR, F0): mean=%.4f folds=%s" % (g["acc_mean"], g["fold_acc"]))

# 5) run_cv 烟囱：LR on F0，25 折
s = run_cv("smoke_lr_F0", X, y, X_test, folds, F0_NUM, F0_CAT,
           lambda seed: make_model("lr", seed=seed))
print("smoke LR F0 25折: acc=%.4f±%.4f se=%.4f logloss=%.4f auc=%.4f"
      % (s["acc_mean"], s["acc_std"], s["acc_se"], s["logloss_mean"], s["auc_mean"]))
oof = np.load(os.path.join(BASE, "oof", "smoke_lr_F0.npy"))
tp = np.load(os.path.join(BASE, "test_preds", "smoke_lr_F0.npy"))
print("oof shape:", oof.shape, "test_preds shape:", tp.shape)
print("STAGE2 DONE")
