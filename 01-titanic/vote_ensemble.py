"""Titanic 新策略：四个模型投票预测。
逻辑回归 + 随机森林 + XGBoost + LightGBM 做投票集成，
软投票（平均预测概率）优先，同时测硬投票（多数表决）。
特征、预处理、5 折划分与对比脚本 compare_models.py 完全一致。
单模型 5 折均值（上轮实测）：逻辑回归 0.7890、随机森林 0.8014、
XGBoost 0.8003、LightGBM 0.8025。
"""
import shutil

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATA = "/home/hatch/workspace/kaggle-data/titanic"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = train[FEATURES]
y = train["Survived"]
X_test = test[FEATURES]

numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical = ["Sex", "Embarked"]

preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

# 与对比脚本一致的 5 折：cross_val_score(cv=5) 默认就是不打乱的 KFold(5)
kf = KFold(n_splits=5, shuffle=False)

# 单模型 5 折均值（上轮 compare_models.py 实测，同一套划分）
SINGLE_MEANS = {"逻辑回归": 0.7890, "随机森林": 0.8014,
                "XGBoost": 0.8003, "LightGBM": 0.8025}
best_single_name = max(SINGLE_MEANS, key=SINGLE_MEANS.get)
best_single_mean = SINGLE_MEANS[best_single_name]


def make_estimators():
    return [
        ("lr", Pipeline([
            ("prep", preprocess),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])),
        ("rf", Pipeline([
            ("prep", preprocess),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ])),
        ("xgb", Pipeline([
            ("prep", preprocess),
            ("clf", XGBClassifier(n_estimators=300, random_state=42, n_jobs=-1,
                                   eval_metric="logloss")),
        ])),
        ("lgbm", Pipeline([
            ("prep", preprocess),
            ("clf", LGBMClassifier(n_estimators=300, random_state=42, n_jobs=-1, verbose=-1)),
        ])),
    ]


voting_scores = {}
for voting in ("soft", "hard"):
    vc = VotingClassifier(estimators=make_estimators(), voting=voting)
    scores = cross_val_score(vc, X, y, cv=kf, scoring="accuracy")
    voting_scores[voting] = scores
    print(f"投票-{voting}: mean={scores.mean():.4f} std={scores.std():.4f} "
          f"folds={[round(s, 4) for s in scores]}")

print(f"最好单模型: {best_single_name} mean={best_single_mean:.4f}")
for voting, scores in voting_scores.items():
    diff = scores.mean() - best_single_mean
    print(f"投票-{voting} vs 最好单模型: {diff:+.4f} "
          f"({'更好' if diff > 0 else '持平' if diff == 0 else '更差'})")

# 选软硬投票里均值更高的那版；优先软投票（均值相同时取软投票）
if voting_scores["soft"].mean() >= voting_scores["hard"].mean():
    chosen_voting = "soft"
else:
    chosen_voting = "hard"
chosen_mean = voting_scores[chosen_voting].mean()
chosen_std = voting_scores[chosen_voting].std()

if chosen_mean >= best_single_mean:
    print(f"采用投票集成 ({chosen_voting}): mean={chosen_mean:.4f} "
          f"std={chosen_std:.4f}，不低于最好单模型 {best_single_mean:.4f}")
    # 先备份当前 LightGBM 版提交文件
    shutil.copyfile(f"{DATA}/submission.csv", f"{DATA}/submission_lightgbm.csv")
    print("已备份当前提交文件为 submission_lightgbm.csv")
    vc = VotingClassifier(estimators=make_estimators(), voting=chosen_voting)
    vc.fit(X, y)
    pred = vc.predict(X_test)
    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred.astype(int)})
    out = f"{DATA}/submission.csv"
    sub.to_csv(out, index=False)
    print("wrote:", out, "rows:", len(sub))
    print("predicted survived:", int(sub["Survived"].sum()), "/", len(sub))
else:
    print(f"投票集成 ({chosen_voting}) mean={chosen_mean:.4f} 低于最好单模型 "
          f"{best_single_mean:.4f}，不覆盖提交文件，当前 submission.csv 仍为 LightGBM 版")
