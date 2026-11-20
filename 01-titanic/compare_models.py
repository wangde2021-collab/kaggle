"""Titanic 策略 C：模型对比再交。
逻辑回归 vs 随机森林 vs XGBoost vs LightGBM，同一套 5 折划分（与基线一致），
选 5 折均值最高的模型重训并生成 submission.csv。
特征和预处理与基线 baseline.py 完全一致。
"""
import shutil

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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

# 与基线一致的 5 折：cross_val_score(cv=5) 默认就是不打乱的 KFold(5)
kf = KFold(n_splits=5, shuffle=False)

models = {
    "逻辑回归": Pipeline([
        ("prep", preprocess),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "随机森林": Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ]),
    "XGBoost": Pipeline([
        ("prep", preprocess),
        ("clf", XGBClassifier(n_estimators=300, random_state=42, n_jobs=-1,
                               eval_metric="logloss")),
    ]),
    "LightGBM": Pipeline([
        ("prep", preprocess),
        ("clf", LGBMClassifier(n_estimators=300, random_state=42, n_jobs=-1, verbose=-1)),
    ]),
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
    results[name] = (scores, model)
    print(f"{name}: mean={scores.mean():.4f} std={scores.std():.4f} "
          f"folds={[round(s, 4) for s in scores]}")

best_name = max(results, key=lambda n: results[n][0].mean())
best_model = results[best_name][1]
print(f"胜出: {best_name} (mean={results[best_name][0].mean():.4f})")

# 备份基线提交文件，再用胜出模型在全量训练集上重训并生成新提交文件
shutil.copyfile(f"{DATA}/submission.csv", f"{DATA}/submission_baseline.csv")
best_model.fit(X, y)
pred = best_model.predict(X_test)

sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred.astype(int)})
out = f"{DATA}/submission.csv"
sub.to_csv(out, index=False)
print("wrote:", out, "rows:", len(sub))
print("predicted survived:", int(sub["Survived"].sum()), "/", len(sub))
print(sub.head(10).to_string(index=False))
