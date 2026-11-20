"""Titanic 基线模型：简单清洗 + 随机森林，5 折交叉验证，生成 submission.csv。"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

model = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
])

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("5-fold CV accuracy: mean=%.4f std=%.4f" % (scores.mean(), scores.std()))
print("fold scores:", [round(s, 4) for s in scores])

model.fit(X, y)
pred = model.predict(X_test)

sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred.astype(int)})
out = f"{DATA}/submission.csv"
sub.to_csv(out, index=False)
print("wrote:", out, "rows:", len(sub))
print("predicted survived:", int(sub["Survived"].sum()), "/", len(sub))
print(sub.head(10).to_string(index=False))
