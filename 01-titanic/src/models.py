"""模型池与预处理器。kind: linear / tree / xgb / lgbm / cat。

早停：xgb/lgbm/cat 在 CV/调参时用验证折早停（early_stopping_rounds=100），
记录每折最佳迭代数；全量重训时去掉早停，n_estimators=平均最佳迭代数*1.15。
"""
import json
import os

import lightgbm as lgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import BASE

N_JOBS = 2
EARLY_STOP = 100


def build_preprocessor(kind, num_cols, cat_cols):
    """返回 (preprocessor, cat_feature_indices 或 None)。
    catboost 用序数编码并返回类别列在变换后矩阵中的位置。"""
    if kind == "cat":
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
        ])
        ct = ColumnTransformer([("num", num_pipe, num_cols),
                                ("cat", cat_pipe, cat_cols)])
        cat_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
        return ct, cat_idx
    if kind == "linear":
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler())])
    else:  # tree / xgb / lgbm：数值只中位数填补，不标准化
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ct = ColumnTransformer([("num", num_pipe, num_cols),
                            ("cat", cat_pipe, cat_cols)])
    return ct, None


def make_model(name, params=None, seed=42, full_train=False, n_estimators=None):
    """返回 (kind, estimator)。params 覆盖默认参数；full_train=True 时去掉早停，
    n_estimators 由调用方按 CV 平均最佳迭代数*1.15 传入。"""
    p = dict(params or {})
    if name == "lr":
        d = dict(C=1.0, max_iter=1000, random_state=seed)
        d.update(p)
        return "linear", LogisticRegression(**d)
    if name == "svc":
        d = dict(C=1.0, gamma="scale", probability=True, random_state=seed)
        d.update(p)
        return "linear", SVC(**d)
    if name == "knn":
        d = dict(n_neighbors=5, weights="uniform")
        d.update(p)
        return "linear", KNeighborsClassifier(**d)
    if name == "rf":
        d = dict(n_estimators=500, max_depth=8, min_samples_leaf=2,
                 max_features="sqrt", random_state=seed, n_jobs=N_JOBS)
        d.update(p)
        if n_estimators:
            d["n_estimators"] = n_estimators
        return "tree", RandomForestClassifier(**d)
    if name == "et":
        d = dict(n_estimators=500, max_depth=8, min_samples_leaf=2,
                 max_features="sqrt", random_state=seed, n_jobs=N_JOBS)
        d.update(p)
        if n_estimators:
            d["n_estimators"] = n_estimators
        return "tree", ExtraTreesClassifier(**d)
    if name == "xgb":
        d = dict(n_estimators=1000, learning_rate=0.05, max_depth=3,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                 reg_lambda=1.0, random_state=seed, n_jobs=N_JOBS,
                 eval_metric="logloss")
        if not full_train:
            d["early_stopping_rounds"] = EARLY_STOP
        d.update(p)
        if n_estimators:
            d["n_estimators"] = n_estimators
        return "xgb", XGBClassifier(**d)
    if name == "lgbm":
        d = dict(n_estimators=1000, learning_rate=0.05, num_leaves=15,
                 max_depth=-1, min_child_samples=20, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                 random_state=seed, n_jobs=N_JOBS, verbose=-1)
        d.update(p)
        if n_estimators:
            d["n_estimators"] = n_estimators
        return "lgbm", LGBMClassifier(**d)
    if name == "cat":
        d = dict(iterations=1000, learning_rate=0.05, depth=6,
                 l2_leaf_reg=3.0, random_seed=seed, verbose=0,
                 thread_count=N_JOBS)
        if not full_train:
            d["early_stopping_rounds"] = EARLY_STOP
        d.update(p)
        if n_estimators:
            d["iterations"] = n_estimators
        return "cat", CatBoostClassifier(**d)
    raise ValueError(f"unknown model {name}")


def load_best_params(name):
    path = os.path.join(BASE, "configs", f"best_params_{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_best_params(name, params):
    path = os.path.join(BASE, "configs", f"best_params_{name}.json")
    with open(path, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True)
    return path
