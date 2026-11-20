"""第 5 阶段：模型池 + Optuna 调参（目标 rep0 的 5 折平均 Accuracy），
调完在全部 25 折上重评并登记；树/boosting 用 5 种子平均（42-46）。

预算缩减说明（机器 4 核、内存紧张，如实记录）：LR/SVC/KNN 各 30/30/20 次，
RF/ET 各 30 次，XGB/LGBM 各 50 次，CAT 30 次（技能原预算 LR/SVC/KNN 50、RF/ET 50、
XGB/LGBM/CAT 100）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "workspace",
                                "kaggle-data", "titanic", "src"))
from datetime import date
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, log_loss

from config import BASE, F_BEST_CAT, F_BEST_NUM, F_BEST_USE_GROUP_AGE
from cv import fold_lists_from_df, load_folds, log_experiment, run_cv
from features import add_nonlabel_features
from models import (load_best_params, make_model, save_best_params)

optuna.logging.set_verbosity(optuna.logging.WARNING)

train = pd.read_csv(os.path.join(BASE, "data", "raw", "train.csv"))
test = pd.read_csv(os.path.join(BASE, "data", "raw", "test.csv"))
y = train["Survived"]
folds = load_folds()
fold_list = fold_lists_from_df(folds)
rep0 = fold_list[:5]
today = str(date.today())
tr, te = add_nonlabel_features(train, test)
X = tr[F_BEST_NUM + F_BEST_CAT]
X_test = te[F_BEST_NUM + F_BEST_CAT]

from cv import _fit_predict_fold


def rep0_score(model_name, params):
    """rep0 5 折平均 Accuracy（调参目标），同时返回平均 LogLoss。"""
    accs, lls = [], []
    for tr_idx, va_idx in rep0:
        p_va, _, _ = _fit_predict_fold(
            X, y, X_test, tr_idx, va_idx, F_BEST_NUM, F_BEST_CAT,
            lambda seed: make_model(model_name, params=params, seed=seed),
            42, False, F_BEST_USE_GROUP_AGE)
        accs.append(accuracy_score(y.iloc[va_idx], (p_va >= 0.5).astype(int)))
        lls.append(log_loss(y.iloc[va_idx], p_va, labels=[0, 1]))
    return float(np.mean(accs)), float(np.mean(lls))


def suggest(trial, model):
    if model == "lr":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if model == "svc":
        return {"C": trial.suggest_float("C", 1e-2, 1e2, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True)}
    if model == "knn":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "p": trial.suggest_categorical("p", [1, 2])}
    if model in ("rf", "et"):
        return {"max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", 0.5, 0.7])}
    if model == "xgb":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True)}
    if model == "lgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 8, 31),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True)}
    if model == "cat":
        return {"iterations": trial.suggest_int("iterations", 200, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "depth": trial.suggest_int("depth", 4, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20)}
    raise ValueError(model)


TRIALS = {"lr": 30, "svc": 30, "knn": 20, "rf": 30, "et": 30,
          "xgb": 50, "lgbm": 50, "cat": 30}
SEED_AVG = {"rf", "et", "xgb", "lgbm", "cat"}

results = {}
for model in ["lr", "svc", "knn", "rf", "et", "xgb", "lgbm", "cat"]:
    if os.path.exists(os.path.join(BASE, "configs", f"best_params_{model}.json")):
        print(f"调参 {model}: 已有 best_params，跳过调参", flush=True)
        results[model] = load_best_params(model)
        continue
    def objective(trial, model=model):
        params = suggest(trial, model)
        acc, ll = rep0_score(model, params)
        trial.set_user_attr("logloss", ll)
        return acc
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=TRIALS[model], show_progress_bar=False)
    best = study.best_params
    # 平分选 LogLoss 低的：取最优 acc 附近（差 <1e-4）的 trial 中 logloss 最低者
    best_acc = study.best_value
    cands = [t for t in study.trials
             if t.value is not None and best_acc - t.value < 1e-4]
    pick = min(cands, key=lambda t: t.user_attrs.get("logloss", 99))
    best = pick.params
    save_best_params(model, best)
    print(f"调参 {model}: {TRIALS[model]} trials, rep0 acc={pick.value:.4f} "
          f"logloss={pick.user_attrs.get('logloss', float('nan')):.4f} params={best}",
          flush=True)
    results[model] = best

# 全部 25 折重评并登记（已有 OOF 文件则跳过，避免断点重跑）
for model in ["lr", "svc", "knn", "rf", "et", "xgb", "lgbm", "cat"]:
    exp_id = f"tuned_{model}"
    if os.path.exists(os.path.join(BASE, "oof", f"{exp_id}.npy")):
        print(f"25折 {model}: 已有 OOF，跳过重评", flush=True)
        continue
    params = load_best_params(model)
    seeds = (42, 43, 44, 45, 46) if model in SEED_AVG else (42,)
    exp_id = f"tuned_{model}"
    s = run_cv(exp_id, X, y, X_test, folds, F_BEST_NUM, F_BEST_CAT,
               lambda seed: make_model(model, params=params, seed=seed),
               seeds=seeds, use_group_age=F_BEST_USE_GROUP_AGE)
    print(f"25折 {model} ({len(seeds)}种子): acc={s['acc_mean']:.4f}±{s['acc_std']:.4f} "
          f"se={s['acc_se']:.4f} logloss={s['logloss_mean']:.4f} auc={s['auc_mean']:.4f} "
          f"best_iter_mean={s['best_iter_mean']}", flush=True)
    log_experiment({"exp_id": exp_id, "date": today,
                    "feature_set": "F_best",
                    "model": model, "params_file": f"configs/best_params_{model}.json",
                    "n_seeds": len(seeds), "cv_scheme": "RSKF 5x5 seed42",
                    "acc_mean": round(s["acc_mean"], 4),
                    "acc_std": round(s["acc_std"], 4),
                    "acc_se": round(s["acc_se"], 4),
                    "logloss_mean": round(s["logloss_mean"], 4),
                    "auc_mean": round(s["auc_mean"], 4), "group_cv_acc": "",
                    "vs": "", "d_mean": "", "wins": "", "submitted": "no",
                    "public_lb": "",
                    "note": f"调参后25折重评，rep0目标；best_iter_mean={s['best_iter_mean']}"})
print("STAGE5 DONE")
