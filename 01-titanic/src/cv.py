"""验证方案：冻结 25 折、统一 run_cv、配对比较 compare、分组 CV。"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     StratifiedGroupKFold,
                                     StratifiedKFold)

from config import BASE, FOLDS_PATH, SEED
from features import GroupSurvivalEncoder, TitleAgeImputer
from models import build_preprocessor

EXP_COLUMNS = ["exp_id", "date", "feature_set", "model", "params_file",
               "n_seeds", "cv_scheme", "acc_mean", "acc_std", "acc_se",
               "logloss_mean", "auc_mean", "group_cv_acc", "vs",
               "d_mean", "wins", "submitted", "public_lb", "note"]


def make_folds(y, path=FOLDS_PATH):
    """生成冻结的 RepeatedStratifiedKFold(5折x5次, seed=42)，存 folds.csv。"""
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    # split() 一次产出 25 个 (tr, va)，按顺序每 5 个为一个 repeat。
    fold_mat = np.zeros((len(y), 5), dtype=int)
    splits = list(rskf.split(np.zeros(len(y)), y))
    for i, (tr, va) in enumerate(splits):
        rep, fold = divmod(i, 5)
        fold_mat[va, rep] = fold
    df = pd.DataFrame(fold_mat, columns=[f"rep{r}_fold" for r in range(5)])
    df.insert(0, "PassengerId", pd.read_csv(
        os.path.join(BASE, "data", "raw", "train.csv"))["PassengerId"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_folds(path=FOLDS_PATH):
    return pd.read_csv(path)


def fold_lists_from_df(folds_df):
    """folds.csv -> 25 个 (tr_idx, va_idx)，顺序 rep0_fold0..rep4_fold4。"""
    out = []
    for rep in range(5):
        col = folds_df[f"rep{rep}_fold"].values
        for fold in range(5):
            va = np.where(col == fold)[0]
            tr = np.where(col != fold)[0]
            out.append((tr, va))
    return out


def _fit_predict_fold(X, y, X_test, tr, va, num_cols, cat_cols,
                      model_fn, seed, use_g8, use_group_age):
    """单折：折内 fit 全部 Transformer/预处理器/模型，返回验证概率、test 概率、最佳迭代数。"""
    X_tr = X.iloc[tr].copy()
    X_va = X.iloc[va].copy()
    X_te = X_test.copy()
    y_tr, y_va = y.iloc[tr], y.iloc[va]
    if use_group_age:
        imp = TitleAgeImputer().fit(X_tr)
        X_tr, X_va, X_te = imp.transform(X_tr), imp.transform(X_va), imp.transform(X_te)
    if use_g8:
        enc = GroupSurvivalEncoder().fit(X_tr, y_tr)
        X_tr, X_va, X_te = enc.transform(X_tr), enc.transform(X_va), enc.transform(X_te)
    kind, model = model_fn(seed)
    pre, cat_idx = build_preprocessor(kind, num_cols, cat_cols)
    Xtr_t = pre.fit_transform(X_tr)
    Xva_t = pre.transform(X_va)
    Xte_t = pre.transform(X_te)
    if kind == "cat":
        # CatBoost 要求类别列为 int 或 str：把变换后的矩阵拼成 DataFrame，
        # 数值列保持 float，类别列（序数编码输出为 float）转 int。
        def _to_cat_frame(arr):
            df = pd.DataFrame(np.asarray(arr),
                              columns=list(num_cols) + list(cat_cols))
            for c in cat_cols:
                df[c] = df[c].astype(int)
            return df
        Xtr_t = _to_cat_frame(Xtr_t)
        Xva_t = _to_cat_frame(Xva_t)
        Xte_t = _to_cat_frame(Xte_t)
    fit_kw = {}
    if kind == "xgb":
        fit_kw = {"eval_set": [(Xva_t, y_va)], "verbose": False}
    elif kind == "lgbm":
        fit_kw = {"eval_set": [(Xva_t, y_va)],
                  "callbacks": [lgb.early_stopping(100, verbose=False)]}
    elif kind == "cat":
        fit_kw = {"eval_set": (Xva_t, y_va),
                  "cat_features": cat_idx, "verbose": 0}
    if kind == "cat" and "cat_features" not in fit_kw:
        fit_kw["cat_features"] = cat_idx
    model.fit(Xtr_t, y_tr, **fit_kw)
    best_iter = None
    if kind == "xgb":
        best_iter = getattr(model, "best_iteration", None)
    elif kind == "lgbm":
        best_iter = getattr(model, "best_iteration_", None)
    elif kind == "cat":
        try:
            best_iter = model.get_best_iteration()
        except Exception:
            best_iter = None
    p_va = model.predict_proba(Xva_t)[:, 1]
    p_te = model.predict_proba(Xte_t)[:, 1]
    return p_va, p_te, best_iter


def run_cv(exp_id, X, y, X_test, folds_df, num_cols, cat_cols, model_fn,
           seeds=(42,), use_g8=False, use_group_age=False, save=True):
    """25 折统一 CV。返回 summary dict；保存 oof/{exp_id}.npy (5,891)，
    test_preds/{exp_id}.npy (418,)，logs/fold_metrics_{exp_id}.csv。"""
    fold_list = fold_lists_from_df(folds_df)
    n = len(y)
    oof = np.zeros((5, n))
    test_sum = np.zeros(len(X_test))
    best_iters = []
    rows = []
    for seed in seeds:
        for i, (tr, va) in enumerate(fold_list):
            rep, fold = divmod(i, 5)
            p_va, p_te, bi = _fit_predict_fold(
                X, y, X_test, tr, va, num_cols, cat_cols, model_fn, seed,
                use_g8, use_group_age)
            oof[rep, va] += p_va / len(seeds)
            test_sum += p_te / (len(seeds) * len(fold_list))
            if bi is not None:
                best_iters.append(bi)
    # 指标以种子平均后的 oof 为准重算每折指标。
    rows = []
    for i, (tr, va) in enumerate(fold_list):
        rep, fold = divmod(i, 5)
        p = oof[rep, va]
        pred = (p >= 0.5).astype(int)
        rows.append({
            "rep": rep, "fold": fold,
            "acc": accuracy_score(y.iloc[va], pred),
            "logloss": log_loss(y.iloc[va], p, labels=[0, 1]),
            "auc": roc_auc_score(y.iloc[va], p),
        })
    fm = pd.DataFrame(rows)
    if save:
        os.makedirs(os.path.join(BASE, "oof"), exist_ok=True)
        os.makedirs(os.path.join(BASE, "test_preds"), exist_ok=True)
        os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
        np.save(os.path.join(BASE, "oof", f"{exp_id}.npy"), oof)
        np.save(os.path.join(BASE, "test_preds", f"{exp_id}.npy"), test_sum)
        fm.to_csv(os.path.join(BASE, "logs", f"fold_metrics_{exp_id}.csv"),
                  index=False)
    acc = fm["acc"].values
    summary = {
        "exp_id": exp_id,
        "acc_mean": float(acc.mean()), "acc_std": float(acc.std(ddof=1)),
        "acc_se": float(acc.std(ddof=1) / np.sqrt(len(acc))),
        "logloss_mean": float(fm["logloss"].mean()),
        "auc_mean": float(fm["auc"].mean()),
        "best_iter_mean": (float(np.mean(best_iters)) if best_iters else None),
        "best_iter_n": len(best_iters),
        "fold_metrics": fm,
    }
    return summary


def run_group_cv(X, y, X_test, groups, num_cols, cat_cols, model_fn,
                 seeds=(42,), use_g8=False, use_group_age=False):
    """StratifiedGroupKFold(5折, 分组=Ticket)稳健性检查，只返回均值与逐折分数。"""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_list = list(sgkf.split(X, y, groups))
    n = len(y)
    oof = np.zeros(n)
    rows = []
    for tr, va in fold_list:
        p_va, _, _ = _fit_predict_fold(
            X, y, X_test, tr, va, num_cols, cat_cols, model_fn, seeds[0],
            use_g8, use_group_age)
        oof[va] = p_va
        pred = (p_va >= 0.5).astype(int)
        rows.append(accuracy_score(y.iloc[va], pred))
    return {"acc_mean": float(np.mean(rows)), "fold_acc": [round(v, 4) for v in rows],
            "oof": oof}


def compare(exp_a, exp_b):
    """配对比较：读两个实验的 fold_metrics，按 25 折逐折差值比较。
    返回 d 均值/标准差/标准误、A 胜折数、配对 t 检验 p 值。"""
    def _load(e):
        p = os.path.join(BASE, "logs", f"fold_metrics_{e}.csv")
        return pd.read_csv(p).sort_values(["rep", "fold"])["acc"].values
    a, b = _load(exp_a), _load(exp_b)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    return {"d_mean": float(d.mean()), "d_std": float(d.std(ddof=1)),
            "d_se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "wins_a": int((d > 0).sum()), "ties": int((d == 0).sum()),
            "n": len(d), "t": float(t), "p": float(p)}


def log_experiment(row):
    """向 logs/experiments.csv 追加一行（建表头如果不存在）。"""
    path = os.path.join(BASE, "logs", "experiments.csv")
    full = {c: "" for c in EXP_COLUMNS}
    full.update({k: v for k, v in row.items() if k in EXP_COLUMNS})
    df = pd.DataFrame([full], columns=EXP_COLUMNS)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
