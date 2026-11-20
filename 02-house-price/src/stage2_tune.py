"""Stage 2: Optuna tuning on rep0 (10 folds), F_best features.
Budgets: linear/SVR 30 trials, trees 50 trials. Best params -> configs/.
Trees are tuned in early-stopping mode; mean best_iter recorded alongside.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optuna
import config as C, cv as CV, models as M

optuna.logging.set_verbosity(optuna.logging.WARNING)
DATE = "2026-11-20"
train, test = CV.load_data()
folds = CV.load_folds(len(train))
with open(os.path.join(C.BASE, "configs", "f_best.json")) as fh:
    GROUPS = json.load(fh)["groups"]
print("tuning on groups:", GROUPS, flush=True)


def objective_factory(make_fn):
    def obj(trial):
        try:
            est_fn = make_fn(trial)
            res = CV.run_cv(est_fn, train, folds, reps=1)
            return res["mean"]
        except Exception:
            return 1.0  # failed fit (e.g. Huber lbfgs) counts as a bad trial
    return obj


def run_study(name, make_fn, n_trials):
    out = os.path.join(C.BASE, "configs", f"best_params_{name}.json")
    if os.path.exists(out):
        print(f"{name}: already tuned, skip", flush=True)
        return None
    t0 = time.time()
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective_factory(make_fn), n_trials=n_trials,
                   show_progress_bar=False)
    print(f"{name}: best rep0 RMSLE {study.best_value:.4f} "
          f"params={study.best_params} ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(C.BASE, "configs", f"best_params_{name}.json"), "w") as fh:
        json.dump({"params": study.best_params, "rep0_rmsle": study.best_value},
                  fh, indent=1)
    return study.best_params


# ---- linear / SVR (30 trials each) ----
def mk_ridge(t):
    return lambda: M.make_ridge(
        dict(alpha=t.suggest_float("alpha", 0.01, 3000, log=True)), GROUPS)


def mk_enet(t):
    return lambda: M.make_elasticnet(
        dict(alpha=t.suggest_float("alpha", 1e-4, 0.05, log=True),
             l1_ratio=t.suggest_float("l1_ratio", 0.05, 0.95)), GROUPS)


def mk_huber(t):
    return lambda: M.make_huber(
        dict(alpha=t.suggest_float("alpha", 1e-4, 0.5, log=True),
             epsilon=t.suggest_float("epsilon", 1.1, 1.8),
             max_iter=500), GROUPS)


def mk_svr(t):
    return lambda: M.make_svr(
        dict(C=t.suggest_float("C", 0.5, 100, log=True),
             gamma=t.suggest_float("gamma", 0.001, 0.05, log=True),
             epsilon=t.suggest_float("epsilon", 0.01, 0.15)), GROUPS)


run_study("ridge", mk_ridge, 30)
run_study("elasticnet", mk_enet, 30)
run_study("huber", mk_huber, 30)
run_study("svr", mk_svr, 30)

# ---- trees (50 trials each, early-stopping mode) ----
def mk_lgbm(t):
    return lambda: M.make_lgbm(
        dict(learning_rate=t.suggest_float("learning_rate", 0.01, 0.08, log=True),
             num_leaves=t.suggest_int("num_leaves", 7, 31),
             max_depth=t.suggest_int("max_depth", 3, 6),
             min_child_samples=t.suggest_int("min_child_samples", 10, 40),
             subsample=t.suggest_float("subsample", 0.6, 1.0),
             colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
             reg_lambda=t.suggest_float("reg_lambda", 0.1, 30, log=True),
             reg_alpha=t.suggest_float("reg_alpha", 1e-3, 3, log=True)), GROUPS)


def mk_xgb(t):
    return lambda: M.make_xgb(
        dict(learning_rate=t.suggest_float("learning_rate", 0.01, 0.08, log=True),
             max_depth=t.suggest_int("max_depth", 2, 5),
             min_child_weight=t.suggest_float("min_child_weight", 0.5, 10, log=True),
             subsample=t.suggest_float("subsample", 0.6, 1.0),
             colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
             reg_lambda=t.suggest_float("reg_lambda", 0.1, 30, log=True),
             reg_alpha=t.suggest_float("reg_alpha", 1e-3, 3, log=True),
             gamma=t.suggest_float("gamma", 1e-3, 3, log=True)), GROUPS)


def mk_cat(t):
    return lambda: M.make_cat(
        dict(learning_rate=t.suggest_float("learning_rate", 0.01, 0.08, log=True),
             depth=t.suggest_int("depth", 3, 6),
             l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 30, log=True),
             random_strength=t.suggest_float("random_strength", 0.1, 3, log=True),
             bagging_temperature=t.suggest_float("bagging_temperature", 0, 3)),
        GROUPS)


run_study("lgbm", mk_lgbm, 50)
run_study("xgb", mk_xgb, 50)
run_study("cat", mk_cat, 50)
print("tuning done", flush=True)
