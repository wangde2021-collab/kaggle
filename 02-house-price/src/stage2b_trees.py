"""Bounded tree tuning: <=30 trials AND <=360s wall per model (optuna timeout).
Order: cat -> lgbm -> xgb. On any failure, fall back to documented defaults.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optuna
import config as C, cv as CV, models as M

optuna.logging.set_verbosity(optuna.logging.WARNING)
train, test = CV.load_data()
folds = CV.load_folds(len(train))
with open(os.path.join(C.BASE, "configs", "f_best.json")) as fh:
    GROUPS = json.load(fh)["groups"]
WALL = 360


def run(name, make_fn, defaults):
    out = os.path.join(C.BASE, "configs", f"best_params_{name}.json")
    if os.path.exists(out):
        print(f"{name}: exists, skip", flush=True); return
    t0 = time.time()
    try:
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=C.SEED))
        def obj(trial):
            try:
                res = CV.run_cv(make_fn(trial), train, folds, reps=1)
                return res["mean"]
            except Exception:
                return 1.0
        study.optimize(obj, n_trials=30, timeout=WALL, show_progress_bar=False)
        done = [t for t in study.trials if t.value is not None and t.value < 0.9]
        if done:
            json.dump({"params": study.best_params, "rep0_rmsle": study.best_value,
                       "n_trials_done": len(done)}, open(out, "w"), indent=1)
            print(f"{name}: best {study.best_value:.4f} in {len(done)} trials "
                  f"({time.time()-t0:.0f}s) {study.best_params}", flush=True)
            return
        raise RuntimeError("no successful trial")
    except Exception as e:
        json.dump({"params": defaults, "rep0_rmsle": None,
                   "source": f"defaults_after_tuning_failure: {e}"},
                  open(out, "w"), indent=1)
        print(f"{name}: DEFAULTS used ({e})", flush=True)


def mk_cat(t):
    return lambda: M.make_cat(
        dict(learning_rate=t.suggest_float("learning_rate", 0.03, 0.1, log=True),
             depth=t.suggest_int("depth", 3, 6),
             l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 30, log=True),
             random_strength=t.suggest_float("random_strength", 0.1, 3, log=True),
             bagging_temperature=t.suggest_float("bagging_temperature", 0, 3)),
        GROUPS)


def mk_lgbm(t):
    return lambda: M.make_lgbm(
        dict(learning_rate=t.suggest_float("learning_rate", 0.02, 0.08, log=True),
             num_leaves=t.suggest_int("num_leaves", 7, 31),
             max_depth=t.suggest_int("max_depth", 3, 6),
             min_child_samples=t.suggest_int("min_child_samples", 10, 40),
             subsample=t.suggest_float("subsample", 0.6, 1.0),
             colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
             reg_lambda=t.suggest_float("reg_lambda", 0.1, 30, log=True),
             reg_alpha=t.suggest_float("reg_alpha", 1e-3, 3, log=True)), GROUPS)


def mk_xgb(t):
    return lambda: M.make_xgb(
        dict(learning_rate=t.suggest_float("learning_rate", 0.02, 0.08, log=True),
             max_depth=t.suggest_int("max_depth", 2, 5),
             min_child_weight=t.suggest_float("min_child_weight", 0.5, 10, log=True),
             subsample=t.suggest_float("subsample", 0.6, 1.0),
             colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
             reg_lambda=t.suggest_float("reg_lambda", 0.1, 30, log=True),
             reg_alpha=t.suggest_float("reg_alpha", 1e-3, 3, log=True),
             gamma=t.suggest_float("gamma", 1e-3, 3, log=True)), GROUPS)


run("cat", mk_cat, dict(learning_rate=0.03, depth=5, l2_leaf_reg=5.0))
run("lgbm", mk_lgbm, dict(learning_rate=0.03, num_leaves=15, max_depth=4,
                          min_child_samples=20, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=5.0))
run("xgb", mk_xgb, dict(learning_rate=0.03, max_depth=3, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=3, reg_lambda=5.0))
print("tree tuning done", flush=True)
