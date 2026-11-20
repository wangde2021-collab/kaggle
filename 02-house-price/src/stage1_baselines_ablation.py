"""Stage 1: baselines on frozen 30 folds (F0) + feature ablation on rep0."""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C, cv as CV, models as M

DATE = "2026-11-20"
GROUP_ORDER = list(C.FEATURE_GROUPS.keys())
train, test = CV.load_data()
folds = CV.load_folds(len(train))


class MedianModel:
    def fit(self, X, y):
        self.m_ = float(np.median(y)); return self
    def predict(self, X):
        return np.full(len(X), self.m_)


def run_and_log(exp_id, make_est, feature_set, model_name, reps=None,
                compare_to=None, base_scores=None, notes=""):
    t0 = time.time()
    res = CV.run_cv(make_est, train, folds, reps=reps)
    CV.save_oof(exp_id, res["oof"]); CV.save_scores(exp_id, res["scores"])
    row = dict(exp_id=exp_id, date=DATE, feature_set=feature_set, model=model_name,
               params_file="", seeds=1,
               cv_scheme=f"{C.N_SPLITS}x{reps or C.N_REPS}", rmsle_mean=round(res["mean"], 4),
               rmsle_std=round(res["std"], 4), rmsle_se=round(res["se"], 4),
               submitted="no", notes=notes)
    if compare_to and base_scores is not None:
        cmp_ = CV.compare(res["scores"], base_scores)
        row.update(compare_to=compare_to, d_mean=round(cmp_["d_mean"], 4),
                   wins=cmp_["wins_a"])
    CV.log_experiment(row)
    print(f"{exp_id}: RMSLE {res['mean']:.4f}±{res['std']:.4f}  ({time.time()-t0:.0f}s)",
          flush=True)
    return res


print("=== Baselines (F0, 30 folds) ===", flush=True)
b_med = run_and_log("B_median", lambda: MedianModel(), "F0", "median")
b_ridge = run_and_log("B_ridge", lambda: M.make_ridge(groups=[]), "F0",
                      "ridge_default", compare_to="B_median",
                      base_scores=b_med["scores"])
b_lgbm = run_and_log("B_lgbm", lambda: M.make_lgbm(groups=[]), "F0",
                     "lgbm_default", compare_to="B_median",
                     base_scores=b_med["scores"])

print("=== Ablation (rep0, 10 folds) ===", flush=True)
CAT_ABL = dict(learning_rate=0.06, depth=5, l2_leaf_reg=5.0)


def abl_models(groups):
    return {
        "ridge": lambda: M.make_ridge(groups=groups),
        "cat": lambda: M.make_cat(params=CAT_ABL, groups=groups, fixed_iters=500),
    }


ablation = {"cumulative": [], "reverse": []}
prev = {}
cur_groups = []
for step, g in enumerate([None] + GROUP_ORDER):
    if g is not None:
        cur_groups = cur_groups + [g]
    entry = {"step": step, "added": g or "F0", "groups": list(cur_groups), "scores": {}}
    for mname, mk in abl_models(cur_groups).items():
        res = CV.run_cv(mk, train, folds, reps=1)
        entry["scores"][mname] = dict(mean=res["mean"], std=res["std"],
                                      folds=res["scores"].tolist())
        if mname in prev:
            cmp_ = CV.compare(res["scores"], prev[mname])
            entry.setdefault("vs_prev", {})[mname] = cmp_
            print(f"  cum {entry['added']} {mname}: {res['mean']:.4f} "
                  f"d={cmp_['d_mean']:+.4f} wins={cmp_['wins_a']}/10", flush=True)
        else:
            print(f"  cum F0 {mname}: {res['mean']:.4f}", flush=True)
        prev[mname] = res["scores"]
    ablation["cumulative"].append(entry)

full_groups = list(GROUP_ORDER)
full_scores = {e["groups"] == full_groups and m: None for e in [ablation["cumulative"][-1]] for m in []}
full_entry = ablation["cumulative"][-1]
full_scores = {m: np.array(full_entry["scores"][m]["folds"]) for m in ("ridge", "cat")}
print("--- reverse ablation (remove one group from full) ---", flush=True)
drop_candidates = []
for g in GROUP_ORDER:
    sub = [x for x in full_groups if x != g]
    entry = {"removed": g, "groups": sub, "scores": {}}
    for mname, mk in abl_models(sub).items():
        res = CV.run_cv(mk, train, folds, reps=1)
        cmp_ = CV.compare(res["scores"], full_scores[mname])
        entry["scores"][mname] = dict(mean=res["mean"], std=res["std"],
                                      folds=res["scores"].tolist(), vs_full=cmp_)
        print(f"  -{g} {mname}: {res['mean']:.4f} d_vs_full={cmp_['d_mean']:+.4f} "
              f"wins={cmp_['wins_a']}/10", flush=True)
    ablation["reverse"].append(entry)
    # removal does not hurt either model -> drop candidate
    if all(entry["scores"][m]["vs_full"]["d_mean"] <= 0.0005 for m in ("ridge", "cat")):
        drop_candidates.append(g)

# cumulative support: group kept if >=1 model improved (d<0, wins>=6) and the
# other model not clearly worse (d <= +0.002)
kept = []
for entry in ablation["cumulative"][1:]:
    g = entry["added"]
    vs = entry["vs_prev"]
    improved = any(vs[m]["d_mean"] < 0 and vs[m]["wins_a"] >= 6 for m in vs)
    not_worse = all(vs[m]["d_mean"] <= 0.002 for m in vs)
    if improved and not_worse:
        kept.append(g)
f_best = [g for g in kept if g not in drop_candidates]
ablation["kept_after_cumulative"] = kept
ablation["drop_after_reverse"] = drop_candidates
ablation["f_best"] = f_best
with open(os.path.join(C.BASE, "logs", "ablation.json"), "w") as fh:
    json.dump(ablation, fh, indent=1, default=float)
with open(os.path.join(C.BASE, "configs", "f_best.json"), "w") as fh:
    json.dump({"groups": f_best}, fh)
print("F_best =", f_best, flush=True)
