"""Stage 3: tuned singles on 30 folds, ensembles, decision gate,
full-train refits, versioned submission file (NOT submitted to Kaggle)."""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, time
import numpy as np
import pandas as pd
from scipy.optimize import nnls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C, cv as CV, models as M

DATE = "2026-11-20"
train, test = CV.load_data()
folds = CV.load_folds(len(train))
fm = CV.fold_matrix(folds)
y_price = train[C.TARGET].values.astype(float)
y_log = np.log1p(y_price)
X = train.drop(columns=[C.TARGET])
X_test = test.copy()
with open(os.path.join(C.BASE, "configs", "f_best.json")) as fh:
    GROUPS = json.load(fh)["groups"]

SINGLES = ["ridge", "elasticnet", "lgbm", "xgb", "cat"]  # huber/svr dropped (see experiments.csv)
MAKERS = {"ridge": M.make_ridge, "elasticnet": M.make_elasticnet,
          "lgbm": M.make_lgbm, "xgb": M.make_xgb, "cat": M.make_cat}
TREE = {"lgbm", "xgb", "cat"}


def load_params(name):
    with open(os.path.join(C.BASE, "configs", f"best_params_{name}.json")) as fh:
        return json.load(fh)["params"]


def run_cv_collect(make_est):
    """Like CV.run_cv but also collects best_iter_ per fold for GBDTs."""
    oof = np.full((C.N_REPS, len(train)), np.nan)
    scores, iters = [], []
    for r in range(C.N_REPS):
        for f in range(C.N_SPLITS):
            va = fm[r] == f; tr = ~va
            Xtr, Xva = X[tr], X[va]
            keep = ~CV.dirty_mask(Xtr, y_price[tr])
            est = make_est()
            est.fit(Xtr[keep], y_log[tr][keep])
            pred = np.asarray(est.predict(Xva), dtype=float)
            oof[r, va] = pred
            scores.append(CV.rmsle_fold(y_price[va], pred))
            iters.append(getattr(est, "best_iter_", None))
    return np.array(scores), oof, iters


results = {}
best_iters = {}
print("=== tuned singles, 30 folds ===", flush=True)
for name in SINGLES:
    params = load_params(name)
    t0 = time.time()
    scores, oof, iters = run_cv_collect(lambda: MAKERS[name](params, GROUPS))
    CV.save_oof(f"{name}_t", oof); CV.save_scores(f"{name}_t", scores)
    iters_ok = [i for i in iters if i]
    if iters_ok:
        best_iters[name] = float(np.mean(iters_ok))
    results[name] = dict(scores=scores, oof=oof, mean=float(np.mean(scores)),
                         std=float(np.std(scores, ddof=1)))
    CV.log_experiment(dict(
        exp_id=f"{name}_t", date=DATE, feature_set="+".join(GROUPS) or "F0",
        model=f"{name}_tuned", params_file=f"configs/best_params_{name}.json",
        seeds=1, cv_scheme="10x3", rmsle_mean=round(results[name]["mean"], 4),
        rmsle_std=round(results[name]["std"], 4),
        rmsle_se=round(results[name]["std"] / np.sqrt(30), 4), submitted="no",
        notes=f"mean best_iter={best_iters.get(name, '')}"))
    print(f"{name}_t: {results[name]['mean']:.4f}±{results[name]['std']:.4f} "
          f"best_iter~{best_iters.get(name)} ({time.time()-t0:.0f}s)", flush=True)

# ---------- diversity ----------
oof_mean = {n: results[n]["oof"].mean(axis=0) for n in SINGLES}
P = np.column_stack([oof_mean[n] for n in SINGLES])
corr = np.corrcoef(P.T)
print("OOF log-pred correlations:\n", np.round(corr, 3), flush=True)
keep = list(SINGLES)
for i in range(len(SINGLES)):
    for j in range(i + 1, len(SINGLES)):
        if corr[i, j] > 0.995 and SINGLES[j] in keep and SINGLES[i] in keep:
            worse = SINGLES[i] if results[SINGLES[i]]["mean"] > results[SINGLES[j]]["mean"] else SINGLES[j]
            keep.remove(worse)
            print(f"drop {worse} (corr>0.995 pair)", flush=True)
print("ensemble members:", keep, flush=True)


def fold_scores_from_oof(pred_by_rep):
    sc = []
    for r in range(C.N_REPS):
        for f in range(C.N_SPLITS):
            va = fm[r] == f
            sc.append(CV.rmsle_fold(y_price[va], pred_by_rep[r][va]))
    return np.array(sc)


# ---------- E1 equal weight ----------
e1_oof = np.mean([results[n]["oof"] for n in keep], axis=0)
e1_scores = fold_scores_from_oof(e1_oof)
CV.save_oof("E1_equal", e1_oof); CV.save_scores("E1_equal", e1_scores)
print(f"E1_equal: {e1_scores.mean():.4f}±{e1_scores.std(ddof=1):.4f}", flush=True)

# ---------- E2 NNLS weighted (weights fit out-of-fold per fold) ----------
def nnls_weights(A, b):
    w, _ = nnls(A, b)
    if w.sum() <= 0:
        w = np.ones(A.shape[1])
    return w / w.sum()


e2_oof = np.zeros_like(e1_oof)
for r in range(C.N_REPS):
    A_all = np.column_stack([results[n]["oof"][r] for n in keep])
    for f in range(C.N_SPLITS):
        va = fm[r] == f
        w = nnls_weights(A_all[~va], y_log[~va])
        e2_oof[r, va] = A_all[va] @ w
e2_scores = fold_scores_from_oof(e2_oof)
CV.save_oof("E2_nnls", e2_oof); CV.save_scores("E2_nnls", e2_scores)
A_full = np.column_stack([oof_mean[n] for n in keep])
w_final = nnls_weights(A_full, y_log)
print(f"E2_nnls: {e2_scores.mean():.4f}±{e2_scores.std(ddof=1):.4f} "
      f"weights={dict(zip(keep, np.round(w_final, 3)))}", flush=True)

best_single = min(SINGLES, key=lambda n: results[n]["mean"])
S = dict(exp=best_single, scores=results[best_single]["scores"],
         mean=results[best_single]["mean"], std=results[best_single]["std"])
for exp_id, sc in [("E1_equal", e1_scores), ("E2_nnls", e2_scores)]:
    cmp_ = CV.compare(sc, S["scores"])
    CV.log_experiment(dict(
        exp_id=exp_id, date=DATE, feature_set="+".join(GROUPS) or "F0",
        model="ensemble", params_file="", seeds=1, cv_scheme="10x3",
        rmsle_mean=round(float(np.mean(sc)), 4),
        rmsle_std=round(float(np.std(sc, ddof=1)), 4),
        rmsle_se=round(float(np.std(sc, ddof=1) / np.sqrt(30)), 4),
        compare_to=S["exp"], d_mean=round(cmp_["d_mean"], 4),
        wins=cmp_["wins_a"], submitted="no",
        notes=("members=" + ",".join(keep)) if exp_id == "E1_equal"
              else ("weights=" + json.dumps(dict(zip(keep, w_final.round(4)))))))
    print(f"{exp_id} vs {S['exp']}: d={cmp_['d_mean']:+.4f} wins={cmp_['wins_a']}/30",
          flush=True)

# ---------- decision gate ----------
def gate(cand_scores, s_scores):
    cmp_ = CV.compare(cand_scores, s_scores)
    ok1 = cmp_["d_mean"] <= -max(0.002, cmp_["d_se"])
    ok2 = cmp_["wins_a"] >= 18
    ok3 = np.std(cand_scores, ddof=1) <= np.std(s_scores, ddof=1) + 0.003
    return (ok1 and ok2 and ok3), cmp_

cands = {"E1_equal": e1_scores, "E2_nnls": e2_scores}
winner, winner_scores, gate_notes = S["exp"], S["scores"], {}
for cid, sc in cands.items():
    ok, cmp_ = gate(sc, S["scores"])
    gate_notes[cid] = (ok, cmp_)
    if ok and np.mean(sc) < np.mean(winner_scores):
        winner, winner_scores = cid, sc
print("WINNER:", winner, flush=True)

# ---------- full-train refits (all singles; trees 5 seeds, fixed iters) ----------
keep_mask = ~CV.dirty_mask(X, y_price)
X_full, y_full = X[keep_mask], y_log[keep_mask]
test_preds = {}
print("=== full-train refits ===", flush=True)
for name in SINGLES:
    params = load_params(name)
    t0 = time.time()
    if name in TREE:
        fi = int(round(best_iters[name] * 1.15))
        preds = []
        for s in range(5):
            p = dict(params)
            if name == "cat":
                p["random_seed"] = C.SEED + s
            else:
                p["random_state"] = C.SEED + s
            est = MAKERS[name](p, GROUPS, fixed_iters=fi)
            est.fit(X_full, y_full)
            preds.append(np.asarray(est.predict(X_test), dtype=float))
        tp = np.mean(preds, axis=0)
    else:
        est = MAKERS[name](params, GROUPS)
        est.fit(X_full, y_full)
        tp = np.asarray(est.predict(X_test), dtype=float)
    test_preds[name] = tp
    CV.save_test_pred(f"{name}_t", tp)
    print(f"{name}: full-train test pred done ({time.time()-t0:.0f}s)", flush=True)

if winner == "E1_equal":
    final_log = np.mean([test_preds[n] for n in keep], axis=0)
elif winner == "E2_nnls":
    final_log = np.column_stack([test_preds[n] for n in keep]) @ w_final
else:
    final_log = test_preds[winner]
CV.save_test_pred(winner if winner in SINGLES else winner, final_log)

price = np.expm1(final_log)
price = np.clip(price, 1000, None)
sub = pd.DataFrame({"Id": test["Id"].values, "SalePrice": price})
assert len(sub) == 1459 and sub["Id"].equals(test["Id"])
assert sub["SalePrice"].notna().all() and (sub["SalePrice"] > 0).all()
med = float(sub["SalePrice"].median())
assert 140000 <= med <= 190000, f"median out of range: {med}"
w_mean = float(np.mean(winner_scores)); w_std = float(np.std(winner_scores, ddof=1))
import glob, re as _re
existing = glob.glob(os.path.join(C.BASE, "submissions", "sub_*.csv"))
nums = [int(_re.search(r"sub_(\d+)", os.path.basename(p)).group(1)) for p in existing]
next_n = (max(nums) + 1) if nums else 1
fname = f"sub_{next_n:03d}_{winner}_cv{w_mean:.4f}.csv"
sub.to_csv(os.path.join(C.BASE, "submissions", fname), index=False)
sub.to_csv(os.path.join(C.BASE, "submission.csv"), index=False)
idx_path = os.path.join(C.BASE, "submissions", "index.csv")
new_row = pd.DataFrame([dict(n=next_n, exp_id=winner, cv_mean=round(w_mean, 4),
                    cv_std=round(w_std, 4), submitted="no", public_lb="",
                    time=DATE, notes=f"file={fname}; median_pred={med:.0f}")])
if os.path.exists(idx_path):
    old = pd.read_csv(idx_path)
    new_row = pd.concat([old, new_row], ignore_index=True)
new_row.to_csv(idx_path, index=False)
print(f"submission written: {fname} median={med:.0f} "
      f"mean={price.mean():.0f} min={price.min():.0f} max={price.max():.0f}",
      flush=True)

# ---------- decision note ----------
lines = [f"# 决策记录 {DATE}", "",
         f"- 当前最优单模 S = {S['exp']}：30 折 RMSLE {S['mean']:.4f}±{S['std']:.4f}",
         f"- 特征集 F_best = {GROUPS}", ""]
for cid, (ok, cmp_) in gate_notes.items():
    lines.append(f"- 候选 {cid}：均值 {np.mean(cands[cid]):.4f}，d={cmp_['d_mean']:+.4f} "
                 f"(se {cmp_['d_se']:.4f})，胜 {cmp_['wins_a']}/30，"
                 f"std {np.std(cands[cid], ddof=1):.4f} → 决策门 {'通过' if ok else '未通过'}")
lines += ["", f"**结论：最终方案 = {winner}**（30 折 RMSLE {w_mean:.4f}±{w_std:.4f}）。"
          f"提交文件 submissions/{fname}，未提交 Kaggle（等用户点头）。"]
if winner == "E2_nnls":
    lines.append(f"E2 权重：{dict(zip(keep, w_final.round(4)))}")
with open(os.path.join(C.BASE, "notes", f"decision_{DATE.replace('-', '')}.md"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("decision note written", flush=True)
