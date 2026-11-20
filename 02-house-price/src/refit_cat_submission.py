"""Local refit: reproduce the server's winning CatBoost submission.

Replicates exactly the full-train refit block of src/stage3_final.py for the
winner (cat) only, using the server-retrieved tuned params and best_iter.
Server run: 2026-11-21 03:17, cat 30-fold RMSLE 0.1196+-0.0197.
Does NOT re-run CV. Does NOT submit to Kaggle.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C, cv as CV, models as M

DATE = "2026-11-20"
train, test = CV.load_data()
y_price = train[C.TARGET].values.astype(float)
y_log = np.log1p(y_price)
X = train.drop(columns=[C.TARGET])
X_test = test.copy()
with open(os.path.join(C.BASE, "configs", "f_best.json")) as fh:
    GROUPS = json.load(fh)["groups"]
with open(os.path.join(C.BASE, "configs", "best_params_cat.json")) as fh:
    params = json.load(fh)["params"]
BEST_ITER = 1838.1666666666667  # server stage3 mean best_iter for cat
fi = int(round(BEST_ITER * 1.15))
keep_mask = ~CV.dirty_mask(X, y_price)
X_full, y_full = X[keep_mask], y_log[keep_mask]
preds = []
t0 = time.time()
for s in range(5):
    p = dict(params); p["random_seed"] = C.SEED + s
    est = M.make_cat(p, GROUPS, fixed_iters=fi)
    est.fit(X_full, y_full)
    preds.append(np.asarray(est.predict(X_test), dtype=float))
    print(f"seed {s} done ({time.time()-t0:.0f}s)", flush=True)
final_log = np.mean(preds, axis=0)
price = np.clip(np.expm1(final_log), 1000, None)
sub = pd.DataFrame({"Id": test["Id"].values, "SalePrice": price})
assert len(sub) == 1459 and sub["Id"].equals(test["Id"])
assert sub["SalePrice"].notna().all() and (sub["SalePrice"] > 0).all()
med = float(sub["SalePrice"].median())
assert 140000 <= med <= 190000, f"median out of range: {med}"
os.makedirs(os.path.join(C.BASE, "submissions"), exist_ok=True)
fname = "sub_001_cat_cv0.1196.csv"
sub.to_csv(os.path.join(C.BASE, "submissions", fname), index=False)
sub.to_csv(os.path.join(C.BASE, "submission.csv"), index=False)
idx_path = os.path.join(C.BASE, "submissions", "index.csv")
new_row = pd.DataFrame([dict(n=1, exp_id="cat", cv_mean=0.1196, cv_std=0.0197,
                    submitted="no", public_lb="", time=DATE,
                    notes=f"file={fname}; median_pred={med:.0f}")])
if os.path.exists(idx_path):
    old = pd.read_csv(idx_path)
    new_row = pd.concat([old, new_row], ignore_index=True)
new_row.to_csv(idx_path, index=False)
print(f"submission written: {fname} median={med:.0f} "
      f"mean={price.mean():.0f} min={price.min():.0f} max={price.max():.0f}",
      flush=True)
