# Watson v1 (CPU): Clean 5-fold XLM-R-large + OOF-tuned Premise-Group Joint Decoding
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)


def find_file(name):
    for root, _dirs, files in os.walk("/kaggle/input", followlinks=True):
        if name in files:
            return Path(root) / name
    return None


def find_comp_dir():
    for root, _dirs, files in os.walk("/kaggle/input", followlinks=True):
        if {"train.csv", "test.csv"} <= set(files):
            return Path(root)
    raise RuntimeError("train.csv and test.csv not found")


def norm_text(s):
    return " ".join(str(s).strip().lower().split())


def joint_decode_subset(probs, norm_premises, known_counts, lam_known, lam_dup):
    log_p = np.log(np.clip(probs.astype(np.float64), 1e-7, 1.0))
    for i, p in enumerate(norm_premises):
        cnt = known_counts.get(p)
        if cnt is not None and lam_known > 0.0:
            log_p[i] -= lam_known * cnt

    preds = log_p.argmax(axis=1).copy()
    groups = {}
    for i, p in enumerate(norm_premises):
        groups.setdefault(p, []).append(i)

    for _p, idxs in groups.items():
        k = len(idxs)
        if k < 2 or lam_dup <= 0.0 or k > 6:
            continue
        best_score = -1e18
        best_assign = None
        for assign in itertools.product((0, 1, 2), repeat=k):
            score = sum(log_p[idxs[j], assign[j]] for j in range(k))
            dups = k - len(set(assign))
            score -= lam_dup * dups
            if score > best_score:
                best_score = score
                best_assign = assign
        for j in range(k):
            preds[idxs[j]] = best_assign[j]
    return preds


def main():
    comp = find_comp_dir()
    train_df = pd.read_csv(comp / "train.csv")
    test_df = pd.read_csv(comp / "test.csv")
    labels = train_df["label"].astype(int).to_numpy()

    oof_xlmr = np.load(find_file("oof_xlmr.npy"))
    test_xlmr = np.load(find_file("test_xlmr.npy"))
    folds = np.load(find_file("folds.npy"))

    tr_prem = train_df["premise"].fillna("").map(norm_text).tolist()
    te_prem = test_df["premise"].fillna("").map(norm_text).tolist()

    base_acc = float((oof_xlmr.argmax(1) == labels).mean())

    def eval_oof(lk, ld):
        preds = np.zeros(len(train_df), dtype=int)
        for f in range(5):
            tr_idx = np.where(folds != f)[0]
            va_idx = np.where(folds == f)[0]
            kc = {}
            for i in tr_idx:
                p = tr_prem[i]
                if p not in kc:
                    kc[p] = np.zeros(3, dtype=np.float64)
                kc[p][labels[i]] += 1.0
            preds[va_idx] = joint_decode_subset(oof_xlmr[va_idx], [tr_prem[i] for i in va_idx], kc, lk, ld)
        return float((preds == labels).mean())

    grid = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0]
    best_lam = (0.0, 0.0)
    best_acc = base_acc
    for lk in grid:
        for ld in grid:
            acc = eval_oof(lk, ld)
            if acc > best_acc:
                best_acc = acc
                best_lam = (lk, ld)

    full_kc = {}
    for i, p in enumerate(tr_prem):
        if p not in full_kc:
            full_kc[p] = np.zeros(3, dtype=np.float64)
        full_kc[p][labels[i]] += 1.0

    final_preds = joint_decode_subset(test_xlmr, te_prem, full_kc, best_lam[0], best_lam[1])
    sub = pd.DataFrame({"id": test_df["id"], "prediction": final_preds.astype(int)})
    assert len(sub) == 5195
    assert sub["id"].tolist() == test_df["id"].tolist()
    assert sub["prediction"].isin([0, 1, 2]).all()
    sub.to_csv(WORK / "submission.csv", index=False)

    summary = {
        "base_oof_acc": round(base_acc, 5),
        "joint_oof_acc": round(best_acc, 5),
        "best_lam": best_lam,
        "test_pred_dist": sub["prediction"].value_counts(normalize=True).round(4).to_dict(),
    }
    (WORK / "summary_v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WATSON V1 SUMMARY:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
