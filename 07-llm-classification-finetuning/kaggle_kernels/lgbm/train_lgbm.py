# LLM Classification Finetuning — Kernel 1 (CPU): Symmetric LightGBM + TF-IDF SVD Baseline
import json
import os
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
SEED = 42
REFUSAL_RE = re.compile(
    r"\b(as an ai|i cannot|i can't|i am unable|i'm sorry|i apologize|cannot fulfill|against my programming)\b",
    re.I,
)
URL_RE = re.compile(r"https?://\S+")
NUM_LIST_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
BULLET_RE = re.compile(r"(?m)^\s*[-*•]\s+")
HEADER_RE = re.compile(r"(?m)^\s*#{1,6}\s+")


def find_input_root():
    for root, _dirs, files in os.walk("/kaggle/input", followlinks=True):
        if {"train.csv", "test.csv"} <= set(files):
            return Path(root)
    raise RuntimeError("train.csv and test.csv not found")


def parse_turns(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return [""]
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return [str(x) if x is not None else "" for x in arr] or [""]
    except Exception:
        pass
    return [str(s)]


def text_stats(txt):
    n_ch = len(txt)
    words = txt.split()
    n_wd = len(words)
    lines = txt.splitlines()
    n_ln = len(lines)
    n_code = txt.count("```") // 2
    n_bold = txt.count("**") // 2
    n_hdr = len(HEADER_RE.findall(txt))
    n_bul = len(BULLET_RE.findall(txt))
    n_num = len(NUM_LIST_RE.findall(txt))
    n_q = txt.count("?")
    n_ex = txt.count("!")
    n_url = len(URL_RE.findall(txt))
    n_ref = len(REFUSAL_RE.findall(txt))
    n_dig = sum(ch.isdigit() for ch in txt)
    n_non_ascii = sum(ord(ch) > 127 for ch in txt)
    avg_wlen = n_ch / max(1, n_wd)
    uniq_ratio = len(set(w.lower() for w in words)) / max(1, n_wd)
    return [
        n_ch, np.log1p(n_ch), n_wd, np.log1p(n_wd), n_ln, np.log1p(n_ln),
        n_code, n_bold, n_hdr, n_bul, n_num, n_q, n_ex, n_url, n_ref,
        n_dig / max(1, n_ch), n_non_ascii / max(1, n_ch), avg_wlen, uniq_ratio,
    ]


def jaccard(a, b):
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def extract_pair_features(df, vec=None, svd=None):
    p_turns = [parse_turns(s) for s in df["prompt"]]
    a_turns = [parse_turns(s) for s in df["response_a"]]
    b_turns = [parse_turns(s) for s in df["response_b"]]

    p_txt = ["\n".join(t) for t in p_turns]
    a_txt = ["\n".join(t) for t in a_turns]
    b_txt = ["\n".join(t) for t in b_turns]

    feats = []
    for pt, at, bt, ptr, atr, btr in zip(p_txt, a_txt, b_txt, p_turns, a_turns, b_turns):
        sp = text_stats(pt)
        sa = text_stats(at)
        sb = text_stats(bt)
        eq_exact = float(at == bt)
        eq_norm = float(" ".join(at.lower().split()) == " ".join(bt.lower().split()))
        len_ratio = np.log((len(at) + 1.0) / (len(bt) + 1.0))
        wd_ratio = np.log((len(at.split()) + 1.0) / (len(bt.split()) + 1.0))
        last_diff = len(atr[-1]) - len(btr[-1])
        jac_pa = jaccard(pt, at)
        jac_pb = jaccard(pt, bt)
        jac_ab = jaccard(at, bt)
        diff_ab = [x - y for x, y in zip(sa, sb)]
        feats.append(
            [len(ptr), eq_exact, eq_norm, len_ratio, wd_ratio, last_diff,
             jac_pa, jac_pb, jac_pa - jac_pb, jac_ab]
            + sp[:6] + sa + sb + diff_ab
        )
    X = np.asarray(feats, dtype=np.float32)

    all_texts = p_txt + a_txt + b_txt
    if vec is None:
        vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
        M = vec.fit_transform(all_texts)
        svd = TruncatedSVD(n_components=24, random_state=SEED)
        Z = svd.fit_transform(M)
    else:
        M = vec.transform(all_texts)
        Z = svd.transform(M)

    n = len(df)
    Zp, Za, Zb = Z[:n], Z[n:2 * n], Z[2 * n:]

    def cos(U, V):
        nu = np.linalg.norm(U, axis=1) + 1e-8
        nv = np.linalg.norm(V, axis=1) + 1e-8
        return ((U * V).sum(axis=1) / (nu * nv)).astype(np.float32)

    c_pa = cos(Zp, Za)
    c_pb = cos(Zp, Zb)
    c_ab = cos(Za, Zb)
    X = np.hstack([X, np.column_stack([c_pa, c_pb, c_pa - c_pb, c_ab])])
    return X, vec, svd


def swap_df(df):
    sw = df.copy()
    sw["response_a"], sw["response_b"] = df["response_b"], df["response_a"]
    return sw


def main():
    t0 = time.monotonic()
    root = find_input_root()
    train = pd.read_csv(root / "train.csv")
    y = np.where(train["winner_model_a"] == 1, 0, np.where(train["winner_model_b"] == 1, 1, 2))

    # Shared 12-fold split (fold 0 = 4,790 val rows)
    skf = StratifiedKFold(n_splits=12, shuffle=True, random_state=SEED)
    val_idx = next(skf.split(train, y))[1]
    tr_mask = np.ones(len(train), dtype=bool)
    tr_mask[val_idx] = False
    tr_idx = np.where(tr_mask)[0]
    np.save(WORK / "val_idx.npy", val_idx)

    tr_df = train.iloc[tr_idx].reset_index(drop=True)
    va_df = train.iloc[val_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    y_va = y[val_idx]

    print(f"Extracting features (tr={len(tr_df)}, va={len(va_df)})...", flush=True)
    X_tr, vec, svd = extract_pair_features(tr_df)
    X_tr_sw, _, _ = extract_pair_features(swap_df(tr_df), vec, svd)
    y_tr_sw = np.where(y_tr == 0, 1, np.where(y_tr == 1, 0, 2))

    X_tr_full = np.vstack([X_tr, X_tr_sw])
    y_tr_full = np.concatenate([y_tr, y_tr_sw])

    X_va, _, _ = extract_pair_features(va_df, vec, svd)
    X_va_sw, _, _ = extract_pair_features(swap_df(va_df), vec, svd)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        subsample_for_bin=100000,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        min_child_samples=40,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr_full,
        y_tr_full,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(200)],
    )

    p_orig = model.predict_proba(X_va)
    p_swap = model.predict_proba(X_va_sw)[:, [1, 0, 2]]
    p_tta = 0.5 * (p_orig + p_swap)

    loss_orig = float(log_loss(y_va, p_orig))
    loss_tta = float(log_loss(y_va, p_tta))
    acc_tta = float(accuracy_score(y_va, p_tta.argmax(1)))

    import pickle
    model.booster_.save_model(str(WORK / "lgbm_model.txt"))
    with open(WORK / "tfidf_svd.pkl", "wb") as f:
        pickle.dump((vec, svd), f)
    np.save(WORK / "oof_val_lgbm.npy", p_tta.astype(np.float32))

    metrics = {
        "val_logloss_orig": round(loss_orig, 5),
        "val_logloss_tta": round(loss_tta, 5),
        "val_acc_tta": round(acc_tta, 5),
        "best_iteration": int(model.best_iteration_),
        "n_features": int(X_tr.shape[1]),
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
    (WORK / "metrics_lgbm.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("LGBM DONE:", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
