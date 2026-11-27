"""Generate offline Kaggle submission notebooks (v1 and v2) for LLM Classification Finetuning."""
import json
from pathlib import Path


INFER_CODE_BODY = r'''import gc
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss

T0 = time.monotonic()
WORK = Path("/kaggle/working")
USE_MODEL2 = __USE_MODEL2__


def find_comp_dir():
    for root, _dirs, files in os.walk("/kaggle/input", followlinks=True):
        if {"train.csv", "test.csv"} <= set(files):
            return Path(root)
    raise RuntimeError("Competition data not found")


def find_artifact_dir(must_files):
    for root, _dirs, files in os.walk("/kaggle/input", followlinks=True):
        if set(must_files) <= set(files):
            return Path(root)
    return None


def find_merged_model_dirs():
    dirs = []
    for root, _dirs, files in os.walk("/kaggle/input", followlinks=True):
        if "config.json" in files and ("model.safetensors" in files or "model.safetensors.index.json" in files or "pytorch_model.bin" in files):
            dirs.append(Path(root))
    dirs.sort(key=lambda p: str(p))
    return dirs


REFUSAL_RE = re.compile(
    r"\b(as an ai|i cannot|i can't|i am unable|i'm sorry|i apologize|cannot fulfill|against my programming)\b",
    re.I,
)
URL_RE = re.compile(r"https?://\S+")
NUM_LIST_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
BULLET_RE = re.compile(r"(?m)^\s*[-*•]\s+")
HEADER_RE = re.compile(r"(?m)^\s*#{1,6}\s+")


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
    n_ln = len(txt.splitlines())
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


def extract_pair_features(df, vec, svd):
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
    M = vec.transform(p_txt + a_txt + b_txt)
    Z = svd.transform(M)
    n = len(df)
    Zp, Za, Zb = Z[:n], Z[n:2 * n], Z[2 * n:]

    def cos(U, V):
        nu = np.linalg.norm(U, axis=1) + 1e-8
        nv = np.linalg.norm(V, axis=1) + 1e-8
        return ((U * V).sum(axis=1) / (nu * nv)).astype(np.float32)

    c_pa, c_pb, c_ab = cos(Zp, Za), cos(Zp, Zb), cos(Za, Zb)
    return np.hstack([X, np.column_stack([c_pa, c_pb, c_pa - c_pb, c_ab])])


WORKER_INFER = r"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

P_MAX, A_MAX, B_MAX = 256, 512, 512


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


def trunc_head_tail(ids, max_len, head_len, sep_id):
    if len(ids) <= max_len:
        return ids
    tail_len = max_len - head_len - 1
    return ids[:head_len] + [sep_id] + ids[-tail_len:]


def tokenize_pairs(df, tokenizer, swap=False):
    prompts = ["\n---\n".join(parse_turns(s)) for s in df["prompt"]]
    col_a = "response_b" if swap else "response_a"
    col_b = "response_a" if swap else "response_b"
    resps_a = ["\n---\n".join(parse_turns(s)) for s in df[col_a]]
    resps_b = ["\n---\n".join(parse_turns(s)) for s in df[col_b]]
    p_tok = tokenizer(prompts, add_special_tokens=False).input_ids
    a_tok = tokenizer(resps_a, add_special_tokens=False).input_ids
    b_tok = tokenizer(resps_b, add_special_tokens=False).input_ids

    hdr_p = tokenizer("<|prompt|>\n", add_special_tokens=False).input_ids
    hdr_a = tokenizer("\n\n<|response_a|>\n", add_special_tokens=False).input_ids
    hdr_b = tokenizer("\n\n<|response_b|>\n", add_special_tokens=False).input_ids
    ftr = tokenizer("\n\nWhich response is better?", add_special_tokens=False).input_ids
    sep_toks = tokenizer.encode("\n...\n", add_special_tokens=False)
    sep_id = sep_toks[0] if sep_toks else (tokenizer.eos_token_id or 0)
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    return [
        hdr_p + trunc_head_tail(pi, P_MAX, 160, sep_id)
        + hdr_a + trunc_head_tail(ai, A_MAX, 320, sep_id)
        + hdr_b + trunc_head_tail(bi, B_MAX, 320, sep_id)
        + ftr + [eos_id]
        for pi, ai, bi in zip(p_tok, a_tok, b_tok)
    ]


@torch.no_grad()
def run_split(model, seqs, pad_id, device):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    out = np.zeros((len(seqs), 3), dtype=np.float32)
    i = 0
    while i < len(order):
        max_len = len(seqs[order[i]])
        bs = 16 if max_len <= 400 else (8 if max_len <= 900 else 4)
        bidx = order[i:i + bs]
        i += bs
        ml = max(len(seqs[k]) for k in bidx)
        ids = torch.full((len(bidx), ml), pad_id, dtype=torch.long)
        mask = torch.zeros((len(bidx), ml), dtype=torch.long)
        for r, k in enumerate(bidx):
            s = seqs[k]
            ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[r, :len(s)] = 1
        ids = ids.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lg = model(input_ids=ids, attention_mask=mask).logits
        out[bidx] = lg.float().cpu().numpy()
    return out


def main():
    model_dir = sys.argv[1]
    test_csv = sys.argv[2]
    shard_id = int(sys.argv[3])
    n_shards = int(sys.argv[4])
    out_prefix = sys.argv[5]

    df = pd.read_csv(test_csv)
    idx = np.array_split(np.arange(len(df)), n_shards)[shard_id]
    sub_df = df.iloc[idx].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    pad_id = tok.pad_token_id

    is_gemma = "gemma" in model_dir.lower()
    kw = {"torch_dtype": torch.float16}
    if is_gemma:
        kw["attn_implementation"] = "eager"
    mod = AutoModelForSequenceClassification.from_pretrained(model_dir, **kw).to("cuda:0")
    mod.config.pad_token_id = pad_id
    mod.eval()

    seqs_orig = tokenize_pairs(sub_df, tok, swap=False)
    seqs_swap = tokenize_pairs(sub_df, tok, swap=True)
    lg_orig = run_split(mod, seqs_orig, pad_id, torch.device("cuda:0"))
    lg_swap = run_split(mod, seqs_swap, pad_id, torch.device("cuda:0"))
    np.save(f"{out_prefix}_orig.npy", lg_orig)
    np.save(f"{out_prefix}_swap.npy", lg_swap)
    print(f"Shard {shard_id} finished {len(sub_df)} rows for {model_dir}", flush=True)


if __name__ == "__main__":
    main()
"""


def softmax_np(logits, temp=1.0):
    v = logits.astype(np.float64) / temp
    v = v - v.max(axis=1, keepdims=True)
    e = np.exp(v)
    return e / e.sum(axis=1, keepdims=True)


def run_parallel_2gpu(model_dir, test_csv, tag):
    import torch
    ngpu = torch.cuda.device_count()
    n_shards = min(2, max(1, ngpu))
    worker_file = Path("/tmp/infer_worker.py")
    worker_file.write_text(WORKER_INFER, encoding="utf-8")

    procs = []
    for r in range(n_shards):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(r)
        p = subprocess.Popen(
            [sys.executable, str(worker_file), str(model_dir), str(test_csv), str(r), str(n_shards), f"/tmp/{tag}_r{r}"],
            env=env,
        )
        procs.append(p)
    for p in procs:
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Worker failed with exit code {rc}")
    lg_o = np.concatenate([np.load(f"/tmp/{tag}_r{r}_orig.npy") for r in range(n_shards)], axis=0)
    lg_s = np.concatenate([np.load(f"/tmp/{tag}_r{r}_swap.npy") for r in range(n_shards)], axis=0)
    return lg_o, lg_s


def main():
    comp_dir = find_comp_dir()
    train_df = pd.read_csv(comp_dir / "train.csv")
    test_df = pd.read_csv(comp_dir / "test.csv")
    y_all = np.where(train_df["winner_model_a"] == 1, 0, np.where(train_df["winner_model_b"] == 1, 1, 2))

    # 1. Load LGBM artifacts + predict test
    lgbm_dir = find_artifact_dir(["lgbm_model.txt", "oof_val_lgbm.npy", "val_idx.npy", "tfidf_svd.pkl"])
    assert lgbm_dir is not None, "Missing LGBM artifacts"
    val_idx = np.load(lgbm_dir / "val_idx.npy")
    y_va = y_all[val_idx]
    oof_lgbm = np.load(lgbm_dir / "oof_val_lgbm.npy").astype(np.float64)

    booster = lgb.Booster(model_file=str(lgbm_dir / "lgbm_model.txt"))
    with open(lgbm_dir / "tfidf_svd.pkl", "rb") as f:
        vec, svd = pickle.load(f)
    X_te = extract_pair_features(test_df, vec, svd)
    te_sw = test_df.copy()
    te_sw["response_a"], te_sw["response_b"] = test_df["response_b"], test_df["response_a"]
    X_te_sw = extract_pair_features(te_sw, vec, svd)
    p_te_lgbm = 0.5 * (booster.predict(X_te) + booster.predict(X_te_sw)[:, [1, 0, 2]])
    print(f"LGBM val logloss={log_loss(y_va, oof_lgbm):.5f}", flush=True)

    # 2. Discover LLM merged_model directories + their val_logits & calib.json
    model_dirs = find_merged_model_dirs()
    assert len(model_dirs) >= 1, "No merged LLM model found"
    if not USE_MODEL2:
        model_dirs = model_dirs[:1]

    oof_list = []
    test_list = []
    for k, mdir in enumerate(model_dirs):
        parent = mdir.parent
        calib = json.loads((parent / "calib.json").read_text(encoding="utf-8"))
        T = float(calib["best_T"])
        vo = np.load(parent / "val_logits_orig.npy")
        vs = np.load(parent / "val_logits_swap.npy")
        p_val = 0.5 * (softmax_np(vo, T) + softmax_np(vs, T)[:, [1, 0, 2]])
        print(f"LLM[{k}] {parent.name}: T={T:.3f}, val_logloss={log_loss(y_va, p_val):.5f}", flush=True)
        oof_list.append(p_val)

        lg_o, lg_s = run_parallel_2gpu(mdir, comp_dir / "test.csv", f"m{k}")
        p_te = 0.5 * (softmax_np(lg_o, T) + softmax_np(lg_s, T)[:, [1, 0, 2]])
        test_list.append(p_te)
        gc.collect()

    oof_list.append(oof_lgbm)
    test_list.append(p_te_lgbm)

    # 3. Simplex blend optimization on shared val_idx
    m = len(oof_list)
    def loss_obj(logits_w):
        e = np.exp(logits_w - np.max(logits_w))
        w = e / e.sum()
        blend = sum(w[i] * oof_list[i] for i in range(m))
        return float(log_loss(y_va, np.clip(blend, 1e-6, 1.0 - 1e-6)))

    res = minimize(loss_obj, np.zeros(m), method="Nelder-Mead")
    ew = np.exp(res.x - np.max(res.x))
    weights = ew / ew.sum()
    oof_blend = sum(weights[i] * oof_list[i] for i in range(m))
    print(f"Ensemble weights={[round(float(x), 4) for x in weights]} | val_logloss={log_loss(y_va, oof_blend):.5f} | val_acc={accuracy_score(y_va, oof_blend.argmax(1)):.5f}", flush=True)

    p_final = sum(weights[i] * test_list[i] for i in range(m))
    p_final = np.clip(p_final, 1e-6, 1.0 - 1e-6)
    p_final = p_final / p_final.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({
        "id": test_df["id"],
        "winner_model_a": p_final[:, 0],
        "winner_model_b": p_final[:, 1],
        "winner_tie": p_final[:, 2],
    })
    assert len(sub) == len(test_df)
    assert sub["id"].tolist() == test_df["id"].tolist()
    assert np.isfinite(sub[["winner_model_a", "winner_model_b", "winner_tie"]].to_numpy()).all()
    sub.to_csv(WORK / "submission.csv", index=False)
    print(f"Wrote submission.csv ({len(sub)} rows) in {(time.monotonic() - T0) / 60:.2f}m", flush=True)
    print(sub.head(3), flush=True)


if __name__ == "__main__":
    main()
'''


def write_nb(path: Path, code: str):
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.splitlines()],
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")


def main():
    base = Path(r"d:\Kaggle\9-LLM-Classification-Finetuning\kaggle_kernels")

    v1_dir = base / "submit_v1"
    v1_dir.mkdir(parents=True, exist_ok=True)
    write_nb(v1_dir / "submit_v1.ipynb", INFER_CODE_BODY.replace("__USE_MODEL2__", "False"))
    (v1_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "dewang2025/llm-cls-submit-v1",
                "title": "LLM Cls Submit v1",
                "code_file": "submit_v1.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "keywords": ["gpu"],
                "dataset_sources": [],
                "kernel_sources": [
                    "dewang2025/llm-cls-train-qwen25-3b",
                    "dewang2025/llm-cls-train-lgbm",
                ],
                "competition_sources": ["llm-classification-finetuning"],
                "model_sources": [],
                "machine_shape": "NvidiaTeslaT4",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    v2_dir = base / "submit_v2"
    v2_dir.mkdir(parents=True, exist_ok=True)
    write_nb(v2_dir / "submit_v2.ipynb", INFER_CODE_BODY.replace("__USE_MODEL2__", "True"))
    (v2_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "dewang2025/llm-cls-submit-v2",
                "title": "LLM Cls Submit v2",
                "code_file": "submit_v2.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": False,
                "keywords": ["gpu"],
                "dataset_sources": [],
                "kernel_sources": [
                    "dewang2025/llm-cls-train-qwen25-3b",
                    "dewang2025/llm-cls-train-model2",
                    "dewang2025/llm-cls-train-lgbm",
                ],
                "competition_sources": ["llm-classification-finetuning"],
                "model_sources": [],
                "machine_shape": "NvidiaTeslaT4",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Generated submit_v1 and submit_v2 notebooks.")


if __name__ == "__main__":
    main()
