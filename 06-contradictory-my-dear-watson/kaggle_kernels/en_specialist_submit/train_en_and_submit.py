# Contradictory, My Dear Watson — Kernel 3: English Specialist (DeBERTa-v3-large fp32)
# + Multi-backbone Ensemble + Premise-Group Joint Decoding + Submission

import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np
import pandas as pd
import torch

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "microsoft/deberta-v3-large"
SEED = 42


def find_file(name, must_contain=None):
    for root, _dirs, files in os.walk("/kaggle/input", followlinks=True):
        if name in files:
            p = Path(root) / name
            if must_contain is None or must_contain in str(p):
                return p
    return None


def find_input_root():
    for root, _dirs, files in os.walk("/kaggle/input", followlinks=True):
        if {"train.csv", "test.csv"} <= set(files):
            return Path(root)
    raise RuntimeError("train.csv and test.csv not found")


def norm_text(s):
    return " ".join(str(s).strip().lower().split())


def build_clean_en_stage1(test_df, out_csv):
    from datasets import load_dataset

    test_premises = set(test_df["premise"].fillna("").map(norm_text))
    test_pairs = set(
        zip(
            test_df["premise"].fillna("").map(norm_text),
            test_df["hypothesis"].fillna("").map(norm_text),
        )
    )
    mnli = load_dataset("nyu-mll/multi_nli", split="train")
    rng = np.random.default_rng(SEED + 777)
    idx = rng.permutation(len(mnli))
    rows = []
    removed = 0
    for i in idx:
        ex = mnli[int(i)]
        p = str(ex["premise"] or "").strip()
        h = str(ex["hypothesis"] or "").strip()
        y = int(ex["label"])
        if y not in (0, 1, 2) or not p or not h:
            continue
        np_ = norm_text(p)
        if (np_, norm_text(h)) in test_pairs or np_ in test_premises:
            removed += 1
            continue
        rows.append((p, h, y))
        if len(rows) >= 18000:
            break
    df = pd.DataFrame(rows, columns=["premise", "hypothesis", "label"])
    df.to_csv(out_csv, index=False)
    print(f"Clean EN Stage-1 dataset: {len(df)} rows | removed={removed}", flush=True)


WORKER_CODE = r'''
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

SEED = 42
MAX_LEN = 160
WORK = Path("/kaggle/working")
TMP = Path("/tmp/watson_en")


class NLIDataset(Dataset):
    def __init__(self, enc, labels=None):
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        item = {"input_ids": self.input_ids[i], "attention_mask": self.attention_mask[i]}
        if self.labels is not None:
            item["labels"] = int(self.labels[i])
        return item


def make_collate(pad_id):
    def collate(batch):
        max_l = max(len(x["input_ids"]) for x in batch)
        bs = len(batch)
        ids = torch.full((bs, max_l), pad_id, dtype=torch.long)
        mask = torch.zeros((bs, max_l), dtype=torch.long)
        for i, x in enumerate(batch):
            l = len(x["input_ids"])
            ids[i, :l] = torch.tensor(x["input_ids"], dtype=torch.long)
            mask[i, :l] = torch.tensor(x["attention_mask"], dtype=torch.long)
        out = {"input_ids": ids, "attention_mask": mask}
        if "labels" in batch[0]:
            out["labels"] = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
        return out
    return collate


def encode_df(df, tokenizer):
    return tokenizer(
        df["premise"].fillna("").astype(str).tolist(),
        df["hypothesis"].fillna("").astype(str).tolist(),
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )


def softmax_np(logits):
    v = logits.astype(np.float64)
    v = v - v.max(axis=1, keepdims=True)
    e = np.exp(v)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


@torch.no_grad()
def predict_local(model, enc, pad_id, batch_size, device):
    model.eval()
    ds = NLIDataset(enc)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=make_collate(pad_id), num_workers=2)
    chunks = []
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        logits = model(input_ids=ids, attention_mask=mask).logits
        chunks.append(logits.float().cpu().numpy())
    return softmax_np(np.concatenate(chunks, axis=0))


def train_epochs(
    init_dir, train_enc, train_labels, epochs, lr, per_gpu_bs, grad_accum,
    pad_id, device, rank, world_size, label_smoothing=0.05,
    val_enc=None, val_labels=None, save_best_dir=None,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        init_dir, num_labels=3, torch_dtype=torch.float32
    ).to(device)
    model.gradient_checkpointing_enable()
    ddp_model = DDP(model, device_ids=[device.index], output_device=device.index) if world_size > 1 else model

    ds = NLIDataset(train_enc, train_labels)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED) if world_size > 1 else None
    loader = DataLoader(
        ds, batch_size=per_gpu_bs, shuffle=(sampler is None), sampler=sampler,
        collate_fn=make_collate(pad_id), num_workers=2, pin_memory=True,
    )

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=lr, eps=1e-6)
    total_steps = max(1, (len(loader) * epochs) // grad_accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_acc = -1.0
    best_probs = None
    for ep in range(epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        steps = 0
        for step, batch in enumerate(loader):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            sync_ctx = ddp_model.no_sync() if (world_size > 1 and (step + 1) % grad_accum != 0 and (step + 1) != len(loader)) else torch.enable_grad()
            with sync_ctx:
                logits = ddp_model(input_ids=ids, attention_mask=mask).logits
                loss = loss_fn(logits.float(), y) / grad_accum
                if ep == 0 and step == 0:
                    if rank == 0:
                        print(f"  step 0 loss={float(loss.item()) * grad_accum:.4f}", flush=True)
                    assert torch.isfinite(loss), f"Non-finite loss on step 0: {loss}"
                loss.backward()
            running_loss += float(loss.item()) * grad_accum
            steps += 1
            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        if world_size > 1:
            dist.barrier()
        if rank == 0:
            msg = f"  epoch {ep + 1}/{epochs} train_loss={running_loss / max(1, steps):.4f}"
            if val_enc is not None:
                vp = predict_local(model, val_enc, pad_id, 32, device)
                acc = float((vp.argmax(1) == val_labels).mean())
                msg += f" val_acc={acc:.5f}"
                if acc >= best_acc:
                    best_acc = acc
                    best_probs = vp
                    if save_best_dir is not None:
                        model.save_pretrained(save_best_dir)
            print(msg, flush=True)
        if world_size > 1:
            dist.barrier()

    if rank == 0 and save_best_dir is not None and val_enc is None:
        model.save_pretrained(save_best_dir)
    del model, ddp_model, optimizer
    torch.cuda.empty_cache()
    return best_acc, best_probs


def main():
    t0 = time.monotonic()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(SEED + local_rank)
    np.random.seed(SEED + local_rank)

    tokenizer = AutoTokenizer.from_pretrained(str(TMP / "base_tok"))
    pad_id = tokenizer.pad_token_id or 0

    stage1_df = pd.read_csv(TMP / "stage1_en.csv")
    train_df = pd.read_csv(TMP / "train.csv")
    test_df = pd.read_csv(TMP / "test.csv")
    folds = np.load(TMP / "folds.npy")

    if local_rank == 0:
        print(f"Stage 1 EN specialist (DeBERTa-v3-large fp32) on {len(stage1_df)} rows", flush=True)
    s1_enc = encode_df(stage1_df, tokenizer)
    s1_y = stage1_df["label"].astype(int).to_numpy()
    stage1_dir = TMP / "stage1_ckpt"
    train_epochs(
        str(TMP / "base_model"), s1_enc, s1_y, epochs=1, lr=1.2e-5, per_gpu_bs=8, grad_accum=2,
        pad_id=pad_id, device=device, rank=local_rank, world_size=world_size,
        label_smoothing=0.05, save_best_dir=str(stage1_dir),
    )
    if world_size > 1:
        dist.barrier()

    en_tr_mask = (train_df["language"].str.lower() == "english").to_numpy()
    en_te_mask = (test_df["language"].str.lower() == "english").to_numpy()
    tr_enc_all = encode_df(train_df, tokenizer)
    te_enc_en = encode_df(test_df[en_te_mask].reset_index(drop=True), tokenizer)
    labels = train_df["label"].astype(int).to_numpy()

    def slice_enc(enc, idxs):
        return {"input_ids": [enc["input_ids"][i] for i in idxs], "attention_mask": [enc["attention_mask"][i] for i in idxs]}

    oof_en = np.zeros((len(train_df), 3), dtype=np.float32)
    test_en = np.zeros((len(test_df), 3), dtype=np.float32)
    en_test_accum = np.zeros((int(en_te_mask.sum()), 3), dtype=np.float32)
    completed = []

    for f in range(5):
        if time.monotonic() - t0 > 95 * 60:
            if local_rank == 0:
                print(f"Wall-clock guard before fold {f}", flush=True)
            break
        tr_idx = np.where((folds != f) & en_tr_mask)[0]
        va_idx = np.where((folds == f) & en_tr_mask)[0]
        if local_rank == 0:
            print(f"=== EN Fold {f} (tr={len(tr_idx)}, va={len(va_idx)}) ===", flush=True)
        fold_dir = TMP / f"fold_{f}_best"
        best_acc, va_probs = train_epochs(
            str(stage1_dir), slice_enc(tr_enc_all, tr_idx), labels[tr_idx],
            epochs=3, lr=1.0e-5, per_gpu_bs=8, grad_accum=2, pad_id=pad_id,
            device=device, rank=local_rank, world_size=world_size, label_smoothing=0.05,
            val_enc=slice_enc(tr_enc_all, va_idx), val_labels=labels[va_idx], save_best_dir=str(fold_dir),
        )
        if local_rank == 0:
            oof_en[va_idx] = va_probs
            best_mod = AutoModelForSequenceClassification.from_pretrained(
                str(fold_dir), num_labels=3, torch_dtype=torch.float32
            ).to(device)
            tp = predict_local(best_mod, te_enc_en, pad_id, 32, device)
            del best_mod
            torch.cuda.empty_cache()
            en_test_accum += tp
            completed.append(f)
            test_en[en_te_mask] = en_test_accum / len(completed)
            np.save(WORK / "oof_deberta_en.npy", oof_en)
            np.save(WORK / "test_deberta_en.npy", test_en)
            valid_mask = np.isin(folds, completed) & en_tr_mask
            cum_acc = float((oof_en[valid_mask].argmax(1) == labels[valid_mask]).mean())
            print(f"EN Fold {f} done: val_acc={best_acc:.5f} | cum_en_oof_acc={cum_acc:.5f} | elapsed={(time.monotonic()-t0)/60:.1f}m", flush=True)
        if world_size > 1:
            dist.barrier()

    if local_rank == 0:
        (WORK / "en_completed.json").write_text(json.dumps(completed), encoding="utf-8")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
'''


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

    for p, idxs in groups.items():
        k = len(idxs)
        if k < 2 or lam_dup <= 0.0:
            continue
        if k <= 6:
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
    t0 = time.monotonic()
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
        "transformers==4.46.3", "tokenizers==0.20.3", "huggingface-hub==0.36.0", "sentencepiece==0.2.1",
    ])
    root = find_input_root()
    train_df = pd.read_csv(root / "train.csv")
    test_df = pd.read_csv(root / "test.csv")
    for df in (train_df, test_df):
        for col in ("premise", "hypothesis"):
            df[col] = df[col].fillna("").astype(str)

    oof_xlmr_p = find_file("oof_xlmr.npy")
    test_xlmr_p = find_file("test_xlmr.npy")
    folds_p = find_file("folds.npy")
    oof_mdeb_p = find_file("oof_mdeberta.npy")
    test_mdeb_p = find_file("test_mdeberta.npy")

    assert oof_xlmr_p and test_xlmr_p and folds_p, "Missing XLM-R outputs!"
    oof_xlmr = np.load(oof_xlmr_p)
    test_xlmr = np.load(test_xlmr_p)
    folds = np.load(folds_p)
    has_mdeb = oof_mdeb_p is not None and test_mdeb_p is not None
    oof_mdeb = np.load(oof_mdeb_p) if has_mdeb else oof_xlmr
    test_mdeb = np.load(test_mdeb_p) if has_mdeb else test_xlmr

    tmp = Path("/tmp/watson_en")
    tmp.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(tmp / "train.csv", index=False)
    test_df.to_csv(tmp / "test.csv", index=False)
    np.save(tmp / "folds.npy", folds)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("Caching EN specialist:", MODEL_NAME, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mod = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, torch_dtype=torch.float32
    )
    tok.save_pretrained(tmp / "base_tok")
    mod.save_pretrained(tmp / "base_model")
    del mod, tok

    build_clean_en_stage1(test_df, tmp / "stage1_en.csv")
    worker_path = tmp / "worker.py"
    worker_path.write_text(WORKER_CODE, encoding="utf-8")
    ngpu = torch.cuda.device_count()
    if ngpu >= 2:
        subprocess.check_call(["torchrun", "--nproc_per_node", str(ngpu), str(worker_path)])
    else:
        subprocess.check_call([sys.executable, str(worker_path)])

    oof_en = np.load(WORK / "oof_deberta_en.npy")
    test_en = np.load(WORK / "test_deberta_en.npy")
    labels = train_df["label"].astype(int).to_numpy()
    en_tr = (train_df["language"].str.lower() == "english").to_numpy()
    en_te = (test_df["language"].str.lower() == "english").to_numpy()

    valid_xlmr = oof_xlmr.sum(1) > 0.5
    valid_mdeb = (oof_mdeb.sum(1) > 0.5) if has_mdeb else valid_xlmr
    valid_en_oof = oof_en.sum(1) > 0.5
    en_fit = en_tr & valid_xlmr & valid_mdeb & valid_en_oof
    non_fit = (~en_tr) & valid_xlmr & valid_mdeb

    # Optimize per-language-group weights on valid OOF rows
    best_en_acc = -1.0
    best_en_w = (0.34, 0.33, 0.33)
    for w1 in np.linspace(0.0, 1.0, 21):
        for w2 in np.linspace(0.0, 1.0 - w1, int(round((1.0 - w1) * 20)) + 1):
            w3 = max(0.0, 1.0 - w1 - w2)
            p = w1 * oof_xlmr[en_fit] + w2 * oof_mdeb[en_fit] + w3 * oof_en[en_fit]
            acc = float((p.argmax(1) == labels[en_fit]).mean())
            if acc > best_en_acc:
                best_en_acc = acc
                best_en_w = (round(float(w1), 3), round(float(w2), 3), round(float(w3), 3))

    best_non_acc = -1.0
    best_non_w = 0.5
    if has_mdeb:
        for w1 in np.linspace(0.0, 1.0, 41):
            p = w1 * oof_xlmr[non_fit] + (1.0 - w1) * oof_mdeb[non_fit]
            acc = float((p.argmax(1) == labels[non_fit]).mean())
            if acc > best_non_acc:
                best_non_acc = acc
                best_non_w = round(float(w1), 3)
    else:
        best_non_w = 1.0
        best_non_acc = float((oof_xlmr[non_fit].argmax(1) == labels[non_fit]).mean())

    # Build OOF ensemble gracefully handling any missing fold if wall-clock guard triggered
    oof_ens = oof_xlmr.copy()
    m_en_full = np.where(
        valid_en_oof[:, None],
        best_en_w[0] * oof_xlmr + best_en_w[1] * np.where(valid_mdeb[:, None], oof_mdeb, oof_xlmr) + best_en_w[2] * oof_en,
        0.5 * oof_xlmr + 0.5 * np.where(valid_mdeb[:, None], oof_mdeb, oof_xlmr),
    )
    oof_ens[en_tr] = m_en_full[en_tr]
    oof_ens[~en_tr] = np.where(
        valid_mdeb[~en_tr, None],
        best_non_w * oof_xlmr[~en_tr] + (1.0 - best_non_w) * oof_mdeb[~en_tr],
        oof_xlmr[~en_tr],
    )

    test_ens = np.zeros_like(test_xlmr)
    test_ens[en_te] = best_en_w[0] * test_xlmr[en_te] + best_en_w[1] * test_mdeb[en_te] + best_en_w[2] * test_en[en_te]
    test_ens[~en_te] = best_non_w * test_xlmr[~en_te] + (1.0 - best_non_w) * test_mdeb[~en_te]

    valid_eval = valid_xlmr & valid_mdeb & ((~en_tr) | valid_en_oof)
    base_oof_acc = float((oof_ens[valid_eval].argmax(1) == labels[valid_eval]).mean())
    print(f"Ensemble OOF Acc={base_oof_acc:.5f} (n={int(valid_eval.sum())}) | EN={best_en_acc:.5f} w={best_en_w} | NonEN={best_non_acc:.5f} w_xlmr={best_non_w}", flush=True)

    # OOF validation of Premise-Group Joint Decoding
    tr_norm_prem = train_df["premise"].map(norm_text).tolist()
    te_norm_prem = test_df["premise"].map(norm_text).tolist()

    def eval_joint_oof(lam_k, lam_d):
        oof_preds = np.zeros(len(train_df), dtype=int)
        for f in range(5):
            tr_idx = np.where(folds != f)[0]
            va_idx = np.where(folds == f)[0]
            kc = {}
            for i in tr_idx:
                p = tr_norm_prem[i]
                if p not in kc:
                    kc[p] = np.zeros(3, dtype=np.float64)
                kc[p][labels[i]] += 1.0
            vp = [tr_norm_prem[i] for i in va_idx]
            oof_preds[va_idx] = joint_decode_subset(oof_ens[va_idx], vp, kc, lam_k, lam_d)
        return float((oof_preds[valid_eval] == labels[valid_eval]).mean())

    best_lam = (0.0, 0.0)
    best_joint_acc = base_oof_acc
    grid = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0]
    for lk in grid:
        for ld in grid:
            acc = eval_joint_oof(lk, ld)
            if acc > best_joint_acc:
                best_joint_acc = acc
                best_lam = (lk, ld)

    use_joint = (best_joint_acc - base_oof_acc) >= 0.002
    print(f"Joint decode OOF: base={base_oof_acc:.5f}, best={best_joint_acc:.5f} at {best_lam}, apply={use_joint}", flush=True)

    if use_joint:
        full_kc = {}
        for i, p in enumerate(tr_norm_prem):
            if p not in full_kc:
                full_kc[p] = np.zeros(3, dtype=np.float64)
            full_kc[p][labels[i]] += 1.0
        final_preds = joint_decode_subset(test_ens, te_norm_prem, full_kc, best_lam[0], best_lam[1])
    else:
        final_preds = test_ens.argmax(1).astype(int)

    sub = pd.DataFrame({"id": test_df["id"], "prediction": final_preds.astype(int)})
    assert len(sub) == 5195
    assert sub["id"].tolist() == test_df["id"].tolist()
    assert sub["prediction"].isin([0, 1, 2]).all()
    sub.to_csv(WORK / "submission.csv", index=False)

    summary = {
        "oof_xlmr": round(float((oof_xlmr[valid_xlmr].argmax(1) == labels[valid_xlmr]).mean()), 5),
        "oof_mdeb": round(float((oof_mdeb[valid_mdeb].argmax(1) == labels[valid_mdeb]).mean()), 5) if has_mdeb else None,
        "oof_deberta_en_only": round(float((oof_en[en_tr & valid_en_oof].argmax(1) == labels[en_tr & valid_en_oof]).mean()), 5),
        "oof_ens_base": round(base_oof_acc, 5),
        "oof_ens_joint": round(best_joint_acc, 5),
        "en_weights": best_en_w,
        "non_en_xlmr_weight": best_non_w,
        "joint_params": best_lam,
        "applied_joint": use_joint,
        "test_pred_dist": sub["prediction"].value_counts(normalize=True).round(4).to_dict(),
        "elapsed_min": round((time.monotonic() - t0) / 60.0, 2),
    }
    (WORK / "summary_ensemble.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("FINAL SUMMARY:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
