# Contradictory, My Dear Watson — Kernel 2: Clean mDeBERTa-v3-base (fp32, Zero Test Leakage)
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
from sklearn.model_selection import StratifiedKFold

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "microsoft/mdeberta-v3-base"
SEED = 42
LANGS = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]


def find_input_root():
    for root, _dirs, files in os.walk("/kaggle/input", followlinks=True):
        if {"train.csv", "test.csv"} <= set(files):
            return Path(root)
    raise RuntimeError("train.csv and test.csv not found under /kaggle/input")


def norm_text(s):
    return " ".join(str(s).strip().lower().split())


def build_clean_stage1(test_df, out_csv):
    from datasets import load_dataset

    test_premises = set(test_df["premise"].fillna("").map(norm_text))
    test_pairs = set(
        zip(
            test_df["premise"].fillna("").map(norm_text),
            test_df["hypothesis"].fillna("").map(norm_text),
        )
    )

    rows = []
    removed_prem = 0
    removed_pair = 0

    def add_split(ds, lang, limit):
        nonlocal removed_prem, removed_pair
        rng = np.random.default_rng(SEED + sum(ord(c) for c in lang))
        idx = rng.permutation(len(ds))
        kept = 0
        for i in idx:
            ex = ds[int(i)]
            p = str(ex["premise"] or "").strip()
            h = str(ex["hypothesis"] or "").strip()
            y = int(ex["label"])
            if y not in (0, 1, 2) or not p or not h:
                continue
            np_ = norm_text(p)
            nh_ = norm_text(h)
            if (np_, nh_) in test_pairs:
                removed_pair += 1
                continue
            if np_ in test_premises:
                removed_prem += 1
                continue
            rows.append((p, h, y, lang))
            kept += 1
            if kept >= limit:
                break
        return kept

    try:
        mnli = load_dataset("nyu-mll/multi_nli", split="train")
        add_split(mnli, "en", 16000)
    except Exception:
        xnli_en = load_dataset("facebook/xnli", "en", split="train")
        add_split(xnli_en, "en", 16000)

    for lang in LANGS:
        if lang == "en":
            continue
        try:
            ds = load_dataset("facebook/xnli", lang, split="train")
            add_split(ds, lang, 2100)
        except Exception as e:
            print(f"WARN xnli[{lang}][train] failed: {e}", flush=True)

    df = pd.DataFrame(rows, columns=["premise", "hypothesis", "label", "lang"])
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"Clean Stage-1 dataset: {len(df)} rows | removed_pair={removed_pair} | removed_premise={removed_prem}", flush=True)


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
MODEL_NAME = "microsoft/mdeberta-v3-base"
WORK = Path("/kaggle/working")
TMP = Path("/tmp/watson_mdeberta")


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
        # DeBERTa-v3 runs in fp32 to prevent T4 fp16 overflow
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
                vp = predict_local(model, val_enc, pad_id, 64, device)
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

    stage1_df = pd.read_csv(TMP / "stage1_clean.csv")
    train_df = pd.read_csv(TMP / "train.csv")
    test_df = pd.read_csv(TMP / "test.csv")
    folds = np.load(WORK / "folds.npy")

    if local_rank == 0:
        print(f"Stage 1: mDeBERTa-v3-base (fp32) on {len(stage1_df)} clean rows (world={world_size})", flush=True)
    s1_enc = encode_df(stage1_df, tokenizer)
    s1_y = stage1_df["label"].astype(int).to_numpy()
    stage1_dir = TMP / "stage1_ckpt"
    train_epochs(
        str(TMP / "base_model"), s1_enc, s1_y, epochs=1, lr=2.5e-5, per_gpu_bs=16, grad_accum=1,
        pad_id=pad_id, device=device, rank=local_rank, world_size=world_size,
        label_smoothing=0.05, save_best_dir=str(stage1_dir),
    )
    if world_size > 1:
        dist.barrier()

    tr_enc_all = encode_df(train_df, tokenizer)
    te_enc = encode_df(test_df, tokenizer)
    labels = train_df["label"].astype(int).to_numpy()

    def slice_enc(enc, idxs):
        return {"input_ids": [enc["input_ids"][i] for i in idxs], "attention_mask": [enc["attention_mask"][i] for i in idxs]}

    oof = np.zeros((len(train_df), 3), dtype=np.float32)
    test_preds = np.zeros((len(test_df), 3), dtype=np.float32)
    completed = []

    for f in range(5):
        if time.monotonic() - t0 > 95 * 60:
            if local_rank == 0:
                print(f"Wall-clock guard before fold {f}", flush=True)
            break
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        if local_rank == 0:
            print(f"=== Fold {f} ===", flush=True)
        fold_dir = TMP / f"fold_{f}_best"
        best_acc, va_probs = train_epochs(
            str(stage1_dir), slice_enc(tr_enc_all, tr_idx), labels[tr_idx],
            epochs=3, lr=2.0e-5, per_gpu_bs=16, grad_accum=1, pad_id=pad_id,
            device=device, rank=local_rank, world_size=world_size, label_smoothing=0.05,
            val_enc=slice_enc(tr_enc_all, va_idx), val_labels=labels[va_idx], save_best_dir=str(fold_dir),
        )
        if local_rank == 0:
            oof[va_idx] = va_probs
            best_mod = AutoModelForSequenceClassification.from_pretrained(
                str(fold_dir), num_labels=3, torch_dtype=torch.float32
            ).to(device)
            tp = predict_local(best_mod, te_enc, pad_id, 64, device)
            del best_mod
            torch.cuda.empty_cache()
            test_preds += tp
            completed.append(f)

            valid_mask = np.isin(folds, completed)
            cur_oof_acc = float((oof[valid_mask].argmax(1) == labels[valid_mask]).mean())
            cur_te = test_preds / len(completed)
            np.save(WORK / "oof_mdeberta.npy", oof)
            np.save(WORK / "test_mdeberta.npy", cur_te)
            print(f"Fold {f} done: val_acc={best_acc:.5f} | cum_oof_acc={cur_oof_acc:.5f} | elapsed={(time.monotonic()-t0)/60:.1f}m", flush=True)
        if world_size > 1:
            dist.barrier()

    if local_rank == 0:
        valid_mask = np.isin(folds, completed)
        en_mask = (train_df["language"].str.lower() == "english").to_numpy()
        metrics = {
            "model": MODEL_NAME,
            "completed_folds": completed,
            "oof_acc_all": float((oof[valid_mask].argmax(1) == labels[valid_mask]).mean()),
            "oof_acc_en": float((oof[valid_mask & en_mask].argmax(1) == labels[valid_mask & en_mask]).mean()),
            "oof_acc_non_en": float((oof[valid_mask & (~en_mask)].argmax(1) == labels[valid_mask & (~en_mask)]).mean()),
            "elapsed_min": round((time.monotonic() - t0) / 60.0, 2),
        }
        (WORK / "metrics_mdeberta.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("METRICS:", json.dumps(metrics), flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
'''


def main():
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

    strat = train_df["language"].astype(str) + "_" + train_df["label"].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = np.zeros(len(train_df), dtype=np.int32)
    for f, (_tr, va) in enumerate(skf.split(train_df, strat)):
        folds[va] = f
    np.save(WORK / "folds.npy", folds)

    tmp = Path("/tmp/watson_mdeberta")
    tmp.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(tmp / "train.csv", index=False)
    test_df.to_csv(tmp / "test.csv", index=False)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("Caching model and tokenizer:", MODEL_NAME, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mod = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, torch_dtype=torch.float32
    )
    tok.save_pretrained(tmp / "base_tok")
    mod.save_pretrained(tmp / "base_model")
    del mod, tok

    build_clean_stage1(test_df, tmp / "stage1_clean.csv")

    worker_path = tmp / "worker.py"
    worker_path.write_text(WORKER_CODE, encoding="utf-8")
    ngpu = torch.cuda.device_count()
    if ngpu >= 2:
        subprocess.check_call(["torchrun", "--nproc_per_node", str(ngpu), str(worker_path)])
    else:
        subprocess.check_call([sys.executable, str(worker_path)])


if __name__ == "__main__":
    main()
