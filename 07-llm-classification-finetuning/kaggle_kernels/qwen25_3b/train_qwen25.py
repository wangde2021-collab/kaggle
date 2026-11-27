# LLM Classification Finetuning — Kernel 2: Qwen2.5-3B-Instruct LoRA (2xT4 DDP, Merged FP16 Output)
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)


def ensure_peft():
    try:
        import peft  # noqa: F401
        print("peft available:", peft.__version__, flush=True)
    except Exception:
        print("Installing peft...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft"])


def find_comp_dir():
    for root, _dirs, files in os.walk("/kaggle/input", followlinks=True):
        if {"train.csv", "test.csv"} <= set(files):
            return Path(root)
    raise RuntimeError("Competition data not found")


def find_model_dir(hints=("qwen2.5", "qwen")):
    candidates = []
    for root, _dirs, files in os.walk("/kaggle/input", followlinks=True):
        if "config.json" in files:
            candidates.append(Path(root))
    for h in hints:
        for p in candidates:
            if h in str(p).lower():
                return p
    if candidates:
        return candidates[0]
    raise RuntimeError("Pretrained model directory with config.json not found under /kaggle/input")


WORKER_CODE = r'''
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

SEED = 42
WORK = Path("/kaggle/working")
P_MAX, A_MAX, B_MAX = 256, 512, 512
MAX_TRAIN_HOURS = 7.0


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
    sep_id = tokenizer.encode("\n...\n", add_special_tokens=False)[0]
    eos_id = tokenizer.eos_token_id

    out = []
    for pi, ai, bi in zip(p_tok, a_tok, b_tok):
        pi = trunc_head_tail(pi, P_MAX, 160, sep_id)
        ai = trunc_head_tail(ai, A_MAX, 320, sep_id)
        bi = trunc_head_tail(bi, B_MAX, 320, sep_id)
        ids = hdr_p + pi + hdr_a + ai + hdr_b + bi + ftr + [eos_id]
        out.append(ids)
    return out


def make_length_bucketed_batches(lengths, bs, seed, world_size, rank):
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    bucket_sz = bs * 128
    batches = []
    for s in range(0, len(idx), bucket_sz):
        chunk = idx[s:s + bucket_sz]
        chunk.sort(key=lambda i: lengths[i], reverse=True)
        for b in range(0, len(chunk), bs):
            batches.append(chunk[b:b + bs])
    rng.shuffle(batches)
    # Trim to multiple of world_size so DDP ranks stay perfectly synced
    n_keep = (len(batches) // world_size) * world_size
    batches = batches[:n_keep]
    return batches[rank::world_size]


def pad_batch(ids_list, pad_id, labels_list=None):
    max_l = max(len(x) for x in ids_list)
    bs = len(ids_list)
    ids = torch.full((bs, max_l), pad_id, dtype=torch.long)
    mask = torch.zeros((bs, max_l), dtype=torch.long)
    for i, seq in enumerate(ids_list):
        l = len(seq)
        ids[i, :l] = torch.tensor(seq, dtype=torch.long)
        mask[i, :l] = 1
    out = {"input_ids": ids, "attention_mask": mask}
    if labels_list is not None:
        out["labels"] = torch.tensor(labels_list, dtype=torch.long)
    return out


@torch.no_grad()
def infer_logits(model, seqs, pad_id, batch_size, device):
    model.eval()
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]), reverse=True)
    logits_out = np.zeros((len(seqs), 3), dtype=np.float32)
    for s in range(0, len(order), batch_size):
        bidx = order[s:s + batch_size]
        batch = pad_batch([seqs[i] for i in bidx], pad_id)
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lg = model(input_ids=ids, attention_mask=mask).logits
        logits_out[bidx] = lg.float().cpu().numpy()
    return logits_out


def softmax_np(logits, temp=1.0):
    v = logits.astype(np.float64) / temp
    v = v - v.max(axis=1, keepdims=True)
    e = np.exp(v)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float64)


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
    random.seed(SEED + local_rank)

    comp_dir = Path(os.environ["COMP_DIR"])
    model_dir = Path(os.environ["MODEL_DIR"])

    train_df = pd.read_csv(comp_dir / "train.csv")
    y = np.where(train_df["winner_model_a"] == 1, 0, np.where(train_df["winner_model_b"] == 1, 1, 2))
    skf = StratifiedKFold(n_splits=12, shuffle=True, random_state=SEED)
    val_idx = next(skf.split(train_df, y))[1]
    tr_mask = np.ones(len(train_df), dtype=bool)
    tr_mask[val_idx] = False
    tr_idx = np.where(tr_mask)[0]

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pad_id = tokenizer.pad_token_id

    if local_rank == 0:
        print(f"Tokenizing tr={len(tr_idx)} and va={len(val_idx)}...", flush=True)
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[val_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    y_va = y[val_idx]

    # Symmetric random 50% A/B swap on train (computed identically across ranks)
    rng_sw = np.random.default_rng(SEED)
    sw_mask = rng_sw.random(len(tr_df)) < 0.5
    tr_df_aug = tr_df.copy()
    tr_df_aug.loc[sw_mask, "response_a"], tr_df_aug.loc[sw_mask, "response_b"] = (
        tr_df.loc[sw_mask, "response_b"].values,
        tr_df.loc[sw_mask, "response_a"].values,
    )
    y_tr_aug = y_tr.copy()
    y_tr_aug[sw_mask] = np.where(y_tr[sw_mask] == 0, 1, np.where(y_tr[sw_mask] == 1, 0, 2))

    tr_seqs = tokenize_pairs(tr_df_aug, tokenizer, swap=False)
    va_seqs_orig = tokenize_pairs(va_df, tokenizer, swap=False)
    va_seqs_swap = tokenize_pairs(va_df, tokenizer, swap=True)
    tr_lens = [len(s) for s in tr_seqs]
    if local_rank == 0:
        print(f"Tokenized lengths p50={np.median(tr_lens):.0f}, p95={np.percentile(tr_lens,95):.0f}, max={max(tr_lens)}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=3, torch_dtype=torch.float16
    )
    model.config.pad_token_id = pad_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_cfg)
    # Keep trainable LoRA & score weights in fp32 for safe GradScaler fp16 training
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model = model.to(device)
    if local_rank == 0:
        model.print_trainable_parameters()

    ddp_model = DDP(model, device_ids=[device.index], output_device=device.index) if world_size > 1 else model

    per_gpu_bs = 2
    grad_accum = 8
    batches = make_length_bucketed_batches(tr_lens, per_gpu_bs, SEED, world_size, local_rank)
    total_opt_steps = max(1, len(batches) // grad_accum)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1.5e-4, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(10, int(0.05 * total_opt_steps)), total_opt_steps)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.CrossEntropyLoss()

    ddp_model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    opt_step = 0
    stop_flag = torch.tensor([0], device=device)

    for step, bidx in enumerate(batches):
        batch = pad_batch([tr_seqs[i] for i in bidx], pad_id, [int(y_tr_aug[i]) for i in bidx])
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        yb = batch["labels"].to(device, non_blocking=True)

        sync_ctx = (
            ddp_model.no_sync()
            if (world_size > 1 and (step + 1) % grad_accum != 0 and (step + 1) != len(batches))
            else torch.enable_grad()
        )
        with sync_ctx:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = ddp_model(input_ids=ids, attention_mask=mask).logits
                loss = loss_fn(logits.float(), yb) / grad_accum
            scaler.scale(loss).backward()
        running_loss += float(loss.item()) * grad_accum

        if (step + 1) % grad_accum == 0 or (step + 1) == len(batches):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            opt_step += 1

            if opt_step % 50 == 0:
                if local_rank == 0:
                    elapsed_m = (time.monotonic() - t0) / 60.0
                    print(f"step {opt_step}/{total_opt_steps} | loss={running_loss/50:.4f} | elapsed={elapsed_m:.1f}m", flush=True)
                    running_loss = 0.0
                    if elapsed_m > MAX_TRAIN_HOURS * 60.0:
                        stop_flag[0] = 1
                else:
                    running_loss = 0.0
                if world_size > 1:
                    dist.broadcast(stop_flag, src=0)
                if stop_flag.item() == 1:
                    if local_rank == 0:
                        print("Hit MAX_TRAIN_HOURS guard; proceeding to merge & validation.", flush=True)
                    break

    if world_size > 1:
        dist.barrier()
    del ddp_model, optimizer, scaler
    torch.cuda.empty_cache()

    if local_rank == 0:
        print("Merging LoRA weights into pure fp16 checkpoint...", flush=True)
        merged = model.merge_and_unload().half()
        merged_dir = WORK / "merged_model"
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)

        # Validation + inference speed benchmark using the pure merged fp16 model
        t_inf0 = time.monotonic()
        lg_orig = infer_logits(merged, va_seqs_orig, pad_id, 8, device)
        lg_swap = infer_logits(merged, va_seqs_swap, pad_id, 8, device)
        inf_s = time.monotonic() - t_inf0
        rows_per_sec_1gpu = (2 * len(va_df)) / max(1e-6, inf_s)
        # Dual-GPU T4 inference is ~1.9x faster for 25k TTA (50k passes)
        proj_25k_tta_min = (50000.0 / (rows_per_sec_1gpu * 1.9)) / 60.0

        np.save(WORK / "val_logits_orig.npy", lg_orig)
        np.save(WORK / "val_logits_swap.npy", lg_swap)
        np.save(WORK / "val_idx.npy", val_idx)

        raw_loss = float(log_loss(y_va, softmax_np(lg_orig, 1.0)))
        best_T, best_tta_loss = 1.0, 99.0
        for T in np.linspace(0.70, 1.50, 81):
            p = 0.5 * (softmax_np(lg_orig, T) + softmax_np(lg_swap, T)[:, [1, 0, 2]])
            l = float(log_loss(y_va, p))
            if l < best_tta_loss:
                best_tta_loss = l
                best_T = float(T)
        p_best = 0.5 * (softmax_np(lg_orig, best_T) + softmax_np(lg_swap, best_T)[:, [1, 0, 2]])
        acc_tta = float(accuracy_score(y_va, p_best.argmax(1)))

        calib = {
            "model_dir_hint": str(model_dir),
            "opt_steps": int(opt_step),
            "total_opt_steps": int(total_opt_steps),
            "val_logloss_orig": round(raw_loss, 5),
            "val_logloss_tta_T1": round(float(log_loss(y_va, 0.5 * (softmax_np(lg_orig, 1.0) + softmax_np(lg_swap, 1.0)[:, [1, 0, 2]]))), 5),
            "val_logloss_tta_cal": round(best_tta_loss, 5),
            "best_T": round(best_T, 3),
            "val_acc_tta": round(acc_tta, 5),
            "proj_25k_tta_2xT4_min": round(proj_25k_tta_min, 1),
            "total_elapsed_min": round((time.monotonic() - t0) / 60.0, 2),
        }
        (WORK / "calib.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")
        print("QWEN2.5-3B DONE:", json.dumps(calib), flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
'''


def main():
    ensure_peft()
    os.environ["COMP_DIR"] = str(find_comp_dir())
    os.environ["MODEL_DIR"] = str(find_model_dir(("qwen2.5", "qwen")))
    print("COMP_DIR:", os.environ["COMP_DIR"], "| MODEL_DIR:", os.environ["MODEL_DIR"], flush=True)

    worker_path = Path("/tmp/qwen25_worker.py")
    worker_path.write_text(WORKER_CODE, encoding="utf-8")
    ngpu = torch.cuda.device_count()
    if ngpu >= 2:
        subprocess.check_call(["torchrun", "--nproc_per_node", str(ngpu), str(worker_path)])
    else:
        subprocess.check_call([sys.executable, str(worker_path)])


if __name__ == "__main__":
    main()
