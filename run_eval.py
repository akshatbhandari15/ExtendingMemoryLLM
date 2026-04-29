#!/usr/bin/env python3
"""
Retention eval launcher for MemoryLLMWithStrategies.

Runs knowledge retention curves: inject a target context, then N distractor
contexts one by one, and query accuracy at each step.

Usage:
    # Single strategy
    python run_eval.py --strategy attention --dataset squad --nuc 20

    # All strategies back-to-back (loads model once, saves time)
    python run_eval.py --strategy all --dataset squad --nuc 20

    # Quick smoke test (5 examples, 3 distractor steps)
    python run_eval.py --strategy random --dataset squad --nuc 3 --num_samples 5

    # Skip already-completed runs (safe to re-run after interruption)
    python run_eval.py --strategy all --dataset squad --nuc 20 --resume
"""

import argparse
import json
import os
import time
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from modeling_memoryllm_strategies import MemoryLLMWithStrategies
from dataset.nq import NQDataset
from dataset.squad import SQuADDataset


STRATEGIES = ["random", "attention", "age", "surprise"]

DATA_PATHS = {
    "squad": "./data/squad/dev-v2.0.json",
    "nq":    "./data/nq/v1.0-simplified_nq-dev-all.jsonl",
}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MemoryLLM retention eval")
    p.add_argument("--model",       default="YuWangX/memoryllm-8b",
                   help="HuggingFace model ID or local path")
    p.add_argument("--strategy",    default="random",
                   choices=STRATEGIES + ["all"],
                   help="Drop strategy, or 'all' to run every strategy sequentially")
    p.add_argument("--dataset",     default="squad",
                   choices=list(DATA_PATHS.keys()))
    p.add_argument("--nuc",         type=int, default=20,
                   help="Number of unrelated (distractor) contexts to inject after the target")
    p.add_argument("--num_samples", type=int, default=100,
                   help="Number of eval examples (None = full dev set)")
    p.add_argument("--num_tokens",  type=int, default=256,
                   help="Tokens per memory block — must match the model config")
    p.add_argument("--output_dir",  default="results")
    p.add_argument("--dtype",       default="bfloat16",
                   choices=["float16", "bfloat16"])
    p.add_argument("--resume",      action="store_true",
                   help="Skip strategies whose output file already exists")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def collate_fn(data, tokenizer, max_length, num_tokens, padding="longest"):
    contexts, questions, answers, unrelated_contexts = zip(*data)

    def tok(texts, max_len, pad):
        return tokenizer(
            list(texts), max_length=max_len, padding=pad,
            truncation=True, return_tensors="pt", add_special_tokens=False,
        )

    ctx   = tok(contexts,  512,        padding)
    quest = tok(questions, max_length, "longest")
    ans   = tok(list(answers), max_length, "longest")

    # unrelated_contexts shape: (batch, nuc) -> transpose to (nuc, batch)
    unrelated = np.array(unrelated_contexts).transpose().tolist()

    def mem_prefix(n):
        return torch.ones(n, num_tokens)

    ctx_mask   = torch.cat([mem_prefix(ctx.input_ids.shape[0]),   ctx.attention_mask],   dim=1)
    quest_mask = torch.cat([mem_prefix(quest.input_ids.shape[0]), quest.attention_mask], dim=1)
    ans_mask   = torch.cat([mem_prefix(ans.input_ids.shape[0]),   ans.attention_mask],   dim=1)

    out = (ctx.input_ids, ctx_mask,
           quest.input_ids, quest_mask,
           ans.input_ids,   ans_mask)

    for uc in unrelated:
        uc_tok  = tok(uc, 512, padding)
        uc_mask = torch.cat([mem_prefix(uc_tok.input_ids.shape[0]), uc_tok.attention_mask], dim=1)
        out += (uc_tok.input_ids, uc_mask)

    return out


def build_dataloader(dataset_name, model_path, nuc, num_samples, num_tokens, tokenizer):
    kwargs = dict(
        num=num_samples,
        num_unrelated_contexts=nuc,
        tokenizer="llama",
        tokenizer_path=model_path,
    )
    if dataset_name == "squad":
        ds = SQuADDataset(filename=DATA_PATHS["squad"], **kwargs)
    else:
        ds = NQDataset(filename=DATA_PATHS["nq"], **kwargs)

    cfn = partial(collate_fn, tokenizer=tokenizer,
                  max_length=256, num_tokens=num_tokens)
    return DataLoader(ds, batch_size=1, shuffle=False,
                      num_workers=2, collate_fn=cfn)


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def _normalize(s):
    """SQuAD-style: drop EOS markers, lowercase, strip punct, collapse whitespace."""
    import re, string
    s = s.replace("</s>", "").replace("<|end_of_text|>", "")
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def exact_hit(predictions, targets):
    """Fraction of predictions that contain the gold answer (normalized substring)."""
    hits = sum(_normalize(t) in _normalize(p) for p, t in zip(predictions, targets))
    return hits / max(len(predictions), 1)


def run_retention_eval(model, tokenizer, dataloader, nuc, device):
    """
    For each example:
      - inject target context
      - inject nuc distractors one by one
      - generate answer after each injection
    Returns:
      accs:        list of float, length nuc+1 (accuracy at each step)
      per_example: list of dicts with raw preds for qualitative inspection
    """
    model.eval()

    # Snapshot the pretrained memory so we can restore it before each example.
    # Resetting to zeros wipes 49/50 of the memory pool (inject_memory only
    # rewrites one block of num_tokens), which destroys retention.
    checkpoint_memory = model.memory.data.detach().clone()

    step_preds = {f"step_{i}": [] for i in range(nuc + 1)}
    step_tgts  = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  eval"):
            batch = [x.to(device) for x in batch]
            ctx_ids, ctx_mask, q_ids, q_mask, a_ids, _a_mask = batch[:6]
            unrel = batch[6:]  # alternating: ids, mask for each distractor

            # Reset memory to pretrained checkpoint state for this example
            model.memory.data.copy_(checkpoint_memory)

            # Injection sequence: target first, then distractors
            seq_ids   = [ctx_ids]  + [unrel[i * 2]     for i in range(nuc)]
            seq_masks = [ctx_mask] + [unrel[i * 2 + 1] for i in range(nuc)]

            # Memory prefix prepended to question mask
            full_q_mask = torch.cat([
                torch.ones(q_ids.shape[0],
                           model.num_tokens * (model.num_blocks - 1),
                           device=device),
                q_mask,
            ], dim=1)

            for step, (ids, mask) in enumerate(zip(seq_ids, seq_masks)):
                model.inject_memory(ids, mask, update_memory=True)

                pred = model.generate(
                    inputs=q_ids,
                    attention_mask=full_q_mask,
                    max_new_tokens=10,
                    pad_token_id=tokenizer.pad_token_id,
                )[:, q_ids.shape[1]:][0].cpu()

                step_preds[f"step_{step}"].append(pred)

            step_tgts.append(a_ids[0].cpu())

    targets_dec = tokenizer.batch_decode(step_tgts)
    preds_dec   = {k: tokenizer.batch_decode(v) for k, v in step_preds.items()}

    accs = [exact_hit(preds_dec[f"step_{i}"], targets_dec) for i in range(nuc + 1)]

    per_example = [
        {
            "target":     targets_dec[i],
            "step_preds": {k: preds_dec[k][i] for k in preds_dec},
        }
        for i in range(len(targets_dec))
    ]

    return accs, per_example


# ---------------------------------------------------------------------------
# Per-strategy runner
# ---------------------------------------------------------------------------

def run_strategy(strategy, model, tokenizer, args, device):
    out_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_{strategy}_nuc{args.nuc}.json",
    )

    if args.resume and os.path.exists(out_path):
        print(f"  [skip] {out_path} already exists")
        return

    print(f"\n{'='*60}")
    print(f"  Strategy : {strategy}")
    print(f"  Dataset  : {args.dataset}  |  NUC: {args.nuc}  |  N: {args.num_samples}")
    print(f"{'='*60}")

    model.set_drop_strategy(strategy)

    dataloader = build_dataloader(
        args.dataset, args.model, args.nuc,
        args.num_samples, args.num_tokens, tokenizer,
    )

    t0 = time.time()
    accs, per_example = run_retention_eval(model, tokenizer, dataloader, args.nuc, device)
    elapsed = time.time() - t0

    auc = float(np.trapezoid(accs))

    print(f"\n  Step accuracies : {[f'{a:.3f}' for a in accs]}")
    print(f"  AUC             : {auc:.4f}")
    print(f"  Elapsed         : {elapsed:.1f}s")

    results = {
        "config": {
            "model":       args.model,
            "strategy":    strategy,
            "dataset":     args.dataset,
            "nuc":         args.nuc,
            "num_samples": args.num_samples,
            "dtype":       args.dtype,
        },
        "timestamp":         time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds":   elapsed,
        "accuracy_per_step": accs,
        "auc":               auc,
        "per_example":       per_example,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    strategies = STRATEGIES if args.strategy == "all" else [args.strategy]

    print(f"Loading {args.model} ...")
    model = MemoryLLMWithStrategies.from_pretrained(args.model).to(device).to(dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded — device: {device}, dtype: {args.dtype}")

    for strategy in strategies:
        run_strategy(strategy, model, tokenizer, args, device)

    print("\nAll done.")


if __name__ == "__main__":
    main()
