#!/usr/bin/env python3
"""
E0 — Memory-presence sanity check for YuWangX/memoryllm-8b.

Verifies that the pretrained checkpoint actually ships trained memory weights
(not random init). If this fails, every downstream retention experiment is
meaningless — the model would be running on its base LLM attention only.

Conditions (all run on the same 30 SQuAD examples):
  1. normal    — memory reset to checkpoint state before each example
  2. zeroed    — memory reset to torch.zeros_like(...) before each example
  3. scrambled — memory reset to a per-layer-permuted copy of the checkpoint

Pass criterion (per plan):
  - LOAD REPORT shows no missing memory/initialized/positional-emb keys
  - step-0 (normal) - step-0 (zeroed) > 0.10 absolute (>10 pts separation)

Usage:
    python run_sanity.py --num_samples 30 --nuc 20
    python run_sanity.py --output_dir /content/drive/MyDrive/ExtendingMemoryLLM/results
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import LlamaTokenizer

from modeling_memoryllm_strategies import MemoryLLMWithStrategies
from run_eval import build_dataloader, exact_hit


CONDITIONS = ("normal", "zeroed", "scrambled")


def parse_args():
    p = argparse.ArgumentParser(description="E0 memory-presence sanity check")
    p.add_argument("--model",       default="YuWangX/memoryllm-8b")
    p.add_argument("--nuc",         type=int, default=20)
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--num_tokens",  type=int, default=256)
    p.add_argument("--output_dir",  default="results")
    p.add_argument("--dtype",       default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def reset_memory(model, condition, checkpoint_memory, rng):
    """Reset model.memory per E0 condition. Mutates model.memory.data in place."""
    if condition == "normal":
        model.memory.data.copy_(checkpoint_memory)
    elif condition == "zeroed":
        model.memory.data.zero_()
    elif condition == "scrambled":
        L, N, _ = checkpoint_memory.shape
        for li in range(L):
            perm = torch.from_numpy(rng.permutation(N)).to(checkpoint_memory.device)
            model.memory.data[li] = checkpoint_memory[li][perm]
    else:
        raise ValueError(f"Unknown condition: {condition}")


def run_condition(condition, model, tokenizer, dataloader, nuc, device,
                  checkpoint_memory, seed):
    rng = np.random.default_rng(seed)
    model.eval()

    step_preds = {f"step_{i}": [] for i in range(nuc + 1)}
    step_tgts  = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  {condition}"):
            batch = [x.to(device) for x in batch]
            ctx_ids, ctx_mask, q_ids, q_mask, a_ids, _ = batch[:6]
            unrel = batch[6:]

            reset_memory(model, condition, checkpoint_memory, rng)

            seq_ids   = [ctx_ids]  + [unrel[i * 2]     for i in range(nuc)]
            seq_masks = [ctx_mask] + [unrel[i * 2 + 1] for i in range(nuc)]

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
    return accs


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print(f"Loading {args.model} ...")
    model, load_info = MemoryLLMWithStrategies.from_pretrained(
        args.model, output_loading_info=True,
    )
    model = model.to(device).to(dtype)
    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(args.model)

    # LOAD REPORT — the core of the sanity check
    missing    = load_info.get("missing_keys", [])
    unexpected = load_info.get("unexpected_keys", [])
    mismatched = load_info.get("mismatched_keys", [])
    memory_key_fragments = ("memory", "initialized", "new_memory_positional_emb", "bos_embedding")
    memory_missing = [k for k in missing if any(f in k for f in memory_key_fragments)]

    mem_mean = float(model.memory.data.mean().item())
    mem_std  = float(model.memory.data.std().item())
    mem_abs  = float(model.memory.data.abs().max().item())
    init_val = int(model.initialized.item()) if hasattr(model, "initialized") else -1

    print("\n" + "=" * 60)
    print("  LOAD REPORT")
    print("=" * 60)
    print(f"  missing_keys        : {len(missing)}")
    print(f"  unexpected_keys     : {len(unexpected)}")
    print(f"  mismatched_keys     : {len(mismatched)}")
    print(f"  memory keys missing : {memory_missing if memory_missing else 'none'}")
    print(f"  initialized buffer  : {init_val}  (1 = memory populated during pretraining)")
    print(f"  memory.data stats   : mean={mem_mean:+.4f}  std={mem_std:.4f}  abs_max={mem_abs:.4f}")
    print(f"  memory.shape        : {tuple(model.memory.shape)}")

    if memory_missing:
        print("\n  [WARN] Memory-related keys MISSING — checkpoint does not ship trained memory.")
        print("         Proceeding with sanity eval, but expect all conditions to tie.")

    # Stash checkpoint memory before any injection dirties it
    checkpoint_memory = model.memory.data.clone()
    model.set_drop_strategy("random")  # irrelevant for E0 — no drops happen at step 0

    dataloader = build_dataloader(
        "squad", args.model, args.nuc, args.num_samples, args.num_tokens, tokenizer,
    )

    results = {
        "config": {
            "model":       args.model,
            "nuc":         args.nuc,
            "num_samples": args.num_samples,
            "seed":        args.seed,
            "dtype":       args.dtype,
        },
        "load_report": {
            "missing_keys_count":    len(missing),
            "unexpected_keys_count": len(unexpected),
            "memory_keys_missing":   memory_missing,
            "initialized":           init_val,
            "memory_stats":          {"mean": mem_mean, "std": mem_std, "abs_max": mem_abs},
            "memory_shape":          list(model.memory.shape),
        },
        "accuracy_per_step_by_condition": {},
    }

    for condition in CONDITIONS:
        print(f"\n  Running condition: {condition}")
        accs = run_condition(
            condition, model, tokenizer, dataloader, args.nuc, device,
            checkpoint_memory, args.seed,
        )
        results["accuracy_per_step_by_condition"][condition] = accs
        print(f"    step-0 acc         : {accs[0]:.3f}")
        print(f"    step-{args.nuc:<2} acc        : {accs[-1]:.3f}")
        print(f"    AUC (trapz)        : {float(np.trapz(accs)):.3f}")

    # Verdict
    normal0  = results["accuracy_per_step_by_condition"]["normal"][0]
    zeroed0  = results["accuracy_per_step_by_condition"]["zeroed"][0]
    scram0   = results["accuracy_per_step_by_condition"]["scrambled"][0]
    gap_nz   = normal0 - zeroed0
    gap_ns   = normal0 - scram0

    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)
    print(f"  step-0 normal   : {normal0:.3f}")
    print(f"  step-0 zeroed   : {zeroed0:.3f}")
    print(f"  step-0 scrambled: {scram0:.3f}")
    print(f"  normal - zeroed : {gap_nz:+.3f}")
    print(f"  normal - scram. : {gap_ns:+.3f}")

    passed = gap_nz > 0.10 and not memory_missing
    results["verdict"] = {
        "passed":           bool(passed),
        "normal_vs_zeroed": gap_nz,
        "normal_vs_scram":  gap_ns,
    }

    if passed:
        print("  [PASS] Memory contributes. Safe to proceed with retention experiments.")
    else:
        print("  [FAIL] Memory appears INERT at step 0 (gap < 0.10 or keys missing).")
        print("         Pivot to fallback narrative: 'Random is the upper bound")
        print("         when memory is untrained' (see plan).")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "sanity_check.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
