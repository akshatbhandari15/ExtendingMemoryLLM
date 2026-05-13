#!/usr/bin/env python3
"""
Position-of-answer bias analysis.

Tests whether a strategy's apparent win is just exploiting where the dataset
puts answers in the context. Specifically: does `age` win because the model
genuinely retains better, or because it preserves the recent memory block
and SQuAD/NQ tend to place gold answers in the late portion of contexts?

For each example we recompute the gold answer's normalized character position
within its source context (0.0 = start, 1.0 = end), bin examples into thirds
(early/mid/late), then compute step-20 accuracy per (strategy, bin) using the
same normalized matcher as analysis/rescore.py.

Outputs:
    results/position_bias.csv  — strategy x bin accuracy table
    figures/position_bias.png  — grouped bar chart, one panel per dataset

No GPU needed. Reads results/perlayer/*.json (the per_example dumps) and
re-instantiates the dataset to get (context, answer) pairs in eval order.

Usage:
    python analysis/position_bias.py
    python analysis/position_bias.py --results_dir results/perlayer \\
        --data_dir data --step 20 --bins 3
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.rescore import normalize, hit  # reuse the canonical matcher


def load_examples(dataset_name: str, data_dir: str, num_samples: int):
    """Return list of (context, answer) for the first `num_samples` eval items,
    in the same order the eval driver iterates."""
    # Use a lightweight tokenizer purely to satisfy the dataset constructor;
    # we don't actually tokenize anything here.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("YuWangX/memoryllm-8b")

    if dataset_name == "squad":
        from dataset.squad import SQuADDataset
        ds = SQuADDataset(
            filename=os.path.join(data_dir, "squad/dev-v2.0.json"),
            num=num_samples,
            num_unrelated_contexts=0,  # we don't need distractors for this analysis
            tokenizer=tok,
        )
        return [(ex["context"], ex["answer"]) for ex in ds.data]

    elif dataset_name == "nq":
        from dataset.nq import NQDataset
        ds = NQDataset(
            filename=os.path.join(data_dir, "nq/v1.0-simplified_nq-dev-all.jsonl"),
            num=num_samples,
            num_unrelated_contexts=0,
            tokenizer=tok,
        )
        return list(zip(ds.long_answers, ds.short_answers))

    raise ValueError(f"unknown dataset {dataset_name}")


def answer_position(context: str, answer: str) -> float:
    """Fractional position (0..1) of the answer's first occurrence in context.
    Uses normalized matching so case/punct differences don't fail. Returns
    None if the answer can't be located (rare for SQuAD, more common for NQ
    where the short answer may not appear verbatim in the long_answer)."""
    if not context or not answer:
        return None
    nc, na = normalize(context), normalize(answer)
    if not na or not nc:
        return None
    idx = nc.find(na)
    if idx < 0:
        return None
    # Fractional position of answer start. Use len(nc) - len(na) so the value
    # is on [0, 1] (i.e. answer that ends exactly at end-of-context => 1.0).
    denom = max(len(nc) - len(na), 1)
    return idx / denom


def bin_of(pos: float, n_bins: int) -> int:
    """Map pos in [0,1] to a bin index in [0, n_bins-1]."""
    if pos is None:
        return -1  # un-locatable
    return min(int(pos * n_bins), n_bins - 1)


def parse_result_filename(path: str):
    """results/perlayer/squad_age_nuc20_perlayer.json -> ('squad', 'age')."""
    base = os.path.basename(path).replace(".json", "")
    if base.endswith("_dropped"):
        return None, None
    parts = base.split("_")
    # squad_age_nuc20[_perlayer][_seedN] -> dataset=parts[0], strategy=parts[1]
    return parts[0], parts[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/perlayer")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_csv", default="results/position_bias.csv")
    ap.add_argument("--out_fig", default="figures/position_bias.png")
    ap.add_argument("--step", type=int, default=20,
                    help="Which step to score accuracy at (default: nuc=20).")
    ap.add_argument("--bins", type=int, default=3,
                    help="Number of position bins (default: 3 = early/mid/late).")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "*.json")))
    files = [f for f in files if not f.endswith("_dropped.json")]
    if not files:
        print(f"No result JSONs found in {args.results_dir}")
        sys.exit(1)

    # Group per (dataset). Position cache keyed by dataset so we load each
    # dataset exactly once.
    pos_cache = {}

    bin_labels = ["early", "mid", "late"] if args.bins == 3 else \
                 [f"q{i+1}" for i in range(args.bins)]

    # rows: (dataset, strategy, bin, n, accuracy)
    rows = []

    for fp in files:
        dataset, strategy = parse_result_filename(fp)
        if dataset is None:
            continue

        with open(fp) as f:
            d = json.load(f)

        per_ex = d.get("per_example")
        if per_ex is None:
            print(f"  [skip] {fp} has no per_example dump")
            continue

        N = len(per_ex)
        if dataset not in pos_cache:
            print(f"Loading {dataset} dataset (N={N}) to get answer positions...")
            ex = load_examples(dataset, args.data_dir, N)
            positions = [answer_position(c, a) for c, a in ex]
            pos_cache[dataset] = positions
        positions = pos_cache[dataset]

        # Step-20 hit per example, then bin
        step_key = f"step_{args.step}"
        per_bin_total = [0] * args.bins
        per_bin_hit   = [0] * args.bins
        unlocatable   = 0

        for i, ex in enumerate(per_ex):
            if i >= len(positions):
                break
            b = bin_of(positions[i], args.bins)
            if b < 0:
                unlocatable += 1
                continue
            per_bin_total[b] += 1
            if hit(ex["step_preds"][step_key], ex["target"]):
                per_bin_hit[b] += 1

        for b in range(args.bins):
            n = per_bin_total[b]
            acc = (per_bin_hit[b] / n) if n > 0 else float("nan")
            rows.append({
                "dataset":  dataset,
                "strategy": strategy,
                "bin":      bin_labels[b] if b < len(bin_labels) else f"b{b}",
                "n":        n,
                "accuracy": acc,
            })

        print(f"  {dataset:6s} {strategy:10s}  bins(n)={per_bin_total}  "
              f"hits={per_bin_hit}  unlocatable={unlocatable}/{N}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "strategy", "bin", "n", "accuracy"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {args.out_csv} ({len(rows)} rows)")

    # Plot — one subplot per dataset, grouped bars over strategies
    datasets = sorted({r["dataset"] for r in rows})
    strategies = sorted({r["strategy"] for r in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5),
                             sharey=True, squeeze=False)
    width = 0.8 / max(len(strategies), 1)
    x = np.arange(args.bins)

    for ax, ds in zip(axes[0], datasets):
        for si, strat in enumerate(strategies):
            ys = []
            for b in range(args.bins):
                ms = [r for r in rows if r["dataset"] == ds
                                       and r["strategy"] == strat
                                       and r["bin"] == bin_labels[b]]
                ys.append(ms[0]["accuracy"] if ms else float("nan"))
            ax.bar(x + (si - len(strategies)/2 + 0.5) * width, ys, width, label=strat)
        ax.set_title(f"{ds} — step {args.step} accuracy by answer position")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels[:args.bins])
        ax.set_xlabel("Answer position in context")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=140)
    print(f"Wrote {args.out_fig}")


if __name__ == "__main__":
    main()
