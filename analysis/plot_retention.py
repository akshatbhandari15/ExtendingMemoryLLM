#!/usr/bin/env python3
"""
Plot retention decay curves: accuracy vs update step, all strategies overlaid.

Reads results/{dataset}_{strategy}_nuc{N}.json and writes:
    figures/retention_squad.png
    figures/retention_nq.png

Usage:
    python analysis/plot_retention.py
    python analysis/plot_retention.py --results_dir results --nuc 20
"""

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt

STRATEGIES = ["random", "attention", "age", "surprise"]
COLORS = {
    "random":    "#888888",
    "attention": "#1f77b4",
    "age":       "#2ca02c",
    "surprise":  "#d62728",
}
DATASETS = ["squad", "nq"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_dataset(dataset, results_dir, nuc, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    found_any = False
    for strat in STRATEGIES:
        path = os.path.join(results_dir, f"{dataset}_{strat}_nuc{nuc}.json")
        if not os.path.exists(path):
            print(f"  [skip] {path} (missing)")
            continue
        d = load_json(path)
        accs = d["accuracy_per_step"]
        steps = list(range(len(accs)))
        ax.plot(steps, accs, marker="o", markersize=4, linewidth=1.8,
                label=f"{strat} (AUC={d['auc']:.2f})", color=COLORS[strat])
        found_any = True

    if not found_any:
        print(f"  [warn] no JSONs found for dataset={dataset}, nuc={nuc}")
        plt.close(fig)
        return False

    ax.set_xlabel("Update step (# distractors injected)")
    ax.set_ylabel("Exact-hit accuracy")
    ax.set_title(f"Knowledge retention — {dataset.upper()} (nuc={nuc})")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(0, nuc + 1, max(1, nuc // 10)))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--figures_dir", default="figures")
    p.add_argument("--nuc", type=int, default=20)
    args = p.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    for ds in DATASETS:
        out = os.path.join(args.figures_dir, f"retention_{ds}.png")
        plot_dataset(ds, args.results_dir, args.nuc, out)


if __name__ == "__main__":
    main()
