#!/usr/bin/env python3
"""
Build the AUC summary table across (dataset, strategy).

Reads results/{dataset}_{strategy}_nuc{N}.json and writes results/auc_summary.csv
with columns: dataset, strategy, nuc, num_samples, auc, step_0, step_final, elapsed_s.

Usage:
    python analysis/auc_table.py
    python analysis/auc_table.py --nuc 20 --out results/auc_summary.csv
"""

import argparse
import csv
import json
import os

STRATEGIES = ["random", "attention", "age", "surprise"]
DATASETS = ["squad", "nq"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--nuc", type=int, default=20)
    p.add_argument("--out", default="results/auc_summary.csv")
    args = p.parse_args()

    rows = []
    for ds in DATASETS:
        for strat in STRATEGIES:
            path = os.path.join(args.results_dir, f"{ds}_{strat}_nuc{args.nuc}.json")
            if not os.path.exists(path):
                print(f"  [skip] {path} (missing)")
                continue
            with open(path) as f:
                d = json.load(f)
            accs = d["accuracy_per_step"]
            rows.append({
                "dataset":     ds,
                "strategy":    strat,
                "nuc":         d["config"]["nuc"],
                "num_samples": d["config"]["num_samples"],
                "auc":         round(d["auc"], 4),
                "step_0":      round(accs[0], 4),
                "step_final":  round(accs[-1], 4),
                "elapsed_s":   round(d.get("elapsed_seconds", -1), 1),
            })

    if not rows:
        print("No JSONs found.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = ["dataset", "strategy", "nuc", "num_samples",
                  "auc", "step_0", "step_final", "elapsed_s"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  -> {args.out}")

    # Pretty-print to stdout
    print("\n  dataset  strategy   nuc   N    AUC   step0  stepN")
    print("  " + "-" * 52)
    for r in rows:
        print(f"  {r['dataset']:<8} {r['strategy']:<10} {r['nuc']:<5} {r['num_samples']:<4}"
              f" {r['auc']:<6} {r['step_0']:<6} {r['step_final']}")


if __name__ == "__main__":
    main()
