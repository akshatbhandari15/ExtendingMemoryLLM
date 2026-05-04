#!/usr/bin/env python3
"""
Layer-Jaccard analysis of dropped token indices.

Reads the *_dropped.json files produced by run_eval.py --log_dropped and
computes:

  Layer-Jaccard (within-strategy): average pairwise Jaccard similarity of the
    dropped-token sets across transformer layers for a given (example, step).
    High  = synchronized drops (bad — layers evict the same tokens, losing
            independent retention chances).
    Low   = diverse drops (good — each layer independently retains tokens).
    Note: in shared-drop mode (drop_memory_per_layer=False), Jaccard = 1.0
    by definition for all strategies.

  Strategy-Jaccard (cross-strategy): average overlap between the sets dropped
    by two different strategies at the same (example, step).
    High = strategies make the same choices (redundant).
    Low  = genuinely different eviction decisions.

Usage:
    # Analyse all *_dropped.json files in results/
    python analysis/dropped_indices.py

    # Specific files
    python analysis/dropped_indices.py --files results/squad_random_nuc20_perlayer_dropped.json \\
                                               results/squad_age_nuc20_perlayer_dropped.json

    # Save CSV summary
    python analysis/dropped_indices.py --out results/jaccard_summary.csv
"""

import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np


# ---------------------------------------------------------------------------
# Jaccard helpers
# ---------------------------------------------------------------------------

def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    u = sa | sb
    if not u:
        return 1.0
    return len(sa & sb) / len(u)


def layer_jaccard_entry(entry: dict) -> float:
    """Mean pairwise Jaccard across all layer pairs for one (example, step) entry."""
    mode = entry.get("mode", "shared")
    if mode == "shared":
        return 1.0  # by definition — same indices applied to every layer

    layers: dict = entry["layers"]
    if len(layers) < 2:
        return 1.0

    keys = sorted(layers.keys(), key=int)
    scores = []
    for i, j in combinations(keys, 2):
        scores.append(jaccard(layers[i], layers[j]))
    return float(np.mean(scores)) if scores else 1.0


def compute_layer_jaccard(drop_log: list) -> dict:
    """
    Returns per-step mean and std of layer-Jaccard.
    Also returns the overall mean across all steps.
    """
    by_step: dict = {}
    for entry in drop_log:
        step = entry["step"]
        lj = layer_jaccard_entry(entry)
        by_step.setdefault(step, []).append(lj)

    per_step = {
        step: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for step, vals in sorted(by_step.items())
    }
    all_vals = [v for vals in by_step.values() for v in vals]
    overall = float(np.mean(all_vals)) if all_vals else float("nan")
    return {"overall": overall, "per_step": per_step}


# ---------------------------------------------------------------------------
# Cross-strategy Jaccard
# ---------------------------------------------------------------------------

def build_index(drop_log: list) -> dict:
    """Index entries by (example, step) → dropped set (shared or layer-0 for per_layer)."""
    idx = {}
    for entry in drop_log:
        key = (entry["example"], entry["step"])
        if entry.get("mode") == "shared":
            idx[key] = set(entry["shared_dropped"])
        else:
            # Use union across all layers as the "total dropped set"
            union = set()
            for positions in entry["layers"].values():
                union.update(positions)
            idx[key] = union
    return idx


def cross_strategy_jaccard(log_a: list, log_b: list) -> float:
    """Mean Jaccard between strategy A and B at matched (example, step) pairs."""
    idx_a = build_index(log_a)
    idx_b = build_index(log_b)
    shared_keys = set(idx_a) & set(idx_b)
    if not shared_keys:
        return float("nan")
    scores = [jaccard(list(idx_a[k]), list(idx_b[k])) for k in shared_keys]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results",
                   help="Directory to glob *_dropped.json from")
    p.add_argument("--files", nargs="+", default=None,
                   help="Explicit file paths (overrides --results_dir)")
    p.add_argument("--out", default=None,
                   help="Optional CSV path to save summary")
    args = p.parse_args()

    if args.files:
        files = args.files
    else:
        files = sorted(glob.glob(os.path.join(args.results_dir, "*_dropped.json")))

    if not files:
        print("No *_dropped.json files found. Run: python run_eval.py --log_dropped")
        return

    logs = {}
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        cfg  = data["config"]
        name = f"{cfg['dataset']}_{cfg['strategy']}_nuc{cfg['nuc']}"
        mode = data["drop_log"][0]["mode"] if data["drop_log"] else "unknown"
        logs[name] = (data["drop_log"], mode, fpath)
        print(f"  Loaded {os.path.basename(fpath)} — {len(data['drop_log'])} entries, mode={mode}")

    print()

    # ---- Layer-Jaccard per strategy ----
    print("Layer-Jaccard (within-strategy, mean across all steps)")
    print("  NOTE: shared mode always gives 1.00 by definition.")
    print(f"  {'Strategy':<45}  {'Mode':<10}  {'Layer-Jaccard':>14}")
    print("  " + "-" * 75)

    layer_jaccards = {}
    rows = []
    for name, (log, mode, _) in sorted(logs.items()):
        lj = compute_layer_jaccard(log)
        layer_jaccards[name] = lj["overall"]
        print(f"  {name:<45}  {mode:<10}  {lj['overall']:>14.4f}")
        rows.append({"name": name, "mode": mode, "layer_jaccard": lj["overall"]})

    # ---- Cross-strategy Jaccard ----
    if len(logs) >= 2:
        print()
        print("Cross-strategy Jaccard (do strategies drop the same tokens?)")
        print(f"  {'Pair':<60}  {'Jaccard':>10}")
        print("  " + "-" * 75)
        names = sorted(logs.keys())
        for a, b in combinations(names, 2):
            cj = cross_strategy_jaccard(logs[a][0], logs[b][0])
            print(f"  {a} vs {b:<20}  {cj:>10.4f}")

    # ---- Save CSV ----
    if args.out and rows:
        import csv
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "mode", "layer_jaccard"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Saved -> {args.out}")


if __name__ == "__main__":
    main()
