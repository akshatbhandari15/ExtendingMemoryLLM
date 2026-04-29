#!/usr/bin/env python3
"""
Statistical significance testing across strategies.

For each (dataset, strategy_pair):
  - Recompute per-example, per-step accuracy from per_example dump
  - Bootstrap 95% CI on per-strategy AUC (resample examples with replacement)
  - Paired permutation test on AUC difference between strategies

Pair tests use 'random' as the baseline (3 comparisons per dataset:
random vs attention, age, surprise). Bonferroni correction for 3 tests
within each dataset.

Reads results/{dataset}_{strategy}_nuc{N}.json (needs per_example dump).
Writes results/significance.csv.

Usage:
    python analysis/significance.py
    python analysis/significance.py --bootstrap_iters 5000 --perm_iters 10000
"""

import argparse
import csv
import json
import os

import numpy as np

STRATEGIES = ["random", "attention", "age", "surprise"]
DATASETS = ["squad", "nq"]
BASELINE = "random"


def _normalize(s):
    import re, string
    s = s.replace("</s>", "").replace("<|end_of_text|>", "")
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def per_example_hit_matrix(per_example, nuc):
    """Return an (N_examples, nuc+1) bool array of hit/miss per example per step."""
    N = len(per_example)
    M = np.zeros((N, nuc + 1), dtype=bool)
    for i, ex in enumerate(per_example):
        t = _normalize(ex["target"])
        for s in range(nuc + 1):
            p = _normalize(ex["step_preds"].get(f"step_{s}", ""))
            M[i, s] = t in p
    return M


def auc_of_acc_curve(acc_curve):
    """np.trapezoid over the curve (matches what run_eval saves)."""
    return float(np.trapezoid(acc_curve))


def bootstrap_auc_ci(hit_matrix, n_iters, rng):
    """Bootstrap 95% CI on AUC by resampling examples with replacement."""
    N = hit_matrix.shape[0]
    aucs = np.empty(n_iters)
    for b in range(n_iters):
        idx = rng.integers(0, N, size=N)
        acc_curve = hit_matrix[idx].mean(axis=0)
        aucs[b] = auc_of_acc_curve(acc_curve)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_permutation_pvalue(hit_a, hit_b, n_iters, rng):
    """
    Two-sided paired permutation test on AUC(b) - AUC(a).
    Same examples for both strategies (paired by example index).
    Permutation = randomly swap a/b assignment per example.
    """
    assert hit_a.shape == hit_b.shape
    obs_diff = auc_of_acc_curve(hit_b.mean(0)) - auc_of_acc_curve(hit_a.mean(0))

    N = hit_a.shape[0]
    extreme = 0
    for _ in range(n_iters):
        swap = rng.random(N) < 0.5
        ma = np.where(swap[:, None], hit_b, hit_a)
        mb = np.where(swap[:, None], hit_a, hit_b)
        d = auc_of_acc_curve(mb.mean(0)) - auc_of_acc_curve(ma.mean(0))
        if abs(d) >= abs(obs_diff):
            extreme += 1
    p = (extreme + 1) / (n_iters + 1)
    return float(obs_diff), float(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--nuc", type=int, default=20)
    p.add_argument("--out", default="results/significance.csv")
    p.add_argument("--bootstrap_iters", type=int, default=2000)
    p.add_argument("--perm_iters", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []

    for ds in DATASETS:
        # Load all strategies for this dataset
        hits = {}
        aucs = {}
        for strat in STRATEGIES:
            path = os.path.join(args.results_dir, f"{ds}_{strat}_nuc{args.nuc}.json")
            if not os.path.exists(path):
                print(f"  [skip] {path}")
                continue
            with open(path) as f:
                d = json.load(f)
            if "per_example" not in d:
                print(f"  [warn] {path} has no per_example dump — cannot do significance")
                continue
            H = per_example_hit_matrix(d["per_example"], args.nuc)
            hits[strat] = H
            aucs[strat] = auc_of_acc_curve(H.mean(0))

        if BASELINE not in hits:
            print(f"  [skip] {ds}: no baseline ({BASELINE}) JSON")
            continue

        # Bootstrap CIs (per-strategy, independent)
        cis = {}
        for strat, H in hits.items():
            lo, hi = bootstrap_auc_ci(H, args.bootstrap_iters, rng)
            cis[strat] = (lo, hi)
            print(f"  {ds}/{strat}:  AUC={aucs[strat]:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

        # Pairwise vs baseline (random)
        n_pairs = sum(1 for s in STRATEGIES if s != BASELINE and s in hits)
        for strat in STRATEGIES:
            if strat == BASELINE or strat not in hits:
                continue
            # Align example sets by truncating to min length (should be equal)
            n = min(hits[BASELINE].shape[0], hits[strat].shape[0])
            diff, p = paired_permutation_pvalue(
                hits[BASELINE][:n], hits[strat][:n],
                args.perm_iters, rng,
            )
            p_bonf = min(1.0, p * n_pairs)
            sig = "**" if p_bonf < 0.01 else ("*" if p_bonf < 0.05 else "ns")
            print(f"  {ds}: {strat} vs {BASELINE}:  ΔAUC={diff:+.3f}  p={p:.4f}  p_bonf={p_bonf:.4f}  {sig}")
            rows.append({
                "dataset":   ds,
                "strategy":  strat,
                "baseline":  BASELINE,
                "auc":       round(aucs[strat], 4),
                "auc_ci_lo": round(cis[strat][0], 4),
                "auc_ci_hi": round(cis[strat][1], 4),
                "delta_auc": round(diff, 4),
                "p_value":   round(p, 5),
                "p_bonf":    round(p_bonf, 5),
                "n_pairs":   n_pairs,
                "sig":       sig,
            })

        # Add baseline row for completeness
        if BASELINE in hits:
            rows.append({
                "dataset":   ds,
                "strategy":  BASELINE,
                "baseline":  "(self)",
                "auc":       round(aucs[BASELINE], 4),
                "auc_ci_lo": round(cis[BASELINE][0], 4),
                "auc_ci_hi": round(cis[BASELINE][1], 4),
                "delta_auc": 0.0,
                "p_value":   1.0,
                "p_bonf":    1.0,
                "n_pairs":   n_pairs,
                "sig":       "—",
            })

    if not rows:
        print("No significance rows produced.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = ["dataset", "strategy", "baseline", "auc", "auc_ci_lo", "auc_ci_hi",
                  "delta_auc", "p_value", "p_bonf", "n_pairs", "sig"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
