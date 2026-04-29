#!/usr/bin/env python3
"""
Plot retention decay curves with bootstrap 95% CI bands.

Reads results/{dataset}_{strategy}_nuc{N}.json and writes:
    figures/retention_squad.png
    figures/retention_nq.png
    figures/retention_combined.png   (side-by-side)

Usage:
    python analysis/plot_retention.py
    python analysis/plot_retention.py --no_ci      # disable CI bands
    python analysis/plot_retention.py --bootstrap_iters 5000
"""

import argparse
import json
import os
import re
import string

import matplotlib.pyplot as plt
import numpy as np

STRATEGIES = ["random", "attention", "age", "surprise"]
# Colorblind-safe palette (Wong 2011); random is gray to read as baseline
COLORS = {
    "random":    "#7f7f7f",   # neutral gray
    "attention": "#0072B2",   # blue
    "age":       "#009E73",   # green
    "surprise":  "#D55E00",   # vermillion
}
LINESTYLES = {
    "random":    "--",        # dashed = baseline
    "attention": "-",
    "age":       "-",
    "surprise":  "-",
}
DATASETS = ["squad", "nq"]


def normalize(s):
    s = s.replace("</s>", "").replace("<|end_of_text|>", "")
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def load_hit_matrix(path, nuc):
    """Return (N, nuc+1) bool matrix of per-example, per-step hits."""
    with open(path) as f:
        d = json.load(f)
    if "per_example" not in d:
        return None, d.get("accuracy_per_step")
    N = len(d["per_example"])
    M = np.zeros((N, nuc + 1), dtype=bool)
    for i, ex in enumerate(d["per_example"]):
        t = normalize(ex["target"])
        for s in range(nuc + 1):
            M[i, s] = t in normalize(ex["step_preds"].get(f"step_{s}", ""))
    return M, d["accuracy_per_step"]


def bootstrap_band(hit_matrix, n_iters, rng):
    """Bootstrap 95% CI band on per-step accuracy by resampling examples."""
    N, T = hit_matrix.shape
    samples = np.empty((n_iters, T))
    for b in range(n_iters):
        idx = rng.integers(0, N, size=N)
        samples[b] = hit_matrix[idx].mean(axis=0)
    lo = np.percentile(samples, 2.5, axis=0)
    hi = np.percentile(samples, 97.5, axis=0)
    return lo, hi


def auc_of(curve):
    return float(np.trapezoid(curve))


def plot_dataset(ax, dataset, results_dir, nuc, draw_ci, n_iters, rng):
    found = False
    y_max = 0.0
    y_min = 1.0

    for strat in STRATEGIES:
        path = os.path.join(results_dir, f"{dataset}_{strat}_nuc{nuc}.json")
        if not os.path.exists(path):
            continue
        H, accs_saved = load_hit_matrix(path, nuc)
        if H is None:
            accs = np.array(accs_saved)
        else:
            accs = H.mean(axis=0)
        steps = np.arange(len(accs))
        auc = auc_of(accs)

        if draw_ci and H is not None:
            lo, hi = bootstrap_band(H, n_iters, rng)
            ax.fill_between(steps, lo, hi, color=COLORS[strat], alpha=0.15, linewidth=0)
            y_min = min(y_min, float(lo.min()))
            y_max = max(y_max, float(hi.max()))

        ax.plot(steps, accs,
                marker="o", markersize=4, linewidth=1.8,
                linestyle=LINESTYLES[strat],
                color=COLORS[strat],
                label=f"{strat} (AUC={auc:.2f})")
        y_min = min(y_min, float(accs.min()))
        y_max = max(y_max, float(accs.max()))
        found = True

    if not found:
        return False

    # Tight y-axis with a little headroom; round to nice ticks
    pad = max(0.05, (y_max - y_min) * 0.15)
    ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))

    ax.set_xlabel("Update step (# distractors injected)")
    ax.set_ylabel("Exact-hit accuracy")
    ax.set_title(f"{dataset.upper()} (N=100, nuc={nuc})")
    ax.set_xticks(range(0, nuc + 1, max(1, nuc // 10)))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.92, fontsize=9)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--figures_dir", default="figures")
    p.add_argument("--nuc", type=int, default=20)
    p.add_argument("--no_ci", action="store_true")
    p.add_argument("--bootstrap_iters", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Per-dataset figures
    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ok = plot_dataset(ax, ds, args.results_dir, args.nuc,
                          draw_ci=not args.no_ci,
                          n_iters=args.bootstrap_iters, rng=rng)
        out = os.path.join(args.figures_dir, f"retention_{ds}.png")
        if ok:
            plt.tight_layout()
            plt.savefig(out, dpi=160)
            print(f"  -> {out}")
        plt.close(fig)

    # Combined side-by-side panel
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=False)
    any_drawn = False
    for ax, ds in zip(axes, DATASETS):
        ok = plot_dataset(ax, ds, args.results_dir, args.nuc,
                          draw_ci=not args.no_ci,
                          n_iters=args.bootstrap_iters, rng=rng)
        any_drawn = any_drawn or ok
    if any_drawn:
        fig.suptitle("Knowledge retention across update steps (95% bootstrap CI shaded)",
                     fontsize=12, y=1.02)
        plt.tight_layout()
        out = os.path.join(args.figures_dir, "retention_combined.png")
        plt.savefig(out, dpi=160, bbox_inches="tight")
        print(f"  -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
