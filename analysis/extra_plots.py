#!/usr/bin/env python3
"""
Five extra paper-quality plots, all from existing JSONs/CSVs on disk:

  P1. Decay-curve overlay: shared-drop vs per-layer per (dataset, strategy)
  P2. Strategy-agreement heatmap: pairwise Jaccard at step-N, one panel per dataset
  P3. AUC bar chart with 95% bootstrap CIs (shared vs per-layer, side by side)
  P4. Robust / forgot / recovered stacked bar per (dataset, strategy)
  P5. Layer-Jaccard horizontal bar chart

Reads:
  - results/{squad,nq}_{strategy}_nuc20.json           (shared-drop baseline)
  - results/perlayer/{squad,nq}_{strategy}_nuc20_perlayer.json
  - results/jaccard_summary.csv

Writes:
  - figures/p1_decay_overlay.png
  - figures/p2_strategy_agreement.png
  - figures/p3_auc_ci_bars.png
  - figures/p4_robust_forgot_recovered.png
  - figures/p5_layer_jaccard.png
"""

import csv
import glob
import json
import os
import re
import string

import matplotlib.pyplot as plt
import numpy as np

STRATEGIES = ["random", "attention", "age", "surprise"]
DATASETS = ["squad", "nq"]
COLORS = {"random": "#1f77b4", "attention": "#ff7f0e",
          "age": "#2ca02c", "surprise": "#d62728"}


def normalize(s):
    s = s.replace("</s>", "").replace("<|end_of_text|>", "")
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return re.sub(r"\s+", " ", s).strip()


def hit(p, t):
    return normalize(t) in normalize(p)


def load(path):
    with open(path) as f:
        return json.load(f)


def hit_matrix(d):
    pe = d["per_example"]
    N = len(pe)
    nuc = d["config"]["nuc"]
    M = np.zeros((N, nuc + 1), dtype=bool)
    for i, ex in enumerate(pe):
        for s in range(nuc + 1):
            M[i, s] = hit(ex["step_preds"][f"step_{s}"], ex["target"])
    return M


def bootstrap_auc_ci(M, n_iters=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    N = M.shape[0]
    aucs = np.empty(n_iters)
    trapz = getattr(np, "trapezoid", np.trapz)
    for i in range(n_iters):
        idx = rng.integers(0, N, size=N)
        aucs[i] = trapz(M[idx].mean(0))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def load_all():
    """Returns dict[mode][dataset][strategy] = json_dict (with _M populated)."""
    out = {"shared": {ds: {} for ds in DATASETS},
           "perlayer": {ds: {} for ds in DATASETS}}
    for ds in DATASETS:
        for st in STRATEGIES:
            # shared
            p = f"results/{ds}_{st}_nuc20.json"
            if os.path.exists(p):
                d = load(p)
                d["_M"] = hit_matrix(d)
                out["shared"][ds][st] = d
            # per-layer
            p = f"results/perlayer/{ds}_{st}_nuc20_perlayer.json"
            if os.path.exists(p):
                d = load(p)
                d["_M"] = hit_matrix(d)
                out["perlayer"][ds][st] = d
    return out


# ---------------------------------------------------------------------------
# P1 — decay curve overlay: shared vs per-layer
# ---------------------------------------------------------------------------
def plot_p1(D, out):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey="row")
    for di, ds in enumerate(DATASETS):
        for si, st in enumerate(STRATEGIES):
            ax = axes[di, si]
            if st in D["shared"][ds]:
                accs = D["shared"][ds][st]["_M"].mean(0)
                ax.plot(accs, "--o", color=COLORS[st], alpha=0.5,
                        label="shared", markersize=4)
            if st in D["perlayer"][ds]:
                accs = D["perlayer"][ds][st]["_M"].mean(0)
                ax.plot(accs, "-o", color=COLORS[st],
                        label="per-layer", markersize=4)
            ax.set_title(f"{ds} — {st}")
            ax.set_xlabel("# distractors")
            if si == 0:
                ax.set_ylabel("step accuracy")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_ylim(bottom=0)
    fig.suptitle("Decay curves: shared-drop (dashed) vs per-layer (solid)", y=1.00)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# P2 — strategy-agreement heatmap (Jaccard at step-N)
# ---------------------------------------------------------------------------
def plot_p2(D, out):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4.5))
    for ax, ds in zip(axes, DATASETS):
        K = len(STRATEGIES)
        J = np.zeros((K, K))
        for i, s1 in enumerate(STRATEGIES):
            for j, s2 in enumerate(STRATEGIES):
                h1 = D["perlayer"][ds][s1]["_M"][:, -1]
                h2 = D["perlayer"][ds][s2]["_M"][:, -1]
                inter = (h1 & h2).sum()
                union = (h1 | h2).sum()
                J[i, j] = (inter / union) if union else 0
        im = ax.imshow(J, vmin=0, vmax=1, cmap="viridis")
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{J[i,j]:.2f}", ha="center", va="center",
                        color="w" if J[i, j] < 0.5 else "k", fontsize=10)
        ax.set_xticks(range(K)); ax.set_xticklabels(STRATEGIES, rotation=30)
        ax.set_yticks(range(K)); ax.set_yticklabels(STRATEGIES)
        ax.set_title(f"{ds} — Jaccard at step 20 (per-layer)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Which examples do strategies agree on retaining?", y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# P3 — AUC bar chart with 95% bootstrap CIs, shared vs per-layer
# ---------------------------------------------------------------------------
def plot_p3(D, out, n_iters=2000):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 5))
    rng = np.random.default_rng(0)
    width = 0.36
    x = np.arange(len(STRATEGIES))
    trapz = getattr(np, "trapezoid", np.trapz)
    for ax, ds in zip(axes, DATASETS):
        shared_aucs, shared_lo, shared_hi = [], [], []
        per_aucs,    per_lo,    per_hi    = [], [], []
        for st in STRATEGIES:
            for tag, mode in [("shared", shared_aucs), ("perlayer", per_aucs)]:
                if st in D[tag][ds]:
                    M = D[tag][ds][st]["_M"]
                    mode.append(float(trapz(M.mean(0))))
                else:
                    mode.append(float("nan"))
            for tag, lo, hi in [("shared", shared_lo, shared_hi),
                                ("perlayer", per_lo, per_hi)]:
                if st in D[tag][ds]:
                    a, b = bootstrap_auc_ci(D[tag][ds][st]["_M"], n_iters, rng)
                    lo.append(a); hi.append(b)
                else:
                    lo.append(float("nan")); hi.append(float("nan"))
        shared_err = [np.array(shared_aucs) - np.array(shared_lo),
                      np.array(shared_hi) - np.array(shared_aucs)]
        per_err = [np.array(per_aucs) - np.array(per_lo),
                   np.array(per_hi) - np.array(per_aucs)]
        ax.bar(x - width/2, shared_aucs, width, yerr=shared_err,
               label="shared", color="#aaaaaa", capsize=4)
        ax.bar(x + width/2, per_aucs, width, yerr=per_err,
               label="per-layer", color="#2ca02c", capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(STRATEGIES)
        ax.set_title(f"{ds} — AUC (with 95% bootstrap CI)")
        ax.set_ylabel("AUC")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("AUC with 95% CIs — shared-drop vs per-layer mode", y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# P4 — robust / forgot / recovered stacked bars (per-layer only)
# ---------------------------------------------------------------------------
def plot_p4(D, out):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 4.5))
    width = 0.6
    for ax, ds in zip(axes, DATASETS):
        labels, robust, forgot, recovered, lost = [], [], [], [], []
        for st in STRATEGIES:
            M = D["perlayer"][ds][st]["_M"]
            start = M[:, 0]; end = M[:, -1]
            labels.append(st)
            robust.append(int((start & end).sum()))
            forgot.append(int((start & ~end).sum()))
            recovered.append(int((~start & end).sum()))
            lost.append(int((~start & ~end).sum()))
        x = np.arange(len(labels))
        ax.bar(x, robust, width, label="robust (right @0 & @20)", color="#2ca02c")
        ax.bar(x, recovered, width, bottom=np.array(robust),
               label="recovered (wrong @0, right @20)", color="#9edcc8")
        ax.bar(x, forgot, width,
               bottom=np.array(robust) + np.array(recovered),
               label="forgot (right @0, wrong @20)", color="#ff7f0e")
        ax.bar(x, lost, width,
               bottom=np.array(robust) + np.array(recovered) + np.array(forgot),
               label="never (wrong @0 & @20)", color="#cccccc")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(f"{ds} — example breakdown (N=100)")
        ax.set_ylabel("# examples")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Where each strategy's accuracy comes from", y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# P5 — Layer-Jaccard horizontal bar chart
# ---------------------------------------------------------------------------
def plot_p5(out, csv_path="results/jaccard_summary.csv"):
    rows = []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    # parse "dataset_strategy_nuc20" into (dataset, strategy)
    parsed = []
    for r in rows:
        m = re.match(r"(squad|nq)_(\w+?)_nuc\d+", r["name"])
        if m:
            parsed.append({"dataset": m.group(1), "strategy": m.group(2),
                           "j": float(r["layer_jaccard"])})
    fig, ax = plt.subplots(figsize=(8, 4))
    parsed.sort(key=lambda r: (r["strategy"], r["dataset"]))
    y = np.arange(len(parsed))
    for i, r in enumerate(parsed):
        ax.barh(i, r["j"], color=COLORS[r["strategy"]],
                alpha=1.0 if r["dataset"] == "squad" else 0.55)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['strategy']} ({r['dataset']})" for r in parsed])
    ax.set_xlabel("Layer-Jaccard (mean across all steps)")
    ax.set_xlim(0, 1)
    ax.axvline(1.0, color="k", linestyle=":", alpha=0.5)
    ax.set_title("Layer-Jaccard — do per-layer drops agree across layers?")
    for i, r in enumerate(parsed):
        ax.text(r["j"] + 0.01, i, f"{r['j']:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  wrote {out}")


def main():
    os.makedirs("figures", exist_ok=True)
    print("Loading JSONs...")
    D = load_all()
    print("Generating plots...")
    plot_p1(D, "figures/p1_decay_overlay.png")
    plot_p2(D, "figures/p2_strategy_agreement.png")
    plot_p3(D, "figures/p3_auc_ci_bars.png")
    plot_p4(D, "figures/p4_robust_forgot_recovered.png")
    plot_p5("figures/p5_layer_jaccard.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
