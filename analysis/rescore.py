#!/usr/bin/env python3
"""
Recompute accuracy_per_step and AUC from per_example dumps with relaxed matching.

The default exact_hit in run_eval.py is case-sensitive substring match — too strict.
Standard QA evals (SQuAD/NQ) lowercase and strip punctuation. This script rescores
existing JSONs in place (backs up original to *.json.strict).

Usage:
    python analysis/rescore.py                       # rescore all JSONs in results/
    python analysis/rescore.py --pattern 'nq_*'      # only NQ
    python analysis/rescore.py --dry_run             # show diff, don't overwrite
"""

import argparse
import glob
import json
import os
import re
import shutil
import string

import numpy as np


def normalize(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punct, collapse whitespace."""
    s = s.replace("</s>", "").replace("<|end_of_text|>", "")
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def hit(pred: str, target: str) -> bool:
    return normalize(target) in normalize(pred)


def rescore_one(path: str, dry_run: bool = False) -> dict:
    with open(path) as f:
        d = json.load(f)

    if "per_example" not in d:
        return {"path": path, "error": "no per_example dump"}

    nuc = d["config"]["nuc"]
    N = len(d["per_example"])
    M = np.zeros((N, nuc + 1), dtype=bool)
    for i, ex in enumerate(d["per_example"]):
        for s in range(nuc + 1):
            M[i, s] = hit(ex["step_preds"][f"step_{s}"], ex["target"])

    new_accs = M.mean(0).tolist()
    new_auc = float(np.trapezoid(new_accs))
    old_accs = d["accuracy_per_step"]
    old_auc = d["auc"]

    summary = {
        "path":     path,
        "old_step0": old_accs[0],
        "new_step0": new_accs[0],
        "old_auc":  old_auc,
        "new_auc":  new_auc,
        "delta_auc": new_auc - old_auc,
    }

    if dry_run:
        return summary

    # Back up original
    backup = path + ".strict"
    if not os.path.exists(backup):
        shutil.copy(path, backup)

    # Rewrite
    d["accuracy_per_step"] = new_accs
    d["auc"] = new_auc
    d["scoring"] = "lowercase+strip_punct (rescored from per_example)"
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--pattern", default="*_nuc*.json")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, args.pattern)))
    files = [f for f in files if not f.endswith(".strict")]

    if not files:
        print(f"No JSONs matching {args.pattern} in {args.results_dir}")
        return

    print(f"{'file':40s}  {'step0':>15s}  {'AUC':>15s}")
    print("-" * 75)
    for f in files:
        s = rescore_one(f, dry_run=args.dry_run)
        if "error" in s:
            print(f"  {os.path.basename(f):38s}  [skip] {s['error']}")
            continue
        name = os.path.basename(f)
        print(f"  {name:38s}  "
              f"{s['old_step0']:.3f}->{s['new_step0']:.3f}  "
              f"{s['old_auc']:.3f}->{s['new_auc']:.3f}  (Δ{s['delta_auc']:+.3f})")

    if args.dry_run:
        print("\n  (dry run — no files modified)")
    else:
        print(f"\n  Rescored {len(files)} JSONs. Originals backed up as *.json.strict")


if __name__ == "__main__":
    main()
