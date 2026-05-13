# Project status — 2026-05-12

## Latest update (2026-05-12 evening) — SQuAD per-layer matrix done

**Phase A2 complete.** All 4 SQuAD strategies ran in per-layer mode (matches the checkpoint's training condition). Total wall time **2.8 hr** on A100 — well under the 5–7 hr estimate thanks to the `age` vectorisation.

### Per-layer SQuAD AUCs (N=100, nuc=20, canonical numbers)

| Strategy | Per-layer AUC | Prior shared-drop AUC | Δ (perlayer − shared) | ΔAUC vs random (perlayer) |
|---|---:|---:|---:|---:|
| random    | **8.18** | 8.01 | +0.17 | — |
| attention | 7.63  | 7.68 | −0.05 | −0.55 |
| age       | **8.46** | 8.47 | −0.01 | +0.28 |
| surprise  | 8.015 | 7.75 | +0.27 | −0.165 |

**Per-layer ranking:** age > random > surprise > attention.

**Initial read (full analysis pending Phase B):** TA's per-layer-independence hypothesis appears supported. Random gains the most from per-layer mode (+0.17), surprise also gains substantially (+0.27), age is flat (predicted — ages sync across layers), attention barely moves. See `PROGRESS_LOG.md` "Session 2026-05-12 evening" for the detailed read.

**Operational note:** during the `attention` strategy, HF auto-falls-back from sdpa to eager attention (sdpa doesn't support `output_attentions=True` which the strategy needs for EMA updates). Benign, expected, no correctness impact.

**Still to do:** Phase A3 (NQ), then Phase B (analysis), then Phase C (writeup hygiene + this doc's headline numbers update).

---

## Earlier (2026-05-12 afternoon) — pipeline re-verified, cleared for per-layer canonical runs

Branch `ketaki` at commit `add22ca`. Three fixes shipped today (commits `68ce9ea`, `b31f6ee`, `add22ca` — full forensics in `PROGRESS_LOG.md` "Session 2026-05-12"):

- **Bug 6 (caught + fixed):** `attn_implementation` was never passed to `from_pretrained`, so HF defaulted to sdpa even when flash-attn was installed. Both drivers now auto-detect flash-attn and print which implementation is active.
- **Decision:** drop flash-attn from the Colab setup path. Current Colab (torch 2.10 + cu128 + py3.12) has no compatible prebuilt wheel; source builds are unreliable. sdpa fallback is ~1.3–1.5× slower at batch=1, fine.
- **Bug 7 (caught + fixed):** `np.trapezoid` (numpy 2.0) didn't exist on pinned `numpy==1.26.4`. Replaced with `getattr(np, "trapezoid", np.trapz)` across 5 files.

**Pipeline state verified on fresh Colab A100:**
- Pins load correctly: `torch 2.5.1+cu124 / transformers 4.48.2 / peft 0.10.0 / accelerate 1.2.0`.
- LoRA loads cleanly: `missing_keys: 0, unexpected_keys: 0` (peft pin still holding, no Bug 2 regression).
- Smoke test (`--drop_per_layer --log_dropped`) produces both result JSON + drop-log JSON.
- Sanity passed: `normal − zeroed = +0.567` (>> 0.10 threshold).

**Cleared for Phase A (per-layer matrix run on SQuAD + NQ).** See "What's next" at the bottom of this doc.

---

# Earlier snapshot — 2026-04-28

## What this project is
Columbia COMS 6998 final project. Fork of MemoryLLM (ICML 2024), which augments Llama-3-8B with a 1.67B-parameter memory pool (50 blocks × 256 tokens × 32 layers × 4096 dim). The paper drops memory tokens at random; we replace that with four importance-aware strategies and measure knowledge retention as a function of distractor injections.

## Strategies implemented (in `modeling_memoryllm_strategies.py`)

| Strategy  | Drops                       | Signal              |
|-----------|-----------------------------|---------------------|
| random    | uniform                     | — (baseline)        |
| attention | lowest EMA-attention        | relevance           |
| age       | oldest, recent-protected    | recency             |
| surprise  | most similar to incoming    | redundancy          |
| fisher    | not implemented (stretch)   | output sensitivity  |

## Bugs found and fixed this session (4 critical + 1 metric)

| # | Bug | Symptom | Fix | Commit |
|---|-----|---------|-----|--------|
| 1 | `LlamaTokenizer` against Llama-3 model | Wrong token IDs | → `AutoTokenizer` everywhere | `30f3961` |
| 2 | peft 0.19 silently drops 384 LoRA decoder adapters | ~0% accuracy across the board | Pin `transformers==4.48.2 peft==0.10.0` + loud error message in subclass | `898b6b7` |
| 3 | NQ wraps context with `"Context: "` and question with `"Questions: ... Answer:"` | Generation produces fragments of context, not answers | Return raw text like SQuAD | `9e12e55` |
| 4 | `run_eval.py` reset memory to **zeros** instead of pretrained checkpoint state | Wiped 49/50 of trained memory pool every example | Snapshot `checkpoint_memory` once, restore per-example | `ca75cac` |
| + | Strict substring match too harsh (case- and punctuation-sensitive) | Under-counted ~50% of NQ accuracy | SQuAD-style normalize: lowercase + strip punct | latest |

The peft + memory-reset bugs together explain why prior eval runs produced AUCs of 0.045 / 0.030 / 0.030 / 0 for NQ. Both were silent failures producing plausible-looking but meaningless numbers. Both were caught by `run_sanity.py` (issue #31, **BLOCKING**), which compares `normal` (checkpoint memory) vs `zeroed` vs `scrambled` memory.

## Sanity check (issue #31) — PASSED ✓

Run on SQuAD, N=30, nuc=5:
- 0 missing keys, 0 unexpected keys, 0 mismatched
- `initialized: 1`; `memory.std=0.336` (real values)
- step-0 accuracy: **normal 0.667, zeroed 0.033, scrambled 0.633**
- gap_normal_minus_zeroed = +0.633 (>>0.10 threshold) → memory contributes substantially
- Side note: scrambled ≈ normal at step 0 (0.63 vs 0.67) — token *ordering* matters less than the *presence* of trained weights and the LoRA decoder. Worth a sentence in the paper.

## Main results (rescored, paper-quality bootstrap iters)

### SQuAD (N=100, nuc=20)

| Strategy   | AUC      | step-0   | step-20  | ΔAUC vs random | p_bonf |
|------------|----------|----------|----------|----------------|--------|
| random     | 8.01     | 0.55     | 0.34     | —              | —      |
| **age**    | **8.47** | 0.49     | **0.42** | **+0.46**      | 0.57   |
| surprise   | 7.75     | 0.42     | 0.35     | −0.26          | 1.00   |
| attention  | 7.68     | 0.53     | 0.36     | −0.33          | 1.00   |

### NQ (N=100, nuc=20)

| Strategy    | AUC      | step-0   | step-20  | ΔAUC vs random | p_bonf |
|-------------|----------|----------|----------|----------------|--------|
| random      | 1.55     | 0.14     | 0.06     | —              | —      |
| surprise    | **1.92** | 0.12     | 0.07     | **+0.37**      | 0.35   |
| age         | 1.88     | 0.10     | 0.07     | +0.33          | 0.24   |
| attention   | 1.78     | 0.20     | 0.06     | +0.23          | 0.33   |

## Three honest findings

1. **`age` is the most consistent strategy** — only one with ΔAUC > 0 on both datasets. On SQuAD, has a distinctively *flat* curve (step-20 ≈ step-0) where the others decay sharply.
2. **Random is a surprisingly strong baseline** — beats `attention` and `surprise` on SQuAD. Confirms the reviewer's structural-advantage hypothesis (random fully decorrelates drops across layers; importance-aware strategies often synchronize).
3. **Strategy ranking is dataset-dependent.** NQ: surprise > age > attention > random. SQuAD: age > random > surprise ≈ attention.

## One important caveat

**No comparison reaches Bonferroni-corrected significance** (all p_bonf > 0.05). With N=100 and bootstrap CIs ±0.6–1.3, the study is underpowered to detect ΔAUC ≈ 0.4. Trends are consistent in direction but not confirmed.

## Code artifacts produced

| File | Purpose |
|------|---------|
| `run_eval.py` | Eval driver (4 strategies × 2 datasets) — patched to use `AutoTokenizer`, normalized scoring, checkpoint memory reset |
| `run_sanity.py` | E0 memory-presence check (issue #31) |
| `dataset/squad.py`, `dataset/nq.py` | Patched: tokenizer, format, dev-fallback for missing train file |
| `modeling_memoryllm.py` | Patched: surface peft load errors loudly |
| `modeling_memoryllm_strategies.py` | The four strategies — unchanged this session |
| `analysis/plot_retention.py` | Retention curves with bootstrap CI bands, side-by-side panel |
| `analysis/auc_table.py` | `results/auc_summary.csv` |
| `analysis/significance.py` | Bootstrap CIs + paired permutation tests, Bonferroni |
| `analysis/rescore.py` | Recompute accuracy with normalized matching from existing per_example dumps |

## Figures + data files

- `figures/retention_squad.png`, `figures/retention_nq.png`, `figures/retention_combined.png`
- `results/auc_summary.csv` (8 rows)
- `results/significance.csv` (6 comparison rows + 2 baseline rows)
- `results/{nq,squad}_{strategy}_nuc20.json` × 8 (rescored; originals saved as `.json.strict`)
- `results/sanity_check.json` (E0 verdict)

## GitHub issues status

| Status | Issues |
|--------|--------|
| ✅ Closed (this session) | #4, #5, #21, #22, #24, #31 |
| ✅ Closed (data complete) | #8, #9, #10, #11, #12 |
| ✅ Closed (analysis done) | #13, #14, #25 |
| ⏳ Open — writing | #16, #17, #18, #26, #27, #28, #29, #30 |
| ⏳ Open — stretch | #15, #19, #20, #23, #32, #33, #34 |
| ⏳ Open — admin | #1, #2, #3, #6, #7 |

## What's left to do

### Writing (the critical path)

1. **#26 Intro + Related Work** — start now, doesn't need numbers
2. **#27 Methods** — formalize 4 strategies, fix "orthogonal → most dissimilar" wording, add 3B vs 8B paragraph
3. **#28 Experiments + Results** — embed combined retention plot, AUC table, significance table; write the three findings
4. **#29 Discussion** — three points: (a) why age wins, (b) why random is strong, (c) limitations
5. **#30 Abstract + final pass** — last
6. **#16 / #17 / #18 Slides** — lift from report

### Optional stretch (with a couple days of buffer)

- **Dropped-indices analysis (#15, #32)** — 2 hr, tests the per-layer-independence hypothesis directly. **Highest ROI extra.**
- **Memory-pool pressure ablation** (`num_blocks ∈ {4, 8, 16, 50}`, `age` vs `random`, SQuAD only) — 7 hr including 6 hr GPU. **Second-highest ROI.**
- **Age noise-scale ablation** (1e-3 vs 1e-6) — closes an open code-fix question, ~4 hr GPU.

### Skip

Fisher (#19, too risky), Hybrid (#34, too risky), full pool ablation matrix (too expensive), N=200 re-runs (marginal payoff).

---

## What's next (as of 2026-05-12)

Ordered checklist — full detail in `CODEBASE_REVIEW.md §7`.

### Phase A — GPU runs (~10–13 hr total)
- [x] **A1.** Backup `results/` to Drive (1 min). _Done 2026-05-12._
- [x] **A2.** SQuAD per-layer matrix (4 JSONs + 4 drop logs, 2.8 hr). _Done 2026-05-12 evening._
- [ ] **A3.** NQ per-layer matrix (same command, `--dataset nq`, ETA ~2.8–3 hr).
- [ ] **A4.** Sync `results/perlayer/` (8 JSONs + 8 drop logs) to Drive after A3.

### Phase B — Analysis (no GPU, ~30 min)
- [ ] **B1.** `analysis/auc_table.py --results_dir results/perlayer --out results/auc_perlayer.csv` — canonical AUC table.
- [ ] **B2.** `analysis/significance.py --results_dir results/perlayer --out results/significance_perlayer.csv --bootstrap_iters 5000 --perm_iters 10000`.
- [ ] **B3.** `analysis/dropped_indices.py --files "results/perlayer/*_dropped.json" --out results/jaccard_summary.csv` — direct answer to TA Q2 (Layer-Jaccard).
- [ ] **B4.** `analysis/plot_retention.py --results_dir results/perlayer --out figures/retention_perlayer` — regenerate curves.
- [ ] **B5.** Position-of-answer histogram (new ~40 LOC script; not yet written) — answers TA Q3 + Tushar #2.

### Phase C — Writeup hygiene
- [ ] **C1.** Fix "orthogonal" → "most dissimilar" in methods text.
- [ ] **C2.** Normalize AUC to [0, 1] in CSVs + figure axes.
- [ ] **C3.** Update this `PROJECT_STATUS.md` with per-layer headline numbers + train/test-mismatch finding + 3-paragraph TA Q1/Q2/Q3 answers.

### Phase D — Optional
- [ ] **D1.** N=300 confirmation for `age` + `random` per-layer only (~6 hr/dataset) if N=100 trends are directional but not Bonferroni-significant.

### Phase E — Report (parallelizable with A)
- [ ] **E1.** Intro + Related Work (#26).
- [ ] **E2.** Methods (#27) — includes C1 wording fix + 3B-vs-8B paragraph.
- [ ] **E3.** Experiments + Results (#28) — after Phase B.
- [ ] **E4.** Discussion (#29).
- [ ] **E5.** Abstract (#30).
- [ ] **E6.** Slides (#16/#17/#18).

**Critical path:** A1 → A2 → A3 → B1/B2/B3 → C3 → E2 → E3 → E4 → E5.
