# ExtendingMemoryLLM — Codebase Review & TA-Feedback Notes

**Review date:** 2026-05-04.
**Scope:** end-to-end read of model code, eval driver, datasets, analysis scripts, results, and progress docs. No code changes — review only, plus a written plan to address TA's feedback.

---

## 1. What the project is (one paragraph)

Columbia COMS 6998 final project. Fork of MemoryLLM (ICML 2024), which augments Llama-3-8B with a 1.67 B-parameter memory pool (`L=32` layers × `num_blocks=50` × `num_tokens=256` × `d=4096`). The original paper drops memory tokens **at random** when new context arrives. We replace that policy with four importance-aware strategies and measure knowledge retention as a function of distractor injections (the AUC of accuracy vs. number-of-unrelated-contexts curve, "NUC"). Datasets: SQuAD v2 and NaturalQuestions. Headline so far: `age` is the only strategy with positive ΔAUC on both datasets, but no comparison clears Bonferroni at N=100. Random is a surprisingly strong baseline — TA's hypothesis (per-layer independence) is supported by the code.

---

## 2. Repository layout — what each file does

### Model code
| File | Role |
|------|------|
| `modeling_memoryllm.py` (93 KB) | Original MemoryLLM (Llama-3-8B + memory pool). Patched to surface peft load errors loudly (commit `898b6b7`). Hosts `update_memory_with_delta_memory` with branches on `self.drop_memory_per_layer` (lines 1651, 1673). |
| `modeling_memoryllm_strategies.py` (20 KB) | **Our contribution.** Subclass `MemoryLLMWithStrategies(MemoryLLM)`. Adds `set_drop_strategy()`, overrides `drop_memory()` and `update_memory_with_delta_memory()`, tracks `_attention_ema`, `_token_ages`, `_fisher_scores`. Strategies: `random`, `attention`, `age`, `surprise`, `fisher` (last is unused — see §5). Also: `enable_drop_logging()`, `mark_new_example()`, `get_drop_log()` for the Layer-Jaccard analysis. |
| `modeling_mplus.py` | M+ variant from upstream — not used in our experiments. |
| `configuration_memoryllm.py` | HF config class. Default `drop_memory_per_layer=False` (line 144). |

### Datasets
| File | Role |
|------|------|
| `dataset/squad.py` | SQuAD v2 loader. Patched: `AutoTokenizer`; dev-fallback for distractor contexts when `train-v2.0.json` is absent. |
| `dataset/nq.py` | NQ loader. Patched: `AutoTokenizer`; returns raw context/question (no `"Context: "` wrapper); dev-fallback for distractors. |
| `data/squad/dev-v2.0.json` (4.2 MB) | 5928 answerable QA pairs. |
| `data/nq/v1.0-simplified_nq-dev-all.jsonl` (6.4 GB) | 7830 examples. |
| `data/{squad,nq}/indices_{squad_3,nq_4}.npy` | Eval-pair indices from `YuWangX/KnowledgeRetention`. |

### Eval / sanity
| File | Role |
|------|------|
| `run_eval.py` | Main eval driver. Flags: `--strategy`, `--dataset`, `--nuc`, `--num_samples`, `--resume`, `--seed`, `--drop_per_layer`, `--log_dropped`. Filename stem encodes `{dataset}_{strategy}_nuc{N}[_perlayer][_seed{S}]`. |
| `run_sanity.py` | E0 — compares `normal` vs `zeroed` vs `scrambled` memory. Catches silent load bugs. |
| `metrics.py` | Helpers for accuracy / AUC. |
| `test_qa_memory.py`, `test_strategies.py`, `test_training_colab.py` | Smoke tests. |

### Analysis
| File | Role |
|------|------|
| `analysis/plot_retention.py` | Retention curves with bootstrap CI bands. Produced `figures/retention_{squad,nq,combined}.png`. |
| `analysis/auc_table.py` | Builds `results/auc_summary.csv`. |
| `analysis/significance.py` | Bootstrap CIs + paired permutation tests vs `random`, Bonferroni. |
| `analysis/rescore.py` | Recompute accuracy with normalized matching (no GPU). |
| `analysis/dropped_indices.py` | Layer-Jaccard (within-strategy cross-layer overlap) + cross-strategy Jaccard. Reads `*_dropped.json`. |

### Outputs
| File | Notes |
|------|-------|
| `results/{squad,nq}_{strategy}_nuc20.json` × 8 | Accuracy curves + per-example dumps. **All in shared-drop mode** (`drop_memory_per_layer=False`). |
| `results/auc_summary.csv` | AUC per (dataset, strategy). |
| `results/significance.csv` | Paired permutation tests. |
| `figures/retention_{squad,nq,combined}.png` | Retention curves. |

### Docs
| File | Notes |
|------|-------|
| `PROJECT_STATUS.md` | Snapshot dated 2026-04-28. AUC table, bug log, what's left. |
| `PROGRESS_LOG.md` | Chronological debugging log. Includes the four-bug debug session (peft, memory-reset, NQ wrapping, tokenizer). |
| `EXPERIMENTS.md` | Run instructions / flag reference. |
| `presentations/*.md` | Speaker notes, walkthrough, Q&A prep. |

---

## 3. End-to-end flow (start → finish)

1. **Setup.** `scripts/setup.sh` pins versions (`transformers==4.48.2`, `peft==0.10.0`, `accelerate==1.2.0`). These pins are load-bearing — peft 0.19 silently drops 384 LoRA decoder adapters and tanks accuracy to ~0%. Don't unpin without re-running sanity.

2. **Data download.** SQuAD v2 dev + `indices_squad_3.npy`; NQ dev + `indices_nq_4.npy`. See `data/README.md`. NQ is 6.4 GB — allow time on Colab.

3. **Sanity (`run_sanity.py`).** Loads the model, compares accuracy under three conditions: `normal` (pretrained checkpoint memory), `zeroed`, `scrambled`. PASS condition: `normal − zeroed > 0.10`. Latest run: `0.667 − 0.033 = +0.633` ✓. Always run before a long eval.

4. **Eval driver (`run_eval.py`).** For each strategy:
   - Load model once (`MemoryLLMWithStrategies.from_pretrained("YuWangX/memoryllm-8b")`).
   - For each example: snapshot pretrained memory, inject the target context, then `nuc=20` distractors one at a time, generating an answer at each step.
   - **Critical fix:** `model.memory.data.copy_(checkpoint_memory)` (line 177). Earlier code reset to zeros, which wiped 49/50 of the trained pool because `inject_memory` only rewrites one block of 256 tokens.
   - Score: normalized substring match (`_normalize` strips `</s>`/`<|end_of_text|>`, lowercases, drops punct) → boolean per example/step → mean over examples per step → `np.trapezoid` → AUC.
   - With `--log_dropped`: writes a companion `*_dropped.json` containing per-(example, step, layer) dropped indices.

5. **Strategies (`modeling_memoryllm_strategies.py:_compute_importance`).**
   - **random** — `torch.rand(N)` per call. Drawn fresh per layer when `drop_memory_per_layer=True`.
   - **attention** — `_attention_ema[layer_idx]` accumulated via `update_attention_scores` after each injection (alpha=0.9). New tokens initialised to mean. Drops lowest EMA.
   - **age** — `_token_ages[layer_idx]`, incremented per drop step. Tokens with age ≤ `protection_window` (=`num_tokens`=256) get `+inf` importance (protected); else `1/(age+1)`. Plus `1e-3` tie-break noise (was `1e-8`; raised in `f186d17` to give per-layer diversity within an age bucket).
   - **surprise** — drops tokens with **highest** cosine similarity to incoming `delta_memory[layer_idx]` (i.e. importance = `1 − max_sim` → lowest importance = most redundant). Wording note: paper says "orthogonal" but code drops "most similar" → fix wording (already flagged).
   - **fisher** — `_fisher_scores` computed by KL-divergence of output distribution after masking each block (`update_fisher_scores`). Implemented but **not used** — too expensive, and TA agreed to skip.

6. **Drop dispatch (`update_memory_with_delta_memory`).**
   - `drop_memory_per_layer=False` (default, what every saved result uses): one call to `drop_memory` with `layer_idx=0`, same `remaining_indices` applied to all 32 layers (lines 268–286). Metadata for all layers updated with the *shared* indices.
   - `drop_memory_per_layer=True`: one `drop_memory` call per layer with its own `layer_idx` (lines 251–266). Each layer gets its own indices.

7. **Analysis.** `auc_table.py` builds the summary CSV; `significance.py` does paired permutation + Bonferroni; `plot_retention.py` makes the figures; `rescore.py` recomputes accuracy from existing per_example dumps with the normalised matcher; `dropped_indices.py` computes Layer-Jaccard from `*_dropped.json`.

---

## 4. Headline results (what's currently reported)

From `results/auc_summary.csv` (`nuc=20`, `N=100`, shared-drop mode):

| Dataset | random | attention | age | surprise |
|--------|--------|-----------|-----|----------|
| SQuAD  | 8.005  | 7.675 (−0.33) | **8.465 (+0.46)** | 7.745 (−0.26) |
| NQ     | 1.55   | 1.78 (+0.23)  | 1.875 (+0.33)     | **1.915 (+0.37)** |

`significance.csv`: nothing clears Bonferroni at p<0.05. All `ns`. Three findings stand:

1. `age` is the only strategy with positive ΔAUC on both datasets.
2. Random is surprisingly strong — beats `attention` and `surprise` on SQuAD.
3. Strategy ranking is dataset-dependent (NQ favours `surprise`, SQuAD favours `age`).

---

## 5. Inconsistencies / things to flag

These are the items I'd raise in a code review.

### 5.1 All saved results use `drop_memory_per_layer=False` — and that's a train/test mismatch
The 8 result JSONs were produced before per-layer mode was wired into `run_eval.py`. Every reported AUC is **shared-drop** mode (`False`), including our four importance-aware strategies.

**Critical correction on the upstream picture.** The Python class default in `configuration_memoryllm.py:144` is `False`, but every shipped *training* config in upstream overrides it to `True`:
- `official-memoryllm/train/MemoryLLM/configs/llama/llama_30x256.yaml:25` → `drop_memory_per_layer: true`
- `official-memoryllm/train/MemoryLLM/configs/openllama/openllama_4x256.yaml:25` → `drop_memory_per_layer: true`
- `official-memoryllm/train/MemoryLLM/configs/openllama/openllama_debug.yaml:25` → `drop_memory_per_layer: true`

So the `YuWangX/memoryllm-8b` checkpoint we load from HuggingFace was **trained with per-layer-independent dropping**. By running inference at `False`, we have a **train/test distribution mismatch**: the model learned to operate when each layer drops independently, and at eval we're forcing every layer to drop the same positions in lockstep. This is a confounder for *all* current results, not just for the TA's question.

What this means concretely:
- Every strategy (`random` included) needs to be re-run with `--drop_per_layer` to match the checkpoint's training condition.
- The existing 8 shared-drop JSONs are still useful — they document the train/test-mismatch behaviour (which itself is interesting evidence for TA Q1: "why does shared random work well even though the model wasn't trained for it?"). Keep them; don't delete.
- The age noise fix (`1e-8 → 1e-3`) was put in to recover per-layer diversity within an age bucket. In shared mode it changes nothing (one noise vector, used once). It only matters when `drop_memory_per_layer=True` — i.e. it was always meant for the per-layer runs we hadn't done yet.

### 5.2 `_compute_importance` ignores the model's device
`torch.rand(N)` and the age tie-break noise are CPU tensors. `surprise` does `importance.cpu()` at the end; the others return CPU tensors implicitly. Fine for the current path (`drop_memory` runs on CPU-ish indices), but if `drop_memory_per_layer=True` is ever vectorised on-GPU it'll need `device=current_memory.device`.

### 5.3 Age strategy: per-token Python loop
`for i in range(N): ...` over N=12,800 in `_compute_importance` for `age` is slow in inner-loop. Not buggy. Visible in elapsed time: SQuAD age took 7582 s vs 2523 s for random (3× slower). Trivially vectorisable with `np.where(ages <= window, np.inf, 1.0/(ages+1))`.

### 5.4 `auc` units (paper / slide hygiene)
`np.trapezoid(accs)` over 21 points with `dx=1` gives AUC ∈ [0, 20]. CSVs and figures keep this scale. Should normalise to `/ 20` for the paper or it'll confuse readers expecting [0, 1].

### 5.5 N=100 underpowers the study
Bootstrap CIs on AUC are ±0.6–1.3. ΔAUC of 0.4 cannot survive a 3-test Bonferroni at this N. Either (a) increase to N≥200/300 (~2–3× GPU time), (b) drop Bonferroni and report uncorrected, or (c) explicitly frame as descriptive, not confirmatory.

### 5.6 `inject_memory` only refreshes attention EMA when `output_attentions=True`
`need_attention = (self._drop_strategy == "attention" and self.initialized)`. So attention EMA is updated only when the active strategy is `attention`. Fine — but means swapping strategies on a loaded model resets nothing; if you change strategies mid-run without `reset_metadata()`, stale state can leak. Eval driver loads fresh per run, so this is theoretical right now.

### 5.7 `dataset/nq.py` distractor pool depends on N
The fallback uses non-eval entries from the loaded dev range. For `num=100` there are ~159 non-eval entries — enough for 20 distractors. If `num_samples` grows past ~150 this will silently break.

### 5.8 Surprise strategy wording
`surprise` drops tokens **most similar** to incoming context (highest cosine), not "orthogonal" tokens (cosine ≈ 0) as the methods text currently says. Already flagged in `PROGRESS_LOG.md`; fix in writeup.

---

## 6. Addressing the TA's feedback

Three asks from the TA, plus the Tushar-list of three.

### 6.1 TA Q1 — "Why does non-independent dropping work well for the random baseline?"

**Translation:** even when *every* layer gets the same dropped indices (shared mode), random is competitive. Why?

**Working hypothesis (write into report §Discussion):**
- Memory tokens within one block (`num_tokens=256`) are highly redundant — they all encode pieces of the same injected context. Losing any one is recoverable from the rest in the same layer.
- The 32 layers themselves are partially redundant for any given fact: a fact tends to be re-encoded across multiple layers via residual streams. So even shared-index dropping doesn't kill a fact unless it happens to land in *every* layer's active representation of it — which is unlikely to coincide with the random index pick.
- Importance-aware strategies, by contrast, *concentrate* drops on tokens that score low on a single signal (e.g. low attention). If that signal correlates across layers (and it does for `age` and arguably for `attention`), they hit the same fact at every layer simultaneously. Worse than random.
- This predicts: **the gap between random and importance-aware should shrink — possibly invert — when we run independent dropping.** That's the test in §6.4.

**How to verify in code (no run needed yet):**
- Look at the per_example dump for `random` at step 20: count examples where the answer is preserved despite shared drop. If preservation is high it confirms intra-block redundancy.
- Run `analysis/dropped_indices.py` cross-strategy Jaccard between `age`/`attention`/`surprise` once we have per-layer logs — if Jaccard is high, importance strategies cluster their drops on the same tokens (the synchronisation hypothesis).

### 6.2 TA Q2 — "Are your methods' drop decisions correlated across layers?" (Layer-Jaccard)

This is exactly what `analysis/dropped_indices.py` computes (Layer-Jaccard within strategy, plus cross-strategy Jaccard). To get usable inputs:

1. Run all 4 strategies × 2 datasets in **per-layer mode with logging**:
   ```bash
   python run_eval.py --strategy all --dataset squad --nuc 20 --num_samples 100 \
                      --drop_per_layer --log_dropped --output_dir results/perlayer/
   python run_eval.py --strategy all --dataset nq --nuc 20 --num_samples 100 \
                      --drop_per_layer --log_dropped --output_dir results/perlayer/
   ```
2. `python analysis/dropped_indices.py --out results/jaccard_summary.csv`.

**Predicted outcomes (write the prediction in the report so we can compare):**
- `random` Layer-Jaccard ≈ 1/num_blocks ≈ 0.02 (purely random independent draws of size N/50 from N).
- `attention` Layer-Jaccard moderate (per-layer attention differs but comes from the same forward pass — partially correlated).
- `surprise` Layer-Jaccard moderate-to-high (per-layer `delta_memory` differs but encodes the same content).
- `age` Layer-Jaccard close to 1 *unless* the `1e-3` noise actually breaks ties — measure it.

### 6.3 TA Q3 — "Are there dropping strategies best suited for particular kinds of tasks?"

We already see this descriptively: NQ favours `surprise`, SQuAD favours `age`. To make the claim usable rather than anecdotal:

- Bin SQuAD examples by **target answer position within context** (early/mid/late). If `age`'s win on SQuAD comes from late-position answers, that confirms recency bias of SQuAD.
- Bin NQ by **answer length** (single-token vs multi-token short answers). NQ targets are often longer/looser; surprise's "drop redundant tokens" might preserve diversity that helps multi-token extraction.
- Cross-tab the `auc_summary` against these bins. No new GPU runs needed — the per_example dumps already have enough data for this if we re-decode targets and contexts.

This is also how to engage Tushar item #2 (age vs recency bias).

### 6.4 Tushar #1 — Re-implement with `drop_memory_per_layer=True`

Status: **infrastructure exists (`--drop_per_layer` in `run_eval.py`, per-layer branch in `modeling_memoryllm_strategies.py:251–266`) but no result has been produced with it yet.** All 8 saved JSONs are shared-drop.

**Why to flip to `True` — two independent reasons:**
1. **Train/test consistency.** Upstream's training YAMLs (see §5.1) all set `drop_memory_per_layer: true`. The `YuWangX/memoryllm-8b` checkpoint was trained under per-layer independent dropping. Inferring at `False` is a distribution mismatch.
2. **TA Q2 only makes sense at `True`.** "Are your methods' drop decisions correlated across layers?" presupposes per-layer independence; in `False` mode Layer-Jaccard is trivially 1.0 by construction.

So the previous "random can stay at `False`" suggestion was wrong — we should flip *all four* strategies, including random, to match the training condition.

**Plan:**
- Run all four strategies (`random`, `attention`, `age`, `surprise`) × 2 datasets with `--drop_per_layer --log_dropped` into `results/perlayer/`. → 8 new JSONs + 8 `*_dropped.json` for Jaccard.
- Keep the existing 8 shared-drop JSONs in `results/` as-is. They become the "train/test-mismatch contrast" condition — useful evidence for TA Q1 (why does shared random work well even though the model wasn't trained for it?).
- Canonical headline numbers come from `results/perlayer/`. Shared-drop numbers move to an appendix as the contrast.

**How `--drop_per_layer` works internally** (so the report can describe it precisely):
- The flag flips one attribute, `model.drop_memory_per_layer`, read inside `update_memory_with_delta_memory`.
- `False`: one `drop_memory` call with `layer_idx=0` produces a single `remaining_indices` tensor; that tensor slices all 32 layers' memory at once, and metadata for all layers is updated with the same indices.
- `True`: a `for idx in range(L)` loop calls `drop_memory(..., layer_idx=idx)` 32 times, each call computing importance from that layer's own `_attention_ema[idx]` / `_token_ages[idx]` / `delta_memory[idx]` and producing its own `remaining_indices`.
- Per-strategy effect of `True`: `random` resamples `torch.rand(N)` per layer (genuinely independent); `attention` uses real per-layer attention so drops diverge; `surprise` uses the per-layer `delta_memory` slice so drops diverge; `age` would still drop the same tokens at every layer because ages are synchronised — the `1e-3` tie-break noise is what breaks ties differently per layer.
- Cost: 32× more `drop_memory` calls per step. Cheap for `random`/`attention`/`surprise`; expensive for `age` because of the per-token Python loop in `_compute_importance` (vectorise before scaling N).

**How `--drop_per_layer` works internally** (so the report can describe it precisely):
- The flag flips one attribute, `model.drop_memory_per_layer`, read inside `update_memory_with_delta_memory`.
- `False`: one `drop_memory` call with `layer_idx=0` produces a single `remaining_indices` tensor; that tensor slices all 32 layers' memory at once, and metadata for all layers is updated with the same indices.
- `True`: a `for idx in range(L)` loop calls `drop_memory(..., layer_idx=idx)` 32 times, each call computing importance from that layer's own `_attention_ema[idx]` / `_token_ages[idx]` / `delta_memory[idx]` and producing its own `remaining_indices`.
- Per-strategy effect of `True`: `random` resamples `torch.rand(N)` per layer (genuinely independent); `attention` uses real per-layer attention so drops diverge; `surprise` uses the per-layer `delta_memory` slice so drops diverge; `age` would still drop the same tokens at every layer because ages are synchronised — the `1e-3` tie-break noise is what breaks ties differently per layer.
- Cost: 32× more `drop_memory` calls per step. Cheap for `random`/`attention`/`surprise`; expensive for `age` because of the per-token Python loop in `_compute_importance` (vectorise before scaling N).

### 6.5 Tushar #2 — "Why is `age` performing better? Is the dataset biased toward recent tokens?"

Two diagnostics, neither needing a new GPU run:

- **Position-of-answer histogram.** For each SQuAD/NQ example, find where the gold answer's first token sits in the tokenised context (e.g. by char-offset → token-index). Plot accuracy at step 20 conditioned on that position. If `age`'s edge concentrates on late-position answers, that's recency bias.
- **Block-injection ordering.** `inject_memory` puts new tokens at the *end* of the pool. Under `age`'s protection window (last 256 tokens always protected), the most recently injected context block is always safe — so the `age` strategy is essentially "always keep the freshest block, drop older blocks first". For SQuAD where the *target context* was injected first and then 20 distractors follow, `age` will preferentially evict the target — yet `age` *wins*. That's interesting and contradicts the naive prediction. Hypothesis: at 20 distractors, the target context has aged past the protection window in every strategy, so the differentiator is *which old tokens are kept*. `age`'s "drop oldest" rule keeps the most recent distractor, which contains the question-relevant lexical priming for the next-token decoder. Worth verifying with a step-by-step accuracy decomposition.

Both diagnostics also handle TA Q3.

### 6.6 Tushar #3 — Run with > 100 samples to confirm trends

Cost estimate: SQuAD 100-sample run took 2,500–7,500 s/strategy on A100 (`age` is 3× slower because of the per-token Python loop). For N=300:
- Random/attention/surprise: ~2.1 hr/strategy.
- Age: ~6.3 hr/strategy.
- Total per dataset (4 strategies, per-layer mode): ~12–13 hr.

Two datasets ≈ 25 hr A100. Tight but feasible. Alternatives:
- Run only `age` and `random` at N=300 (the two that matter for the headline). 6 hr total per dataset.
- Or vectorise the age loop first (replace `for i in range(N)` with `np.where(ages <= window, np.inf, 1.0/(ages+1))`) — should drop age runtime to par.

### 6.7 Skip Fisher (confirmed)

`fisher` is implemented in `modeling_memoryllm_strategies.py:208–213` and `update_fisher_scores` exists, but it's never called from `run_eval.py` and there's no result file. TA agreed to skip — leave the code in place but don't run.

---

## 7. Concrete plan (step-by-step)

Updated with: (a) checkpoint was trained at `True`, so we have a train/test mismatch; (b) all four strategies including `random` need the per-layer run; (c) existing shared-drop JSONs are retained as contrast, not replaced.

| # | Step | Why | Outcome |
|---|------|-----|---------|
| 1 | Vectorise `age`'s per-token loop in `modeling_memoryllm_strategies.py:_compute_importance` (`np.where(ages <= window, np.inf, 1.0/(ages+1))` instead of `for i in range(N)`). | `age` is 3× slower than other strategies; per-layer mode runs `drop_memory` 32× per step, which would multiply that. | Brings age runtime to par with random/attention/surprise. ~5 min change. |
| 2 | Smoke test: `python run_eval.py --strategy random --dataset squad --nuc 3 --num_samples 5 --drop_per_layer --log_dropped --output_dir results/perlayer_smoke`. | Verifies the per-layer flag actually flips `model.drop_memory_per_layer`, drop logging writes a `*_dropped.json`, and nothing crashes. | ~2 min on A100. |
| 3 | Sanity: `python run_sanity.py --num_samples 30 --nuc 5`. | Catches peft / tokenizer / memory-load drift before paying for a 13-hr run. Must show `normal − zeroed > 0.10`. | ~15 min on A100. |
| 4 | **Per-layer matrix run.** All four strategies × both datasets, `N=100`, `nuc=20`, `--drop_per_layer --log_dropped --output_dir results/perlayer --resume`. | This is the canonical experiment — matches the checkpoint's training condition and answers TA Q2. | 8 new JSONs + 8 `*_dropped.json`. ~10–13 hr on A100 (less if step 1 is done). |
| 5 | `python analysis/auc_table.py --results_dir results/perlayer --out results/auc_perlayer.csv`. | Per-layer headline AUC table. | New canonical AUC numbers. |
| 6 | `python analysis/significance.py --results_dir results/perlayer --out results/significance_perlayer.csv --bootstrap_iters 5000 --perm_iters 10000`. | Per-layer significance with paper-quality iter counts. | Updated `significance_perlayer.csv`. |
| 7 | `python analysis/dropped_indices.py --files results/perlayer/*_dropped.json --out results/jaccard_summary.csv`. | Direct numeric answer to TA Q2 — Layer-Jaccard within each strategy + cross-strategy Jaccard. | `jaccard_summary.csv` with the prediction in §6.2 confirmed or refuted. |
| 8 | New ~40-LOC script: position-of-answer histogram over existing per_example dumps. Bin by where the gold answer sits in the source context, compute step-20 accuracy per bin. | Tests TA Q3 + Tushar #2 (recency bias). No GPU needed. | One figure + one CSV; finding goes into Discussion. |
| 9 | Update `analysis/plot_retention.py` to overlay per-layer vs shared on the same axes (one panel per dataset). | Visual answer to TA Q1 — shows whether random's lead over importance strategies survives the flip to per-layer mode. | Updated `figures/retention_{squad,nq}.png`. |
| 10 | Fix surprise wording ("orthogonal" → "most dissimilar" / "most redundant") in any draft text. | Reviewer flagged this previously; trivial. | Wording corrected. |
| 11 | Normalise AUC by `/20` in CSVs / figures for the paper. | AUC ∈ [0, 20] is confusing for readers expecting [0, 1]. | One-line change in `auc_table.py` + figure axes. |
| 12 | (Optional) N=300 confirmation for `age` + `random` only, per-layer. | If trends from step 5 are directional but not Bonferroni-significant, more N can push them over. | ~6 hr per dataset (after step 1). |
| 13 | Update `PROJECT_STATUS.md` with: train/test mismatch finding, per-layer headline numbers, Layer-Jaccard table, position-bias finding, 3-paragraph answers to TA Q1/Q2/Q3. | Final write-up. | Replaces §3 of `PROJECT_STATUS.md`. |

**What we are explicitly *not* doing:** Fisher (TA agreed to skip), hybrid strategies, full pool ablation matrix, retraining the model under our strategies.

**Critical-path subset if GPU budget is tight:** steps 1, 3, 4, 5, 6, 7, 13. That's the minimum set that answers all of TA Q1/Q2/Q3 and Tushar #1/#3.

---

## 8. Open questions to bring back to the TA / team

1. Is "per-layer independent" the *only* comparison we should report, or do we keep shared-mode in an appendix as a contrast?
2. For Layer-Jaccard, do we want the within-strategy number averaged over (example, step) — current implementation — or stratified by step? Stratified might show whether synchronisation worsens over time.
3. For task-suitability, which slicing variable does the TA care about most? My guess is answer-position (recency proxy) for SQuAD and answer-length for NQ; happy to use whatever they prefer.
4. Do we have GPU budget for N=300 across all 4 strategies × 2 datasets × per-layer (~25 hr A100)? If not, the N=300 confirmation is `age` + `random` only.

---

## 9. Bottom line

Code is in good shape. The four-bug fix session in late April closed all silent-failure paths. The single biggest gap is that **every saved result is shared-drop**, which directly under-tests the TA's per-layer-independence question and is inconsistent with the upstream MemoryLLM paper's random baseline. Running the matrix in `--drop_per_layer` mode is the highest-leverage next step — it answers TA Q1 and Q2, gives us the Layer-Jaccard numbers, and rebases the headline AUCs onto the canonical comparison. The per-token Python loop in `age` is the only real engineering wart and is a one-line vectorisation away.
