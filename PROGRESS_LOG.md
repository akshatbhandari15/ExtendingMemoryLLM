# ExtendingMemoryLLM — Progress Log

Track problems encountered, approaches tried, and outcomes. Updated throughout.

---

## Issue #4: [INFRA] Download and verify SQuAD v2 eval data + index file — DONE ✓

### Attempt log (2026-04-21)

**Step 1: Create directory structure**
- Created `data/squad/`
- Problem: `.gitignore` had `./data` (ignores the whole directory), so `data/README.md` couldn't be committed
- Fix: Changed to `data/**` + `!data/README.md`

**Step 2: Download `dev-v2.0.json`**
- `curl -L "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json" -o data/squad/dev-v2.0.json`
- Result: 4.2MB ✓

**Step 3: Download `indices_squad_3.npy`**
- Problem: `hf_hub_download(..., filename='indices_squad_3.npy')` → 404
- Root cause: file lives at `squad/indices_squad_3.npy` inside the HF dataset, not at root
- Fix: listed files with `list_repo_files('YuWangX/KnowledgeRetention', repo_type='dataset')` to find path, then downloaded with `filename='squad/indices_squad_3.npy'`
- Also needed: `pip3 install huggingface-hub` (not installed by default)

**Step 4: Verify — PASSED**
- Shape: `(2250,)`, dtype `int64`, range `[0, 5927]`
- 5928 answerable QA pairs in dev set → max index valid ✓

**Committed:** `data/README.md` + updated `.gitignore` (commit `fddc3a5`)

---

## Issue #5: [INFRA] Download and verify NaturalQA eval data + index file — DONE ✓

### Attempt log (2026-04-21)

Both files are on HuggingFace `YuWangX/KnowledgeRetention` — same `hf_hub_download` approach as SQuAD, no need to go through Google's NLQ portal (which requires account signup).

- `indices_nq_4.npy` — downloaded from `nq/indices_nq_4.npy` on HF, 8.3KB
- `v1.0-simplified_nq-dev-all.jsonl` — downloaded from `nq/v1.0-simplified_nq-dev-all.jsonl` on HF, **6.4GB** (large — allow time on Colab)

**Verify — PASSED**
- Index shape: `(1044,)`, dtype `int64`, range `[2, 2562]`
- JSONL has 7830 lines → max index 2562 is valid ✓
- Spot-checked first 3 examples: `question_text` and `annotations.short_answers` keys present ✓

`data/README.md` updated with NQ download commands and verification snippet.

---

## Issue #12: [EVAL] NaturalQA retention curves — all 4 strategies

Depends on #5 (done ✓). Run 100 NQ examples × 20 distractor steps for each of: random, attention, age, surprise. Save to `results/nq_*.json`.

### Bug fix (2026-04-27)

`dataset/nq.py` assumed a train JSONL file existed for distractor contexts — we only have the dev file. Fixed: save all loaded long answers before index re-selection, then use non-eval dev entries as distractor contexts when train file is absent. For `num=100` there are ~159 non-eval entries in the loaded range — plenty for 20 distractor contexts.

### To run (on Colab A100)

```bash
python run_eval.py --strategy all --dataset nq --nuc 20 --num_samples 100 --output_dir results/ --resume
```

Saves: `results/nq_random_nuc20.json`, `results/nq_attention_nuc20.json`, `results/nq_age_nuc20.json`, `results/nq_surprise_nuc20.json`

Estimated runtime: ~8–10 hr on A100 (4 strategies × 100 examples × 20 steps).

Status: Fix committed — ready to run on Colab.

---

## Issue #13: [ANALYSIS] Plot retention decay curves

Depends on eval results (#8, #9, #10, #11, #12). Plot EM accuracy vs update step for all strategies on both datasets. Save to `figures/retention_squad.png` and `figures/retention_nq.png`.

Status: Blocked on evals.

---

## Issue #14: [ANALYSIS] AUC per strategy

Depends on #13. Compute `np.trapz(accuracy_per_step)` per strategy×dataset. Save to `results/auc_summary.csv` for paper Table 1.

Status: Blocked on #13.

---

## Issue #18: [WRITING] Presentation slides

Status: Not started.

---

## Issue #25: [ANALYSIS] Statistical significance testing

Status: Blocked on eval results.

---

## Issue #28: [WRITING] Final report — Experiments + Results

Status: Not started.

---

## Reviewer Feedback — 2026-04-21

### 1. Surprise-gating: "orthogonal" wording is wrong
**Feedback:** Orthogonal = cosine similarity of 0. Lowest $s_i = -1$ = concepts pointing in opposite directions, not orthogonal.
- We say we drop "orthogonal" tokens but we mean the *most dissimilar* (lowest cosine sim).
- **Fix:** Change "orthogonal" to "most dissimilar" or "anti-correlated" in writeup.

### 2. Why random works well — per-layer independence
**Feedback:** Random dropping each layer independently means the probability of a token being dropped at ALL layers is much lower. The model gets many chances to retain any given memory token.
- This is a structural advantage of random that our strategies might not share.
- **Confirmed in code:** see Analysis #3 below.

### 3. Age-stratified: does it pick the same tokens at every layer? — CONFIRMED YES

**Code analysis of `modeling_memoryllm_strategies.py`:**

There are two modes controlled by `drop_memory_per_layer`:

**Mode A — `drop_memory_per_layer=False` (shared dropping, lines 263–281):**
- `drop_memory()` is called ONCE with `layer_idx=0`
- Age scores come from `_token_ages[0]` only
- The SAME `remaining_indices` is applied to all L layers
- **All layers drop identical tokens.** This fully confirms the reviewer's concern.

**Mode B — `drop_memory_per_layer=True` (per-layer, lines 246–261):**
- `drop_memory()` is called separately for each layer with its own `layer_idx`
- BUT `_token_ages` are initialized identically for all layers (all zeros), and updated with the same `remaining_indices` sequence at each step
- Since ages are deterministic and identical across layers, importance scores are identical → same drop order at every layer
- Tie-breaking noise is `1e-8` (microscopic), so in practice same tokens are dropped at every layer here too

**Contrast with random:** `torch.rand(N)` is called fresh for each layer, fully independent.

**Contrast with attention:** `_attention_ema[layer_idx]` accumulates real per-layer attention weights from the transformer — these genuinely differ across layers, so attention-based dropping IS layer-diverse.

**Contrast with surprise:** Uses `delta_memory[layer_idx]` — per-layer representations differ, so surprise IS layer-diverse.

**Bottom line:**
| Strategy | Per-layer diverse? |
|----------|-------------------|
| random | ✓ fully independent |
| attention | ✓ real per-layer attention differs |
| surprise | ✓ per-layer delta_memory differs |
| age | ✗ ages synchronized across layers → same drops everywhere |
| fisher | ✗ KL scores assigned identically to all layers (lines 390–391) |

**Proposed fix for age strategy:** Add a small layer-indexed noise term to importance scores in `_compute_importance` (around line 154):
```python
# current (bad — identical across layers):
importance += torch.rand(N) * 1e-8

# fix — layer-specific noise breaks synchronization:
importance += torch.rand(N) * 1e-8 + layer_idx * 1e-6
```
This gives each layer a slightly different ranking without destroying the age-based ordering signal. The key is the `layer_idx * 1e-6` offset — layers that agree on age-based importance will still disagree on which exact tokens to drop, recovering some of the per-layer diversity that random gets for free.

### 4. Why 3B parameter setup?
**Answer to add to writeup:** The 3B model (`openlm-research/open_llama_3b_v2`) is used for compute profiling and strategy development because it fits on a Colab T4 without quantization and doesn't require gated HF access. The 8B MemoryLLM pretrained checkpoint is used for the actual retention eval — it has trained memory integration, whereas the 3B is initialized from scratch for profiling purposes only.

---

## Open Action Items

- [ ] **Fix writeup:** "orthogonal" → "most dissimilar" / "anti-correlated" in surprise strategy description
- [x] **Code fix:** Age strategy noise scale raised from `1e-8` → `1e-3` in `modeling_memoryllm_strategies.py:167` — each layer now independently reorders same-age tokens, recovering per-layer diversity
- [ ] **Writeup:** Add clear explanation of 3B (profiling) vs 8B (eval) model choice
- [x] **Issue #5:** Download NaturalQA data files — DONE
- [ ] **Issues #12→#13→#14:** Run NQ evals, plot retention curves, compute AUC

---

## Debug session 2026-04-28 — fixing the zero-accuracy eval

The previous NQ run produced AUCs of 0.045 / 0.030 / 0.030 / 0.000 across the four strategies — essentially noise. Spent a session bisecting the cause. Found **four distinct bugs**, all now fixed on `ketaki`. Tracking each below for future-us when something breaks again.

### Bug 1 — `LlamaTokenizer` against a Llama-3 model

`run_eval.py`, `run_sanity.py`, `dataset/nq.py`, `dataset/squad.py` all imported `LlamaTokenizer`. MemoryLLM-8B is built on Llama-3 (tiktoken-based BPE), but `LlamaTokenizer` is the SentencePiece tokenizer for Llama-1/2 — different vocab, different special tokens.

**Fix (commit `30f3961`):** Swapped to `AutoTokenizer` everywhere. Also dropped the `+ "</s>"` hack appended to gold answers in `run_eval.py:89` (that's the Llama-2 EOS string; Llama-3 uses `<|end_of_text|>`). Updated `exact_hit` to strip both for safety.

This alone wasn't the main bug — but it would have produced wrong token IDs at inject time and corrupted comparisons.

### Bug 2 — peft version drift silently dropping 384 LoRA decoder adapters (the critical one)

The repo locks `transformers==4.48.2`, `peft==0.10.0`. Colab now ships `transformers 5.0.0`, `peft 0.19.1`, `torch 2.10`. Under the new peft, `get_peft_model(self.model, peft_config, adapter_name="decoder_adapter")` raises `AttributeError`. The old code in `modeling_memoryllm.py:1554-1566` swallowed it silently with a warning that said:

> "Skipping LoRA wrapping due to peft version incompatibility. This is fine for inference with pretrained checkpoints."

**That comment was wrong.** The checkpoint ships LoRA weights as **separate keys** (`model.layers.{0..31}.{q,k,v,gate,up,down}_proj.lora_{A,B}.decoder_adapter.weight` = 32 × 12 = **384 keys**). Skipping the peft wrapping means those keys land in `unexpected_keys` and are dropped — the model loses its trained pathway from memory pool → output. That is exactly why NQ accuracy was ~0%: memory was loaded but the decoder couldn't read it.

**Diagnosis:** `run_sanity.py` (issue #31) printed `unexpected_keys: 384` with `LoRA decoder_adapter` keys all listed. That's how we caught it.

**Fix:** Pin compatible versions on Colab — `pip install "transformers==4.48.2" "peft==0.10.0" "accelerate==1.2.0"`, then restart runtime. After this: `unexpected_keys: 0`, sanity step-0 jumped from 0.03 → 0.667.

Also patched the warning in `modeling_memoryllm.py:1554-1566` (commit `898b6b7`) to surface the actual `AttributeError` and warn that retention will be near-zero — so this never sneaks back silently.

### Bug 3 — `dataset/squad.py` had no fallback when `train-v2.0.json` is missing

NQ already had this fallback (commit `5c8472c`); SQuAD didn't. Sanity check failed on `FileNotFoundError: train-v2.0.json` once the LoRA bug was fixed and we tried to actually run.

**Fix (commit `898b6b7`):** Mirrored the NQ pattern — when train file is absent, build distractor contexts from non-eval entries in the dev file. Uses `eval_indices_set` to avoid leakage between target and distractors.

### Bug 4 — NQ wrapping context/question, breaking generation format

`dataset/nq.py:170-171` returned `"Context: " + long_answer` and `"Questions: " + question + "? Answer:"`. SQuAD returned raw text. The wrapped format pushed the model into "Q&A page completion" mode — it produced fragments of the long_answer paragraph instead of extracting the short answer.

Sample broken output (Mo Farah question, before fix):
```
target: 'Mo Farah'
step_0: ' The New Yorkshire of the National Health Care to'
step_1: ' Sports Personality of the Year in the Year 201'  # ← regurgitating context
```

**Fix (commit `9e12e55`):** Match SQuAD format — return raw `long_answer` and raw `question` text. After this, predictions became semantically related to the question (e.g., "Gareth Barry" target → step_0 contained "Gareth Barry").

### Bug 5 — Memory reset to zeros instead of pretrained checkpoint state (also critical)

`run_eval.py:166` had:
```python
model.memory.data = torch.zeros_like(model.memory.data)
```

Every example, this wiped the pretrained memory pool to zeros. `inject_memory` only overwrites **one block** of `num_tokens=256` per call — but the pool has `num_blocks=50` blocks for a total of 12,800 tokens. So **49/50 of the pool stayed zero** for the entire example, instead of being the trained checkpoint state.

The sanity check actually proved this was destructive: its `zeroed` condition (same operation) scored 0.033 step-0; `normal` (reset to checkpoint) scored 0.667. **Our eval was running under the `zeroed` condition.** That explains why it appeared everything was broken even after fixing 1–4.

**Fix (commit `ca75cac`):** Snapshot `checkpoint_memory = model.memory.data.detach().clone()` once at the top of the eval loop, then `model.memory.data.copy_(checkpoint_memory)` before each example.

NQ smoke after this fix: step-0 accuracy 0.30, predictions semantically aligned with targets. Pipeline working.

### E0 sanity check (issue #31) — PASSED ✓

Run: `python run_sanity.py --num_samples 30 --nuc 5` on SQuAD with the patched stack.

**LOAD REPORT:**
- `missing_keys: 0`, `unexpected_keys: 0`, `mismatched_keys: 0`
- memory keys missing: none
- `initialized: 1` (memory populated during pretraining)
- `memory.shape: (32, 12800, 4096)`, mean=-0.0009, std=0.3359, abs_max=125.0

**VERDICT:**

| Condition | step-0 | step-5 | AUC |
|---|---|---|---|
| normal | **0.667** | 0.600 | 2.500 |
| zeroed | 0.033 | 0.000 | 0.050 |
| scrambled | 0.633 | 0.433 | 2.233 |

`normal − zeroed = +0.633` (>>0.10 threshold). Memory contributes substantially to retention. Pipeline measures something real. **PASS** — closes #31.

Side note: scrambled performs almost as well as normal at step-0 (0.633 vs 0.667). Ordering of memory tokens matters less than the LoRA decoder adapter and the *presence* of trained weights in the pool. Worth a sentence in the report's analysis section — interesting and unexpected.

### Files touched in this session

| Commit | What |
|---|---|
| `30f3961` | `LlamaTokenizer` → `AutoTokenizer` in run_eval.py, run_sanity.py, dataset/{nq,squad}.py; drop `</s>` hack |
| `898b6b7` | squad.py dev-fallback for distractors; surface peft load error in modeling_memoryllm.py |
| `9e12e55` | NQ returns raw context/question (drop "Context: " / "Questions: ... Answer:" wrapper) |
| `ca75cac` | run_eval.py reset memory to checkpoint not zeros |

### Lesson for future debugging

Both critical bugs (peft and memory-reset) were silent failures producing plausible-looking-but-wrong numbers. The sanity check (issue #31) caught both — it's the only diagnostic that compares known-good (`normal`) to known-bad (`zeroed`) conditions. **Always run sanity before spending GPU hours.** It's a 15-min run on A100 and would have saved this whole session if it had been run before the previous 8-hr eval.

---

## Phase D analysis scripts (added 2026-04-28)

Three scripts in `analysis/`, ready to run as soon as the 8 JSONs land:

- **`analysis/plot_retention.py`** → `figures/retention_{squad,nq}.png` — overlaid 4-strategy decay curves per dataset. Closes #13.
- **`analysis/auc_table.py`** → `results/auc_summary.csv` — table for paper. Closes #14.
- **`analysis/significance.py`** → `results/significance.csv` — bootstrap 95% CIs on AUC + paired permutation tests vs `random`, Bonferroni-corrected over 3 comparisons. Closes #25.

Run with defaults; for paper-final numbers bump iters: `python analysis/significance.py --bootstrap_iters 5000 --perm_iters 10000`.

---

## Current run status (2026-04-28 evening)

- ▶ NQ all-strategies eval running on Colab (Ketaki's account, ~16 hr ETA). Saving to Drive at `MyDrive/ExtendingMemoryLLM/results/`. Uses `--resume` so disconnects don't kill it.
- ⏸ SQuAD all-strategies eval — to be kicked off by Akshat (random + attention) and Tushar (age + surprise) on their accounts using the same `ketaki` branch at commit `ca75cac` or newer. They MUST pin the same `transformers==4.48.2 peft==0.10.0` versions.
- Once 8 JSONs land in `results/`, run the three analysis scripts; embed outputs into `report/experiments.md` (#28).
- nuc=20, num_samples=100 is the only configuration we're running — additional NUC values are redundant since `accuracy_per_step[k]` for k<20 is already in the nuc=20 curve. N=100 gives ~±10pp 95% CI per point.

### Updated open actions

- [ ] Wait for NQ run to finish; download JSONs from Drive
- [ ] Confirm Akshat/Tushar SQuAD results land
- [ ] Run analysis scripts → produce figures + CSVs
- [ ] Draft #26 Intro/Related Work and #27 Methods *now* while runs go (no numbers needed)
- [ ] After analysis: #28 Experiments, #29 Discussion, #30 Abstract
- [ ] Slides #16/#17/#18
