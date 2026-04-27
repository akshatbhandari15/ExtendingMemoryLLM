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
