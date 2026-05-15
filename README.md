# Beyond Random Forgetting: Importance-Aware Memory Management for MemoryLLM

**COMS 6998 — Continual Learning and Memory Models, Columbia University (Spring 2026)**

This is a research fork of [MemoryLLM](https://arxiv.org/abs/2402.04624) (Wang et al., ICML 2024). The original paper augments Llama-3-8B with a 1.67 B-parameter memory pool (32 layers × 50 blocks × 256 tokens × 4096 dim) and drops memory tokens **at random** when new context arrives. We replace that policy with **four importance-aware drop strategies** and ask: do principled eviction signals retain knowledge better than random?

**TL;DR:** at the per-layer-independent dropping condition the checkpoint was actually trained under, random is a surprisingly strong baseline. Age beats it on SQuAD, mainly on mid-position answers (not recency bias). Layer-Jaccard confirms why: age drops the same tokens across all 32 layers (≈0.95), while random fully decorrelates (≈0.01) — that structural independence is most of what gives random its edge.


## Credits & relationship to upstream

This fork builds directly on the official MemoryLLM implementation by Yu Wang et al. The model code (`modeling_memoryllm.py`, `modeling_mplus.py`, `configuration_memoryllm.py`, dataset loaders, the LongBench eval, and the pretrained `YuWangX/memoryllm-8b` checkpoint) is unchanged or minimally patched from upstream. The original MemoryLLM README is preserved below for completeness, with full citations at the bottom of this file. All credit for the base architecture, training, and pretrained weights belongs to the original authors.

What this fork adds:
- `modeling_memoryllm_strategies.py` — `MemoryLLMWithStrategies` subclass that overrides `drop_memory()` with four importance-aware policies.
- `run_eval.py`, `run_sanity.py` — retention eval driver + pre-eval health check.
- `analysis/` — AUC table, bootstrap CIs + permutation tests, Layer-Jaccard, retention plots, position-of-answer bias, plus five extra paper-quality plots.
- `CODEBASE_REVIEW.md`, `PROGRESS_LOG.md`, `PROJECT_STATUS.md`, `EXPERIMENTS.md` — full project documentation.

---

## Headline results (per-layer mode, N=100 examples, 20 distractor steps)

| Strategy | SQuAD AUC | SQuAD vs random | NQ AUC | NQ vs random |
|---|---:|---:|---:|---:|
| random | 8.18 | — | **1.765** | — |
| attention | 7.63 | −0.55 (p=0.072) | 1.63 | −0.135 (ns) |
| **age** | **8.46** | +0.28 (ns) | 1.71 | −0.055 (ns) |
| surprise | 8.015 | −0.165 (ns) | 1.63 | −0.135 (ns) |

AUC is `np.trapz(accuracy_per_step)` over 21 points; normalised values (AUC / nuc, on [0, 1]) are in `results/auc_perlayer.csv`. No comparison clears Bonferroni at N=100 — trends are descriptive, not confirmatory. Layer-Jaccard numbers (TA Q2):

- random: **0.01** (drops fully decorrelated across layers)
- attention: **0.01** (per-layer attention EMAs diverge sharply too)
- surprise: **0.60** (per-layer `delta_memory` partially correlated)
- age: **0.95** (ages are synchronised; tie-break noise barely diversifies)

Full numbers: `results/{auc_perlayer,significance_perlayer,jaccard_summary,position_bias}.csv`. Figures: `figures/p1_decay_overlay.png`, `p2_strategy_agreement.png`, `p3_auc_ci_bars.png`, `p4_robust_forgot_recovered.png`, `p5_layer_jaccard.png`, `position_bias.png`, `retention_{squad,nq,combined}.png`.

---

## Repository layout

| Path | What |
|---|---|
| `modeling_memoryllm.py` | Upstream MemoryLLM model (lightly patched to surface peft load errors loudly). |
| `modeling_memoryllm_strategies.py` | **Our contribution.** `MemoryLLMWithStrategies(MemoryLLM)` with four drop strategies + drop logging. |
| `modeling_mplus.py`, `configuration_memoryllm.py` | Upstream, unchanged in essentials. |
| `run_eval.py` | Retention eval driver. Auto-detects flash-attn else falls back to sdpa; supports `--drop_per_layer` and `--log_dropped`. |
| `run_sanity.py` | E0 health check — `normal` vs `zeroed` vs `scrambled` memory. Must pass before long runs. |
| `dataset/squad.py`, `dataset/nq.py` | Loaders, patched to use `AutoTokenizer` and fall back to dev-only distractor contexts when train files are absent. |
| `analysis/auc_table.py` | Builds `results/auc_*.csv` with raw + normalised AUC. |
| `analysis/significance.py` | Bootstrap 95% CIs + paired permutation tests vs `random`, Bonferroni-corrected. |
| `analysis/dropped_indices.py` | Layer-Jaccard (within-strategy cross-layer overlap) + cross-strategy Jaccard. |
| `analysis/plot_retention.py` | Retention decay curves with bootstrap CI bands. |
| `analysis/position_bias.py` | Step-N accuracy by where the gold answer sits in the source context. |
| `analysis/extra_plots.py` | Five paper-quality plots (P1–P5). |
| `scripts/setup.sh` | Pins `torch 2.5.1 / transformers 4.48.2 / peft 0.10.0 / accelerate 1.2.0`. **Do not drift these pins** (see `PROGRESS_LOG.md` Bug 2). |
| `EXPERIMENTS.md` | Command reference. |
| `CODEBASE_REVIEW.md` | Full repo audit + plan-of-record. |
| `PROGRESS_LOG.md` | Chronological debugging log incl. the four-bug session and per-layer matrix runs. |
| `PROJECT_STATUS.md` | Current headline numbers + Phase checklist. |

---

## Quick start

### 1. Clone

```bash
git clone https://github.com/akshatbhandari15/ExtendingMemoryLLM.git
cd ExtendingMemoryLLM
```

### 2. Install (Colab A100 recommended)

```bash
export HF_TOKEN=hf_xxx           # for YuWangX/memoryllm-8b access
bash scripts/setup.sh --no-data  # installs pinned deps
# Then RESTART the Python kernel so the pins take effect.
```

Verify after restart:
```bash
python -c "import torch, transformers, peft, accelerate; \
print(torch.__version__, transformers.__version__, peft.__version__, accelerate.__version__)"
```
Must print `2.5.1+cu124 4.48.2 0.10.0 1.2.0`. If any drift, **stop** — the LoRA decoder adapters silently fail to load under `peft 0.19+`.

### 3. Download data

```bash
mkdir -p data/squad data/nq
wget -q https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O data/squad/dev-v2.0.json
python -c "
from huggingface_hub import hf_hub_download; import shutil
for fn in ['squad/indices_squad_3.npy','nq/indices_nq_4.npy','nq/v1.0-simplified_nq-dev-all.jsonl']:
    p = hf_hub_download('YuWangX/KnowledgeRetention', fn, repo_type='dataset')
    shutil.copy(p, 'data/' + fn)
"
```
NQ jsonl is 6.4 GB — allow time.

### 4. Sanity (always before long evals)

```bash
python run_sanity.py --num_samples 30 --nuc 5
```
Pass condition: `normal − zeroed > 0.10`. Historical pass values: +0.633 (Apr), +0.567 (May).

### 5. Smoke test the per-layer flag

```bash
python run_eval.py --strategy random --dataset squad --nuc 3 --num_samples 5 \
    --drop_per_layer --log_dropped --output_dir results/perlayer_smoke
```
~2 min. Produces a results JSON + a `*_dropped.json` companion (the layer-by-layer drop log).

### 6. Canonical per-layer matrix run (the headline experiment)

```bash
# SQuAD: ~5-7 hr on A100
python run_eval.py --strategy all --dataset squad --nuc 20 --num_samples 100 \
    --drop_per_layer --log_dropped --output_dir results/perlayer --resume

# NQ: ~5-7 hr on A100
python run_eval.py --strategy all --dataset nq --nuc 20 --num_samples 100 \
    --drop_per_layer --log_dropped --output_dir results/perlayer --resume
```

`--resume` skips strategies whose output JSON already exists, so reruns are safe across Colab disconnects. The `_dropped.json` files are ~100 MB each (per-layer mode); they're gitignored — keep them on Google Drive or rerun to regenerate.

### 7. Analysis (no GPU, ~30 min)

```bash
python analysis/auc_table.py     --results_dir results/perlayer --stem_suffix _perlayer --out results/auc_perlayer.csv
python analysis/significance.py  --results_dir results/perlayer --stem_suffix _perlayer --out results/significance_perlayer.csv \
                                 --bootstrap_iters 5000 --perm_iters 10000
python analysis/dropped_indices.py --results_dir results/perlayer --out results/jaccard_summary.csv
python analysis/plot_retention.py  --results_dir results/perlayer --figures_dir figures
python analysis/position_bias.py
python analysis/extra_plots.py
```

---

## Strategies

| Strategy | What it drops | Signal | Per-layer behaviour (Layer-Jaccard) |
|---|---|---|:---:|
| `random` | Uniform random (baseline) | none | 0.01 (fully decorrelated) |
| `attention` | Lowest accumulated attention (EMA, α=0.9) | relevance | 0.01 (per-layer EMAs diverge sharply) |
| `age` | Oldest tokens; protects last 256 (one full block) | recency | 0.95 (ages synchronised across layers) |
| `surprise` | Most similar to incoming `delta_memory` (i.e. most redundant) | redundancy | 0.60 (partially correlated) |
| `fisher` | Implemented but not run (KL-divergence on masked memory; agreed to skip) | output sensitivity | — |

Reference: `modeling_memoryllm_strategies.py:_compute_importance`. Surprise drops the **most similar** tokens to the incoming context (interpretation: drop the redundant ones), not "orthogonal" tokens — a wording bug noted in the earlier code review.

---

## Programmatic use

```python
import torch
from modeling_memoryllm_strategies import MemoryLLMWithStrategies
from transformers import AutoTokenizer

# Auto-detect flash-attn; falls back to sdpa if absent (Colab default).
try:
    import flash_attn; attn = "flash_attention_2"
except ImportError:
    attn = "sdpa"

model = MemoryLLMWithStrategies.from_pretrained(
    "YuWangX/memoryllm-8b", torch_dtype=torch.bfloat16, attn_implementation=attn,
).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained("YuWangX/memoryllm-8b")

# Pick a strategy and per-layer mode (the checkpoint was trained at True).
model.set_drop_strategy("age")
model.drop_memory_per_layer = True

ctx = "The capital of France is Paris."
model.inject_memory(
    tokenizer(ctx, return_tensors="pt", add_special_tokens=False).input_ids.cuda(),
    update_memory=True,
)
```

---

## Documentation

- **`EXPERIMENTS.md`** — runnable command reference and all flags.
- **`CODEBASE_REVIEW.md`** — full repo map, known inconsistencies, ordered plan.
- **`PROGRESS_LOG.md`** — chronological debugging log (read "Debug session 2026-04-28" and "Session 2026-05-12" entries for context on the four-bug fix and per-layer matrix runs).
- **`PROJECT_STATUS.md`** — canonical numbers + Phase checklist.

---

## Original MemoryLLM & M+ README

This is the official implementation of paper **MemoryLLM: Towards Self-Updatable Large Language Models** and **M+: Extending MemoryLLM with Scalable Long-Term Memory**.

<p align="center" width="100%">
<!-- put the image "memoryllm.png" -->
<img src="assets/memoryllm.png" width="80%" height="80%">
</p>

## Official Links

[![Static Badge](https://img.shields.io/badge/memoryllm-paper-green)](https://arxiv.org/abs/2402.04624)
[![Static Badge](https://img.shields.io/badge/m+-paper-green)](https://arxiv.org/abs/2502.00592)  


[![MemoryLLM Checkpoint](https://img.shields.io/badge/memoryllm_7b-checkpoints-blue)](https://huggingface.co/YuWangX/memoryllm-7b)
[![MemoryLLM Checkpoint](https://img.shields.io/badge/memoryllm_8b-checkpoints-blue)](https://huggingface.co/YuWangX/memoryllm-8b)
[![MemoryLLM Checkpoint](https://img.shields.io/badge/memoryllm_8b_chat-checkpoints-blue)](https://huggingface.co/YuWangX/memoryllm-8b-chat)
[![MemoryLLM Checkpoint](https://img.shields.io/badge/mplus_8b-checkpoints-blue)](https://huggingface.co/YuWangX/mplus-8b)

<!-- This is the official code for the paper: **MemoryLLM: Towards Self-Updatable Large Language Models**.   
The model is open-sourced at https://huggingface.co/YuWangX/memoryllm-7b -->

## Release Notes
- [2025/07/27] 🔥 Updated the training code of `mplus-8b` and open-sourced at [mplus-8b-branch](https://github.com/wangyu-ustc/MemoryLLM/tree/mplus).
- [2025/02/07] 🔥 The model `mplus-8b` has been uploaded to [mplus-8b](https://huggingface.co/YuWangX/mplus-8b).
- [2025/02/01] 🔥 New paper [M+: Extending MemoryLLM with Scalable Long-Term Memory](https://arxiv.org/abs/2502.00592) is on Arxiv! 
- [2024/08/30] 🔥 We release [memoryllm-8b-chat](https://huggingface.co/YuWangX/memoryllm-8b-chat), the chat model built on top of [memoryllm-8b](https://huggingface.co/YuWangX/memoryllm-8b).
- [2024/08/23] 🔥 We release [memoryllm-8b](https://huggingface.co/YuWangX/memoryllm-8b) with 1.67B memory equipped on Llama3! 
- [2024/06/21] 🔥 Training code is provided in the folder `train`.
- [2024/06/02] 🔥 **MemoryLLM** checkpoint is [released](https://huggingface.co/YuWangX/memoryllm-7b)!
- [2024/05/02] 🔥 **MemoryLLM** is accepted to ICML 2024!

## Getting Started

### Environment Setup
```
conda create --name memoryllm
conda activate memoryllm
pip install -r requirements.txt
```

**Note:** In most cases, directly using `requirements.txt` should work well. However, if you encounter any compatibility issues, you can use `requirements_infer_only.txt` which contains locked versions that have been personally tested and verified to work. The testing environment used CUDA version 12.2 with H100-80GB-HBM3 GPUs.

### Load Model
First clone the repository and get into the repository: 
```
git clone git@github.com:wangyu-ustc/MemoryLLM.git
cd MemoryLLM
```

Then to load `MPlus-8B`, please use the following code: 
```python
import torch
from transformers import AutoTokenizer
from modeling_mplus import MPlus

# load the model mplus-8b (currently we only have the pretrained version)
model = MPlus.from_pretrained("YuWangX/mplus-8b", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("YuWangX/mplus-8b")
model = model.to(torch.bfloat16) # need to call it again to cast the `inv_freq` in rotary_emb to bfloat16 as well
model.put_ltm_to_numpy() # We include ltm as modules so that it can be uploaded to huggingface, but for inference we need to put ltm on CPU and cast ltm_ags to numpy. 
model = model.cuda()
# After this, the usage of MPlus is the same as MemoryLLM-8B, please check "How to use the model" below. 
```

To load `MemoryLLM-8B` and `MemoryLLM-8B-chat`, please use the following code:
```python
import torch
from transformers import AutoTokenizer
from modeling_memoryllm import MemoryLLM

# load pretrained model
model = MemoryLLM.from_pretrained("YuWangX/memoryllm-8b", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("YuWangX/memoryllm-8b")
model = model.cuda()

# load chat model
model = MemoryLLM.from_pretrained("YuWangX/memoryllm-8b-chat", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("YuWangX/memoryllm-8b-chat")
model = model.cuda()
```
If you want to use MemoryLLM-7B (the last version), please go to the branch `memoryllm-7b`. 

### How to use the model
Inject a piece of context into the model using the following script:
```python

# Self-Update with the new context
ctx = "Last week, John had a wonderful picnic with David. During their conversation, David mentioned multiple times that he likes eating apples. Though he didn't mention any other fruits, John says he can infer that David also like bananas."

# please make sure the context to inject into the memory is larger than 16 tokens, this is the hard minimum when training the model. The memory will be disturbed when less than 16 tokens are injected into the memory. 
model.inject_memory(tokenizer(ctx, return_tensors='pt', add_special_tokens=False).input_ids.cuda(), update_memory=True)
```

Then for chat model, use the following template: 
```python
# Generation
messages = [{
    'role': 'user', "content": "What fruits does David like?",
}]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)[:, 1:] # remove bos tokens as the model has its own trained bos embeddings.
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(input_ids=inputs.cuda(),
                         max_new_tokens=20,
                         eos_token_id=terminators)

response = tokenizer.decode(outputs[0])
```

For the pretrained model, use the following template:
```python
inputs = tokenizer("Question: What fruits does David like? Answer: David likes", return_tensors='pt', add_special_tokens=False).input_ids.cuda()
outputs = model.generate(input_ids=inputs, max_new_tokens=20)
response = tokenizer.decode(outputs[0][inputs.shape[1]:])
```

### Evaluation

#### Model Editing Evaluations
We put our reimplementation of various model-editing baselines and `MemoryLLM` in the repo [EditingLlama](https://github.com/wangyu-ustc/EditingLlama). 

#### Customized Experiments
To prepare the dataset, please download from [here](https://huggingface.co/datasets/YuWangX/KnowledgeRetention). Please download the dataset and put them as the following structure: 
```
- data
  - squad
    - indices_squad_3.npy
    - dev-v2.0.json
    - train-v2.0.json
  - nq 
    - indices_nq_4.npy
    - v1.0-simplified_nq-dev-all.jsonl
    - v1.0-simplified_simplified-nq-train.jsonl
```
We will evaluate our model on the validation set where the unrelated contexts are sampled from the training set. To evaluate the model, we could use the following script: 

```
mkdir results
python test_qa_memory.py --model YuWangX/memoryllm-7b --nuc 10 --datasets naturalqa squad --num_samples 100
```
here `nuc` means the number of irrelevant contexts, and `naturalqa squad` means the datasets to evaluate the model on.

#### Evaluation on Longbench

```
python longbench_pred.py --model memoryllm-7b --datasets hotpotqa --max_length 16384
```
Here `max_length` is the maximum length used when truncating the context.
Then the generated results are all saved in the folder `longbench` for evaluation.

#### Evaluation results on MemoryLLM-8B
Evaluation results on the knowledge-retention tasks are as follows: (we updated the evaluation dataset by filtering out the examples whose questions can be answered by Llama3-8B. The new dataset is [here](https://huggingface.co/datasets/YuWangX/KnowledgeRetentionProcessed))
<p align="center" width="100%">
<!-- put the image "memoryllm.png" -->
<img src="assets/nqa_comparison.png" width="100%" height="80%">
</p>
<p align="center" width="100%">
<!-- put the image "memoryllm.png" -->
<img src="assets/squad_comparison.png" width="100%" height="80%">
</p>

Evaluation results on LongBench are as follows:
<p align="center" width="100%">
<!-- put the image "memoryllm.png" -->
<img src="assets/longbench.png" width="100%" height="80%">
</p>

### Training
In our implementations, we train Llama2-7B on C4 dataset. However, this may lead to the poor performance on the benchmark `qasper` (see Figure 4 in the [paper](https://arxiv.org/pdf/2402.04624)). Thus we put the script of training on red-pajama here, which is the dataset we have been using in the models we are currently exploring. 

Please check the folder `train` using the following command:
```
cd train
```
#### Dataset Preparation
Please follow the instructions below to prepare the datasets: (make sure you have the datasets from [here](https://github.com/wangyu-ustc/MemoryLLM?tab=readme-ov-file#customized-experiments) prepared.)
```
cd data

# Please use the softlink to link the validation datasets into the current directory.
ln -s ../../data/nq ./
ln -s ../../data/squad ./

# Then please download the redpajama dataset
cd redpajama
sh download.sh
```

After preparing all the datasets, you can run the following code to start training:
```
python main.py -t --base MemoryLLM/configs/llama/llama_30x256.yaml
```
We have not conducted training on openllama but we do have the script on openllama for debugging purposes. So if you want to see the training on openllama, please run the following command:
```
python main.py -t --base MemoryLLM/configs/openllama/openllama_4x256.yaml
```

## Citations
If you find this repo helpful, please consider cite our paper:
```
@inproceedings{memoryllm,
  author       = {Yu Wang and
                  Yifan Gao and
                  Xiusi Chen and
                  Haoming Jiang and
                  Shiyang Li and
                  Jingfeng Yang and
                  Qingyu Yin and
                  Zheng Li and
                  Xian Li and
                  Bing Yin and
                  Jingbo Shang and
                  Julian J. McAuley},
  title        = {{MEMORYLLM:} Towards Self-Updatable Large Language Models},
  booktitle    = {Forty-first International Conference on Machine Learning, {ICML} 2024,
                  Vienna, Austria, July 21-27, 2024},
  publisher    = {OpenReview.net},
  year         = {2024},
  url          = {https://openreview.net/forum?id=p0lKWzdikQ},
  timestamp    = {Fri, 06 Dec 2024 12:46:25 +0100},
  biburl       = {https://dblp.org/rec/conf/icml/WangGCJLYYLLYSM24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@misc{wang2025mextendingmemoryllmscalable,
      title={M+: Extending MemoryLLM with Scalable Long-Term Memory}, 
      author={Yu Wang and Dmitry Krotov and Yuanzhe Hu and Yifan Gao and Wangchunshu Zhou and Julian McAuley and Dan Gutfreund and Rogerio Feris and Zexue He},
      year={2025},
      eprint={2502.00592},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.00592}, 
}
```
