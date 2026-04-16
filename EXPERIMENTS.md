# Running Experiments

## Quick start

```bash
# 1. Clone and set up
git clone https://github.com/akshatbhandari15/ExtendingMemoryLLM.git
cd ExtendingMemoryLLM
bash scripts/setup.sh

# 2. Smoke test (should finish in ~2 min)
python run_eval.py --strategy random --dataset squad --nuc 3 --num_samples 5

# 3. Full eval — all strategies, SQuAD v2, 20 distractor steps
python run_eval.py --strategy all --dataset squad --nuc 20 --num_samples 100

# 4. Same for NaturalQA
python run_eval.py --strategy all --dataset nq --nuc 20 --num_samples 100
```

## `run_eval.py` — all options

| Flag | Default | Description |
|------|---------|-------------|
| `--strategy` | `random` | `random` / `attention` / `age` / `surprise` / `all` |
| `--dataset` | `squad` | `squad` or `nq` |
| `--nuc` | `20` | Number of distractor contexts injected after the target |
| `--num_samples` | `100` | Number of eval examples (use `5` for smoke tests) |
| `--model` | `YuWangX/memoryllm-8b` | HuggingFace model ID or local path |
| `--output_dir` | `results/` | Where to save JSON result files |
| `--dtype` | `bfloat16` | `bfloat16` or `float16` |
| `--resume` | off | Skip strategies whose output file already exists |

**`--strategy all` is the recommended way to run** — it loads the model once and runs all strategies sequentially, which avoids the ~2 min model load overhead per strategy.

## Output format

Results are saved as `results/{dataset}_{strategy}_nuc{nuc}.json`:

```json
{
  "config": { "strategy": "attention", "dataset": "squad", "nuc": 20, ... },
  "timestamp": "2026-04-16 10:30:00",
  "elapsed_seconds": 1842.3,
  "accuracy_per_step": [0.72, 0.68, 0.61, ...],
  "auc": 11.43,
  "per_example": [
    { "target": "Paris", "step_preds": {"step_0": "Paris", "step_1": "London", ...} },
    ...
  ]
}
```

`accuracy_per_step[0]` = accuracy right after injecting the target (upper bound).  
`accuracy_per_step[N]` = accuracy after N distractors have been injected.  
`auc` = `np.trapz(accuracy_per_step)` — higher is better.

## Platform setup

### Google Colab (current)

Open `notebooks/colab_setup.ipynb` in Colab.

1. Set runtime to **A100 GPU**: Runtime → Change runtime type → A100
2. Run cells 1–4 (setup, ~5 min)
3. Run cell 5 (smoke test)
4. Run cells 6–7 (full eval, ~1 hr total)

Results are saved to **Google Drive** at `MyDrive/ExtendingMemoryLLM/results/` so they survive session restarts.

> **Colab gotcha:** Sessions disconnect after ~12 hrs. Use `--resume` so interrupted runs pick up where they left off. The `all` strategy runner saves after each strategy completes.

### GCP (if upgrading)

```bash
# On a new VM with GPU (e.g. n1-standard-8 + A100)
git clone https://github.com/akshatbhandari15/ExtendingMemoryLLM.git
cd ExtendingMemoryLLM
HF_TOKEN=hf_xxx bash scripts/setup.sh

# Run with results saved to GCS bucket
python run_eval.py --strategy all --dataset squad --nuc 20 \
    --output_dir gs://your-bucket/results   # requires gcsfuse or gsutil cp after
```

Or mount a GCS bucket with `gcsfuse` and point `--output_dir` at the mount point.

### Vast.ai (if upgrading)

1. Rent an A100 instance with PyTorch pre-installed
2. In the instance terminal:
```bash
git clone https://github.com/akshatbhandari15/ExtendingMemoryLLM.git
cd ExtendingMemoryLLM
HF_TOKEN=hf_xxx bash scripts/setup.sh
python run_eval.py --strategy all --dataset squad --nuc 20
```
3. Copy results off before destroying the instance:
```bash
scp -P <port> root@<host>:~/ExtendingMemoryLLM/results/ ./results/
```

> **Vast.ai tip:** Use a persistent storage volume so results survive if the instance is interrupted.

## Colab → GCP/Vast.ai migration checklist

The experiment script (`run_eval.py`) is identical on all platforms. Only the setup differs:

- [ ] `scripts/setup.sh` installs all deps — run this first on any new machine
- [ ] HF login: set `HF_TOKEN` env var to avoid interactive prompt
- [ ] Data: `data/squad/` and `data/nq/` — copy from Drive or re-download
- [ ] Results: copy existing `results/` folder to new machine before running with `--resume`

## Data files required

| File | Source |
|------|--------|
| `data/squad/dev-v2.0.json` | Auto-downloaded by `setup.sh` |
| `data/squad/indices_squad_3.npy` | Auto-downloaded from `YuWangX/KnowledgeRetention` |
| `data/nq/indices_nq_4.npy` | Auto-downloaded from `YuWangX/KnowledgeRetention` |
| `data/nq/v1.0-simplified_nq-dev-all.jsonl` | **Manual:** https://ai.google.com/research/NaturalQuestions/download |

## Experiment schedule

| Issue | Strategy | Dataset | NUC | Assigned |
|-------|----------|---------|-----|----------|
| #8  | random | squad | 20 | Akshat |
| #9  | attention | squad | 20 | Akshat |
| #10 | age | squad | 20 | Tushar |
| #11 | surprise | squad | 20 | Tushar |
| #12 | all | nq | 20 | Ketaki |

See [GitHub Issues](https://github.com/akshatbhandari15/ExtendingMemoryLLM/issues) and [Project Board](https://github.com/users/akshatbhandari15/projects/3) for full task tracking.
