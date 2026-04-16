#!/usr/bin/env bash
# =============================================================================
# setup.sh — Environment setup for ExtendingMemoryLLM
#
# Works on any Linux instance: Google Colab, GCP, Vast.ai, etc.
# Run once per session (or once after first clone on persistent instances).
#
# Usage:
#   bash scripts/setup.sh                        # standard install
#   bash scripts/setup.sh --no-data             # skip data download
#   HF_TOKEN=hf_xxx bash scripts/setup.sh       # non-interactive HF login
# =============================================================================

set -e

SKIP_DATA=false
for arg in "$@"; do
  [[ "$arg" == "--no-data" ]] && SKIP_DATA=true
done

echo ">>> Installing Python dependencies..."
pip install -q -r requirements_infer_only.txt
pip install -q omegaconf scikit-learn pandas tqdm

echo ">>> HuggingFace login..."
if [[ -n "$HF_TOKEN" ]]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
  huggingface-cli login
fi

if [[ "$SKIP_DATA" == false ]]; then
  echo ">>> Downloading eval data..."

  mkdir -p data/squad data/nq

  # SQuAD v2
  if [[ ! -f data/squad/dev-v2.0.json ]]; then
    echo "  Downloading SQuAD v2 dev set..."
    wget -q https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json \
         -O data/squad/dev-v2.0.json
  else
    echo "  SQuAD dev set already present, skipping."
  fi

  # Index files from HuggingFace
  if [[ ! -f data/squad/indices_squad_3.npy ]]; then
    echo "  Downloading SQuAD index file..."
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='YuWangX/KnowledgeRetention', filename='indices_squad_3.npy', repo_type='dataset')
shutil.copy(path, 'data/squad/indices_squad_3.npy')
print('  Saved to data/squad/indices_squad_3.npy')
"
  fi

  if [[ ! -f data/nq/indices_nq_4.npy ]]; then
    echo "  Downloading NaturalQA index file..."
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='YuWangX/KnowledgeRetention', filename='indices_nq_4.npy', repo_type='dataset')
shutil.copy(path, 'data/nq/indices_nq_4.npy')
print('  Saved to data/nq/indices_nq_4.npy')
"
  fi

  echo "  NOTE: NaturalQA jsonl must be downloaded manually from:"
  echo "  https://ai.google.com/research/NaturalQuestions/download"
  echo "  Place at: data/nq/v1.0-simplified_nq-dev-all.jsonl"
fi

echo ""
echo ">>> Setup complete. Run a smoke test with:"
echo "    python run_eval.py --strategy random --dataset squad --nuc 3 --num_samples 5"
