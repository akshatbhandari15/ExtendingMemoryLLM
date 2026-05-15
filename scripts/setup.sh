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
# Build prereqs must be present BEFORE flash-attn so it can compile / pick a wheel.
pip install -q packaging ninja wheel

# Install torch + transformers + peft stack first (flash-attn needs torch present at build/import).
pip install -q \
  "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
  "transformers==4.48.2" "peft==0.10.0" "accelerate==1.2.0" \
  "numpy==1.26.4" "torchmetrics==1.3.2" \
  cachetools einops nltk omegaconf PyYAML tqdm scikit-learn pandas

# NOTE: flash-attn is intentionally NOT installed here. Colab's torch/cuda/python combo
# moves faster than flash-attn's wheel releases. Eval falls back to sdpa attention,
# which is correct and only ~1.3-1.5x slower at our batch=1 inference workload.
# To try flash-attn manually: pip install flash-attn --no-build-isolation (slow, may fail).
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available())"

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
