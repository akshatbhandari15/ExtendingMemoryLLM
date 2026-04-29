# data/

Data files for retention eval. **Do not commit data files** — they are gitignored.

## data/squad/

| File | Source | Notes |
|------|--------|-------|
| `dev-v2.0.json` | https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json | SQuAD v2 dev set |
| `train-v2.0.json` | https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json | Only needed when `num_unrelated_contexts > 0` |
| `indices_squad_3.npy` | HuggingFace `YuWangX/KnowledgeRetention`, path `squad/indices_squad_3.npy` | Selects 2250 QA pairs from the 5928 answerable examples in dev |

### Download commands

```bash
mkdir -p data/squad

# SQuAD v2 dev (required)
curl -L "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json" \
     -o data/squad/dev-v2.0.json

# Index file (required)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='YuWangX/KnowledgeRetention',
    filename='squad/indices_squad_3.npy',
    repo_type='dataset',
    local_dir='/tmp/hf_dl'
)
"
cp /tmp/hf_dl/squad/indices_squad_3.npy data/squad/

# SQuAD v2 train (optional — only if num_unrelated_contexts > 0)
curl -L "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json" \
     -o data/squad/train-v2.0.json
```

### Verification

```python
import numpy as np
idx = np.load('data/squad/indices_squad_3.npy')
# Expected: shape (2250,), dtype int64, max index 5927
print(idx.shape, idx.dtype, idx.max())
```

## data/nq/

| File | Source | Notes |
|------|--------|-------|
| `v1.0-simplified_nq-dev-all.jsonl` | HuggingFace `YuWangX/KnowledgeRetention`, path `nq/v1.0-simplified_nq-dev-all.jsonl` | NaturalQA dev set (7830 examples, 6.4GB) |
| `indices_nq_4.npy` | HuggingFace `YuWangX/KnowledgeRetention`, path `nq/indices_nq_4.npy` | Selects 1044 examples, max index 2562 |

### Download commands

```bash
mkdir -p data/nq

python3 -c "
from huggingface_hub import hf_hub_download
import shutil

hf_hub_download(repo_id='YuWangX/KnowledgeRetention', filename='nq/v1.0-simplified_nq-dev-all.jsonl', repo_type='dataset', local_dir='/tmp/hf_dl_nq')
hf_hub_download(repo_id='YuWangX/KnowledgeRetention', filename='nq/indices_nq_4.npy', repo_type='dataset', local_dir='/tmp/hf_dl_nq')
"
cp /tmp/hf_dl_nq/nq/v1.0-simplified_nq-dev-all.jsonl data/nq/
cp /tmp/hf_dl_nq/nq/indices_nq_4.npy data/nq/
```

Note: The JSONL file is 6.4GB — allow 5–10 minutes on a fast connection.

### Verification

```python
import numpy as np
idx = np.load('data/nq/indices_nq_4.npy')
# Expected: shape (1044,), dtype int64, max index 2562
print(idx.shape, idx.dtype, idx.max())
```
