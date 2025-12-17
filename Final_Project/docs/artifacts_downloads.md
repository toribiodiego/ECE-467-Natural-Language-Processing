# Downloaded Artifacts Log

This file tracks large artifacts that have been downloaded from W&B but are gitignored due to size.

## Neutral Label Ablation Experiments

**Downloaded**: 2025-12-16

### 1. DistilBERT Neutral-ON (28 labels)
- **W&B Run**: cslpcnl8
- **Run URL**: https://wandb.ai/Cooper-Union/GoEmotions_Classification/runs/cslpcnl8
- **Location**: `artifacts/models/distilbert-neutral-on/`
- **Size**: ~506 MB (checkpoint + predictions + metrics)
- **Files**:
  - Model checkpoint (pytorch_model.bin, config.json)
  - Tokenizer files
  - Test predictions CSV
  - 10 validation epoch predictions CSVs
  - Per-class metrics CSV (28 emotions)
- **Download Method**: W&B Python API (see `docs/tools/wandb/downloading_files.md`)

```python
# Download command used:
api = wandb.Api()
run = api.run('Cooper-Union/GoEmotions_Classification/cslpcnl8')
for file in run.files():
    if any(prefix in file.name for prefix in ['distilbert', 'predictions/', 'stats/']):
        if 'artifact/' not in file.name:
            file.download(root='artifacts/models/distilbert-neutral-on', replace=True)
```

### 2. DistilBERT Neutral-OFF (27 labels)
- **W&B Run**: g5h44tsf
- **Run URL**: https://wandb.ai/Cooper-Union/GoEmotions_Classification/runs/g5h44tsf
- **Location**: `artifacts/models/distilbert-neutral-off/`
- **Size**: ~506 MB (checkpoint + predictions + metrics)
- **Files**:
  - Model checkpoint (pytorch_model.bin, config.json)
  - Tokenizer files
  - Test predictions CSV
  - 10 validation epoch predictions CSVs
  - Per-class metrics CSV (27 emotions, neutral excluded)
- **Download Method**: W&B Python API (see `docs/tools/wandb/downloading_files.md`)

```python
# Download command used:
api = wandb.Api()
run = api.run('Cooper-Union/GoEmotions_Classification/g5h44tsf')
for file in run.files():
    if any(prefix in file.name for prefix in ['distilbert', 'predictions/', 'stats/']):
        if 'artifact/' not in file.name:
            file.download(root='artifacts/models/distilbert-neutral-off', replace=True)
```

## Re-downloading

If these artifacts are not present locally, re-download using:

```bash
# Use the Python script from docs/tools/wandb/downloading_files.md
python << 'EOF'
import wandb

def download_run_files(run_id, output_dir):
    api = wandb.Api()
    run = api.run(f'Cooper-Union/GoEmotions_Classification/{run_id}')
    for file in run.files():
        if any(prefix in file.name for prefix in ['distilbert', 'predictions/', 'stats/']):
            if 'artifact/' not in file.name:
                file.download(root=output_dir, replace=True)

# Neutral-ON
download_run_files('cslpcnl8', 'artifacts/models/distilbert-neutral-on')

# Neutral-OFF
download_run_files('g5h44tsf', 'artifacts/models/distilbert-neutral-off')
EOF
```

## Other Downloaded Artifacts

### RoBERTa-Large Production Checkpoint
- **Location**: `artifacts/models/roberta-large/`
- **W&B Run**: (to be documented)
- **Status**: Downloaded from training run

### DistilBERT Multi-Seed Checkpoints
- **Location**: `artifacts/models/distilbert/`
- **W&B Runs**: (to be documented)
- **Status**: Downloaded from training runs

---

**Note**: All model checkpoints are gitignored in `.gitignore` under `artifacts/models/`. They should be downloaded from W&B when needed, not stored in version control.
