# Weights & Biases Integration

This directory contains documentation for the Weights & Biases (W&B) integration in the GoEmotions Classification project.

## Documentation Files

- **[file_organization.md](file_organization.md)** - Complete guide to W&B Files tab organization
- **[downloading_files.md](downloading_files.md)** - How to download files from W&B (UI and API)
- **[metrics_guide.md](metrics_guide.md)** - What metrics are logged and where to find them

## Quick Start

### Default Configuration

- **Project Name**: `GoEmotions_Classification`
- **Entity**: Your W&B username (auto-detected)
- **Logging**: Enabled by default

### Basic Usage

```bash
# Train with W&B logging (default)
python -m src.training.train --model distilbert-base --epochs 5

# Disable W&B logging
python -m src.training.train --model distilbert-base --epochs 5 --no-wandb

# Custom project name
python -m src.training.train --model distilbert-base --epochs 5 --wandb-project MyProject

# Custom run name
python -m src.training.train --model distilbert-base --epochs 5 --run-name my-experiment
```

## What Gets Logged

### Metrics (to W&B Charts)

**Training metrics (per epoch)**:
- `train/loss` - Average training loss
- `train/loss_std` - Training loss standard deviation
- `train/learning_rate` - Current learning rate
- `train/epoch_time` - Time per epoch
- `train/samples_per_sec` - Training throughput

**Validation metrics (per epoch)**:
- `val/loss` - Validation loss
- `val/auc` - Validation AUC (used for early stopping)
- `val/f1_micro` - Validation F1 micro-average
- `val/f1_macro` - Validation F1 macro-average

**Test metrics (final)**:
- `test/auc_micro`, `test/auc_macro` - Final test AUC scores
- `test/f1_micro`, `test/f1_macro` - Final test F1 scores
- `test/precision_macro`, `test/recall_macro` - Final test precision/recall

### Files (to W&B Files Tab)

**Checkpoint directory**:
- `{model-name-timestamp}/` - Complete model checkpoint
  - `pytorch_model.bin` - Model weights
  - `config.json` - Model configuration
  - `tokenizer.json` - Tokenizer
  - `vocab.txt` - Vocabulary
  - `metrics.json` - Training metrics summary

**Predictions**:
- `predictions/test_predictions_*.csv` - Test set predictions with probabilities
- `predictions/val_epoch{N}_predictions_*.csv` - Validation predictions per epoch

**Metrics**:
- `stats/per_class_metrics_*.csv` - Per-emotion F1, precision, recall, support

### Artifacts (to W&B Artifacts Tab)

All of the above files are also saved as versioned artifacts:
- `{model}-checkpoint-{run_id}` - Model checkpoint artifact
- `{model}-test-predictions-{run_id}` - Test predictions artifact
- `{model}-val-predictions-{run_id}` - Validation predictions artifact
- `{model}-per-class-metrics-{run_id}` - Metrics artifact

## Accessing Your Data

### Web UI (Easiest)
1. Go to https://wandb.ai
2. Navigate to `GoEmotions_Classification` project
3. Click on your run
4. View metrics in Charts tab
5. Download files from Files tab

### Python API
```python
import wandb

api = wandb.Api()
run = api.run('username/GoEmotions_Classification/run_id')

# Download files
for file in run.files():
    if file.name.startswith('predictions/'):
        file.download(root='./downloads')
```

See [downloading_files.md](downloading_files.md) for detailed examples.

## Common Tasks

### Compare Multiple Runs
1. Go to project page
2. Select runs to compare
3. Click "Compare" button
4. View metrics side-by-side

### Download Best Checkpoint
```python
import wandb

api = wandb.Api()
runs = api.runs('username/GoEmotions_Classification')

# Find run with best validation AUC
best_run = max(runs, key=lambda r: r.summary.get('val/auc', 0))
print(f'Best run: {best_run.name} (AUC: {best_run.summary["val/auc"]})')

# Download checkpoint
for file in best_run.files():
    if 'distilbert' in file.name or 'roberta' in file.name:
        file.download(root='./best_checkpoint')
```

### Analyze Predictions
```python
import wandb
import pandas as pd

api = wandb.Api()
run = api.run('username/GoEmotions_Classification/run_id')

# Download test predictions
for file in run.files():
    if 'test_predictions' in file.name:
        file.download(root='./analysis')

# Load and analyze
df = pd.read_csv('./analysis/predictions/test_predictions_*.csv')
print(df.head())
```

## Integration Details

### Code Structure

**W&B utilities**: `src/training/wandb_utils.py`
- `init_wandb()` - Initialize W&B run
- `log_training_metrics()` - Log per-epoch metrics
- `log_evaluation_metrics()` - Log final test metrics
- `log_artifact_checkpoint()` - Upload checkpoint
- `log_artifact_predictions()` - Upload predictions
- `log_artifact_metrics()` - Upload metrics CSV
- `finish_wandb()` - Finalize run

**Training integration**: `src/training/train.py`
- W&B initialization at start of training
- Metric logging after each epoch
- Artifact upload after training completes

### Configuration

All W&B settings can be configured via command-line arguments:

```bash
python -m src.training.train \
    --wandb-project MyProject \
    --wandb-entity my-team \
    --run-name my-experiment \
    --no-wandb  # Disable W&B
```

## Troubleshooting

### Issue: Login required
```bash
wandb login
# Paste your API key from https://wandb.ai/authorize
```

### Issue: Offline mode
```bash
# Run in offline mode (sync later)
export WANDB_MODE=offline
python -m src.training.train ...

# Sync after run completes
wandb sync wandb/offline-run-*
```

### Issue: Disable W&B completely
```bash
python -m src.training.train --no-wandb ...
```

## Additional Resources

- W&B Documentation: https://docs.wandb.ai
- W&B Python API: https://docs.wandb.ai/ref/python
- GoEmotions Dataset: https://github.com/google-research/google-research/tree/master/goemotions
