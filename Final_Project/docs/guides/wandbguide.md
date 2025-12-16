# Weights & Biases Integration Guide

This guide covers W&B integration for experiment tracking and how to access training artifacts both from W&B cloud and local repository paths.

## Overview

This project uses Weights & Biases (W&B) for:
- Experiment tracking and metrics logging
- Model checkpoint storage and versioning
- Artifact management (predictions, metrics, models)
- Visualization and comparison of runs

However, **local artifact access does not require W&B**. After training, all artifacts are saved locally and can be accessed directly from repository paths.

## Local Artifact Access (No W&B Required)

### Generated During Training

When you run training, artifacts are automatically saved to local paths:

**Model Checkpoints:**
```
artifacts/models/{model}/{model}-{timestamp}/
├── pytorch_model.bin
├── config.json
├── metrics.json
└── tokenizer files
```

**Predictions:**
```
artifacts/predictions/
├── test_predictions_{model}_{timestamp}.csv
└── val_epoch{N}_predictions_{model}_{timestamp}.csv
```

**Metrics:**
```
artifacts/stats/
├── per_class_metrics_{model}_{timestamp}.csv
├── test_metrics_{model}.json
└── metric_summary.csv
```

**Figures:**
```
output/figures/
├── 00_class_distribution.png
├── 07_per_emotion_f1_scores.png
└── ... (all visualization PNGs)
```

### Accessing Local Artifacts

You can use local artifacts directly without W&B:

```bash
# Export predictions from a local checkpoint
python -m src.training.export_predictions \
  --checkpoint artifacts/models/roberta/roberta-large-20251212-211010 \
  --output artifacts/predictions/

# Export per-class metrics from a local checkpoint
python -m src.analysis.export_per_class_metrics \
  --checkpoint artifacts/models/roberta/roberta-large-20251212-211010 \
  --output artifacts/stats/per_class_metrics.csv

# Validate artifact integrity
python -m tests.validate_artifacts
```

### When to Use Local vs W&B

**Use Local Artifacts When:**
- Running analysis scripts on already-trained models
- You have the checkpoint files from training
- Working offline or without W&B credentials
- Regenerating predictions with different thresholds
- Quick validation and testing

**Use W&B When:**
- Downloading artifacts from previous training runs
- Accessing training from different machines
- Comparing multiple experiment runs
- Sharing results with team members
- Long-term artifact storage and versioning

## Downloading from W&B (Optional)

If you need to download artifacts that were uploaded to W&B:

### Setup

```bash
# Install W&B
pip install wandb

# Login (one-time setup)
wandb login
```

### Download Checkpoint

```python
import wandb

# Initialize API
api = wandb.Api()

# Get run
run = api.run("your-entity/goemotions-emotion-classification/run-id")

# Download checkpoint artifact
artifact = run.use_artifact('roberta-large-checkpoint-a71b9ddo:latest')
artifact_dir = artifact.download('./artifacts/models/roberta/')
```

### Download Predictions

```python
# Download prediction files
for file in run.files():
    if file.name.startswith('predictions/'):
        file.download(root='./artifacts/')
```

### Download Metrics

```python
# Download stats files
for file in run.files():
    if file.name.startswith('stats/'):
        file.download(root='./artifacts/')
```

## W&B Artifact Organization

If artifacts were uploaded to W&B, they follow this structure:

**Checkpoint Artifacts:**
- Name: `{model}-checkpoint-{run_id}`
- Contains: Model weights, config, tokenizer, metrics.json

**Prediction Artifacts:**
- Name: `{model}-test-predictions-{run_id}`
- Contains: Test set prediction CSVs
- Name: `{model}-val-predictions-{run_id}`
- Contains: Validation prediction CSVs

**Metrics Artifacts:**
- Name: `{model}-per-class-metrics-{run_id}`
- Contains: Per-class precision/recall/F1 CSVs

## Training with W&B

Enable W&B logging during training:

```bash
# With W&B (uploads artifacts to cloud)
python -m src.training.train \
  --model roberta-large \
  --lr 2e-5 \
  --batch-size 16 \
  --epochs 10 \
  --wandb-project goemotions-emotion-classification

# Without W&B (local only)
python -m src.training.train \
  --model roberta-large \
  --lr 2e-5 \
  --batch-size 16 \
  --epochs 10 \
  --no-wandb
```

When W&B is enabled, artifacts are:
1. Saved locally to `artifacts/` and `output/`
2. Uploaded to W&B cloud for remote access
3. Versioned and tracked in the W&B web interface

When W&B is disabled (`--no-wandb`):
1. Artifacts are only saved locally
2. No upload or cloud storage
3. W&B credentials not required

## Common Workflows

### Workflow 1: Train Locally, Analyze Locally (No W&B)

```bash
# Train without W&B
python -m src.training.train --model distilbert-base --no-wandb

# Artifacts saved to local paths
# artifacts/models/distilbert/distilbert-base-{timestamp}/
# artifacts/predictions/
# artifacts/stats/

# Analyze using local paths
python -m src.analysis.per_emotion_metrics \
  --predictions artifacts/predictions/test_predictions_distilbert-base_*.csv \
  --output artifacts/stats/per_emotion_scores.csv
```

### Workflow 2: Train with W&B, Download Later

```bash
# Train with W&B (on remote GPU)
python -m src.training.train --model roberta-large

# Later, download artifacts from W&B (on local machine)
python scripts/download_wandb_checkpoint.py --run-id a71b9ddo

# Use downloaded artifacts
python -m src.analysis.export_per_class_metrics \
  --checkpoint artifacts/models/roberta/roberta-large-20251212-211010 \
  --output artifacts/stats/per_class_metrics.csv
```

### Workflow 3: Hybrid (W&B for Tracking, Local for Analysis)

```bash
# Train with W&B for experiment tracking
python -m src.training.train --model distilbert-base

# Artifacts automatically saved locally during training
# Use local artifacts immediately for analysis (no download needed)
python -m src.training.export_predictions \
  --checkpoint artifacts/models/distilbert/distilbert-base-{timestamp} \
  --output artifacts/predictions/
```

## Artifact Path Reference

For downstream scripts, use these relative paths:

```python
# Predictions
prediction_path = "artifacts/predictions/test_predictions_roberta-large_20251212-211009.csv"

# Per-class metrics
metrics_path = "artifacts/stats/per_class_metrics_roberta-large_20251212-211010.csv"

# Test metrics (JSON)
json_path = "artifacts/stats/test_metrics_roberta-large.json"

# Figures
figure_path = "output/figures/07_per_emotion_f1_scores.png"

# Checkpoint
checkpoint_path = "artifacts/models/roberta/roberta-large-20251212-211010"
```

## Validating Local Artifacts

Before running downstream analysis, validate that local artifacts are properly formatted:

```bash
# Run validation
python -m tests.validate_artifacts

# With verbose output
python -m tests.validate_artifacts --verbose
```

This checks:
- Prediction CSVs have correct schema
- Per-class metrics are properly formatted
- JSON files are valid
- Figures exist and have reasonable sizes
- All values are in expected ranges

## Troubleshooting

### "Artifact not found" when using local paths

**Problem:** Script can't find local artifact files

**Solution:**
- Ensure training completed successfully
- Check `artifacts/predictions/` and `artifacts/stats/` directories exist
- Use `python -m tests.validate_artifacts` to verify artifact integrity
- Regenerate artifacts if needed using export scripts

### W&B authentication errors

**Problem:** Can't download from W&B

**Solution:**
- Run `wandb login` and enter your API key
- Check you have access to the project
- Verify run ID is correct
- Consider using local artifacts instead if available

### Missing prediction files

**Problem:** Predictions not saved during training

**Solution:**
- Regenerate from checkpoint:
```bash
python -m src.training.export_predictions \
  --checkpoint artifacts/models/{model}/{model}-{timestamp} \
  --output artifacts/predictions/
```

## Additional Resources

For more detailed W&B documentation, see:
- `docs/tools/wandb/README.md` - W&B overview
- `docs/tools/wandb/downloading_files.md` - Download methods
- `docs/tools/wandb/file_organization.md` - W&B file structure

For local artifact usage, see:
- `docs/guides/replication.md` - Training and export scripts
- `artifacts/README.md` - Local artifact organization
- `tests/README.md` - Validation scripts
