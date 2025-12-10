# Weights & Biases Guide

This guide covers all aspects of W&B (Weights & Biases) integration for the GoEmotions project, including setup, logging requirements, and artifact retrieval.

## Table of Contents

1. [Setup](#setup)
2. [Project Configuration](#project-configuration)
3. [Logging Requirements](#logging-requirements)
4. [Artifact Management](#artifact-management)
5. [Retrieving Results](#retrieving-results)
6. [Troubleshooting](#troubleshooting)

---

## Setup

### Account Creation

1. Create a free W&B account at https://wandb.ai/
2. Verify your email address
3. Access your API key from https://wandb.ai/authorize

### Local Configuration

```bash
# Authenticate with W&B
wandb login

# Enter your API key when prompted
# Or set via environment variable:
export WANDB_API_KEY=your_api_key_here
```

### Verify Setup

```bash
python -c "import wandb; wandb.init(project='test'); wandb.finish()"
```

If successful, you should see a run appear in your W&B dashboard.

---

## Project Configuration

### Project Details

**Project Name:** `goemotions-emotion-classification`

**Entity:** [Your W&B username]

**Run Naming Convention:**
```
{model_name}-{timestamp}
Examples:
- roberta-large-20241210-143022
- distilbert-base-20241210-154533
```

**Tags:**
- Model type: `roberta-large`, `distilbert-base`
- Phase: `training`, `evaluation`, `ablation`
- Status: `baseline`, `experimental`, `final`

### Configuration Schema

All runs must log the following configuration parameters:

```python
config = {
    # Model Configuration
    'model_name': str,          # e.g., 'roberta-large'
    'model_params': int,        # Number of parameters
    'num_labels': int,          # 28 for GoEmotions

    # Training Hyperparameters
    'learning_rate': float,     # e.g., 2e-5
    'batch_size': int,          # e.g., 16
    'num_epochs': int,          # e.g., 4
    'dropout': float,           # e.g., 0.1
    'max_seq_length': int,      # e.g., 128
    'warmup_steps': int,        # e.g., 500
    'weight_decay': float,      # e.g., 0.01
    'random_seed': int,         # e.g., 42

    # Dataset Configuration
    'dataset_name': str,        # 'go_emotions'
    'dataset_version': str,     # HuggingFace dataset version
    'train_samples': int,       # 43410
    'val_samples': int,         # 5426
    'test_samples': int,        # 5427

    # System Information
    'gpu_type': str,            # e.g., 'NVIDIA A100'
    'cuda_version': str,        # e.g., '11.8'
    'pytorch_version': str,     # e.g., '2.0.1'
    'transformers_version': str # e.g., '4.30.2'
}
```

---

## Logging Requirements

### Training Metrics (Per Epoch)

Log these metrics at the end of each training epoch:

```python
wandb.log({
    # Loss
    'train/loss': float,
    'train/loss_std': float,      # Standard deviation across batches
    'val/loss': float,

    # Performance Metrics
    'train/auc': float,           # AUC on train set (sampled)
    'val/auc': float,             # AUC on validation set

    # Learning Rate
    'train/learning_rate': float, # Current LR

    # Gradient Statistics
    'train/grad_norm': float,     # Gradient norm

    # Timing
    'train/epoch_time': float,    # Seconds per epoch
    'train/samples_per_sec': float
}, step=epoch)
```

### Evaluation Metrics (After Training)

Log these metrics after final evaluation on test set:

```python
wandb.log({
    # Overall Metrics
    'test/auc': float,
    'test/macro_f1': float,
    'test/micro_f1': float,
    'test/macro_precision': float,
    'test/macro_recall': float,
    'test/micro_precision': float,
    'test/micro_recall': float,

    # Per-Class Metrics (for each of 28 emotions)
    f'test/f1_{emotion_name}': float,
    f'test/precision_{emotion_name}': float,
    f'test/recall_{emotion_name}': float,
    f'test/support_{emotion_name}': int,

    # Training Summary
    'total_training_time': float,  # Total seconds
    'best_epoch': int,
    'final_epoch': int
})
```

### System Information

Log system information at run initialization:

```python
import torch
import platform
import transformers

wandb.config.update({
    'system/platform': platform.platform(),
    'system/python_version': platform.python_version(),
    'system/pytorch_version': torch.__version__,
    'system/cuda_available': torch.cuda.is_available(),
    'system/cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
    'system/gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    'system/gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'system/transformers_version': transformers.__version__
})
```

---

## Artifact Management

### Required Artifacts

All training runs must upload the following artifacts:

#### 1. Model Checkpoint

**Type:** `model`

**Name:** `{model_name}-checkpoint-{run_id}`

**Contents:**
- `config.json` - Model configuration
- `pytorch_model.bin` - Model weights
- `tokenizer_config.json` - Tokenizer configuration
- `vocab.txt` / `merges.txt` - Tokenizer vocabulary
- `special_tokens_map.json` - Special tokens

**Metadata:**
```python
metadata = {
    'model_name': str,
    'final_test_auc': float,
    'final_test_f1': float,
    'best_epoch': int,
    'training_time_hours': float,
    'hyperparameters': dict,
    'random_seed': int
}
```

#### 2. Validation Predictions

**Type:** `dataset`

**Name:** `{model_name}-val-predictions-{run_id}`

**Format:** CSV with columns:
- `text`: str - Input text
- `true_labels`: str - Comma-separated true emotion labels
- `pred_labels`: str - Comma-separated predicted labels (at threshold)
- `pred_probs_{emotion}`: float - Predicted probability for each emotion (28 columns)

#### 3. Test Predictions

**Type:** `dataset`

**Name:** `{model_name}-test-predictions-{run_id}`

**Format:** Same as validation predictions

#### 4. Per-Class Metrics

**Type:** `dataset`

**Name:** `{model_name}-per-class-metrics-{run_id}`

**Format:** CSV with columns:
- `emotion`: str - Emotion name
- `f1_score`: float
- `precision`: float
- `recall`: float
- `support`: int - Number of samples
- `tp`: int - True positives
- `fp`: int - False positives
- `fn`: int - False negatives
- `tn`: int - True negatives

### Uploading Artifacts

```python
# Example: Upload model checkpoint
artifact = wandb.Artifact(
    name=f'roberta-large-checkpoint-{wandb.run.id}',
    type='model',
    description='Best RoBERTa-Large checkpoint based on validation AUC',
    metadata={
        'model_name': 'roberta-large',
        'final_test_auc': 0.957,
        'best_epoch': 3,
        'training_time_hours': 2.5
    }
)
artifact.add_dir('artifacts/models/roberta-large-best/')
wandb.log_artifact(artifact)

# Example: Upload predictions CSV
artifact = wandb.Artifact(
    name=f'roberta-large-test-predictions-{wandb.run.id}',
    type='dataset',
    description='Test set predictions with probabilities'
)
artifact.add_file('output/predictions/test_predictions.csv')
wandb.log_artifact(artifact)
```

---

## Retrieving Results

### Accessing Runs

**View all runs:**
https://wandb.ai/[username]/goemotions-emotion-classification/runs

**Filter by tag:**
- Click "Add filter" → "Tags" → Select tag (e.g., `roberta-large`, `final`)

**Compare runs:**
- Select multiple runs (checkboxes)
- Click "Compare" button

### Downloading Artifacts

#### Via Web Interface

1. Navigate to run page
2. Click "Artifacts" tab
3. Click on artifact name
4. Click "Files" → Download individual files or "Download" for entire artifact

#### Via Python API

```python
import wandb

# Initialize API
api = wandb.Api()

# Get specific run
run = api.run('username/goemotions-emotion-classification/run_id')

# List artifacts
for artifact in run.logged_artifacts():
    print(f"{artifact.name} ({artifact.type})")

# Download specific artifact
artifact = api.artifact('username/goemotions-emotion-classification/roberta-large-checkpoint-abc123:v0')
artifact_dir = artifact.download()
print(f"Downloaded to: {artifact_dir}")

# Load model from artifact
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
```

#### Via Command Line

```bash
# Download artifact
wandb artifact get username/goemotions-emotion-classification/roberta-large-checkpoint-abc123:v0 --root ./artifacts/

# List artifact versions
wandb artifact list username/goemotions-emotion-classification --type model
```

### Retrieving Metrics

```python
import wandb
import pandas as pd

api = wandb.Api()

# Get run
run = api.run('username/goemotions-emotion-classification/run_id')

# Get config
print(run.config)

# Get summary metrics
print(run.summary)

# Get full metrics history
history = run.history()
df = pd.DataFrame(history)
print(df[['epoch', 'train/loss', 'val/auc']])

# Get specific metric
test_auc = run.summary['test/auc']
print(f"Test AUC: {test_auc:.4f}")
```

---

## Troubleshooting

### Common Issues

#### 1. Authentication Failed

**Error:** `wandb: ERROR Error authenticating`

**Solution:**
```bash
# Re-login
wandb login --relogin

# Or set API key directly
export WANDB_API_KEY=your_key_here
```

#### 2. Artifact Upload Timeout

**Error:** `wandb: ERROR Failed to upload artifact`

**Solution:**
```bash
# Increase timeout
export WANDB_HTTP_TIMEOUT=300

# Or disable artifact upload temporarily
export WANDB_MODE=offline
```

#### 3. Run Not Appearing in Dashboard

**Solution:**
```python
# Ensure wandb.finish() is called
wandb.finish()

# Or use context manager
with wandb.init(project='goemotions-emotion-classification') as run:
    # Training code here
    pass  # wandb.finish() called automatically
```

#### 4. Large Artifact Upload Failure

**Solution:**
```python
# Upload in chunks
artifact.add_file('large_file.bin', is_tmp=False)

# Or compress first
import shutil
shutil.make_archive('checkpoint', 'zip', 'artifacts/models/roberta-large-best/')
artifact.add_file('checkpoint.zip')
```

### Disabling W&B for Testing

```bash
# Disable W&B entirely (for local testing)
export WANDB_MODE=disabled

# Or use offline mode (log locally, sync later)
export WANDB_MODE=offline
wandb sync wandb/latest-run  # Sync later
```

```python
# Or in code
wandb.init(mode='disabled')  # For testing
wandb.init(mode='offline')   # For local logging
```

---

## Best Practices

1. **Always tag runs appropriately** - Makes filtering easier
2. **Use descriptive run names** - Include model and timestamp
3. **Log system info** - Helps reproduce results
4. **Upload all artifacts** - Checkpoints, predictions, metrics
5. **Add metadata to artifacts** - Final metrics, hyperparameters, etc.
6. **Test artifact retrieval** - Ensure you can download and reload models
7. **Use offline mode for testing** - Avoid cluttering dashboard during development

---

**See Also:**
- `replication.md` - How to run training with W&B enabled
- `model_performance.md` - W&B run URLs for final models
- `ablation_studies/README.md` - W&B artifacts for ablation experiments

---

**Last Updated:** December 2024

**Maintainer:** ECE-467 Final Project Team
