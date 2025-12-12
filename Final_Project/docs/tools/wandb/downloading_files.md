# Downloading Files from W&B

This guide covers all the ways to download your training outputs from Weights & Biases.

## Table of Contents

- [Web UI Download](#web-ui-download)
- [Python API Download](#python-api-download)
- [Complete Download Scripts](#complete-download-scripts)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting](#troubleshooting)

## Web UI Download

The easiest way to download files for quick access.

### Step 1: Navigate to Your Run

1. Go to https://wandb.ai
2. Navigate to `GoEmotions_Classification` project
3. Click on your run name

### Step 2: Download from Files Tab

1. Click the "Files" tab
2. Navigate to the directory you want:
   - `checkpoint/{model-timestamp}/` for model files
   - `predictions/` for prediction CSVs
   - `stats/` for metrics CSV
3. Click on a file to download it

### Step 3: Download from Artifacts Tab

1. Click the "Artifacts" tab
2. Select an artifact:
   - `{model}-checkpoint-{run_id}` for full checkpoint
   - `{model}-test-predictions-{run_id}` for test predictions
   - `{model}-per-class-metrics-{run_id}` for metrics
3. Click "Download" button

## Python API Download

For programmatic access and bulk downloads.

### Setup

```python
import wandb
from pathlib import Path

# Login (one-time setup)
wandb.login()

# Initialize API
api = wandb.Api()
```

### Download All Files from a Run

```python
def download_all_files(run_id, output_dir='./downloads'):
    """Download all files from a W&B run."""

    api = wandb.Api()
    run = api.run(f'Cooper-Union/GoEmotions_Classification/{run_id}')

    output_path = Path(output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    print(f'Downloading from run: {run.name}')
    print(f'Run URL: {run.url}\n')

    for file in run.files():
        print(f'Downloading {file.name}...')
        file.download(root=str(output_path), replace=True)

    print(f'\nAll files downloaded to {output_path}')

# Usage
download_all_files('p5io8s9z')
```

### Download Specific Files

```python
def download_predictions(run_id, output_dir='./predictions'):
    """Download only prediction CSVs."""

    api = wandb.Api()
    run = api.run(f'Cooper-Union/GoEmotions_Classification/{run_id}')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in run.files():
        if file.name.startswith('predictions/'):
            print(f'Downloading {file.name}...')
            file.download(root=str(output_path), replace=True)

    print(f'Predictions downloaded to {output_path}')

# Usage
download_predictions('p5io8s9z')
```

### Download Checkpoint Only

```python
def download_checkpoint(run_id, output_dir='./checkpoint'):
    """Download only the model checkpoint."""

    api = wandb.Api()
    run = api.run(f'Cooper-Union/GoEmotions_Classification/{run_id}')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in run.files():
        # Match checkpoint directory (distilbert-*, roberta-*, bert-*)
        if any(model in file.name for model in ['distilbert', 'roberta', 'bert']):
            if 'artifact/' not in file.name:  # Skip artifact manifests
                print(f'Downloading {file.name}...')
                file.download(root=str(output_path), replace=True)

    print(f'Checkpoint downloaded to {output_path}')

# Usage
download_checkpoint('p5io8s9z')
```

### Download via Artifacts

```python
def download_artifact(run_id, artifact_type='checkpoint'):
    """Download a specific artifact."""

    api = wandb.Api()
    run = api.run(f'Cooper-Union/GoEmotions_Classification/{run_id}')

    # Find artifact by type
    for artifact in run.logged_artifacts():
        if artifact_type in artifact.name:
            print(f'Downloading artifact: {artifact.name}')
            artifact_dir = artifact.download(root=f'./artifacts/{artifact.name}')
            print(f'Downloaded to {artifact_dir}')
            return artifact_dir

    print(f'No artifact found with type: {artifact_type}')

# Usage
download_artifact('p5io8s9z', 'checkpoint')
download_artifact('p5io8s9z', 'predictions')
download_artifact('p5io8s9z', 'metrics')
```

## Complete Download Scripts

### Script 1: Download Training Outputs Only

This script downloads only your important training outputs, skipping W&B metadata and artifact manifests.

```python
#!/usr/bin/env python3
"""Download important training outputs from a W&B run."""

import wandb
from pathlib import Path

def download_training_outputs(run_id, output_dir='./downloads'):
    """Download checkpoint, predictions, and metrics from a run."""

    api = wandb.Api()
    run = api.run(f'Cooper-Union/GoEmotions_Classification/{run_id}')

    output_path = Path(output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    print(f'Downloading from run: {run.name} ({run.id})')
    print(f'Run URL: {run.url}\n')

    # Define important file prefixes
    important_prefixes = [
        'predictions/',
        'stats/',
        'distilbert',
        'roberta',
        'bert'
    ]

    # Download only important files
    downloaded = []
    for file in run.files():
        # Skip artifact manifest files
        if 'artifact/' in file.name and 'wandb_manifest' in file.name:
            continue

        # Download important files
        if any(file.name.startswith(prefix) for prefix in important_prefixes):
            print(f'Downloading {file.name}...')
            file.download(root=str(output_path), replace=True)
            downloaded.append(file.name)

    print(f'\n{len(downloaded)} files downloaded to {output_path}')

    # Show directory structure
    print('\nDownloaded files:')
    for path in sorted(output_path.rglob('*')):
        if path.is_file():
            rel_path = path.relative_to(output_path)
            size = path.stat().st_size
            print(f'  {rel_path} ({size:,} bytes)')

if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        print('Usage: python download_training_outputs.py <run_id>')
        print('Example: python download_training_outputs.py p5io8s9z')
        sys.exit(1)

    download_training_outputs(sys.argv[1])
```

**Usage**:
```bash
python download_training_outputs.py p5io8s9z
```

### Script 2: Download Best Run from Project

This script finds the best run by validation AUC and downloads its outputs.

```python
#!/usr/bin/env python3
"""Download outputs from the best run in a project."""

import wandb
from pathlib import Path

def download_best_run(project='GoEmotions_Classification',
                      metric='val/auc',
                      output_dir='./best_run'):
    """Find and download the best run based on a metric."""

    api = wandb.Api()
    runs = api.runs(f'Cooper-Union/{project}')

    # Find best run
    best_run = None
    best_value = -float('inf')

    for run in runs:
        value = run.summary.get(metric, -float('inf'))
        if value > best_value:
            best_value = value
            best_run = run

    if best_run is None:
        print(f'No runs found in project: {project}')
        return

    print(f'Best run: {best_run.name} ({best_run.id})')
    print(f'{metric}: {best_value:.4f}')
    print(f'URL: {best_run.url}\n')

    # Download outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in best_run.files():
        if any(prefix in file.name for prefix in ['predictions/', 'stats/', 'distilbert', 'roberta', 'bert']):
            if 'artifact/' not in file.name:
                print(f'Downloading {file.name}...')
                file.download(root=str(output_path), replace=True)

    print(f'\nBest run outputs downloaded to {output_path}')

if __name__ == '__main__':
    download_best_run()
```

**Usage**:
```bash
python download_best_run.py
```

### Script 3: Bulk Download Multiple Runs

```python
#!/usr/bin/env python3
"""Download outputs from multiple runs."""

import wandb
from pathlib import Path

def download_multiple_runs(run_ids, output_dir='./downloads'):
    """Download outputs from multiple runs."""

    api = wandb.Api()

    for run_id in run_ids:
        try:
            run = api.run(f'Cooper-Union/GoEmotions_Classification/{run_id}')
            print(f'\nDownloading run: {run.name} ({run_id})')

            output_path = Path(output_dir) / run_id
            output_path.mkdir(parents=True, exist_ok=True)

            for file in run.files():
                if any(prefix in file.name for prefix in ['predictions/', 'stats/', 'distilbert', 'roberta']):
                    if 'artifact/' not in file.name:
                        file.download(root=str(output_path), replace=True)

            print(f'Downloaded to {output_path}')

        except Exception as e:
            print(f'Error downloading run {run_id}: {e}')

    print(f'\nAll runs downloaded to {output_dir}')

if __name__ == '__main__':
    # List of run IDs to download
    runs_to_download = [
        'p5io8s9z',
        '152sl67o',
        # Add more run IDs here
    ]

    download_multiple_runs(runs_to_download)
```

## Common Use Cases

### Use Case 1: Quick Error Analysis

```python
import wandb
import pandas as pd

# Download test predictions
api = wandb.Api()
run = api.run('Cooper-Union/GoEmotions_Classification/p5io8s9z')

for file in run.files():
    if 'test_predictions' in file.name:
        file.download(root='./analysis')

# Analyze
df = pd.read_csv('./analysis/predictions/test_predictions_*.csv')
errors = df[df['true_labels'] != df['pred_labels']]
print(f'Error rate: {len(errors) / len(df):.2%}')
print(errors[['text', 'true_labels', 'pred_labels']].head())
```

### Use Case 2: Load Checkpoint for Inference

```python
import wandb
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Download checkpoint
api = wandb.Api()
run = api.run('Cooper-Union/GoEmotions_Classification/p5io8s9z')

for file in run.files():
    if 'distilbert' in file.name and 'artifact/' not in file.name:
        file.download(root='./model')

# Load model
checkpoint_dir = './model/distilbert-12-11-2025-221314'
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

# Test
text = "I'm so excited about this!"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
print(outputs.logits)
```

### Use Case 3: Compare Predictions Across Runs

```python
import wandb
import pandas as pd

def get_predictions(run_id):
    """Download and load test predictions from a run."""
    api = wandb.Api()
    run = api.run(f'Cooper-Union/GoEmotions_Classification/{run_id}')

    for file in run.files():
        if 'test_predictions' in file.name:
            file.download(root=f'./compare/{run_id}')
            return pd.read_csv(f'./compare/{run_id}/predictions/test_predictions_*.csv')

# Compare two runs
df1 = get_predictions('run_id_1')
df2 = get_predictions('run_id_2')

# Find disagreements
disagree = df1[df1['pred_labels'] != df2['pred_labels']]
print(f'Predictions differ on {len(disagree)} / {len(df1)} examples')
```

## Troubleshooting

### Authentication Error

```bash
# Login to W&B
wandb login

# Or set API key programmatically
import os
os.environ['WANDB_API_KEY'] = 'your_api_key'
```

### Large File Download Timeout

```python
# Increase timeout
import wandb

api = wandb.Api(timeout=300)  # 5 minutes
```

### Resume Interrupted Download

```python
# Downloads are resumable - just run again
for file in run.files():
    file.download(root='./downloads', replace=True)  # replace=True overwrites
```

### Check Download Progress

```python
# For large downloads, see progress
for file in run.files():
    print(f'Downloading {file.name} ({file.size} bytes)...')
    file.download(root='./downloads', replace=True)
```

### Download Specific Version of Artifact

```python
# Download specific version
api = wandb.Api()
artifact = api.artifact('Cooper-Union/GoEmotions_Classification/checkpoint-p5io8s9z:v0')
artifact.download(root='./checkpoint')
```

## Additional Resources

- W&B Python API Reference: https://docs.wandb.ai/ref/python
- W&B Artifacts Guide: https://docs.wandb.ai/guides/artifacts
- File organization: [file_organization.md](file_organization.md)
