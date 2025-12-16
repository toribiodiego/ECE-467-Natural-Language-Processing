# Artifacts Directory

This directory contains large binary files that should not be committed to version control.

## Subdirectories

- `models/` - Trained model checkpoints (.pt, .pth, .ckpt files)
- `predictions/` - Model prediction CSVs (validation and test sets)
- `stats/` - Per-class metrics, evaluation statistics, and analysis results

## Usage

### Saving Model Checkpoints

Training scripts should save model checkpoints to `artifacts/models/`:

```python
import torch
import os

model_path = "artifacts/models/roberta-large-best.pt"
torch.save(model.state_dict(), model_path)
```

### Loading Model Checkpoints

Analysis scripts can load checkpoints from this directory:

```python
import torch

model_path = "artifacts/models/roberta-large-best.pt"
model.load_state_dict(torch.load(model_path))
```

### Generating Predictions

Export predictions from checkpoints using the export script:

```bash
python -m src.training.export_predictions \
  --checkpoint artifacts/models/roberta/roberta-large-20251212-211010 \
  --output artifacts/predictions/
```

This creates timestamped CSV files with predictions for validation and test sets.

### Exporting Per-Class Metrics

Export per-class metrics (precision, recall, F1) from checkpoints:

```bash
python -m src.analysis.export_per_class_metrics \
  --checkpoint artifacts/models/roberta/roberta-large-20251212-211010 \
  --output artifacts/stats/per_class_metrics.csv
```

This generates a ranked CSV showing model performance for each emotion.

## Important Notes

- All files in this directory are gitignored to avoid committing large binary files
- Model checkpoints can be several hundred MB to multiple GB
- Download pre-trained models from cloud storage or retrain using notebooks
- The `.gitkeep` files ensure the directory structure is tracked even when empty
