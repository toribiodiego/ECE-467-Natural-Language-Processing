# Artifacts Directory

This directory contains large binary files that should not be committed to version control.

## Subdirectories

- `models/` - Trained model checkpoints (.pt, .pth, .ckpt files)

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

## Important Notes

- All files in this directory are gitignored to avoid committing large binary files
- Model checkpoints can be several hundred MB to multiple GB
- Download pre-trained models from cloud storage or retrain using notebooks
- The `.gitkeep` files ensure the directory structure is tracked even when empty
