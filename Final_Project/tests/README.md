# Tests Directory

This directory contains test scripts for validating code and artifacts.

## Test Scripts

### `validate_artifacts.py`

Validates that saved prediction CSVs, per-class metrics, and figures are properly formatted and can be loaded correctly.

**Usage:**
```bash
# Run validation
python -m tests.validate_artifacts

# Run with verbose output
python -m tests.validate_artifacts --verbose
```

**What it validates:**
- Prediction CSVs have required columns (text, true_labels, pred_labels, pred_prob_*)
- Per-class metrics CSVs have correct schema and value ranges
- JSON metrics files are valid and parseable
- Figure files exist and have reasonable sizes
- All probability values are in [0, 1] range
- All metric values (precision, recall, F1) are in [0, 1] range

**Artifact Paths:**

Predictions:
- `artifacts/predictions/test_predictions_{model}_{timestamp}.csv`
- `artifacts/predictions/val_epoch{N}_predictions_{model}_{timestamp}.csv`

Per-Class Metrics:
- `artifacts/stats/per_class_metrics_{model}_{timestamp}.csv`
- `artifacts/stats/per_emotion_scores.csv`

Test Metrics (JSON):
- `artifacts/stats/test_metrics_{model}.json`

Figures:
- `output/figures/*.png`

### `test_experiments.py`

Contains experimental test code and utilities.

### `training/`

Contains tests for training utilities and data loaders.

### `verification/`

Contains verification scripts for W&B artifacts and model outputs.
