# W&B Artifact Verification

This directory contains verification scripts for validating that all expected artifacts were successfully uploaded to Weights & Biases during training runs.

## Scripts

### verify_wandb_artifacts.py

Verifies that all expected artifacts (checkpoint, predictions, metrics) were uploaded to W&B for a training run.

**Usage:**
```bash
python tests/verification/verify_wandb_artifacts.py <run_path>
```

**Example:**
```bash
python tests/verification/verify_wandb_artifacts.py Cooper-Union/GoEmotions_Classification/a71b9ddo
```

**Expected Artifacts:**
- Model checkpoint files (pytorch_model.bin, config.json, metrics.json, tokenizer files)
- Final validation predictions CSV (val_epoch10_predictions)
- Test predictions CSV (test_predictions)
- Per-class metrics CSV (per_class_metrics)

**Output:**
- Prints verification results showing which artifacts were found
- Returns exit code 0 if all artifacts present, 1 if any are missing

## RoBERTa-Large Verification Results

Run ID: `a71b9ddo`
Run URL: https://wandb.ai/Cooper-Union/GoEmotions_Classification/runs/a71b9ddo

All expected artifacts verified successfully:
- Checkpoint: 1355.86 MB (pytorch_model.bin + config files)
- Validation predictions: 1.75 MB (epoch 10)
- Test predictions: 1.75 MB
- Per-class metrics: 0.00 MB

Total files uploaded: 26
Total size: 1379.81 MB
