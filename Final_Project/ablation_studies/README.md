# Ablation Studies

This directory contains results and analysis from systematic ablation experiments designed to validate design decisions and understand model behavior.

## Table of Contents

1. [Overview](#overview)
2. [Quick Comparison](#quick-comparison)
3. [Individual Studies](#individual-studies)
4. [Artifact Retrieval](#artifact-retrieval)
5. [Cross-References](#cross-references)

---

## Overview

**Purpose:** Systematic experiments to understand:
- Impact of design decisions on model performance
- Sensitivity to hyperparameter choices
- Robustness to data perturbations
- Efficiency trade-offs

**Methodology:**
- Controlled experiments with single variable changes
- Consistent random seeds for reproducibility
- All experiments logged to W&B
- Results stored as W&B artifacts (some local CSVs gitignored)

**See:** `docs/design_decisions.md` for rationale behind ablation choices

---

## Quick Comparison

Summary of most impactful ablations (TO BE POPULATED after experiments):

| Ablation | Baseline | Variant | Δ AUC | Δ F1 | Key Finding |
|----------|----------|---------|-------|------|-------------|
| [TBA]    | [TBA]    | [TBA]   | [TBA] | [TBA]| [TBA]       |

**Most Impactful:**
- [TO BE ADDED]

**Least Impactful:**
- [TO BE ADDED]

---

## Individual Studies

### Neutral Label Inclusion

**Question:** Should we train with or without the neutral label?

**Hypothesis:** Neutral may improve calibration but could bias predictions toward neutral

**Experiment:**
- Baseline: Training with all 28 emotions (including neutral)
- Variant: Training with 27 emotions (excluding neutral)
- Controlled: Same hyperparameters, seed, model architecture

**Results:**
- With neutral: [TO BE ADDED]
- Without neutral: [TO BE ADDED]
- Metric delta: [TO BE ADDED]

**W&B Runs:**
- With neutral: [TO BE ADDED]
- Without neutral: [TO BE ADDED]

**Artifacts:**
- Metrics CSV: [TO BE ADDED]
- Predictions: [TO BE ADDED]

**Decision:** [TO BE ADDED]

**See:** `docs/design_decisions.md#neutral-emotion-handling` for context

---

### Loss Weighting

**Question:** Does class-weighted BCE or focal loss improve rare-label performance?

**Hypothesis:** Weighting should help rare emotions (grief, pride) without hurting common ones

**Experiment:**
- Baseline: Standard BCEWithLogitsLoss
- Variant 1: Class-weighted BCE (inverse frequency)
- Variant 2: Focal loss (gamma=2.0)

**Results:**
- Baseline: [TO BE ADDED]
- Class-weighted: [TO BE ADDED]
- Focal loss: [TO BE ADDED]

**Per-Class Impact:**
- Rare emotions (grief, pride, relief): [TO BE ADDED]
- Common emotions (neutral, admiration): [TO BE ADDED]

**W&B Runs:**
- [TO BE ADDED]

**Artifacts:**
- Per-class metrics: [TO BE ADDED]
- Confusion matrices: [TO BE ADDED]

**Decision:** [TO BE ADDED]

---

### Threshold Strategy

**Question:** What threshold strategy optimizes F1 scores?

**Hypothesis:** Per-label thresholds should outperform global threshold

**Experiment:**
- Strategy 1: Global threshold (0.5)
- Strategy 2: Per-label threshold (optimized on validation)
- Strategy 3: Top-k selection (k=1, k=2)

**Results:**
- Global (0.5): [TO BE ADDED]
- Per-label: [TO BE ADDED]
- Top-k (k=1): [TO BE ADDED]
- Top-k (k=2): [TO BE ADDED]

**Threshold Values (Per-Label):**
- [TO BE ADDED - CSV with optimal threshold per emotion]

**W&B Runs:**
- [TO BE ADDED]

**Artifacts:**
- Threshold values: [TO BE ADDED]
- Precision-recall curves: [TO BE ADDED]

**Decision:** [TO BE ADDED]

**See:** `docs/model_performance.md#threshold-selection` for final choice

---

### Sequence Length

**Question:** How does max sequence length affect performance and efficiency?

**Hypothesis:** 128 tokens covers most samples; 256/512 may help long texts

**Experiment:**
- Baseline: max_length=128
- Variant 1: max_length=256
- Variant 2: max_length=512

**Results:**
- 128: [TO BE ADDED]
- 256: [TO BE ADDED]
- 512: [TO BE ADDED]

**Token Coverage:**
- 128: ~95% of samples (no truncation)
- 256: ~99% of samples
- 512: ~100% of samples

**Efficiency Impact:**
- Training time: [TO BE ADDED]
- Memory usage: [TO BE ADDED]
- Inference latency: [TO BE ADDED]

**W&B Runs:**
- [TO BE ADDED]

**Decision:** [TO BE ADDED]

**See:** `docs/dataset_analysis.md` for text length distributions

---

### Text Robustness

**Question:** How robust is the model to noisy inputs?

**Hypothesis:** Model should be somewhat robust to typos but sensitive to punctuation removal

**Experiment:**
- Baseline: Clean test set
- Perturbation 1: Random typos (10% character swaps)
- Perturbation 2: Punctuation removed
- Perturbation 3: Emoji stripped

**Results:**
- Clean: [TO BE ADDED]
- With typos: [TO BE ADDED]
- No punctuation: [TO BE ADDED]
- No emoji: [TO BE ADDED]

**Metric Deltas:**
- [TO BE ADDED - table showing performance degradation]

**W&B Runs:**
- [TO BE ADDED]

**Artifacts:**
- Perturbed test sets: [TO BE ADDED]
- Per-class robustness: [TO BE ADDED]

**Findings:** [TO BE ADDED]

---

### Training Augmentation

**Question:** Does text augmentation during training improve generalization?

**Hypothesis:** Light augmentation may help rare emotions without hurting overall performance

**Experiment:**
- Baseline: No augmentation
- Variant: Synonym replacement + random insertion (prob=0.1)

**Results:**
- No augmentation: [TO BE ADDED]
- With augmentation: [TO BE ADDED]

**Impact on Rare Emotions:**
- [TO BE ADDED]

**W&B Runs:**
- [TO BE ADDED]

**Decision:** [TO BE ADDED]

---

### Tokenization Choices

**Question:** Cased vs uncased tokenization? How to handle URLs/emoji?

**Hypothesis:** Cased may preserve emotion signals; emoji should be kept

**Experiment:**
- Baseline: Uncased, URLs/emoji kept
- Variant 1: Cased
- Variant 2: URLs masked
- Variant 3: Emoji stripped

**Results:**
- [TO BE ADDED]

**W&B Runs:**
- [TO BE ADDED]

**Decision:** [TO BE ADDED]

**See:** `docs/design_decisions.md#tokenization` for final choice

---

## Artifact Retrieval

All ablation artifacts are stored in W&B. Local CSV files may be gitignored for large files.

### Downloading Artifacts

```bash
# List all ablation artifacts
wandb artifact list username/goemotions-emotion-classification --type dataset | grep ablation

# Download specific ablation results
wandb artifact get username/goemotions-emotion-classification/neutral-ablation-metrics:v0
```

### Using Python API

```python
import wandb

api = wandb.Api()

# Get ablation run
run = api.run('username/goemotions-emotion-classification/run_id')

# Download artifacts
for artifact in run.logged_artifacts():
    if 'ablation' in artifact.name:
        artifact.download()
```

**See:** `docs/w_and_b_guide.md#retrieving-results` for detailed instructions

---

## Cross-References

### To Other Documentation

- **`docs/design_decisions.md`** - Context for what we're ablating and why
- **`docs/dataset_analysis.md`** - Dataset characteristics relevant to ablations
- **`docs/model_performance.md`** - How ablations informed final model selection
- **`docs/w_and_b_guide.md`** - How to access ablation artifacts

### From Other Documentation

This ablation archive is referenced in:
- `docs/design_decisions.md` - Links to ablations validating decisions
- `docs/model_performance.md` - References threshold and loss ablations
- `README.md` - Navigation to ablation results

---

**Last Updated:** December 2024

**Maintainer:** ECE-467 Final Project Team
