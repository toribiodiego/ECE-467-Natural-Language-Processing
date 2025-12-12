# Model Performance

This document provides comprehensive performance metrics, comparisons, and analysis for all trained models in the GoEmotions emotion classification project.

## Table of Contents

1. [Overview](#overview)
2. [RoBERTa-Large Performance](#roberta-large-performance)
3. [DistilBERT Performance](#distilbert-performance)
4. [Model Comparison](#model-comparison)
5. [Per-Emotion Performance](#per-emotion-performance)
6. [Threshold Selection](#threshold-selection)
7. [Model Artifacts](#model-artifacts)

---

## Overview

This section will contain a summary of all trained models, their configurations, and key performance metrics.

**Models Trained:**
- RoBERTa-Large (355M parameters)
- DistilBERT (66M parameters)

**Evaluation Metrics:**
- Primary: AUC (Area Under ROC Curve)
- Secondary: Macro/Micro F1, Precision, Recall
- Per-class: F1 scores for all 28 emotions

**See Also:**
- `design_decisions.md` - Rationale for model selection and hyperparameters
- `ablation_studies/README.md` - Experiments informing final model choices

---

## RoBERTa-Large Performance

**Configuration:**
- Model: `roberta-large`
- Parameters: 355M
- Learning Rate: 2e-5
- Batch Size: 16
- Dropout: 0.1
- Epochs: 10 (best at epoch 1)
- Max Sequence Length: 128

**Performance Metrics:**
- **Test AUC (micro): 0.9045**
- **Test AUC (macro): 0.8294**
- Macro F1: 0.1600
- Micro F1: 0.4001
- Macro Precision: 0.2691
- Micro Precision: 0.7278
- Macro Recall: 0.1367
- Micro Recall: 0.2759
- **Best Validation AUC: 0.9038** (epoch 1)

**Training Details:**
- Training Duration: 2.05 hours (7,363 seconds)
- Hardware: NVIDIA A100-SXM4-80GB (1x GPU, 80GB VRAM)
- Platform: Linux (CUDA 12.x)
- W&B Run: [roberta-large-12-12-2025-190654](https://wandb.ai/Cooper-Union/GoEmotions_Classification/runs/a71b9ddo)
- Run ID: `a71b9ddo`
- Checkpoint: `artifacts/models/roberta/roberta-large-20251212-211010/`

**Key Observations:**
- Best performance achieved at epoch 1, suggesting early convergence
- High AUC scores (0.9045 micro, 0.8294 macro) indicate excellent ranking ability
- Lower F1 scores indicate conservative prediction threshold (default 0.5 may not be optimal)
- High precision (0.7278 micro) but lower recall (0.2759 micro) suggests model is conservative
- Validation AUC slightly declined after epoch 1, indicating potential overfitting

---

## DistilBERT Performance

**Configuration:**
- Model: `distilbert-base-uncased`
- Parameters: 66M
- Learning Rate: 3e-5
- Batch Size: 32
- Dropout: 0.1
- Epochs: 4
- Max Sequence Length: 128

**Performance Metrics:**
- Test AUC: [TO BE ADDED]
- Macro F1: [TO BE ADDED]
- Micro F1: [TO BE ADDED]
- Precision: [TO BE ADDED]
- Recall: [TO BE ADDED]

**Training Details:**
- Training Duration: [TO BE ADDED]
- Hardware: [TO BE ADDED]
- W&B Run: [TO BE ADDED]
- Checkpoint: [TO BE ADDED]

---

## Model Comparison

| Metric     | RoBERTa-Large | DistilBERT | Δ     | % Difference |
|------------|---------------|------------|-------|--------------|
| Test AUC   | [TBA]         | [TBA]      | [TBA] | [TBA]        |
| Macro F1   | [TBA]         | [TBA]      | [TBA] | [TBA]        |
| Micro F1   | [TBA]         | [TBA]      | [TBA] | [TBA]        |
| Parameters | 355M          | 66M        | -289M | -81%         |
| Train Time | [TBA]         | [TBA]      | [TBA] | [TBA]        |

**Key Findings:**
- [TO BE ADDED after training]

**Efficiency vs Performance Trade-off:**
- [TO BE ADDED - analysis of whether DistilBERT's efficiency gains justify performance loss]

**See:** `design_decisions.md#model-selection` for selection rationale

---

## Per-Emotion Performance

### Top 5 Best Performing Emotions

| Rank | Emotion    | F1 Score | Precision | Recall | Support |
|------|------------|----------|-----------|--------|---------|
| 1    | [TBA]      | [TBA]    | [TBA]     | [TBA]  | [TBA]   |
| 2    | [TBA]      | [TBA]    | [TBA]     | [TBA]  | [TBA]   |
| 3    | [TBA]      | [TBA]    | [TBA]     | [TBA]  | [TBA]   |
| 4    | [TBA]      | [TBA]    | [TBA]     | [TBA]  | [TBA]   |
| 5    | [TBA]      | [TBA]    | [TBA]     | [TBA]  | [TBA]   |

### Bottom 5 Worst Performing Emotions

| Rank | Emotion    | F1 Score | Precision | Recall | Support |
|------|------------|----------|-----------|--------|---------|
| 24   | [TBA]      | [TBA]    | [TBA]     | [TBA]  | [TBA]   |
| 25   | [TBA]      | [TBA]    | [TBA]     | [TBA]  | [TBA]   |
| 26   | [TBA]      | [TBA]    | [TBA]     | [TBA]  | [TBA]   |
| 27   | [TBA]      | [TBA]    | [TBA]     | [TBA]  | [TBA]   |
| 28   | [TBA]      | [TBA]    | [TBA]     | [TBA]  | [TBA]   |

### Full Per-Emotion Breakdown

**Data Location:** `output/stats/per_emotion_f1_scores.csv`

**W&B Artifact:** [TO BE ADDED]

**Analysis:**
- [TO BE ADDED - correlation between performance and sample size]
- [TO BE ADDED - impact of multi-label complexity on performance]

**See:** `dataset_analysis.md#per-emotion-statistics` for sample distribution context

---

## Threshold Selection

### Strategy Comparison

Different threshold strategies evaluated:

1. **Global Threshold (0.5):**
   - Metrics: [TO BE ADDED]
   - Pros/Cons: [TO BE ADDED]

2. **Per-Label Threshold (Optimized):**
   - Metrics: [TO BE ADDED]
   - Thresholds per emotion: [TO BE ADDED]
   - Pros/Cons: [TO BE ADDED]

3. **Top-K Selection:**
   - Metrics: [TO BE ADDED]
   - Pros/Cons: [TO BE ADDED]

### Selected Strategy

**Final Choice:** [TO BE ADDED]

**Rationale:** [TO BE ADDED]

**Performance Impact:** [TO BE ADDED]

**See:** `ablation_studies/README.md#threshold-strategy-ablation` for detailed ablation results

---

## Model Artifacts

### RoBERTa-Large

**Local Checkpoint:**
- Location: `artifacts/models/roberta/roberta-large-20251212-211010/`
- Files:
  - `pytorch_model.bin` (1.36 GB) - Model weights
  - `config.json` (2.3 KB) - Model configuration
  - `metrics.json` (8.4 KB) - Training metrics and per-class results
  - `vocab.json`, `merges.txt`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` - Tokenizer files

**W&B Artifacts:**
- Run URL: https://wandb.ai/Cooper-Union/GoEmotions_Classification/runs/a71b9ddo
- Run ID: `a71b9ddo`
- Validation Predictions: `artifacts/predictions/val_epoch10_predictions_roberta-large_20251212-210940.csv`
- Test Predictions: `artifacts/predictions/test_predictions_roberta-large_20251212-211009.csv`
- Per-Class Metrics: `artifacts/stats/per_class_metrics_roberta-large_20251212-211010.csv`

**Download Instructions:**
```bash
# Download checkpoint, predictions, and metrics
python scripts/download_wandb_checkpoint.py Cooper-Union/GoEmotions_Classification/a71b9ddo
```

**Retrieval Instructions:** See `docs/wandb_checkpoint_download.md` for detailed download guide

### DistilBERT

**Local Checkpoint:**
- Location: `artifacts/models/distilbert-base-[timestamp]/`
- Files: `config.json`, `pytorch_model.bin`, `tokenizer files`

**W&B Artifacts:**
- Model Checkpoint: [TO BE ADDED]
- Validation Predictions: [TO BE ADDED]
- Test Predictions: [TO BE ADDED]
- Per-Class Metrics: [TO BE ADDED]

**Retrieval Instructions:** See `w_and_b_guide.md#artifact-retrieval`

---

**Last Updated:** December 2024

**Maintainer:** ECE-467 Final Project Team
