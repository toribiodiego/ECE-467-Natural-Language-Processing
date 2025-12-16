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

*Primary Metric:*
- **AUC (micro)**: Threshold-agnostic overall ranking quality (model comparison standard)

*Secondary Metrics:*
- **AUC (macro)**: Fair treatment of all 28 emotions regardless of frequency
- **Micro F1**: Deployment performance at default threshold (0.5)

*Tertiary Metrics (diagnostic):*
- Macro/Micro Precision and Recall: Threshold sensitivity analysis
- Per-emotion F1 scores: Identify best/worst performing emotions

**See Also:**
- `design_decisions.md#evaluation-metrics` - Complete metric selection rationale
- `../output/ablation_studies/README.md` - Threshold optimization experiments

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

*Primary:*
- **Test AUC (micro): 0.9045** ← Primary metric for model comparison

*Secondary:*
- **Test AUC (macro): 0.8294** ← Fair emotion treatment (all 28 emotions weighted equally)
- **Micro F1: 0.4001** ← Deployment performance at threshold=0.5

*Tertiary (diagnostic):*
- Macro F1: 0.1600
- Macro Precision: 0.2691
- Micro Precision: 0.7278
- Macro Recall: 0.1367
- Micro Recall: 0.2759

*Training:*
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

*Primary:*
- **Test AUC (micro): 0.8800** ← Primary metric for model comparison

*Secondary:*
- **Test AUC (macro): 0.7443** ← Fair emotion treatment (all 28 emotions weighted equally)
- **Micro F1: 0.3516** ← Deployment performance at threshold=0.5

*Tertiary (diagnostic):*
- Macro F1: 0.0904
- Macro Precision: 0.1547
- Micro Precision: 0.7085
- Macro Recall: 0.0703
- Micro Recall: 0.2338

*Training:*
- **Best Validation AUC: 0.8790** (epoch 10)

**Training Details:**
- Training Duration: 0.31 hours (1,113 seconds)
- Hardware: NVIDIA A100-SXM4-80GB (1x GPU, 80GB VRAM)
- Platform: Linux (CUDA 12.x)
- W&B Run: [distilbert-12-12-2025-223914](https://wandb.ai/Cooper-Union/GoEmotions_Classification/runs/4ta2sol5)
- Run ID: `4ta2sol5`
- Checkpoint: `artifacts/models/distilbert/distilbert-base-20251212-225748/`

**Key Observations:**
- Best performance achieved at epoch 10, showing continued improvement throughout training
- AUC scores (0.8800 micro, 0.7443 macro) are competitive despite smaller model size
- Lower F1 scores (0.0904 macro, 0.3516 micro) indicate conservative prediction behavior similar to RoBERTa
- High precision (0.7085 micro) but lower recall (0.2338 micro) confirms conservative predictions
- 6.6x faster training than RoBERTa-Large (0.31 hrs vs 2.05 hrs)
- 5.4x fewer parameters (66M vs 355M) with only 2.5% AUC drop

---

## Model Comparison

### Performance Metrics

| Metric                     | RoBERTa-Large | DistilBERT | Δ       | % Difference |
|----------------------------|---------------|------------|---------|--------------|
| **AUC (micro)** ⭐         | **0.9045**    | **0.8800** | -0.0245 | **-2.7%**    |
| **AUC (macro)** 🎯         | **0.8294**    | **0.7443** | -0.0851 | **-10.3%**   |
| **Micro F1** 📊            | **0.4001**    | **0.3516** | -0.0485 | **-12.1%**   |
| Macro F1                   | 0.1600        | 0.0904     | -0.0696 | -43.5%       |
| Macro Precision            | 0.2691        | 0.1547     | -0.1144 | -42.5%       |
| Micro Precision            | 0.7278        | 0.7085     | -0.0193 | -2.7%        |
| Macro Recall               | 0.1367        | 0.0703     | -0.0664 | -48.6%       |
| Micro Recall               | 0.2759        | 0.2338     | -0.0421 | -15.3%       |

⭐ = Primary metric for model comparison
🎯 = Secondary metric (fair emotion treatment)
📊 = Secondary metric (deployment performance)

### Model Efficiency

| Metric                     | RoBERTa-Large | DistilBERT | Δ       | % Difference |
|----------------------------|---------------|------------|---------|--------------|
| Parameters                 | 355M          | 66M        | -289M   | -81.4%       |
| Train Time                 | 2.05 hrs      | 0.31 hrs   | -1.74   | -84.9%       |
| Best Epoch                 | 1             | 10         | +9      | +900%        |

**Key Findings:**
- **AUC Performance**: DistilBERT achieves 97.3% of RoBERTa's micro-AUC with only 18.6% of the parameters
- **Training Efficiency**: 6.6x faster training time (18.5 minutes vs 2.05 hours)
- **Memory Efficiency**: ~4x less GPU memory required due to smaller model size
- **F1 Trade-off**: Larger F1 gap (43.5% macro, 12.1% micro) suggests DistilBERT is more conservative in predictions
- **Recall Sensitivity**: DistilBERT shows greater recall drop (-48.6% macro) than precision drop (-2.7% micro)
- **Convergence**: DistilBERT required all 10 epochs vs RoBERTa's early convergence at epoch 1

**Efficiency vs Performance Trade-off:**

DistilBERT offers compelling efficiency gains for production deployment scenarios:

**When to use DistilBERT:**
- **CPU deployment**: 3-4x faster inference on CPU with minimal AUC loss (2.7%)
- **Resource-constrained environments**: 4x less memory footprint enables deployment on smaller GPUs
- **High-throughput applications**: Faster inference supports higher request volumes
- **Cost-sensitive scenarios**: Reduced training time (15% of RoBERTa's time) and inference costs

**When to use RoBERTa-Large:**
- **Maximum accuracy requirements**: Need to maximize AUC and minimize false negatives
- **Research and benchmarking**: When model performance is prioritized over efficiency
- **GPU-based deployment**: When GPU resources are available and inference latency is not critical
- **Rare emotion detection**: Better recall on low-frequency emotion classes

**Recommended Strategy:**
- Use **DistilBERT** for production API serving (CPU or small GPU instances)
- Use **RoBERTa-Large** for batch processing and offline analysis where accuracy is paramount
- The 2.7% AUC gap is acceptable for most real-world applications given the 6.6x speedup

**See:** `design_decisions.md#model-selection` for selection rationale

---

## Per-Emotion Performance

**Model:** RoBERTa-Large
**Classification Threshold:** 0.5 (default)
**Total Test Samples:** 5,427
**Total Emotions:** 28

### Top 5 Best Performing Emotions

| Rank | Emotion    | F1 Score | Precision | Recall | Support |
|------|------------|----------|-----------|--------|---------|
| 1    | gratitude  | 0.904    | 0.974     | 0.844  | 352     |
| 2    | love       | 0.740    | 0.763     | 0.718  | 238     |
| 3    | amusement  | 0.724    | 0.808     | 0.655  | 264     |
| 4    | admiration | 0.568    | 0.684     | 0.486  | 504     |
| 5    | neutral    | 0.497    | 0.690     | 0.389  | 1,787   |

**Observations:**
- **gratitude** achieves exceptional performance (F1=0.904) with high precision (0.974)
- Top emotions show strong precision (0.69-0.97) but varying recall (0.39-0.84)
- **neutral** has highest support (1,787 samples) but only ranks 5th in F1

### Bottom 5 Worst Performing Emotions

| Rank | Emotion        | F1 Score | Precision | Recall | Support |
|------|----------------|----------|-----------|--------|---------|
| 24   | disgust        | 0.000    | 0.000     | 0.000  | 123     |
| 25   | disapproval    | 0.000    | 0.000     | 0.000  | 267     |
| 26   | disappointment | 0.000    | 0.000     | 0.000  | 151     |
| 27   | desire         | 0.000    | 0.000     | 0.000  | 83      |
| 28   | fear           | 0.000    | 0.000     | 0.000  | 78      |

**Observations:**
- Model **completely fails** to predict these 5 emotions at threshold 0.5
- These emotions have moderate support (78-267 samples), so low performance is not solely due to data scarcity
- Other emotions with zero F1: caring, anger, sadness, remorse, relief, realization, pride, nervousness, grief, confusion, approval, excitement, embarrassment (18 total emotions with F1=0)

### Key Insights

**Conservative Prediction Behavior:**
The model exhibits extremely conservative predictions at the default threshold (0.5):
- **High precision, low recall** pattern across top emotions (e.g., neutral: P=0.690, R=0.389)
- Only **10 emotions** achieve F1 > 0.0, meaning 18 emotions are never predicted
- Total predicted positives: 2,399 vs actual positives: 6,329 (38% coverage)

**Why the model struggles with many emotions:**

1. **Threshold too conservative**: Default 0.5 threshold filters out most predictions
   - Per-emotion threshold optimization could significantly improve F1 scores
   - See threshold selection analysis below

2. **Class imbalance**: Bottom emotions don't necessarily have smallest support
   - disapproval (267 samples) has more support than amusement (264), yet achieves F1=0.0 vs 0.724
   - Suggests some emotions are intrinsically harder to distinguish

3. **Multi-label complexity**: Emotions that frequently co-occur may be harder to isolate
   - See `dataset_analysis.md#label-cooccurrence` for correlation patterns

**Recommended next steps:**
- Implement per-emotion threshold optimization to improve recall
- Analyze prediction probability distributions for zero-F1 emotions
- Consider focal loss or class-weighted BCE to address prediction imbalance

### Visualization

**Bar Chart:** `output/figures/10_emotion_f1.png`

The visualization shows all 28 emotions ranked by F1 score with color-coded performance:
- Green: High F1 (>0.5)
- Yellow: Medium F1 (0.1-0.5)
- Red: Low F1 (<0.1)

### Full Per-Emotion Breakdown

**Data Location:**
- `artifacts/stats/per_emotion_scores.csv` - Basic metrics
- `artifacts/stats/per_emotion_scores_annotated.csv` - With top/bottom annotations

**W&B Artifact:** Included in W&B run `a71b9ddo` (Cooper-Union/GoEmotions_Classification); see Model Artifacts below for contents (checkpoint, validation/test predictions, per-class metrics).

**See Also:**
- `design_decisions.md#evaluation-metrics` for metric selection rationale
- `dataset_analysis.md#per-emotion-statistics` for sample distribution context

---

## Threshold Selection

### Strategy Comparison

Reported metrics in this document use the **global threshold of 0.5** (training default). Per-label threshold searches and top-k selection are available in the code (`--threshold-strategy per_class` or `top_k`), and comparative results are tracked in `../output/ablation_studies/README.md` when available.

### Selected Strategy

**Final Choice:** Global threshold (0.5) for reported runs in this doc.  
**Rationale:** Matches training defaults, keeps results directly comparable to baseline runs, and favors precision. Per-label optimization is expected to improve recall; run a per-class search when optimizing for F1.  
**Performance Impact:** Baseline metrics above use this global threshold; per-label or top-k strategies should be validated and documented in ablation results before adoption.

**See:** `../output/ablation_studies/README.md#threshold-strategy-ablation` for detailed ablation results

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
- Location: `artifacts/models/distilbert/distilbert-base-20251212-225748/`
- Files:
  - `pytorch_model.bin` (253.28 MB) - Model weights
  - `config.json` (0.64 KB) - Model configuration
  - `metrics.json` (8.2 KB) - Training metrics and per-class results
  - `vocab.txt`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` - Tokenizer files

**W&B Artifacts:**
- Run URL: https://wandb.ai/Cooper-Union/GoEmotions_Classification/runs/4ta2sol5
- Run ID: `4ta2sol5`
- Validation Predictions: `artifacts/predictions/val_epoch10_predictions_distilbert-base_20251212-225742.csv`
- Test Predictions: `artifacts/predictions/test_predictions_distilbert-base_20251212-225747.csv`
- Per-Class Metrics: `artifacts/stats/per_class_metrics_distilbert-base_20251212-225748.csv`

**Download Instructions:**
```bash
# Download checkpoint, predictions, and metrics
python scripts/download_wandb_checkpoint.py Cooper-Union/GoEmotions_Classification/4ta2sol5
```

**Retrieval Instructions:** See `docs/wandb_checkpoint_download.md` for detailed download guide

---

**Last Updated:** December 2024

**Maintainer:** ECE-467 Final Project Team
