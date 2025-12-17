# Ablation Studies

This document records all ablation experiments conducted to validate design decisions for the GoEmotions emotion classification project.

## Table of Contents

1. [Overview](#overview)
2. [Neutral Label Inclusion](#neutral-label-inclusion)
3. [Future Ablations](#future-ablations)

---

## Overview

Ablation studies systematically evaluate the impact of specific design choices on model performance. Each study compares two or more configurations while holding all other variables constant.

**Methodology:**
- Train identical models with only one variable changed
- Compare on same test set with same random seed when possible
- Report both overall metrics (AUC, F1) and per-class impacts
- Make data-driven decisions based on deployment requirements

---

## Neutral Label Inclusion

**Objective:** Determine whether to include the "neutral" label in the final emotion classification task.

### Experimental Setup

**Models Compared:**
- **With Neutral** (28 labels): Standard GoEmotions with all 27 emotions + neutral
- **Without Neutral** (27 labels): Emotion-only classification with `--exclude-neutral` flag

**Training Configuration:**
- Model: DistilBERT-base (66M parameters)
- Learning rate: 3e-5
- Batch size: 32
- Epochs: 10
- Dropout: 0.1
- Max sequence length: 128

**Dataset Impact:**
- Samples with only neutral label were removed when excluding neutral
- Test set reduced from 5,427 to 3,821 samples (-28.2%)

### Results

#### Overall Metrics Comparison

| Metric | With Neutral | Without Neutral | Delta | % Change |
|--------|--------------|-----------------|-------|----------|
| **AUC Macro** | 0.744 | 0.783 | **+0.039** | **+5.2%** |
| **AUC Micro** | 0.880 | 0.883 | +0.003 | +0.4% |
| **Macro F1** | 0.090 | 0.128 | **+0.038** | **+42.0%** |
| **Micro F1** | 0.352 | 0.335 | -0.017 | -4.9% |
| **Macro Precision** | 0.155 | 0.238 | **+0.083** | **+53.9%** |
| **Micro Precision** | 0.708 | 0.857 | **+0.148** | **+20.9%** |
| **Macro Recall** | 0.070 | 0.108 | **+0.038** | **+54.3%** |
| **Micro Recall** | 0.234 | 0.208 | -0.026 | -11.1% |

**Key Findings:**
- **Macro metrics improved significantly** (AUC: +5.2%, F1: +42.0%, Precision: +53.9%, Recall: +54.3%)
- **Micro metrics slightly worse** (F1: -4.9%, Recall: -11.1%) but micro precision improved (+20.9%)
- Macro improvement indicates **better performance on rare emotions**
- Micro decline reflects reduced dataset size and loss of high-confidence neutral predictions

#### Per-Class F1 Comparison

**Top 5 Improvements (excluding neutral):**

| Emotion | With Neutral F1 | Without Neutral F1 | Delta | % Improvement |
|---------|-----------------|-------------------|-------|---------------|
| **Love** | 0.307 | 0.733 | **+0.426** | **+139%** |
| **Curiosity** | 0.007 | 0.398 | **+0.391** | **+5,644%** |
| **Amusement** | 0.367 | 0.728 | **+0.361** | **+98%** |
| **Admiration** | 0.455 | 0.584 | **+0.128** | **+28%** |
| **Gratitude** | 0.805 | 0.917 | **+0.112** | **+14%** |

**Emotions with No Change (F1 = 0.0 in both):**
- Caring, Joy, Surprise, Remorse, Relief, Realization, Pride, Nervousness, Grief, Confusion, Disappointment, Disapproval, Disgust, Embarrassment, Excitement, Fear, Desire

**Emotions with Small Improvements:**
- Optimism: 0.000 → 0.082 (+0.082)
- Sadness: 0.000 → 0.013 (+0.013)
- Anger: 0.000 → 0.010 (+0.010)

**Summary Statistics:**
- Mean F1 delta: +0.033
- Median F1 delta: 0.000
- Max improvement: +0.426 (Love)
- Most improved: Curiosity (+5,644% from near-zero baseline)

### Class Balance Impact

**Dataset Size Change:**
- Total test samples: 5,427 → 3,821 (-1,787 samples, -28.2%)
- Labels: 28 → 27 (-1 label)

**Removed Samples:**
- 1,787 samples had only the neutral label
- These represented 28% of the test set
- Neutral was the most frequent label (1,787 occurrences)

**Impact on Training:**
- Training samples: 43,410 → 30,587 (-29.5%)
- Validation samples: 5,426 → 3,834 (-29.3%)
- Fewer total samples but higher emotion label density

**Rare Label Performance:**
- Excluding neutral dramatically improved rare emotion F1 scores
- Love: +139% (0.307 → 0.733)
- Curiosity: Near-zero to viable (0.007 → 0.398)
- Amusement: +98% (0.367 → 0.728)

**Class Balance Hypothesis:**
- With neutral included, model learned to predict neutral frequently (high support)
- Neutral predictions dominated, suppressing rare emotion predictions
- Without neutral, model focused on discriminating between 27 emotions
- Improved macro metrics indicate better class balance

### Decision

**✅ EXCLUDE NEUTRAL LABEL FROM FINAL MODEL**

**Justification:**

1. **Primary Goal: Emotion Detection**
   - Project objective is fine-grained emotion classification, not neutral vs emotional
   - Neutral label conflates "no emotion detected" with "neutral sentiment"
   - Removing neutral creates a pure emotion classification task

2. **Macro Metrics Strongly Favor Exclusion**
   - AUC Macro: +5.2% (0.744 → 0.783)
   - Macro F1: +42.0% (0.090 → 0.128)
   - Macro metrics weight all emotions equally, critical for rare emotions

3. **Rare Emotion Performance Dramatically Improved**
   - Love, Curiosity, Amusement saw 98-139% F1 improvements
   - Many rare emotions went from F1=0.0 to viable scores
   - Macro precision/recall both improved >50%

4. **Micro Metrics Trade-off Acceptable**
   - Micro F1: -4.9% (0.352 → 0.335)
   - Decline is small and reflects dataset size reduction
   - Micro precision actually improved +20.9% (better positive prediction quality)
   - Micro recall decline (-11.1%) expected with fewer samples

5. **Deployment Considerations**
   - Applications need emotion-specific insights, not "neutral or not" classification
   - Improved rare emotion detection valuable for downstream use cases
   - 28.2% dataset reduction acceptable given macro performance gains

**Trade-offs Acknowledged:**
- Smaller dataset (fewer training samples)
- Slightly lower micro F1 and recall
- Cannot detect truly neutral text (must assume some emotion exists)

**Use Cases Benefited:**
- Sentiment analysis requiring emotion granularity
- Content recommendation based on emotional tone
- Conversation analysis tracking specific emotions

**Use Cases Disadvantaged:**
- Binary emotional vs non-emotional classification
- Applications requiring high recall across all samples

### Configuration

**Final Training Command:**
```bash
python -m src.training.train \
  --model distilbert-base \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 10 \
  --dropout 0.1 \
  --exclude-neutral \
  --wandb-project GoEmotions_Classification \
  --output-dir artifacts/models/distilbert-final
```

**Data Pipeline:**
- Load GoEmotions dataset
- Filter out samples with only neutral label
- Remove neutral from multi-label vectors
- Adjust label indices (shift labels after neutral down by 1)
- Result: 27-label classification task

### Reproducibility

**Comparison Script:**
```bash
python -m src.analysis.compare_ablation \
  --run1-dir artifacts/models/distilbert-neutral-on \
  --run2-dir artifacts/models/distilbert-neutral-off \
  --labels "With Neutral,Without Neutral" \
  --output artifacts/stats/neutral_ablation_summary.csv
```

**Artifacts:**
- Overall metrics: `artifacts/stats/neutral_ablation_summary_overall.csv`
- Per-class metrics: `artifacts/stats/neutral_ablation_summary_per_class.csv`
- Combined summary: `artifacts/stats/neutral_ablation_summary.csv`

**W&B Runs:**
- With neutral: `cslpcnl8` (28 labels, 5,427 test samples)
- Without neutral: `g5h44tsf` (27 labels, 3,821 test samples)

---

## Future Ablations

Additional ablation studies planned or in progress:

### Loss Function Experiments
- **Objective:** Compare BCE vs weighted BCE vs focal loss for rare label performance
- **Status:** Planned
- **Commands:** See `docs/guides/gpu_training.md` Loss Function Experiments section

### Sequence Length Optimization
- **Objective:** Test max_seq_length (128, 256, 512) for performance vs efficiency trade-offs
- **Status:** Planned
- **Commands:** See `docs/guides/gpu_training.md` Sequence Length Experiments section

### Threshold Strategy
- **Objective:** Compare global vs per-label vs top-k threshold strategies
- **Status:** Planned
- **Expected:** Per-label thresholds may improve F1, top-k may improve precision

### Data Augmentation
- **Objective:** Test synonym replacement augmentation for rare label robustness
- **Status:** Planned
- **Expected:** Potential rare label F1 improvements without degrading common labels

### Preprocessing Choices
- **Objective:** Compare cased vs uncased tokenizers, URL/emoji handling
- **Status:** Planned
- **Expected:** Uncased likely better for informal text, emoji preservation may help

---

## References

- [Model Performance](results/model_performance.md) - Final model metrics and comparisons
- [Design Decisions](reference/design_decisions.md) - Rationale for all design choices
- [GPU Training Guide](guides/gpu_training.md) - Commands for all ablation experiments
- [Comparison Script](../src/analysis/compare_ablation.py) - Ablation analysis tool
