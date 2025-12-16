# Design Decisions and Rationale

This document records key design decisions made throughout the project, including the reasoning behind each choice and alternatives considered.

## Table of Contents

1. [Data Preprocessing](#data-preprocessing)
2. [Visualization Design](#visualization-design)
3. [Model Architecture](#model-architecture)
4. [Training Configuration](#training-configuration)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Classification Thresholds](#classification-thresholds)

---

## Data Preprocessing

### Neutral Emotion Handling

**Decision:** Retain neutral emotion in all dataset processing, training, and CSV exports, but create dual visualizations (with and without neutral).

**Rationale:**

Neutral is a valid emotion label in the GoEmotions dataset and represents a significant portion of real-world text. Removing it would:
- Reduce dataset size by ~30% (17,772 out of 58,009 total samples)
- Bias the model toward non-neutral emotional content
- Limit applicability to real-world scenarios where neutral text is common

**Visualization Exception:**

We generate two versions of the class distribution bar chart:
1. **With neutral** (`class_distribution_stacked.png`) - Shows complete dataset
2. **Without neutral** (`class_distribution_stacked_no_neutral.png`) - Better visual clarity

**Why the exception?**

Neutral's sample count (17,772) is 3.5x larger than the next emotion (admiration: 5,122). This causes the y-axis to scale to ~18,000, compressing the other 27 emotions and making their multi-label patterns difficult to see. The no-neutral visualization allows the y-axis to scale to ~5,000, providing better visibility of patterns in the remaining emotions.

**Neutral's Multi-Label Distribution:**
- Total samples: 17,772
- Single-label: 16,021 (90.1%) - highest single-label rate among all emotions
- Two-label: 1,671 (9.4%)
- Three+ label: 80 (0.5%) - lowest multi-label rate

**Impact on Training:**

Neutral's high single-label rate (90.1%) makes it easier to learn compared to emotions with higher multi-label co-occurrence. This may require:
- Class-weighted loss functions to balance rare emotions
- Per-class threshold tuning for optimal performance
- Monitoring for neutral-dominant predictions

**Alternatives Considered:**
- Remove neutral entirely (rejected - reduces dataset utility)
- Downsample neutral to match other emotions (rejected - loses valuable training data)
- Always exclude from visualization (rejected - need complete dataset view)

---

## Visualization Design

### Color Scheme for Multi-Label Breakdown

**Decision:** Use pure RGB colors (#0000ff blue, #00ff00 green, #ff0000 red) for 1-label, 2-labels, and 3+ labels respectively.

**Rationale:**

Pure RGB colors provide:
- **High visibility**: Maximum contrast when figures are scaled down
- **Distinct separation**: Clear visual distinction between categories
- **Presentation-ready**: Work well in slides, papers, and web display

**Color Mapping:**
- 1 label: `#0000ff` (pure blue)
- 2 labels: `#00ff00` (pure green)
- 3+ labels: `#ff0000` (pure red)

**Accessibility Considerations:**

The current color scheme is NOT optimized for colorblindness. Alternative schemes are available in the code:
- `colorblind`: Okabe-Ito palette, safe for deuteranopia
- `sequential`: Blue gradient, good for grayscale printing

For publication, consider using the colorblind-safe palette.

**Alternatives Considered:**
- Matplotlib defaults (rejected - less visible when scaled)
- Colorblind-safe palette (available as option)
- Sequential gradient (available as option)

### Legend and Text Sizing

**Decision:** Increase legend font size to 11pt (from default ~9pt).

**Rationale:**

Figures are often scaled down in presentations, papers, or web display. Larger legend text ensures readability at smaller sizes without requiring viewers to zoom in.

**Trade-offs:**
- Larger legend takes more space
- Worth it for improved readability in final presentation

---

## Model Architecture

### Model Selection

**Decision:** Train and evaluate both RoBERTa-Large and DistilBERT.

**Rationale:**

**RoBERTa-Large:**
- **Best performance**: Target AUC ~95.7%
- **Use case**: Best-possible metrics for portfolio
- **Constraints**: Requires 12GB+ VRAM, ~2-3 hours training time

**DistilBERT:**
- **Best efficiency**: Target AUC ~94.8% (only 0.9% lower)
- **Use case**: Practical deployment, faster inference
- **Constraints**: Requires 6GB+ VRAM, ~1-2 hours training time
- **Size**: 66M params vs. RoBERTa-Large 355M params (81% reduction)

**Portfolio Strategy:**

Both models provide valuable comparison:
- RoBERTa-Large shows upper bound of performance
- DistilBERT shows efficiency/performance trade-off
- Difference quantifies the cost of model compression

**Alternatives Considered:**
- BERT-base (rejected - RoBERTa generally superior)
- RoBERTa-base (considered for future comparison)
- ELECTRA (rejected - less common, harder to reproduce)

---

## Training Configuration

### Hyperparameters

**RoBERTa-Large Configuration:**
```python
{
    'learning_rate': 2e-5,
    'batch_size': 16,
    'num_epochs': 10,
    'dropout': 0.1,
    'max_length': 128
}
```

**DistilBERT Configuration:**
```python
{
    'learning_rate': 3e-5,
    'batch_size': 32,
    'num_epochs': 10,
    'dropout': 0.1,
    'max_length': 128
}
```

**Rationale:**

These hyperparameters are based on:
- Literature recommendations for multi-label text classification
- Empirical testing in preliminary experiments
- GPU memory constraints (batch size)
- Training time budget (epochs)

**Learning Rate:**
- 2e-5 for large models (RoBERTa-Large): Conservative to avoid catastrophic forgetting
- 3e-5 for smaller models (DistilBERT): Slightly higher for faster convergence

**Batch Size:**
- 16 for RoBERTa-Large: Maximum that fits in 12GB VRAM
- 32 for DistilBERT: Larger batches for more stable gradients

**Sequence Length:**
- 128 tokens: Covers ~95% of samples without truncation
- Trade-off between coverage and memory/speed

**Planned Ablations:**

The following hyperparameters will be studied in ablation experiments:
- Learning rate: {1e-5, 2e-5, 3e-5}
- Dropout: {0.0, 0.1, 0.2}
- Sequence length: {128, 256, 512}
- Loss weighting: {None, class-weighted, focal}
- Threshold strategies: {global, per-label, top-k}

See `../../output/ablation_studies/README.md` for results and analysis.

### Checkpoint Saving Strategy

**Decision:** Save the model checkpoint from the epoch with the highest validation AUC, not the final epoch.

**Rationale:**

Training does not always monotonically improve validation performance. Models may overfit in later epochs, leading to:
- Decreased validation AUC despite continued decrease in training loss
- Suboptimal final checkpoints if using the last epoch

**Implementation:**

During training, the system:
1. Tracks validation AUC after each epoch
2. Saves a deep copy of the model state when validation AUC improves
3. After training completes, restores the best model state
4. Saves this best-performing model as the final checkpoint

**Benefits:**
- Guarantees saved checkpoint has the best validation performance
- Prevents overfitting from degrading the final model
- Matches the original project methodology (validation AUC as primary metric)

**Example:**
```
Epoch 1: Val AUC = 0.503 → Saved as best
Epoch 2: Val AUC = 0.504 → Saved as new best
Epoch 3: Val AUC = 0.502 → Not saved (worse than epoch 2)
...
Final: Restored epoch 2 model (Val AUC = 0.504)
```

**Alternatives Considered:**
- Save final epoch (rejected - may overfit)
- Save all epochs (rejected - excessive storage)
- Early stopping (considered for future - adds complexity)

---

## Evaluation Metrics

### Metric Selection for Multi-Label Emotion Classification

**Decision:** Use AUC (micro) as the primary metric, with AUC (macro) and Micro F1 as secondary metrics for comprehensive evaluation.

**Rationale:**

Multi-label emotion classification with 28 imbalanced emotion classes (ranging from neutral: 17,772 samples to grief: 96 samples) requires careful metric selection to balance:
1. Overall model performance across all emotions
2. Fair treatment of rare vs. frequent emotions
3. Threshold-agnostic evaluation during training
4. Practical utility for deployment scenarios

### Primary Metric: AUC (micro)

**Why AUC (micro) is primary:**

1. **Threshold-agnostic**: Measures ranking quality independent of classification threshold, allowing model comparison without threshold tuning
2. **Dataset-level performance**: Micro-averaging treats each prediction equally, providing an overall measure of the model's ability to rank emotions correctly across the entire test set
3. **Robust to multi-label complexity**: Works naturally with multi-label scenarios where samples can have 1-3+ emotion labels
4. **Comparable across models**: Enables direct comparison between models (RoBERTa vs DistilBERT) without threshold sensitivity

**Empirical results validate this choice:**
- RoBERTa-Large: AUC (micro) = 0.9045
- DistilBERT: AUC (micro) = 0.8800
- Clear 2.5% performance gap despite 81% parameter reduction

**Why not accuracy?**

Multi-label accuracy is poorly defined and highly sensitive to threshold choice. In our 28-class setting, a model predicting only neutral would appear accurate but have poor utility.

**Why not Macro F1 as primary?**

Macro F1 is threshold-dependent and shows extreme sensitivity in our results:
- RoBERTa Macro F1: 0.1600
- DistilBERT Macro F1: 0.0904
- 43.5% relative difference vs. only 2.7% AUC difference

This gap indicates threshold mismatch, not fundamental model quality difference. F1 requires threshold tuning before meaningful comparison.

### Secondary Metrics

#### 1. AUC (macro) - Fair Emotion Treatment

**Purpose:** Measures per-emotion ranking quality, treating all 28 emotions equally regardless of frequency.

**Why it matters:**
- Ensures rare emotions (grief: 96, pride: 234) are not ignored in favor of frequent ones (neutral: 17,772)
- Macro-averaging gives each emotion equal weight in the final score
- Helps identify if models learn balanced representations vs. just predicting frequent classes

**Results interpretation:**
- RoBERTa AUC (macro): 0.8294 vs AUC (micro): 0.9045
- Gap indicates some emotions are harder to classify than others
- Larger gap = more uneven performance across emotions

#### 2. Micro F1 - Deployment Performance Indicator

**Purpose:** Measures classification performance at a fixed threshold (default 0.5), representing expected behavior in production deployment.

**Why it matters:**
- **Production relevance**: Real systems must make binary predictions at some threshold
- **Overall classification quality**: Balances precision and recall across all predictions
- **Comparison baseline**: Enables comparison with prior work using F1 scores

**Results interpretation:**
- RoBERTa Micro F1: 0.4001 (Precision: 0.7278, Recall: 0.2759)
- High precision, low recall = conservative predictions at default threshold
- Indicates need for threshold tuning to balance precision/recall trade-off

**Why Micro F1 over Macro F1?**
- Micro F1 reflects dataset-level performance (aligns with primary metric philosophy)
- Macro F1 is very low (0.0904-0.1600) due to poor performance on rare emotions, making it less useful for overall quality assessment
- Per-emotion F1 scores (reported separately) provide granular rare-emotion insights

### Tertiary Metrics for Analysis

These metrics are reported but not used for primary model comparison:

**3. Macro Precision/Recall:**
- Identifies systematic biases (e.g., overly conservative predictions)
- Per-emotion breakdown reveals which emotions are hardest to detect

**4. Micro Precision/Recall:**
- Decomposes Micro F1 to diagnose threshold issues
- High precision + low recall = threshold too high (our case)

**5. Per-Emotion F1 Scores:**
- Identify best/worst performing emotions
- Guide future improvements (e.g., data augmentation for rare emotions)
- See `docs/results/model_performance.md#per-emotion-performance`

### Metric Hierarchy Summary

```
Primary:   AUC (micro) → Model comparison, training checkpointing
Secondary: AUC (macro) → Fair emotion treatment verification
           Micro F1    → Deployment performance estimation
Tertiary:  Precision/Recall (macro/micro) → Diagnostic analysis
           Per-emotion F1 → Granular performance insights
```

### Threshold Selection Strategy

**Decision:** Use default threshold (0.5) for initial reporting, with per-label threshold optimization in ablation studies.

**Rationale:**

Our empirical results show threshold sensitivity:
- High precision (0.7085-0.7278) with low recall (0.2338-0.2759)
- Indicates default 0.5 threshold is too conservative
- Optimized thresholds could significantly improve F1 scores

**Threshold strategies to evaluate:**
1. **Global threshold** (current: 0.5): Simple, but suboptimal
2. **Per-label threshold**: Optimize threshold independently for each emotion to maximize F1
3. **Top-k selection**: Always predict k most confident emotions (UX guarantee)

**Future work:**
The ablation study will quantify F1 improvements from threshold optimization and inform production deployment choices.

### Why These Metrics Align with Best Practices

**Multi-label classification literature** (Zhang & Zhou 2014, Tsoumakas & Katakis 2007) recommends:
1. ✓ **Threshold-agnostic metrics** (AUC) for model development
2. ✓ **Both micro and macro averaging** to balance overall and per-class performance
3. ✓ **Complementary metrics** (AUC + F1) to assess ranking and classification quality
4. ✓ **Per-class analysis** to identify systematic biases

**GoEmotions baseline paper** (Demszky et al. 2020) uses macro F1 as primary metric, but:
- Our dataset has more extreme imbalance after preprocessing
- Macro F1 instability (43.5% gap) makes it unsuitable for our model comparison
- AUC (micro) provides more stable and interpretable comparisons
- We still report macro metrics for completeness and comparison with literature

---

## Future Decisions

The following design decisions are pending and will be documented after ablation studies:

### Data Augmentation
- Whether to use text augmentation during training
- Impact on rare-label performance vs. clean-text metrics

### Loss Function
- Class-weighted BCE vs. standard BCE
- Focal loss vs. BCE for rare emotions

### Text Preprocessing
- Cased vs. lowercased tokenization
- URL/emoji handling strategies (keep, mask, strip)

### Model Ensemble
- Whether to ensemble RoBERTa-Large and DistilBERT
- Potential performance gains vs. increased complexity

---

## Classification Thresholds

### Optimal Threshold Selection

**Decision:** Use a global threshold of 0.1 for RoBERTa-Large predictions on the validation set, with per-class threshold optimization available for deployment scenarios.

**Rationale:**

The default 0.5 threshold commonly used for binary classification is suboptimal for multi-label emotion classification with highly imbalanced classes. Analysis of RoBERTa-Large validation predictions revealed:

**Problem with 0.5 Threshold:**
- Only 8 out of 28 emotions were predicted (71% of emotions had zero predictions)
- Macro F1 score: 0.133
- Many rare emotions (fear, grief, embarrassment, desire) never surpassed 0.5 probability
- Model outputs lower confidence scores for rare classes even when they are present

**Threshold Sweep Analysis:**

Evaluated thresholds from 0.1 to 0.9 on validation set (5,426 samples):

| Threshold | Macro F1 | Micro F1 | Labels Predicted |
|-----------|----------|----------|------------------|
| 0.1       | 0.253    | 0.491    | 19/28 (68%)      |
| 0.2       | 0.234    | 0.515    | 13/28 (46%)      |
| 0.3       | 0.190    | 0.490    | 13/28 (46%)      |
| 0.4       | 0.157    | 0.459    | 9/28 (32%)       |
| 0.5       | 0.133    | 0.403    | 8/28 (29%)       |

**Optimal Global Threshold: 0.1**
- Macro F1: 0.253 (89% improvement over 0.5)
- Micro F1: 0.491 (22% improvement)
- Recall (Macro): 0.307 vs. 0.113 at 0.5 threshold
- Enables prediction of 19 out of 28 emotions (vs. 8 at 0.5)

**Trade-offs:**
- Lower threshold increases false positives (precision drops from 0.696 to 0.240)
- However, in emotion classification, missing rare emotions (false negatives) is more problematic than occasional false positives
- The macro F1 improvement indicates better balance across all emotion classes

**Per-Class Optimal Thresholds:**

For production deployment, per-class thresholds can be used:

**High-Performing Emotions (threshold 0.2-0.3):**
- gratitude: 0.30 (F1=0.899)
- amusement: 0.20 (F1=0.761)
- love: 0.20 (F1=0.746)
- neutral: 0.30 (F1=0.639)
- admiration: 0.30 (F1=0.618)

**Moderate Emotions (threshold 0.1):**
- Most emotions perform best at 0.1 threshold
- Examples: anger, surprise, joy, sadness, confusion

**Challenging Emotions (threshold 0.5, F1=0.0):**
- fear, grief, embarrassment, desire, disappointment
- These emotions have insufficient training data or are too difficult to distinguish
- Even at optimal thresholds, the model cannot reliably detect them

**Implementation:**

The threshold sweep script (`src/analysis/threshold_sweep.py`) provides:
1. Global threshold optimization across all classes
2. Per-class threshold optimization for fine-tuned control
3. Visualization of threshold vs. performance trade-offs
4. JSON export of optimal thresholds for deployment

**Artifacts:**
- `artifacts/stats/threshold_sweep/threshold_summary.json` - Optimal thresholds
- `output/figures/11_threshold_macro.png` - Macro-averaged metrics vs threshold
- `output/figures/12_threshold_micro.png` - Micro-averaged metrics vs threshold
- `output/figures/13_threshold_f1.png` - F1 score comparison across averaging methods
- `output/figures/14_threshold_coverage.png` - Label coverage vs threshold
- `output/figures/15_per_class_thresholds.png` - Per-emotion optimal thresholds

**Alternatives Considered:**
- Fixed 0.5 threshold (rejected - poor performance on rare emotions)
- Top-k prediction strategy (considered - may over-predict for single-label samples)
- Calibrated probabilities (future work - requires additional validation data)

**Deployment Recommendations:**
1. Use 0.1 global threshold for balanced performance across all emotions
2. Consider per-class thresholds if precision is critical for specific emotions
3. Accept that 9 emotions (grief, fear, embarrassment, etc.) may never be reliably predicted with current training data
4. Future work: Collect more training samples for rare emotions or use data augmentation

### Model Calibration

**Decision:** Document calibration quality through Brier score and Expected Calibration Error (ECE) analysis to understand the reliability of predicted probabilities.

**Rationale:**

While threshold optimization determines when to predict an emotion, calibration analysis evaluates whether the model's confidence scores are trustworthy. A well-calibrated model's 70% prediction should be correct approximately 70% of the time.

**Calibration Analysis Results:**

Analysis of RoBERTa-Large validation predictions (5,426 samples) reveals:

**Overall Calibration:**
- Brier Score: 0.0285 (scale: 0=perfect, 1=worst)
- Expected Calibration Error (ECE): 0.0094 (scale: 0=perfect)

Both metrics indicate the model is reasonably well-calibrated overall. The low Brier score suggests predicted probabilities closely match actual outcomes, and the low ECE confirms minimal systematic bias between predicted confidence and observed accuracy.

**Per-Emotion Calibration:**

**Best Calibrated Emotions (lowest Brier score):**
- grief: 0.0024
- pride: 0.0028
- relief: 0.0033
- nervousness: 0.0039
- embarrassment: 0.0064

Interestingly, the best-calibrated emotions are those with the poorest prediction performance (F1=0.0). This occurs because the model correctly predicts low probabilities for these rare emotions - when it says 1% chance, the actual rate is indeed near 1%.

**Worst Calibrated Emotions (highest Brier score):**
- neutral: 0.1704
- approval: 0.0671
- disapproval: 0.0503
- annoyance: 0.0493
- admiration: 0.0493

Neutral has the poorest calibration despite being the most common emotion. This suggests the model struggles to accurately estimate confidence for the dominant class, possibly due to class imbalance effects during training.

**Reliability Diagram Insights:**

For top-performing emotions (gratitude, amusement, love), reliability diagrams show:
- Predictions closely follow the perfect calibration diagonal
- Minor tendency toward overconfidence in mid-range probabilities (0.3-0.7)
- High-confidence predictions (>0.8) are well-calibrated

**Practical Implications:**

1. **Trust high-confidence predictions**: When the model predicts >80% probability, it is reliably correct
2. **Mid-range uncertainty**: Predictions in 30-70% range may be slightly overconfident
3. **Rare emotion probabilities are accurate**: Low probabilities (<10%) for rare emotions can be trusted
4. **Neutral requires caution**: Neutral predictions may be less reliable than other high-frequency emotions

**Calibration vs. Threshold Findings:**

The calibration analysis complements threshold optimization:
- Low threshold (0.1) is appropriate given well-calibrated low probabilities
- Model's probability estimates are trustworthy, supporting confidence-based decision-making
- Poor-performing emotions are correctly uncertain (low probabilities), not miscalibrated

**Implementation:**

The calibration analysis script (`src/analysis/calibration.py`) computes:
1. Brier score (mean squared error between predictions and outcomes)
2. Expected Calibration Error (average miscalibration across probability bins)
3. Reliability diagrams showing predicted vs observed frequencies

**Artifacts:**
- `artifacts/stats/calibration/calibration_summary.json` - Overall and per-class metrics
- `artifacts/stats/calibration/per_class_calibration.csv` - Ranked calibration results
- `output/figures/17_reliability_diagram.png` - Reliability curves for top 6 emotions
- `output/figures/18_calibration_summary.png` - Brier score and ECE by emotion

**Future Work:**
- Apply temperature scaling or Platt scaling to improve calibration for neutral
- Investigate overconfidence in mid-range probabilities
- Test calibration stability across different validation sets

---

**Last Updated:** December 2024

**Maintainer:** ECE-467 Final Project Team
