# Design Decisions and Rationale

This document records key design decisions made throughout the project, including the reasoning behind each choice and alternatives considered.

## Table of Contents

1. [Data Preprocessing](#data-preprocessing)
2. [Visualization Design](#visualization-design)
3. [Model Architecture](#model-architecture)
4. [Training Configuration](#training-configuration)
5. [Evaluation Metrics](#evaluation-metrics)

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

See `ablation_studies/README.md` for results and analysis.

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

### Primary Metric: AUC (Area Under ROC Curve)

**Decision:** Use macro-averaged AUC as the primary evaluation metric.

**Rationale:**

For multi-label classification with class imbalance:
- **AUC is threshold-agnostic**: Measures ranking quality independent of classification threshold
- **Macro-averaging**: Treats all emotions equally (vs. micro-averaging which favors frequent classes)
- **Robust to imbalance**: Works well even with rare emotions (grief: 96 samples)

**Why not accuracy?**

Multi-label accuracy is poorly defined and sensitive to threshold choice. A model that always predicts neutral would have high accuracy but poor utility.

**Why not F1 score?**

F1 score requires a threshold decision. We report F1 scores at optimized thresholds, but use AUC for model comparison during training.

**Secondary Metrics:**

- **Per-emotion F1 scores**: Identify strong/weak emotion predictions
- **Precision/Recall curves**: Understand threshold trade-offs
- **Confusion analysis**: Identify common co-occurrence errors

### Threshold Selection Strategy

**Decision:** Evaluate multiple threshold strategies in ablation studies, then select based on use case.

**Strategies to Compare:**
1. **Global threshold** (e.g., 0.5): Simple, interpretable
2. **Per-label threshold**: Optimized for each emotion independently
3. **Top-k selection**: Always predict k most confident emotions

**Rationale:**

Different deployment scenarios require different threshold strategies:
- **Global threshold**: Simplest for production deployment
- **Per-label threshold**: Best metrics, but more complex
- **Top-k**: Guarantees at least k predictions (useful for UX)

The ablation study will quantify performance differences and inform the final choice.

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

**Last Updated:** December 2024

**Maintainer:** ECE-467 Final Project Team
