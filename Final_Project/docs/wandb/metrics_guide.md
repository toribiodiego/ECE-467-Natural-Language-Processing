# W&B Metrics Guide

This document explains what metrics are logged to Weights & Biases and where to find them.

## Overview

Metrics are organized into two categories:
1. **Chart Metrics** - Time-series data visible in W&B Charts tab
2. **Summary Metrics** - Final values visible in W&B Summary and Runs table

## Chart Metrics (Time-Series)

These metrics are logged after each epoch and displayed as charts in the W&B Charts tab.

### Training Metrics

Logged after each training epoch.

| Metric | Description | Typical Range | Use |
|--------|-------------|---------------|-----|
| `train/loss` | Average BCE loss across all batches | 0.3-0.7 | Monitor convergence |
| `train/loss_std` | Standard deviation of loss across batches | 0.0-0.1 | Monitor training stability |
| `train/learning_rate` | Current learning rate (after scheduler) | 1e-6 to 5e-5 | Track scheduler behavior |
| `train/epoch_time` | Time taken for epoch (seconds) | 10-300s | Monitor training speed |
| `train/samples_per_sec` | Training throughput (samples/sec) | 1-100 | Detect slowdowns |

**Example Chart**: Train loss over epochs
```
0.7 |
    |     *
0.6 |    *  *
    |   *    *
0.5 |  *      *
    | *        *
0.4 |*__________*___
    0  1  2  3  4  5
        Epoch
```

### Validation Metrics

Logged after validation at the end of each epoch.

| Metric | Description | Typical Range | Use |
|--------|-------------|---------------|-----|
| `val/loss` | Validation BCE loss | 0.3-0.7 | Detect overfitting |
| `val/auc` | Validation AUC (micro-average) | 0.5-0.9 | Primary metric for early stopping |
| `val/f1_micro` | Validation F1 (micro-average) | 0.0-0.8 | Track overall performance |
| `val/f1_macro` | Validation F1 (macro-average) | 0.0-0.6 | Track per-class performance |

**Key Metric**: `val/auc` is used for:
- Early stopping (stop if no improvement for N epochs)
- Checkpoint selection (save best model)
- Run comparison (which run performed best?)

### Test Metrics (Final Evaluation)

Logged once after training completes.

| Metric | Description | Typical Range | Use |
|--------|-------------|---------------|-----|
| `test/auc_micro` | Test AUC (micro-average) | 0.5-0.9 | Overall performance |
| `test/auc_macro` | Test AUC (macro-average) | 0.5-0.85 | Per-class performance |
| `test/f1_micro` | Test F1 (micro-average) | 0.0-0.8 | Overall F1 score |
| `test/f1_macro` | Test F1 (macro-average) | 0.0-0.6 | Average F1 across emotions |
| `test/precision_macro` | Test precision (macro-average) | 0.0-0.7 | Macro-averaged precision |
| `test/recall_macro` | Test recall (macro-average) | 0.0-0.7 | Macro-averaged recall |

**Note**: These appear as single points in charts since they're only logged once.

## Summary Metrics (Table View)

These metrics are saved to W&B summary and visible in the Runs table for filtering and sorting.

### Metadata

| Metric | Description | Use |
|--------|-------------|-----|
| `best_epoch` | Epoch with best val/auc | Identify when model peaked |
| `final_epoch` | Final epoch number | Check if training completed |
| `total_training_time` | Total training time (seconds) | Compare training efficiency |

### Per-Class Test Metrics

For each of the 28 emotions, the following metrics are saved:

| Metric Pattern | Description | Example |
|----------------|-------------|---------|
| `test/f1_{emotion}` | F1 score for specific emotion | `test/f1_joy` |
| `test/precision_{emotion}` | Precision for specific emotion | `test/precision_joy` |
| `test/recall_{emotion}` | Recall for specific emotion | `test/recall_joy` |
| `test/support_{emotion}` | Number of test samples | `test/support_joy` |

**Emotions**: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, neutral, optimism, pride, realization, relief, remorse, sadness, surprise

**Use Cases**:
- Filter runs by specific emotion performance
- Sort runs by F1 score on hard emotions
- Compare per-emotion performance across models

## Viewing Metrics in W&B

### Charts Tab

Shows time-series metrics as line charts.

**To view**:
1. Go to run page
2. Click "Charts" tab
3. Select metrics to display

**Useful views**:
- Train vs validation loss (detect overfitting)
- Validation AUC over time (track improvement)
- Learning rate schedule (verify warmup/decay)

**Grouping**: Metrics are grouped by prefix:
- `train/*` - All training metrics
- `val/*` - All validation metrics
- `test/*` - All test metrics

### Summary Tab

Shows final metric values.

**To view**:
1. Go to run page
2. Click "Overview" tab
3. Scroll to "Summary" section

**Contains**:
- All test metrics
- Metadata (best_epoch, training_time)
- Per-class metrics (all 28 emotions)

### Runs Table

Compare metrics across multiple runs.

**To view**:
1. Go to project page
2. See table of all runs

**Features**:
- Sort by any metric (click column header)
- Filter by metric value (use search bar)
- Group by hyperparameter
- Export to CSV

**Example - Find best run by emotion**:
1. Click `test/f1_joy` column to sort
2. Top run has best performance on "joy" emotion

## Metric Calculations

### AUC (Area Under ROC Curve)

**Micro-average**: Aggregate predictions across all emotions, then compute AUC
```python
auc_micro = roc_auc_score(all_labels, all_probs, average='micro')
```

**Macro-average**: Compute AUC per emotion, then average
```python
auc_macro = roc_auc_score(all_labels, all_probs, average='macro')
```

**Interpretation**:
- 0.5 = Random guessing
- 0.7 = Decent performance
- 0.8 = Good performance
- 0.9+ = Excellent performance

### F1 Score

**Micro-average**: Aggregate TP/FP/FN across all emotions
```python
f1_micro = f1_score(true_labels, pred_labels, average='micro')
```

**Macro-average**: Compute F1 per emotion, then average
```python
f1_macro = f1_score(true_labels, pred_labels, average='macro')
```

**Interpretation**:
- Micro F1 weighted by class frequency (favors common emotions)
- Macro F1 treats all emotions equally (better for imbalanced data)

### Per-Class Metrics

For each emotion:
```python
TP = true positives
FP = false positives
FN = false negatives
TN = true negatives

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)
```

## Metric Selection Guide

### For Early Stopping

**Use**: `val/auc` (micro-average)

**Why**: Most stable metric, less sensitive to threshold choice

**Config**:
```python
patience = 3  # Stop if no improvement for 3 epochs
```

### For Model Comparison

**Primary**: `test/auc_macro`

**Why**: Treats all emotions equally, better for imbalanced dataset

**Secondary**: `test/f1_macro`

**Why**: Easier to interpret, shows practical performance

### For Error Analysis

**Use**: Per-class metrics in summary + predictions CSV

**Workflow**:
1. Check `test/f1_{emotion}` to find worst emotions
2. Download predictions CSV
3. Filter by that emotion
4. Analyze misclassified examples

### For Hyperparameter Tuning

**Track**:
- `val/auc` - Primary objective
- `train/loss` - Check for underfitting
- `val/loss` - Check for overfitting
- `train/epoch_time` - Monitor training cost

**Compare**:
```python
import wandb

api = wandb.Api()
runs = api.runs('Cooper-Union/GoEmotions_Classification')

# Find best hyperparameters
best_auc = 0
best_config = None

for run in runs:
    auc = run.summary.get('val/auc', 0)
    if auc > best_auc:
        best_auc = auc
        best_config = run.config

print(f'Best config: {best_config}')
print(f'Best AUC: {best_auc}')
```

## Custom Metrics

If you need to log additional metrics, edit `src/training/wandb_utils.py`:

### Add Training Metric

```python
def log_training_metrics(..., custom_metric=None):
    metrics = {
        'train/loss': train_loss,
        # ... existing metrics
    }

    if custom_metric is not None:
        metrics['train/custom_metric'] = custom_metric

    wandb.log(metrics, step=epoch)
```

### Add Test Metric

```python
def log_evaluation_metrics(...):
    chart_metrics = {
        'test/auc_micro': ...,
        # ... existing metrics
        'test/custom_metric': test_results.get('custom_metric', 0.0)
    }
    wandb.log(chart_metrics)
```

## Interpreting Training Curves

### Healthy Training

```
Train Loss:  \_____    (Decreasing, plateaus)
Val Loss:    \_____    (Decreases with train loss)
Val AUC:     /‾‾‾‾‾    (Increasing, plateaus)
```

**Action**: Continue training or increase capacity

### Overfitting

```
Train Loss:  \______   (Still decreasing)
Val Loss:    \__/‾‾    (Starts increasing after epoch 3)
Val AUC:     /‾‾\_     (Peaks then decreases)
```

**Action**: Stop at peak, use regularization, reduce model size

### Underfitting

```
Train Loss:  ‾‾‾‾‾‾    (High and flat)
Val Loss:    ‾‾‾‾‾‾    (High and flat)
Val AUC:     ______    (Low and flat around 0.5-0.6)
```

**Action**: Increase model size, train longer, reduce regularization

### Learning Rate Too High

```
Train Loss:  /\/\/\    (Oscillating)
Val Loss:    /\/\/\    (Oscillating)
```

**Action**: Reduce learning rate, use warmup

### Learning Rate Too Low

```
Train Loss:  ‾‾\___    (Very slow decrease)
Val AUC:     __/‾‾‾    (Slow improvement)
```

**Action**: Increase learning rate

## Additional Resources

- Metrics in code: `src/training/wandb_utils.py`
- Evaluation code: `src/training/evaluation.py`
- W&B Metrics Guide: https://docs.wandb.ai/guides/track/log
