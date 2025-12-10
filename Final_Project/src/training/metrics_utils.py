"""
Metrics and threshold utilities for multi-label classification evaluation.

This module provides reusable functions for computing evaluation metrics
with configurable thresholds, including F1 scores, AUC, precision, recall,
confusion summaries, and optional PR curves.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    auc as compute_auc
)


# Configure logging
logger = logging.getLogger(__name__)


def apply_threshold(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    per_class_thresholds: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply threshold(s) to convert probabilities to binary predictions.

    Args:
        probabilities: Probability scores [n_samples, n_labels]
        threshold: Global threshold (used if per_class_thresholds is None)
        per_class_thresholds: Per-class thresholds [n_labels] (optional)

    Returns:
        Binary predictions [n_samples, n_labels]
    """
    if per_class_thresholds is not None:
        # Apply per-class thresholds
        predictions = (probabilities >= per_class_thresholds[np.newaxis, :]).astype(int)
    else:
        # Apply global threshold
        predictions = (probabilities >= threshold).astype(int)

    return predictions


def apply_topk_threshold(
    probabilities: np.ndarray,
    k: int = 1
) -> np.ndarray:
    """
    Apply top-k threshold: select top k predictions per sample.

    Args:
        probabilities: Probability scores [n_samples, n_labels]
        k: Number of top predictions to select per sample

    Returns:
        Binary predictions [n_samples, n_labels]
    """
    n_samples, n_labels = probabilities.shape
    predictions = np.zeros_like(probabilities, dtype=int)

    for i in range(n_samples):
        # Get indices of top-k probabilities
        top_k_indices = np.argsort(probabilities[i])[-k:]
        predictions[i, top_k_indices] = 1

    return predictions


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    label_names: List[str],
    average_types: List[str] = ['macro', 'micro']
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth binary labels [n_samples, n_labels]
        y_pred: Predicted binary labels [n_samples, n_labels]
        y_probs: Prediction probabilities [n_samples, n_labels]
        label_names: List of label names
        average_types: Types of averaging for aggregate metrics

    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}

    # Compute AUC (micro and macro)
    try:
        metrics['auc_micro'] = roc_auc_score(y_true, y_probs, average='micro')
    except ValueError as e:
        logger.warning(f"Could not compute micro AUC: {e}")
        metrics['auc_micro'] = 0.0

    try:
        metrics['auc_macro'] = roc_auc_score(y_true, y_probs, average='macro')
    except ValueError as e:
        logger.warning(f"Could not compute macro AUC: {e}")
        metrics['auc_macro'] = 0.0

    # Compute aggregate metrics for each averaging type
    for avg in average_types:
        metrics[f'f1_{avg}'] = f1_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f'precision_{avg}'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f'recall_{avg}'] = recall_score(y_true, y_pred, average=avg, zero_division=0)

    # Compute per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    metrics['per_class_f1'] = {label_names[i]: float(per_class_f1[i]) for i in range(len(label_names))}
    metrics['per_class_precision'] = {label_names[i]: float(per_class_precision[i]) for i in range(len(label_names))}
    metrics['per_class_recall'] = {label_names[i]: float(per_class_recall[i]) for i in range(len(label_names))}

    return metrics


def compute_confusion_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str]
) -> Dict[str, Any]:
    """
    Compute confusion matrix summary statistics.

    Args:
        y_true: Ground truth binary labels [n_samples, n_labels]
        y_pred: Predicted binary labels [n_samples, n_labels]
        label_names: List of label names

    Returns:
        Dictionary with confusion matrix summaries per class
    """
    n_labels = len(label_names)
    confusion_summary = {}

    for i in range(n_labels):
        # Compute confusion matrix for this class
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])

        # Extract TP, TN, FP, FN
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        confusion_summary[label_names[i]] = {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'support': int(tp + fn)  # Actual positive samples
        }

    return confusion_summary


def compute_pr_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    label_names: List[str]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute precision-recall curves and AUPRC for each class.

    Args:
        y_true: Ground truth binary labels [n_samples, n_labels]
        y_probs: Prediction probabilities [n_samples, n_labels]
        label_names: List of label names

    Returns:
        Dictionary with PR curve data and AUPRC per class
    """
    n_labels = len(label_names)
    pr_curves = {}

    for i in range(n_labels):
        try:
            precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])
            auprc = compute_auc(recall, precision)

            pr_curves[label_names[i]] = {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds,
                'auprc': float(auprc)
            }
        except ValueError as e:
            logger.warning(f"Could not compute PR curve for {label_names[i]}: {e}")
            pr_curves[label_names[i]] = {
                'precision': np.array([]),
                'recall': np.array([]),
                'thresholds': np.array([]),
                'auprc': 0.0
            }

    return pr_curves


def optimize_thresholds_f1(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    label_names: List[str],
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    n_steps: int = 81
) -> Dict[str, float]:
    """
    Find optimal per-class thresholds that maximize F1 score.

    Args:
        y_true: Ground truth binary labels [n_samples, n_labels]
        y_probs: Prediction probabilities [n_samples, n_labels]
        label_names: List of label names
        threshold_range: Range of thresholds to search (min, max)
        n_steps: Number of threshold values to try

    Returns:
        Dictionary mapping label names to optimal thresholds
    """
    n_labels = len(label_names)
    optimal_thresholds = {}

    thresholds_to_try = np.linspace(threshold_range[0], threshold_range[1], n_steps)

    for i in range(n_labels):
        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in thresholds_to_try:
            y_pred = (y_probs[:, i] >= threshold).astype(int)
            f1 = f1_score(y_true[:, i], y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        optimal_thresholds[label_names[i]] = float(best_threshold)

    return optimal_thresholds


def evaluate_with_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    label_names: List[str],
    threshold: float = 0.5,
    per_class_thresholds: Optional[Dict[str, float]] = None,
    top_k: Optional[int] = None,
    compute_pr_curves_flag: bool = False,
    compute_confusion: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive evaluation with configurable thresholds.

    Args:
        y_true: Ground truth binary labels [n_samples, n_labels]
        y_probs: Prediction probabilities [n_samples, n_labels]
        label_names: List of label names
        threshold: Global threshold (default: 0.5)
        per_class_thresholds: Optional per-class thresholds dict
        top_k: Optional top-k threshold (overrides other thresholds)
        compute_pr_curves_flag: Whether to compute PR curves
        compute_confusion: Whether to compute confusion matrices

    Returns:
        Dictionary with all evaluation results
    """
    results = {
        'threshold_strategy': 'top_k' if top_k else ('per_class' if per_class_thresholds else 'global'),
        'threshold_value': top_k if top_k else (per_class_thresholds if per_class_thresholds else threshold)
    }

    # Apply thresholding
    if top_k is not None:
        y_pred = apply_topk_threshold(y_probs, k=top_k)
    elif per_class_thresholds is not None:
        # Convert dict to array in correct order
        threshold_array = np.array([per_class_thresholds[label] for label in label_names])
        y_pred = apply_threshold(y_probs, per_class_thresholds=threshold_array)
    else:
        y_pred = apply_threshold(y_probs, threshold=threshold)

    # Compute classification metrics
    metrics = compute_classification_metrics(y_true, y_pred, y_probs, label_names)
    results.update(metrics)

    # Compute confusion matrix summaries
    if compute_confusion:
        confusion_summary = compute_confusion_summary(y_true, y_pred, label_names)
        results['confusion_summary'] = confusion_summary

    # Compute PR curves
    if compute_pr_curves_flag:
        pr_curves = compute_pr_curves(y_true, y_probs, label_names)
        results['pr_curves'] = pr_curves

    return results


def log_evaluation_results(
    results: Dict[str, Any],
    logger_instance: logging.Logger = None
) -> None:
    """
    Log evaluation results in a readable format.

    Args:
        results: Results dictionary from evaluate_with_threshold()
        logger_instance: Logger to use (defaults to module logger)
    """
    log = logger_instance or logger

    log.info("")
    log.info("=" * 70)
    log.info("Evaluation Results")
    log.info("=" * 70)

    # Log threshold strategy
    log.info(f"Threshold Strategy: {results.get('threshold_strategy', 'unknown')}")
    threshold_val = results.get('threshold_value')
    if isinstance(threshold_val, dict):
        log.info(f"Using per-class thresholds (showing first 5):")
        for i, (label, thresh) in enumerate(list(threshold_val.items())[:5]):
            log.info(f"  {label}: {thresh:.3f}")
        if len(threshold_val) > 5:
            log.info(f"  ... and {len(threshold_val) - 5} more")
    else:
        log.info(f"Threshold Value: {threshold_val}")

    # Log AUC scores
    log.info("")
    log.info("AUC Scores:")
    log.info(f"  Micro AUC: {results.get('auc_micro', 0.0):.4f}")
    log.info(f"  Macro AUC: {results.get('auc_macro', 0.0):.4f}")

    # Log aggregate metrics
    log.info("")
    log.info("Aggregate Metrics:")
    for metric in ['f1', 'precision', 'recall']:
        for avg in ['macro', 'micro']:
            key = f'{metric}_{avg}'
            if key in results:
                log.info(f"  {key.replace('_', ' ').title():20s}: {results[key]:.4f}")

    # Log per-class F1 scores
    if 'per_class_f1' in results:
        log.info("")
        log.info("Per-Class F1 Scores:")
        sorted_f1 = sorted(results['per_class_f1'].items(), key=lambda x: x[1], reverse=True)
        for label, score in sorted_f1:
            log.info(f"  {label:20s}: {score:.4f}")

    # Log confusion summary if available
    if 'confusion_summary' in results:
        log.info("")
        log.info("Confusion Summary (Top 5 by support):")
        sorted_conf = sorted(
            results['confusion_summary'].items(),
            key=lambda x: x[1]['support'],
            reverse=True
        )[:5]
        for label, conf in sorted_conf:
            log.info(f"  {label:20s}: TP={conf['true_positives']:4d}, "
                    f"FP={conf['false_positives']:4d}, "
                    f"FN={conf['false_negatives']:4d}, "
                    f"Support={conf['support']:4d}")

    log.info("=" * 70)
