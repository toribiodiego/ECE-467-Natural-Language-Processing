#!/usr/bin/env python3
"""
Threshold Sweep Analysis

Evaluates model performance across different classification thresholds to find
optimal thresholds for multi-label emotion classification. This addresses the
issue where a fixed 0.5 threshold causes many rare emotions to never be predicted.

The script:
- Loads validation predictions (probabilities and ground truth labels)
- Evaluates metrics (F1, precision, recall) at each threshold
- Generates threshold vs performance curves
- Identifies optimal threshold(s) that maximize F1 score

Usage:
    python -m src.analysis.threshold_sweep \
        --predictions artifacts/predictions/val_epoch10_predictions_roberta-large_*.csv \
        --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
        --output artifacts/stats/threshold_sweep/

    # Or use checkpoint to generate predictions on-the-fly
    python -m src.analysis.threshold_sweep \
        --checkpoint artifacts/models/roberta/roberta-large-20251212-211010 \
        --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
        --output artifacts/stats/threshold_sweep/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, f1_score
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GoEmotions label list (28 emotions)
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


def load_validation_predictions(predictions_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load validation predictions CSV and extract ground truth + probabilities.

    Args:
        predictions_path: Path to validation predictions CSV

    Returns:
        Tuple of (y_true [n_samples, n_labels], y_probs [n_samples, n_labels])
    """
    logger.info(f"Loading validation predictions from: {predictions_path}")
    df = pd.read_csv(predictions_path)
    logger.info(f"Loaded {len(df)} validation samples")

    # Extract ground truth labels
    y_true = []
    for true_labels_str in df['true_labels']:
        # Parse comma-separated labels
        if pd.isna(true_labels_str) or true_labels_str == '':
            labels = []
        else:
            labels = [label.strip() for label in str(true_labels_str).split(',')]

        # Convert to binary vector
        label_vector = np.zeros(len(EMOTION_LABELS), dtype=int)
        for label in labels:
            if label in EMOTION_LABELS:
                idx = EMOTION_LABELS.index(label)
                label_vector[idx] = 1
        y_true.append(label_vector)

    y_true = np.array(y_true)

    # Extract probabilities
    prob_cols = [f'pred_prob_{emotion}' for emotion in EMOTION_LABELS]
    y_probs = df[prob_cols].values

    logger.info(f"Ground truth shape: {y_true.shape}")
    logger.info(f"Probabilities shape: {y_probs.shape}")
    logger.info(f"Label distribution: {y_true.sum(axis=0)}")

    return y_true, y_probs


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Evaluate performance at a specific threshold.

    Args:
        y_true: Ground truth binary labels [n_samples, n_labels]
        y_probs: Predicted probabilities [n_samples, n_labels]
        threshold: Classification threshold

    Returns:
        Dictionary with metrics at this threshold
    """
    # Apply threshold
    y_pred = (y_probs >= threshold).astype(int)

    # Compute metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )

    # Weighted F1 (by support)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Sample-wise F1 (average F1 across samples)
    sample_f1s = []
    for i in range(len(y_true)):
        if y_true[i].sum() == 0 and y_pred[i].sum() == 0:
            # Both empty - perfect match
            sample_f1s.append(1.0)
        elif y_true[i].sum() == 0 or y_pred[i].sum() == 0:
            # One empty, one not - zero F1
            sample_f1s.append(0.0)
        else:
            tp = (y_true[i] & y_pred[i]).sum()
            precision = tp / y_pred[i].sum() if y_pred[i].sum() > 0 else 0
            recall = tp / y_true[i].sum() if y_true[i].sum() > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            sample_f1s.append(f1)

    f1_sample = np.mean(sample_f1s)

    # Count how many labels are actually predicted
    labels_predicted = (y_pred.sum(axis=0) > 0).sum()

    return {
        'threshold': threshold,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'f1_sample': f1_sample,
        'labels_predicted': int(labels_predicted)
    }


def per_class_threshold_analysis(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: List[float]
) -> pd.DataFrame:
    """
    Analyze optimal threshold for each emotion class independently.

    Args:
        y_true: Ground truth binary labels [n_samples, n_labels]
        y_probs: Predicted probabilities [n_samples, n_labels]
        thresholds: List of thresholds to try

    Returns:
        DataFrame with per-class optimal thresholds
    """
    logger.info("Running per-class threshold analysis...")

    results = []
    for i, emotion in enumerate(EMOTION_LABELS):
        best_f1 = 0.0
        best_threshold = 0.5
        best_precision = 0.0
        best_recall = 0.0

        support = y_true[:, i].sum()

        for threshold in thresholds:
            y_pred = (y_probs[:, i] >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true[:, i], y_pred, average='binary', zero_division=0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall

        results.append({
            'emotion': emotion,
            'optimal_threshold': best_threshold,
            'best_f1': best_f1,
            'precision': best_precision,
            'recall': best_recall,
            'support': int(support)
        })

    df = pd.DataFrame(results)
    df = df.sort_values('best_f1', ascending=False).reset_index(drop=True)

    return df


def plot_threshold_curves(
    sweep_results: pd.DataFrame,
    output_dir: str
):
    """
    Plot threshold vs performance curves as separate figures.

    Args:
        sweep_results: DataFrame with sweep results
        output_dir: Directory to save output PNGs
    """
    logger.info("Generating threshold curves plots...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot 1: Macro metrics
    logger.info("  Creating macro-averaged metrics plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sweep_results['threshold'], sweep_results['f1_macro'],
             'o-', linewidth=2, markersize=6, label='F1', color='#0000FF')
    ax.plot(sweep_results['threshold'], sweep_results['precision_macro'],
             's--', linewidth=1.5, markersize=5, label='Precision', color='#00FF00')
    ax.plot(sweep_results['threshold'], sweep_results['recall_macro'],
             '^--', linewidth=1.5, markersize=5, label='Recall', color='#FF0000')
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Macro Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    macro_file = output_path / 'threshold_macro_metrics.png'
    plt.savefig(macro_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"    Saved to: {macro_file}")

    # Plot 2: Micro metrics
    logger.info("  Creating micro-averaged metrics plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sweep_results['threshold'], sweep_results['f1_micro'],
             'o-', linewidth=2, markersize=6, label='F1', color='#0000FF')
    ax.plot(sweep_results['threshold'], sweep_results['precision_micro'],
             's--', linewidth=1.5, markersize=5, label='Precision', color='#00FF00')
    ax.plot(sweep_results['threshold'], sweep_results['recall_micro'],
             '^--', linewidth=1.5, markersize=5, label='Recall', color='#FF0000')
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Micro Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    micro_file = output_path / 'threshold_micro_metrics.png'
    plt.savefig(micro_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"    Saved to: {micro_file}")

    # Plot 3: All F1 variants
    logger.info("  Creating F1 comparison plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sweep_results['threshold'], sweep_results['f1_macro'],
             'o-', linewidth=2, markersize=6, label='F1 (Macro)', color='#0000FF')
    ax.plot(sweep_results['threshold'], sweep_results['f1_micro'],
             's-', linewidth=2, markersize=6, label='F1 (Micro)', color='#00FF00')
    ax.plot(sweep_results['threshold'], sweep_results['f1_weighted'],
             '^-', linewidth=2, markersize=6, label='F1 (Weighted)', color='#FF0000')
    ax.plot(sweep_results['threshold'], sweep_results['f1_sample'],
             'd-', linewidth=2, markersize=6, label='F1 (Sample)', color='#FF00FF')
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    f1_file = output_path / 'threshold_f1_comparison.png'
    plt.savefig(f1_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"    Saved to: {f1_file}")

    # Plot 4: Number of labels predicted
    logger.info("  Creating label coverage plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sweep_results['threshold'], sweep_results['labels_predicted'],
             'o-', linewidth=2, markersize=6, color='#0000FF', label='Labels Predicted')
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Labels Predicted', fontsize=12, fontweight='bold')
    ax.set_title('Label Coverage vs Threshold', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 30)  # Increased to 30 to show the line at 28
    ax.axhline(y=28, color='#FF0000', linestyle='--', linewidth=2, label='All 28 labels')
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    coverage_file = output_path / 'threshold_label_coverage.png'
    plt.savefig(coverage_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"    Saved to: {coverage_file}")


def plot_per_class_thresholds(
    per_class_results: pd.DataFrame,
    output_path: str
):
    """
    Plot per-class optimal thresholds.

    Args:
        per_class_results: DataFrame with per-class results
        output_path: Path to save output PNG
    """
    logger.info("Generating per-class threshold plot...")

    fig, ax = plt.subplots(figsize=(14, 10))

    emotions = per_class_results['emotion'].values
    thresholds = per_class_results['optimal_threshold'].values
    f1_scores = per_class_results['best_f1'].values

    # Create positions for bars
    y_pos = np.arange(len(emotions))

    # Normalize F1 scores for color mapping
    max_f1 = f1_scores.max()
    if max_f1 > 0:
        normalized_f1 = f1_scores / max_f1
    else:
        normalized_f1 = f1_scores

    # Color by F1 score
    colors = plt.cm.RdYlGn(normalized_f1)

    # Create horizontal bar chart
    bars = ax.barh(y_pos, thresholds, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(emotions, fontsize=9)
    ax.set_xlabel('Optimal Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_title('Per-Emotion Optimal Thresholds (Ranked by F1)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)

    # Add threshold and F1 labels
    for i, (bar, thresh, f1) in enumerate(zip(bars, thresholds, f1_scores)):
        label_x = thresh + 0.02
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
               f'{thresh:.2f} (F1={f1:.3f})',
               ha='left', va='center', fontsize=8, fontweight='bold')

    # Add reference lines
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Default (0.5)')

    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper right', framealpha=0.9)

    # Add colorbar to show F1 score scale
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=plt.cm.RdYlGn, norm=Normalize(vmin=0, vmax=max_f1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('F1 Score', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Per-class threshold plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run threshold sweep analysis on validation predictions'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        help='Path to validation predictions CSV (with probabilities)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (alternative to --predictions, will generate predictions)'
    )
    parser.add_argument(
        '--thresholds',
        type=str,
        default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
        help='Comma-separated list of thresholds to try (default: 0.1,0.2,...,0.9)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Parse thresholds
    thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    logger.info(f"Testing thresholds: {thresholds}")

    # Load predictions
    if args.predictions:
        y_true, y_probs = load_validation_predictions(args.predictions)
    elif args.checkpoint:
        logger.error("Checkpoint-based prediction generation not yet implemented. Please use --predictions.")
        return 1
    else:
        logger.error("Must provide either --predictions or --checkpoint")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run threshold sweep
    logger.info("\n" + "="*70)
    logger.info("THRESHOLD SWEEP ANALYSIS")
    logger.info("="*70)

    sweep_results = []
    for threshold in thresholds:
        logger.info(f"\nEvaluating threshold: {threshold:.2f}")
        metrics = evaluate_at_threshold(y_true, y_probs, threshold)
        sweep_results.append(metrics)

        logger.info(f"  F1 (Macro):  {metrics['f1_macro']:.4f}")
        logger.info(f"  F1 (Micro):  {metrics['f1_micro']:.4f}")
        logger.info(f"  Labels predicted: {metrics['labels_predicted']}/28")

    # Convert to DataFrame
    df_sweep = pd.DataFrame(sweep_results)

    # Save sweep results
    sweep_csv_path = output_dir / 'threshold_sweep_results.csv'
    df_sweep.to_csv(sweep_csv_path, index=False)
    logger.info(f"\nSweep results saved to: {sweep_csv_path}")

    # Find optimal threshold
    optimal_idx = df_sweep['f1_macro'].idxmax()
    optimal_threshold = df_sweep.loc[optimal_idx, 'threshold']
    optimal_f1_macro = df_sweep.loc[optimal_idx, 'f1_macro']

    logger.info("\n" + "="*70)
    logger.info("OPTIMAL THRESHOLD (by Macro F1)")
    logger.info("="*70)
    logger.info(f"Threshold: {optimal_threshold:.2f}")
    logger.info(f"F1 (Macro): {optimal_f1_macro:.4f}")
    logger.info(f"F1 (Micro): {df_sweep.loc[optimal_idx, 'f1_micro']:.4f}")
    logger.info(f"Precision (Macro): {df_sweep.loc[optimal_idx, 'precision_macro']:.4f}")
    logger.info(f"Recall (Macro): {df_sweep.loc[optimal_idx, 'recall_macro']:.4f}")
    logger.info(f"Labels predicted: {df_sweep.loc[optimal_idx, 'labels_predicted']}/28")

    # Per-class threshold analysis
    logger.info("\n" + "="*70)
    logger.info("PER-CLASS OPTIMAL THRESHOLDS")
    logger.info("="*70)

    df_per_class = per_class_threshold_analysis(y_true, y_probs, thresholds)

    per_class_csv_path = output_dir / 'per_class_optimal_thresholds.csv'
    df_per_class.to_csv(per_class_csv_path, index=False)
    logger.info(f"Per-class results saved to: {per_class_csv_path}")

    logger.info("\nTop 5 emotions (by F1 at optimal threshold):")
    for _, row in df_per_class.head(5).iterrows():
        logger.info(f"  {row['emotion']:15s} threshold={row['optimal_threshold']:.2f}  F1={row['best_f1']:.3f}")

    logger.info("\nBottom 5 emotions (by F1 at optimal threshold):")
    for _, row in df_per_class.tail(5).iterrows():
        logger.info(f"  {row['emotion']:15s} threshold={row['optimal_threshold']:.2f}  F1={row['best_f1']:.3f}")

    # Generate visualizations
    logger.info("\n" + "="*70)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("="*70)

    plot_threshold_curves(df_sweep, output_dir)
    plot_per_class_thresholds(df_per_class, output_dir / 'per_class_thresholds.png')

    # Save summary JSON
    summary = {
        'optimal_global_threshold': float(optimal_threshold),
        'optimal_f1_macro': float(optimal_f1_macro),
        'optimal_f1_micro': float(df_sweep.loc[optimal_idx, 'f1_micro']),
        'optimal_precision_macro': float(df_sweep.loc[optimal_idx, 'precision_macro']),
        'optimal_recall_macro': float(df_sweep.loc[optimal_idx, 'recall_macro']),
        'labels_predicted_at_optimal': int(df_sweep.loc[optimal_idx, 'labels_predicted']),
        'per_class_thresholds': df_per_class.set_index('emotion')['optimal_threshold'].to_dict()
    }

    summary_json_path = output_dir / 'threshold_summary.json'
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_json_path}")

    logger.info("\n" + "="*70)
    logger.info("THRESHOLD SWEEP COMPLETE")
    logger.info("="*70)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - threshold_sweep_results.csv")
    logger.info(f"  - per_class_optimal_thresholds.csv")
    logger.info(f"  - threshold_macro_metrics.png")
    logger.info(f"  - threshold_micro_metrics.png")
    logger.info(f"  - threshold_f1_comparison.png")
    logger.info(f"  - threshold_label_coverage.png")
    logger.info(f"  - per_class_thresholds.png")
    logger.info(f"  - threshold_summary.json")

    return 0


if __name__ == '__main__':
    exit(main())
