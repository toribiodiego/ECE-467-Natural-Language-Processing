#!/usr/bin/env python3
"""
Calibration Analysis for Multi-Label Classification

Evaluates how well-calibrated the model's predicted probabilities are by
computing calibration metrics and generating reliability diagrams.

Calibration measures whether predicted probabilities match observed frequencies.
For example, when the model predicts 70% probability, the event should occur
approximately 70% of the time.

Metrics:
- Brier Score: Mean squared difference between predicted probabilities and actual outcomes (lower is better)
- Expected Calibration Error (ECE): Average difference between predicted probability and empirical accuracy across bins
- Reliability Diagram: Visual plot showing predicted vs observed probabilities

Usage:
    python -m src.analysis.calibration \
        --predictions artifacts/predictions/val_epoch10_predictions_roberta-large_*.csv \
        --output artifacts/stats/calibration/

    # Or use checkpoint to generate predictions
    python -m src.analysis.calibration \
        --checkpoint artifacts/models/roberta/roberta-large-20251212-211010 \
        --output artifacts/stats/calibration/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
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

    return y_true, y_probs


def compute_brier_score(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    """
    Compute Brier score for each class and overall.

    Brier score measures the mean squared difference between predicted
    probabilities and actual outcomes. Lower is better (0 = perfect, 1 = worst).

    Args:
        y_true: Ground truth binary labels [n_samples, n_labels]
        y_probs: Predicted probabilities [n_samples, n_labels]

    Returns:
        Dictionary with per-class and overall Brier scores
    """
    logger.info("Computing Brier scores...")

    # Per-class Brier scores
    per_class_brier = {}
    for i, emotion in enumerate(EMOTION_LABELS):
        brier = brier_score_loss(y_true[:, i], y_probs[:, i])
        per_class_brier[emotion] = float(brier)

    # Overall Brier score (average across all classes)
    overall_brier = np.mean(list(per_class_brier.values()))

    # Macro-average (treating each class equally)
    macro_brier = overall_brier

    # Micro-average (flattening all predictions)
    micro_brier = brier_score_loss(y_true.ravel(), y_probs.ravel())

    return {
        'overall': float(overall_brier),
        'macro': float(macro_brier),
        'micro': float(micro_brier),
        'per_class': per_class_brier
    }


def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted probability and observed
    frequency across bins. Perfect calibration = 0.

    Args:
        y_true: Ground truth binary labels [n_samples, n_labels]
        y_probs: Predicted probabilities [n_samples, n_labels]
        n_bins: Number of bins for calibration (default: 10)

    Returns:
        Dictionary with per-class and overall ECE
    """
    logger.info(f"Computing Expected Calibration Error (ECE) with {n_bins} bins...")

    per_class_ece = {}
    for i, emotion in enumerate(EMOTION_LABELS):
        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_probs[:, i], bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Compute ECE
        ece = 0.0
        total_samples = len(y_true[:, i])

        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if mask.sum() > 0:
                bin_accuracy = y_true[:, i][mask].mean()
                bin_confidence = y_probs[:, i][mask].mean()
                bin_size = mask.sum()
                ece += (bin_size / total_samples) * abs(bin_accuracy - bin_confidence)

        per_class_ece[emotion] = float(ece)

    # Overall ECE (average across all classes)
    overall_ece = np.mean(list(per_class_ece.values()))

    return {
        'overall': float(overall_ece),
        'per_class': per_class_ece
    }


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_path: str,
    n_bins: int = 10,
    emotions_to_plot: List[str] = None
):
    """
    Generate reliability diagram showing predicted vs observed probabilities.

    Args:
        y_true: Ground truth binary labels [n_samples, n_labels]
        y_probs: Predicted probabilities [n_samples, n_labels]
        output_path: Path to save output PNG
        n_bins: Number of bins for calibration curve
        emotions_to_plot: List of emotions to plot (default: top 6 by support)
    """
    logger.info("Generating reliability diagram...")

    # Select emotions to plot (top 6 by support if not specified)
    if emotions_to_plot is None:
        support = y_true.sum(axis=0)
        top_indices = np.argsort(support)[-6:][::-1]
        emotions_to_plot = [EMOTION_LABELS[i] for i in top_indices]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, emotion in enumerate(emotions_to_plot):
        ax = axes[idx]
        emotion_idx = EMOTION_LABELS.index(emotion)

        # Compute calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true[:, emotion_idx],
                y_probs[:, emotion_idx],
                n_bins=n_bins,
                strategy='uniform'
            )

            # Plot calibration curve
            ax.plot(mean_predicted_value, fraction_of_positives, 'o-', linewidth=2,
                   markersize=6, label='Model', color='blue')

            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], '--', linewidth=1.5, color='gray', alpha=0.7,
                   label='Perfect Calibration')

            # Compute Brier score and ECE for this emotion
            brier = brier_score_loss(y_true[:, emotion_idx], y_probs[:, emotion_idx])

            # Simple ECE calculation
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_probs[:, emotion_idx], bin_edges[:-1]) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            ece = 0.0
            total = len(y_true[:, emotion_idx])
            for bin_idx in range(n_bins):
                mask = bin_indices == bin_idx
                if mask.sum() > 0:
                    bin_acc = y_true[:, emotion_idx][mask].mean()
                    bin_conf = y_probs[:, emotion_idx][mask].mean()
                    ece += (mask.sum() / total) * abs(bin_acc - bin_conf)

            support = y_true[:, emotion_idx].sum()

            # Customize plot
            ax.set_xlabel('Predicted Probability', fontsize=10, fontweight='bold')
            ax.set_ylabel('Observed Frequency', fontsize=10, fontweight='bold')
            ax.set_title(f'{emotion.capitalize()}\nBrier={brier:.3f}, ECE={ece:.3f}, n={int(support)}',
                        fontsize=11, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=9)

        except Exception as e:
            logger.warning(f"Failed to generate calibration curve for {emotion}: {e}")
            ax.text(0.5, 0.5, f'Insufficient data\nfor {emotion}',
                   ha='center', va='center', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    plt.suptitle('Reliability Diagrams (Top 6 Emotions by Support)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Reliability diagram saved to: {output_file}")


def plot_calibration_summary(
    brier_scores: Dict[str, float],
    ece_scores: Dict[str, float],
    output_path: str
):
    """
    Generate summary plot of calibration metrics across all emotions.

    Args:
        brier_scores: Per-class Brier scores
        ece_scores: Per-class ECE scores
        output_path: Path to save output PNG
    """
    logger.info("Generating calibration summary plot...")

    # Sort emotions by Brier score
    emotions = list(brier_scores.keys())
    brier_values = [brier_scores[e] for e in emotions]
    ece_values = [ece_scores[e] for e in emotions]

    # Sort by Brier score
    sorted_indices = np.argsort(brier_values)
    emotions_sorted = [emotions[i] for i in sorted_indices]
    brier_sorted = [brier_values[i] for i in sorted_indices]
    ece_sorted = [ece_values[i] for i in sorted_indices]

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Plot 1: Brier scores
    ax1 = axes[0]
    y_pos = np.arange(len(emotions_sorted))
    bars1 = ax1.barh(y_pos, brier_sorted, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(emotions_sorted, fontsize=9)
    ax1.set_xlabel('Brier Score (lower is better)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Emotion', fontsize=11, fontweight='bold')
    ax1.set_title('Brier Score by Emotion', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_xlim(0, max(brier_sorted) * 1.1)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, brier_sorted)):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')

    # Plot 2: ECE scores
    ax2 = axes[1]
    bars2 = ax2.barh(y_pos, ece_sorted, color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(emotions_sorted, fontsize=9)
    ax2.set_xlabel('Expected Calibration Error (lower is better)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Emotion', fontsize=11, fontweight='bold')
    ax2.set_title('ECE by Emotion', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, max(ece_sorted) * 1.1)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, ece_sorted)):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Calibration summary plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute calibration metrics and generate reliability diagrams'
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
        '--output',
        type=str,
        required=True,
        help='Output directory for calibration results'
    )
    parser.add_argument(
        '--n-bins',
        type=int,
        default=10,
        help='Number of bins for calibration (default: 10)'
    )

    args = parser.parse_args()

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

    # Compute calibration metrics
    logger.info("\n" + "="*70)
    logger.info("CALIBRATION ANALYSIS")
    logger.info("="*70)

    # Brier score
    logger.info("\n--- Brier Score ---")
    brier_results = compute_brier_score(y_true, y_probs)
    logger.info(f"Overall Brier Score: {brier_results['overall']:.4f}")
    logger.info(f"Macro Brier Score: {brier_results['macro']:.4f}")
    logger.info(f"Micro Brier Score: {brier_results['micro']:.4f}")

    # ECE
    logger.info("\n--- Expected Calibration Error (ECE) ---")
    ece_results = compute_ece(y_true, y_probs, n_bins=args.n_bins)
    logger.info(f"Overall ECE: {ece_results['overall']:.4f}")

    # Best and worst calibrated emotions
    brier_per_class = brier_results['per_class']
    ece_per_class = ece_results['per_class']

    sorted_brier = sorted(brier_per_class.items(), key=lambda x: x[1])
    logger.info("\nBest calibrated emotions (lowest Brier score):")
    for emotion, score in sorted_brier[:5]:
        logger.info(f"  {emotion:15s} Brier={score:.4f}")

    logger.info("\nWorst calibrated emotions (highest Brier score):")
    for emotion, score in sorted_brier[-5:]:
        logger.info(f"  {emotion:15s} Brier={score:.4f}")

    # Save results to CSV
    logger.info("\n--- Saving Results ---")

    # Per-class metrics CSV
    df_calibration = pd.DataFrame({
        'emotion': EMOTION_LABELS,
        'brier_score': [brier_per_class[e] for e in EMOTION_LABELS],
        'ece': [ece_per_class[e] for e in EMOTION_LABELS],
        'support': y_true.sum(axis=0)
    })
    df_calibration = df_calibration.sort_values('brier_score').reset_index(drop=True)

    calibration_csv = output_dir / 'per_class_calibration.csv'
    df_calibration.to_csv(calibration_csv, index=False)
    logger.info(f"Per-class calibration metrics saved to: {calibration_csv}")

    # Summary JSON
    summary = {
        'overall_brier_score': brier_results['overall'],
        'macro_brier_score': brier_results['macro'],
        'micro_brier_score': brier_results['micro'],
        'overall_ece': ece_results['overall'],
        'n_bins': args.n_bins,
        'best_calibrated': sorted_brier[0][0],
        'worst_calibrated': sorted_brier[-1][0],
        'per_class_brier': brier_per_class,
        'per_class_ece': ece_per_class
    }

    summary_json = output_dir / 'calibration_summary.json'
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Calibration summary saved to: {summary_json}")

    # Generate visualizations
    logger.info("\n--- Generating Visualizations ---")

    # Reliability diagrams for top 6 emotions
    reliability_plot = output_dir / 'reliability_diagram.png'
    plot_reliability_diagram(y_true, y_probs, reliability_plot, n_bins=args.n_bins)

    # Calibration summary plot
    summary_plot = output_dir / 'calibration_summary.png'
    plot_calibration_summary(brier_per_class, ece_per_class, summary_plot)

    logger.info("\n" + "="*70)
    logger.info("CALIBRATION ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"  - per_class_calibration.csv")
    logger.info(f"  - calibration_summary.json")
    logger.info(f"  - reliability_diagram.png")
    logger.info(f"  - calibration_summary.png")

    logger.info("\n--- Key Findings ---")
    logger.info(f"Overall Brier Score: {brier_results['overall']:.4f} (0=perfect, 1=worst)")
    logger.info(f"Overall ECE: {ece_results['overall']:.4f} (0=perfect calibration)")
    logger.info(f"Best calibrated: {sorted_brier[0][0]} (Brier={sorted_brier[0][1]:.4f})")
    logger.info(f"Worst calibrated: {sorted_brier[-1][0]} (Brier={sorted_brier[-1][1]:.4f})")

    return 0


if __name__ == '__main__':
    exit(main())
