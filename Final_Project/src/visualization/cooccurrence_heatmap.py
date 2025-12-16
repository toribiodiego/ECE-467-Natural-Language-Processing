#!/usr/bin/env python3
"""
Co-occurrence Heatmap Visualization

Generates heatmap comparing emotion co-occurrence patterns between ground truth
dataset and model predictions to assess whether the model preserves natural
emotion correlation patterns.

Usage:
    python -m src.visualization.cooccurrence_heatmap \
        --dataset-cooccurrence artifacts/stats/cooccurrence/dataset_cooccurrence.csv \
        --prediction-cooccurrence artifacts/stats/cooccurrence/prediction_cooccurrence.csv \
        --output artifacts/stats/cooccurrence/cooccurrence_heatmap.png
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GoEmotions emotion labels (28 emotions)
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


def load_cooccurrence_data(dataset_file: str, prediction_file: str) -> tuple:
    """Load co-occurrence data from both dataset and predictions.

    Args:
        dataset_file: Path to dataset co-occurrence CSV
        prediction_file: Path to prediction co-occurrence CSV

    Returns:
        Tuple of (dataset_df, prediction_df)
    """
    logger.info(f"Loading dataset co-occurrence: {dataset_file}")
    df_dataset = pd.read_csv(dataset_file)

    logger.info(f"Loading prediction co-occurrence: {prediction_file}")
    df_pred = pd.read_csv(prediction_file)

    logger.info(f"Dataset pairs: {len(df_dataset)}")
    logger.info(f"Prediction pairs: {len(df_pred)}")

    return df_dataset, df_pred


def create_cooccurrence_matrix(df: pd.DataFrame, n_emotions: int) -> np.ndarray:
    """Create co-occurrence matrix from pair data.

    Args:
        df: DataFrame with label1, label2, cooccurrence_count columns
        n_emotions: Number of emotions (for matrix size)

    Returns:
        Symmetric co-occurrence matrix
    """
    # Create label to index mapping
    label_to_idx = {label: i for i, label in enumerate(EMOTION_LABELS[:n_emotions])}

    # Initialize matrix
    matrix = np.zeros((n_emotions, n_emotions))

    # Fill matrix
    for _, row in df.iterrows():
        label1 = row['label1']
        label2 = row['label2']
        count = row['cooccurrence_count']

        if label1 in label_to_idx and label2 in label_to_idx:
            i = label_to_idx[label1]
            j = label_to_idx[label2]
            matrix[i, j] = count
            matrix[j, i] = count  # Symmetric

    return matrix


def create_comparison_heatmap(
    dataset_matrix: np.ndarray,
    prediction_matrix: np.ndarray,
    output_file: str,
    top_n: int = 15
):
    """Create side-by-side heatmap comparing dataset vs predictions.

    Args:
        dataset_matrix: Ground truth co-occurrence matrix
        prediction_matrix: Prediction co-occurrence matrix
        output_file: Path to save output figure
        top_n: Number of top emotions to show (by total co-occurrence)
    """
    # Select top-N emotions by total co-occurrence count
    total_cooccurrence = dataset_matrix.sum(axis=0) + dataset_matrix.sum(axis=1)
    top_indices = np.argsort(total_cooccurrence)[-top_n:][::-1]
    top_labels = [EMOTION_LABELS[i] for i in top_indices]

    # Extract submatrices for top emotions
    dataset_sub = dataset_matrix[np.ix_(top_indices, top_indices)]
    prediction_sub = prediction_matrix[np.ix_(top_indices, top_indices)]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Common colormap and normalization
    vmax = max(dataset_sub.max(), prediction_sub.max())
    vmin = 0

    # Plot 1: Dataset ground truth
    sns.heatmap(
        dataset_sub,
        xticklabels=top_labels,
        yticklabels=top_labels,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={'label': 'Co-occurrence Count'},
        ax=ax1
    )
    ax1.set_title('Ground Truth Co-occurrence (Test Set)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Emotion', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Emotion', fontsize=11, fontweight='bold')

    # Plot 2: Model predictions
    sns.heatmap(
        prediction_sub,
        xticklabels=top_labels,
        yticklabels=top_labels,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={'label': 'Co-occurrence Count'},
        ax=ax2
    )
    ax2.set_title('Model Predictions (Threshold 0.5)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Emotion', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Emotion', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Heatmap saved to: {output_path}")

    plt.close()


def create_difference_heatmap(
    dataset_matrix: np.ndarray,
    prediction_matrix: np.ndarray,
    output_file: str,
    top_n: int = 15
):
    """Create heatmap showing difference (predictions - dataset).

    Args:
        dataset_matrix: Ground truth co-occurrence matrix
        prediction_matrix: Prediction co-occurrence matrix
        output_file: Path to save output figure
        top_n: Number of top emotions to show
    """
    # Select top-N emotions by total co-occurrence count
    total_cooccurrence = dataset_matrix.sum(axis=0) + dataset_matrix.sum(axis=1)
    top_indices = np.argsort(total_cooccurrence)[-top_n:][::-1]
    top_labels = [EMOTION_LABELS[i] for i in top_indices]

    # Extract submatrices
    dataset_sub = dataset_matrix[np.ix_(top_indices, top_indices)]
    prediction_sub = prediction_matrix[np.ix_(top_indices, top_indices)]

    # Compute difference (negative = under-predicted, positive = over-predicted)
    difference = prediction_sub - dataset_sub

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use diverging colormap centered at 0
    vmax = max(abs(difference.min()), abs(difference.max()))
    sns.heatmap(
        difference,
        xticklabels=top_labels,
        yticklabels=top_labels,
        annot=True,
        fmt='.0f',
        cmap='RdBu_r',  # Red for under-prediction, blue for over-prediction
        center=0,
        vmin=-vmax,
        vmax=vmax,
        square=True,
        cbar_kws={'label': 'Difference (Pred - Dataset)'},
        ax=ax
    )
    ax.set_title('Prediction Error: Under-prediction (Red) vs Over-prediction (Blue)',
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Emotion', fontsize=11, fontweight='bold')
    ax.set_ylabel('Emotion', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Difference heatmap saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate co-occurrence heatmap comparing dataset vs predictions'
    )
    parser.add_argument(
        '--dataset-cooccurrence',
        type=str,
        required=True,
        help='Path to dataset co-occurrence CSV'
    )
    parser.add_argument(
        '--prediction-cooccurrence',
        type=str,
        required=True,
        help='Path to prediction co-occurrence CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output PNG file path'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=15,
        help='Number of top emotions to display (default: 15)'
    )
    parser.add_argument(
        '--difference-heatmap',
        action='store_true',
        help='Also generate difference heatmap'
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("CO-OCCURRENCE HEATMAP VISUALIZATION")
    logger.info("="*70)

    # Load data
    df_dataset, df_pred = load_cooccurrence_data(
        args.dataset_cooccurrence,
        args.prediction_cooccurrence
    )

    # Create co-occurrence matrices
    logger.info("Creating co-occurrence matrices...")
    n_emotions = len(EMOTION_LABELS)
    dataset_matrix = create_cooccurrence_matrix(df_dataset, n_emotions)
    prediction_matrix = create_cooccurrence_matrix(df_pred, n_emotions)

    # Generate comparison heatmap
    logger.info(f"Generating comparison heatmap (top {args.top_n} emotions)...")
    create_comparison_heatmap(
        dataset_matrix,
        prediction_matrix,
        args.output,
        top_n=args.top_n
    )

    # Optionally generate difference heatmap
    if args.difference_heatmap:
        logger.info("Generating difference heatmap...")
        difference_file = str(Path(args.output).parent / 'cooccurrence_difference.png')
        create_difference_heatmap(
            dataset_matrix,
            prediction_matrix,
            difference_file,
            top_n=args.top_n
        )

    logger.info("\n" + "="*70)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("="*70)
    logger.info(f"Output saved to: {args.output}")


if __name__ == '__main__':
    main()
