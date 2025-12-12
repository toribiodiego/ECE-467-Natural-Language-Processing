"""
Per-Emotion Metrics Script

Computes per-emotion precision, recall, and F1 scores from model test predictions
to identify which emotions are classified well vs. poorly.

Usage:
    python -m src.analysis.per_emotion_metrics \
        --predictions artifacts/predictions/roberta-large-test-predictions.csv \
        --output artifacts/stats/per_emotion_scores.csv
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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


def load_predictions(predictions_path: str) -> pd.DataFrame:
    """Load predictions CSV file.

    Args:
        predictions_path: Path to predictions CSV

    Returns:
        DataFrame with predictions
    """
    logger.info(f"Loading predictions from: {predictions_path}")
    df = pd.read_csv(predictions_path)
    logger.info(f"Loaded {len(df)} predictions")
    return df


def parse_labels(label_str):
    """Parse comma-separated label string into list.

    Args:
        label_str: Comma-separated string of labels or empty string

    Returns:
        List of label strings
    """
    if pd.isna(label_str) or label_str == '':
        return []
    return [label.strip() for label in str(label_str).split(',')]


def create_binary_matrix(df: pd.DataFrame, column: str) -> np.ndarray:
    """Convert label strings to binary matrix.

    Args:
        df: DataFrame with label column
        column: Name of column containing labels

    Returns:
        Binary matrix (n_samples, n_emotions)
    """
    n_samples = len(df)
    n_emotions = len(EMOTION_LABELS)
    matrix = np.zeros((n_samples, n_emotions), dtype=int)

    for i, label_str in enumerate(df[column]):
        labels = parse_labels(label_str)
        for label in labels:
            if label in EMOTION_LABELS:
                idx = EMOTION_LABELS.index(label)
                matrix[i, idx] = 1

    return matrix


def compute_per_emotion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Compute precision, recall, and F1 for each emotion.

    Args:
        y_true: Binary matrix of true labels (n_samples, n_emotions)
        y_pred: Binary matrix of predicted labels (n_samples, n_emotions)

    Returns:
        DataFrame with per-emotion metrics
    """
    logger.info("Computing per-emotion metrics...")

    # Compute precision, recall, F1 for each emotion
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Create results dataframe
    results = []
    for i, emotion in enumerate(EMOTION_LABELS):
        results.append({
            'emotion': emotion,
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        })

    df_results = pd.DataFrame(results)

    # Sort by F1 score descending
    df_results = df_results.sort_values('f1', ascending=False).reset_index(drop=True)

    # Add rank column
    df_results.insert(0, 'rank', range(1, len(df_results) + 1))

    logger.info("Metrics computed successfully")
    return df_results


def apply_threshold(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Apply threshold to probability columns to get predicted labels.

    Args:
        df: DataFrame with pred_prob_* columns
        threshold: Classification threshold

    Returns:
        DataFrame with pred_labels column populated
    """
    logger.info(f"Applying threshold {threshold} to predictions...")

    pred_labels_list = []

    for _, row in df.iterrows():
        predicted_emotions = []
        for emotion in EMOTION_LABELS:
            prob_col = f'pred_prob_{emotion}'
            if prob_col in df.columns and row[prob_col] >= threshold:
                predicted_emotions.append(emotion)

        # Join predicted emotions into comma-separated string
        pred_labels_list.append(','.join(predicted_emotions) if predicted_emotions else '')

    df = df.copy()
    df['pred_labels'] = pred_labels_list

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-emotion precision/recall/F1 scores from predictions'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV file for per-emotion scores'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold for predictions (default: 0.5)'
    )

    args = parser.parse_args()

    # Load predictions
    df_pred = load_predictions(args.predictions)

    # Apply threshold to get predicted labels if needed
    if 'pred_labels' not in df_pred.columns or df_pred['pred_labels'].isna().any():
        df_pred = apply_threshold(df_pred, threshold=args.threshold)

    # Convert to binary matrices
    y_true = create_binary_matrix(df_pred, 'true_labels')
    y_pred = create_binary_matrix(df_pred, 'pred_labels')

    logger.info(f"True labels shape: {y_true.shape}")
    logger.info(f"Predicted labels shape: {y_pred.shape}")
    logger.info(f"Total true positives (across all emotions): {y_true.sum()}")
    logger.info(f"Total predicted positives (across all emotions): {y_pred.sum()}")

    # Compute metrics
    df_metrics = compute_per_emotion_metrics(y_true, y_pred)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(output_path, index=False)

    logger.info("\n" + "="*70)
    logger.info("PER-EMOTION METRICS SUMMARY")
    logger.info("="*70)
    logger.info(f"\nTop 5 Emotions (by F1 score):")
    logger.info(df_metrics.head(5).to_string(index=False))
    logger.info(f"\nBottom 5 Emotions (by F1 score):")
    logger.info(df_metrics.tail(5).to_string(index=False))
    logger.info(f"\nFull results saved to: {output_path}")


if __name__ == '__main__':
    main()
