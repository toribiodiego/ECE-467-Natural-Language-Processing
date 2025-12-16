#!/usr/bin/env python3
"""
Co-occurrence Analysis for Multi-Label Emotion Classification

Computes label co-occurrence frequencies to analyze how often emotion pairs/triples
appear together in the dataset or model predictions. This helps understand:
- Ground truth emotion correlation patterns
- Whether model predictions preserve co-occurrence patterns from training data
- Which emotion combinations are common vs rare

Usage:
    # Compute dataset baseline (ground truth co-occurrence)
    python -m src.analysis.cooccurrence \
        --mode dataset \
        --split test \
        --output artifacts/stats/cooccurrence/dataset_cooccurrence.csv

    # Compute model prediction co-occurrence
    python -m src.analysis.cooccurrence \
        --mode predictions \
        --predictions artifacts/predictions/test_predictions_roberta-large_*.csv \
        --output artifacts/stats/cooccurrence/model_cooccurrence.csv
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from collections import Counter

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


def parse_label_string(label_str: str) -> List[str]:
    """Parse comma-separated label string into list of labels.

    Args:
        label_str: Comma-separated string of labels (e.g., "joy,love,admiration")

    Returns:
        List of individual labels, or empty list if no labels
    """
    if pd.isna(label_str) or label_str.strip() == '':
        return []
    return [label.strip() for label in label_str.split(',')]


def compute_cooccurrence_from_dataset(split: str = 'test') -> pd.DataFrame:
    """Compute label co-occurrence from GoEmotions dataset.

    Args:
        split: Dataset split to analyze ('train', 'validation', or 'test')

    Returns:
        DataFrame with co-occurrence statistics
    """
    logger.info(f"Loading GoEmotions {split} split...")
    from src.data.load_dataset import load_go_emotions

    dataset = load_go_emotions()
    split_data = dataset[split if split != 'val' else 'validation']

    logger.info(f"Computing co-occurrence for {len(split_data)} samples...")

    # Count individual labels and pairs
    single_counts = Counter()
    pair_counts = Counter()
    triple_counts = Counter()

    total_samples = len(split_data)
    samples_with_multiple_labels = 0

    for example in split_data:
        labels = example['labels']

        if len(labels) > 1:
            samples_with_multiple_labels += 1

        # Count individual labels
        for label_idx in labels:
            label_name = EMOTION_LABELS[label_idx]
            single_counts[label_name] += 1

        # Count pairs (combinations, not permutations)
        if len(labels) >= 2:
            for i, label_i in enumerate(labels):
                for label_j in labels[i+1:]:
                    label_i_name = EMOTION_LABELS[label_i]
                    label_j_name = EMOTION_LABELS[label_j]
                    # Sort to ensure consistent ordering
                    pair = tuple(sorted([label_i_name, label_j_name]))
                    pair_counts[pair] += 1

        # Count triples
        if len(labels) >= 3:
            for i, label_i in enumerate(labels):
                for j, label_j in enumerate(labels[i+1:], start=i+1):
                    for label_k in labels[j+1:]:
                        label_i_name = EMOTION_LABELS[label_i]
                        label_j_name = EMOTION_LABELS[label_j]
                        label_k_name = EMOTION_LABELS[label_k]
                        # Sort to ensure consistent ordering
                        triple = tuple(sorted([label_i_name, label_j_name, label_k_name]))
                        triple_counts[triple] += 1

    logger.info(f"Multi-label samples: {samples_with_multiple_labels}/{total_samples} "
                f"({100*samples_with_multiple_labels/total_samples:.1f}%)")

    # Create DataFrame with pair statistics
    pair_data = []
    for (label1, label2), count in pair_counts.most_common():
        freq1 = single_counts[label1]
        freq2 = single_counts[label2]

        # Conditional probabilities
        p_label2_given_label1 = count / freq1 if freq1 > 0 else 0
        p_label1_given_label2 = count / freq2 if freq2 > 0 else 0

        # Lift: how much more likely are they to co-occur than by chance
        p_label1 = freq1 / total_samples
        p_label2 = freq2 / total_samples
        p_both = count / total_samples
        expected_both = p_label1 * p_label2
        lift = p_both / expected_both if expected_both > 0 else 0

        pair_data.append({
            'label1': label1,
            'label2': label2,
            'cooccurrence_count': count,
            'label1_count': freq1,
            'label2_count': freq2,
            'p_label2_given_label1': p_label2_given_label1,
            'p_label1_given_label2': p_label1_given_label2,
            'lift': lift,
            'pmi': np.log(lift) if lift > 0 else -np.inf  # Pointwise Mutual Information
        })

    df = pd.DataFrame(pair_data)

    logger.info(f"Found {len(df)} unique emotion pairs")
    logger.info(f"Most common pairs:")
    for _, row in df.head(10).iterrows():
        logger.info(f"  {row['label1']:15s} + {row['label2']:15s}: "
                    f"{row['cooccurrence_count']:4d} occurrences (lift={row['lift']:.2f})")

    return df


def compute_cooccurrence_from_predictions(predictions_file: str) -> pd.DataFrame:
    """Compute label co-occurrence from model predictions CSV.

    Args:
        predictions_file: Path to predictions CSV with 'pred_labels' column

    Returns:
        DataFrame with co-occurrence statistics
    """
    logger.info(f"Loading predictions from: {predictions_file}")
    df_pred = pd.read_csv(predictions_file)

    if 'pred_labels' not in df_pred.columns:
        raise ValueError("Predictions file must contain 'pred_labels' column")

    logger.info(f"Computing co-occurrence for {len(df_pred)} predictions...")

    # Count individual labels and pairs
    single_counts = Counter()
    pair_counts = Counter()

    total_samples = len(df_pred)
    samples_with_multiple_labels = 0
    samples_with_predictions = 0

    for _, row in df_pred.iterrows():
        labels = parse_label_string(row['pred_labels'])

        if len(labels) == 0:
            continue

        samples_with_predictions += 1

        if len(labels) > 1:
            samples_with_multiple_labels += 1

        # Count individual labels
        for label in labels:
            single_counts[label] += 1

        # Count pairs
        if len(labels) >= 2:
            for i, label_i in enumerate(labels):
                for label_j in labels[i+1:]:
                    pair = tuple(sorted([label_i, label_j]))
                    pair_counts[pair] += 1

    logger.info(f"Samples with predictions: {samples_with_predictions}/{total_samples}")
    logger.info(f"Multi-label predictions: {samples_with_multiple_labels}/{samples_with_predictions} "
                f"({100*samples_with_multiple_labels/samples_with_predictions:.1f}%)")

    # Create DataFrame with pair statistics
    pair_data = []
    for (label1, label2), count in pair_counts.most_common():
        freq1 = single_counts[label1]
        freq2 = single_counts[label2]

        # Conditional probabilities
        p_label2_given_label1 = count / freq1 if freq1 > 0 else 0
        p_label1_given_label2 = count / freq2 if freq2 > 0 else 0

        # Lift
        p_label1 = freq1 / samples_with_predictions
        p_label2 = freq2 / samples_with_predictions
        p_both = count / samples_with_predictions
        expected_both = p_label1 * p_label2
        lift = p_both / expected_both if expected_both > 0 else 0

        pair_data.append({
            'label1': label1,
            'label2': label2,
            'cooccurrence_count': count,
            'label1_count': freq1,
            'label2_count': freq2,
            'p_label2_given_label1': p_label2_given_label1,
            'p_label1_given_label2': p_label1_given_label2,
            'lift': lift,
            'pmi': np.log(lift) if lift > 0 else -np.inf
        })

    df = pd.DataFrame(pair_data)

    logger.info(f"Found {len(df)} unique predicted emotion pairs")
    logger.info(f"Most common predicted pairs:")
    for _, row in df.head(10).iterrows():
        logger.info(f"  {row['label1']:15s} + {row['label2']:15s}: "
                    f"{row['cooccurrence_count']:4d} occurrences (lift={row['lift']:.2f})")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Compute label co-occurrence statistics for multi-label emotion data'
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['dataset', 'predictions'],
        help='Compute co-occurrence from dataset or model predictions'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'validation', 'test'],
        help='Dataset split (only used in dataset mode)'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        help='Path to predictions CSV (only used in predictions mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'predictions' and not args.predictions:
        parser.error("--predictions is required when mode is 'predictions'")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute co-occurrence
    logger.info("="*70)
    logger.info("CO-OCCURRENCE ANALYSIS")
    logger.info("="*70)

    if args.mode == 'dataset':
        logger.info(f"Mode: Dataset ground truth ({args.split} split)")
        df = compute_cooccurrence_from_dataset(split=args.split)
    else:
        logger.info(f"Mode: Model predictions")
        df = compute_cooccurrence_from_predictions(args.predictions)

    # Save results
    df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    # Summary statistics
    logger.info("\n" + "="*70)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*70)
    logger.info(f"Total unique pairs: {len(df)}")
    logger.info(f"Mean co-occurrence count: {df['cooccurrence_count'].mean():.1f}")
    logger.info(f"Median co-occurrence count: {df['cooccurrence_count'].median():.1f}")
    logger.info(f"Max co-occurrence count: {df['cooccurrence_count'].max()}")
    logger.info(f"Mean lift: {df['lift'].mean():.2f}")
    logger.info(f"Pairs with lift > 1.0 (positive correlation): {(df['lift'] > 1.0).sum()}")
    logger.info(f"Pairs with lift < 1.0 (negative correlation): {(df['lift'] < 1.0).sum()}")


if __name__ == '__main__':
    main()
