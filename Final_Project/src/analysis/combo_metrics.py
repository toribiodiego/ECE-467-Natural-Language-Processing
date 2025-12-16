#!/usr/bin/env python3
"""
Combination-Level Performance Metrics

Evaluates how well the model predicts specific emotion combinations (pairs/triples)
by computing recall and Jaccard similarity for the most frequent multi-label patterns
in the test set.

Metrics:
- Combo Recall: What fraction of ground truth combination occurrences were correctly predicted
- Jaccard Similarity: Intersection over union of predicted vs true labels for samples with this combo
- Exact Match Rate: What fraction of predictions exactly matched the ground truth combination

Usage:
    python -m src.analysis.combo_metrics \
        --predictions artifacts/predictions/roberta-large-test-predictions.csv \
        --output artifacts/stats/cooccurrence/combo_metrics.csv \
        --top-n 20
"""

import argparse
import logging
from pathlib import Path
from typing import List, Set, Tuple, Dict
from collections import Counter
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GoEmotions emotion labels
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


def parse_label_string(label_str: str) -> Set[str]:
    """Parse comma-separated label string into set of labels."""
    if pd.isna(label_str) or label_str.strip() == '':
        return set()
    return set(label.strip() for label in label_str.split(','))


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity (intersection over union)."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_combo_metrics(
    predictions_file: str,
    top_n: int = 20
) -> pd.DataFrame:
    """Compute combination-level metrics for top-N most frequent combos.

    Args:
        predictions_file: Path to predictions CSV with true_labels and pred_labels
        top_n: Number of top combinations to analyze

    Returns:
        DataFrame with combo-level metrics
    """
    logger.info(f"Loading predictions from: {predictions_file}")
    df = pd.read_csv(predictions_file)

    if 'true_labels' not in df.columns or 'pred_labels' not in df.columns:
        raise ValueError("Predictions file must contain 'true_labels' and 'pred_labels' columns")

    logger.info(f"Processing {len(df)} samples...")

    # Find all multi-label combinations in ground truth
    combo_counts = Counter()
    combo_samples = {}  # Maps combo -> list of sample indices

    for idx, row in df.iterrows():
        true_labels = parse_label_string(row['true_labels'])

        if len(true_labels) < 2:
            continue  # Skip single-label samples

        # Create sorted tuple for consistent representation
        combo = tuple(sorted(true_labels))
        combo_counts[combo] += 1

        if combo not in combo_samples:
            combo_samples[combo] = []
        combo_samples[combo].append(idx)

    logger.info(f"Found {len(combo_counts)} unique multi-label combinations")
    logger.info(f"Total multi-label samples: {sum(combo_counts.values())}")

    # Get top-N most frequent combinations
    top_combos = combo_counts.most_common(top_n)

    logger.info(f"\nAnalyzing top {min(top_n, len(top_combos))} combinations:")
    for combo, count in top_combos[:5]:
        logger.info(f"  {' + '.join(combo):40s}: {count:3d} occurrences")

    # Compute metrics for each combination
    results = []

    for combo, count in top_combos:
        combo_set = set(combo)
        sample_indices = combo_samples[combo]

        # Metrics
        correct_predictions = 0  # Predicted combo is exactly the true combo
        partial_matches = 0      # Predicted labels overlap with true combo
        jaccard_scores = []

        for idx in sample_indices:
            pred_labels = parse_label_string(df.loc[idx, 'pred_labels'])

            # Check if prediction exactly matches the combo
            if pred_labels == combo_set:
                correct_predictions += 1

            # Check if there's any overlap
            if len(pred_labels & combo_set) > 0:
                partial_matches += 1

            # Compute Jaccard similarity
            jaccard = jaccard_similarity(combo_set, pred_labels)
            jaccard_scores.append(jaccard)

        # Recall: fraction of combo occurrences that were correctly predicted
        recall = correct_predictions / count if count > 0 else 0

        # Partial recall: fraction with at least one correct label
        partial_recall = partial_matches / count if count > 0 else 0

        # Mean Jaccard similarity across all occurrences
        mean_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0

        results.append({
            'combination': ' + '.join(combo),
            'combo_size': len(combo),
            'frequency': count,
            'exact_match_count': correct_predictions,
            'exact_match_rate': recall,
            'partial_match_count': partial_matches,
            'partial_match_rate': partial_recall,
            'mean_jaccard': mean_jaccard,
            'median_jaccard': np.median(jaccard_scores) if jaccard_scores else 0
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Compute combination-level performance metrics'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions CSV with true_labels and pred_labels'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Number of top combinations to analyze (default: 20)'
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("COMBINATION-LEVEL PERFORMANCE ANALYSIS")
    logger.info("="*70)

    # Compute metrics
    df = compute_combo_metrics(
        predictions_file=args.predictions,
        top_n=args.top_n
    )

    # Save results
    df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    # Summary statistics
    logger.info("\n" + "="*70)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*70)
    logger.info(f"Combinations analyzed: {len(df)}")
    logger.info(f"\nExact Match Performance:")
    logger.info(f"  Mean exact match rate: {df['exact_match_rate'].mean():.3f}")
    logger.info(f"  Median exact match rate: {df['exact_match_rate'].median():.3f}")
    logger.info(f"  Combinations with 0% exact match: {(df['exact_match_rate'] == 0).sum()}")

    logger.info(f"\nPartial Match Performance:")
    logger.info(f"  Mean partial match rate: {df['partial_match_rate'].mean():.3f}")
    logger.info(f"  Median partial match rate: {df['partial_match_rate'].median():.3f}")

    logger.info(f"\nJaccard Similarity:")
    logger.info(f"  Mean Jaccard: {df['mean_jaccard'].mean():.3f}")
    logger.info(f"  Median Jaccard: {df['mean_jaccard'].median():.3f}")

    # Best and worst combinations
    logger.info("\n" + "="*70)
    logger.info("TOP 5 BEST PREDICTED COMBINATIONS (by exact match rate)")
    logger.info("="*70)
    best = df.nlargest(5, 'exact_match_rate')
    for _, row in best.iterrows():
        logger.info(f"{row['combination']:40s}: {row['exact_match_rate']:.1%} "
                    f"({row['exact_match_count']}/{row['frequency']}), "
                    f"Jaccard={row['mean_jaccard']:.3f}")

    logger.info("\n" + "="*70)
    logger.info("TOP 5 WORST PREDICTED COMBINATIONS (by exact match rate)")
    logger.info("="*70)
    worst = df.nsmallest(5, 'exact_match_rate')
    for _, row in worst.iterrows():
        logger.info(f"{row['combination']:40s}: {row['exact_match_rate']:.1%} "
                    f"({row['exact_match_count']}/{row['frequency']}), "
                    f"Jaccard={row['mean_jaccard']:.3f}")


if __name__ == '__main__':
    main()
