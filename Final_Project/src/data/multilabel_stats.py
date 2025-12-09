"""
Multi-label distribution statistics for GoEmotions dataset.

This module provides functions to analyze the GoEmotions dataset and calculate
statistics about the distribution of samples by number of labels (single vs.
multiple emotions).
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
from datasets import Dataset, DatasetDict
import pandas as pd
from dotenv import load_dotenv

from .load_dataset import load_go_emotions

# Configure logging
logger = logging.getLogger(__name__)


def calculate_multilabel_distribution(dataset_split: Dataset) -> Dict[str, Any]:
    """
    Calculate the distribution of samples by number of labels.

    Analyzes a dataset split to determine how many samples contain exactly
    1 label, exactly 2 labels, or 3+ labels. This is useful for understanding
    the multi-label nature of the GoEmotions dataset.

    Args:
        dataset_split: A HuggingFace Dataset split (train, validation, or test)
                      containing samples with a 'labels' field.

    Returns:
        Dictionary containing distribution statistics:
        {
            'total_samples': int,
            'one_label_count': int,
            'two_labels_count': int,
            'three_plus_labels_count': int,
            'one_label_pct': float,
            'two_labels_pct': float,
            'three_plus_labels_pct': float,
            'detailed_distribution': Dict[int, int]  # Maps num_labels -> count
        }

    Example:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("go_emotions")
        >>> stats = calculate_multilabel_distribution(dataset['train'])
        >>> print(f"{stats['one_label_pct']:.1f}% samples have 1 label")
    """
    if 'labels' not in dataset_split.features:
        raise ValueError("Dataset split must contain 'labels' field")

    label_counts = []

    # Count number of labels per sample
    for sample in dataset_split:
        num_labels = len(sample["labels"])
        label_counts.append(num_labels)

    # Count occurrences
    count_distribution = Counter(label_counts)
    total_samples = len(label_counts)

    if total_samples == 0:
        logger.warning("Dataset split is empty")
        return {
            'total_samples': 0,
            'one_label_count': 0,
            'two_labels_count': 0,
            'three_plus_labels_count': 0,
            'one_label_pct': 0.0,
            'two_labels_pct': 0.0,
            'three_plus_labels_pct': 0.0,
            'detailed_distribution': {}
        }

    # Calculate aggregated statistics
    one_label = count_distribution.get(1, 0)
    two_labels = count_distribution.get(2, 0)
    three_plus_labels = sum(count for k, count in count_distribution.items() if k >= 3)

    return {
        'total_samples': total_samples,
        'one_label_count': one_label,
        'two_labels_count': two_labels,
        'three_plus_labels_count': three_plus_labels,
        'one_label_pct': (one_label / total_samples) * 100,
        'two_labels_pct': (two_labels / total_samples) * 100,
        'three_plus_labels_pct': (three_plus_labels / total_samples) * 100,
        'detailed_distribution': dict(sorted(count_distribution.items()))
    }


def calculate_all_splits(dataset: DatasetDict) -> Dict[str, Dict[str, Any]]:
    """
    Calculate multi-label distribution statistics for all splits.

    Args:
        dataset: DatasetDict with 'train', 'validation', and 'test' splits.

    Returns:
        Dictionary mapping split names to their statistics:
        {
            'train': {...},
            'validation': {...},
            'test': {...}
        }

    Example:
        >>> dataset = load_go_emotions()
        >>> all_stats = calculate_all_splits(dataset)
        >>> print(f"Train samples: {all_stats['train']['total_samples']}")
    """
    splits = ['train', 'validation', 'test']
    results = {}

    for split_name in splits:
        if split_name not in dataset:
            logger.warning(f"Split '{split_name}' not found in dataset, skipping")
            continue

        logger.info(f"Analyzing {split_name} split...")
        stats = calculate_multilabel_distribution(dataset[split_name])
        results[split_name] = stats

        logger.info(
            f"  {split_name}: {stats['total_samples']} samples - "
            f"1 label: {stats['one_label_pct']:.1f}%, "
            f"2 labels: {stats['two_labels_pct']:.1f}%, "
            f"3+ labels: {stats['three_plus_labels_pct']:.1f}%"
        )

    return results


def create_summary_dataframe(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a summary DataFrame from multi-label statistics.

    Args:
        results: Dictionary mapping split names to their statistics
                (as returned by calculate_all_splits).

    Returns:
        Pandas DataFrame with columns:
        ['Split', '1 Label (%)', '2 Labels (%)', '3+ Labels (%)', 'Total Samples']

    Example:
        >>> results = calculate_all_splits(dataset)
        >>> df = create_summary_dataframe(results)
        >>> print(df)
    """
    summary_data = []

    for split_name in ['train', 'validation', 'test']:
        if split_name not in results:
            continue

        stats = results[split_name]
        summary_data.append({
            'Split': split_name.capitalize(),
            '1 Label (%)': f"{stats['one_label_pct']:.1f}",
            '2 Labels (%)': f"{stats['two_labels_pct']:.1f}",
            '3+ Labels (%)': f"{stats['three_plus_labels_pct']:.1f}",
            'Total Samples': stats['total_samples']
        })

    return pd.DataFrame(summary_data)


def save_statistics_csv(
    results: Dict[str, Dict[str, Any]],
    output_path: str = 'multi_label_stats.csv'
) -> None:
    """
    Save multi-label statistics to a CSV file.

    Args:
        results: Dictionary mapping split names to their statistics.
        output_path: Path where CSV file should be saved.

    Example:
        >>> results = calculate_all_splits(dataset)
        >>> save_statistics_csv(results, 'output/stats/multi_label_stats.csv')
    """
    df = create_summary_dataframe(results)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Statistics exported to: {output_path}")


def print_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Print a formatted summary of multi-label statistics.

    Args:
        results: Dictionary mapping split names to their statistics.
    """
    df = create_summary_dataframe(results)

    print("=" * 70)
    print("SUMMARY: Multi-Label Distribution Across Splits")
    print("=" * 70)
    print(df.to_string(index=False))
    print()

    # Print key finding for caption writing
    if 'train' in results:
        train_stats = results['train']
        total_multi_label_pct = (
            train_stats['two_labels_pct'] + train_stats['three_plus_labels_pct']
        )
        print("KEY FINDING for caption:")
        print(f"  {total_multi_label_pct:.1f}% of training samples contain multiple emotions")
        print(
            f"  (Breakdown: {train_stats['two_labels_pct']:.1f}% have 2 labels, "
            f"{train_stats['three_plus_labels_pct']:.1f}% have 3+ labels)"
        )


def main() -> None:
    """
    CLI entry point for calculating and saving multi-label statistics.

    Loads the GoEmotions dataset, calculates statistics for all splits,
    prints a summary, and saves results to CSV.

    Environment variables (loaded from .env):
        OUTPUT_DIR: Base directory for output files (default: 'output')
    """
    # Load environment variables from .env file
    load_dotenv()

    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Loading GoEmotions dataset...")
    dataset = load_go_emotions()

    print("\nCalculating multi-label distribution for each split...\n")
    results = calculate_all_splits(dataset)

    # Print detailed statistics for each split
    for split_name in ['train', 'validation', 'test']:
        if split_name not in results:
            continue

        stats = results[split_name]
        print(f"{split_name.capitalize()} split:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  1 label:  {stats['one_label_count']:5d} ({stats['one_label_pct']:.1f}%)")
        print(f"  2 labels: {stats['two_labels_count']:5d} ({stats['two_labels_pct']:.1f}%)")
        print(f"  3+ labels: {stats['three_plus_labels_count']:5d} ({stats['three_plus_labels_pct']:.1f}%)")
        print(f"  Detailed distribution: {stats['detailed_distribution']}")
        print()

    # Print summary table
    print_summary(results)

    # Save to CSV (use environment variable if set, otherwise default)
    output_dir = os.getenv('OUTPUT_DIR', 'output')
    output_path = os.path.join(output_dir, 'stats', 'multi_label_stats.csv')

    save_statistics_csv(results, output_path)
    print(f"\nStatistics exported to: {output_path}")


if __name__ == "__main__":
    main()
