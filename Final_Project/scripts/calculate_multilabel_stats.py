"""
Calculate multi-label distribution statistics for GoEmotions dataset.

This script analyzes the GoEmotions dataset to determine what percentage of samples
contain 1 label, 2 labels, or 3+ labels. These statistics will be used for creating
an enhanced class distribution figure with a multi-label inset.
"""

from datasets import load_dataset
import numpy as np
import pandas as pd
from collections import Counter


def calculate_multilabel_distribution(dataset_split):
    """
    Calculate the distribution of samples by number of labels.

    Args:
        dataset_split: A dataset split (train, validation, or test)

    Returns:
        dict: Distribution counts and percentages
    """
    label_counts = []

    for sample in dataset_split:
        num_labels = len(sample["labels"])
        label_counts.append(num_labels)

    # Count occurrences
    count_distribution = Counter(label_counts)
    total_samples = len(label_counts)

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


def main():
    print("Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions")

    print("\nCalculating multi-label distribution for each split...\n")

    splits = ['train', 'validation', 'test']
    results = {}

    for split_name in splits:
        print(f"Analyzing {split_name} split...")
        stats = calculate_multilabel_distribution(dataset[split_name])
        results[split_name] = stats

        print(f"  Total samples: {stats['total_samples']}")
        print(f"  1 label:  {stats['one_label_count']:5d} ({stats['one_label_pct']:.1f}%)")
        print(f"  2 labels: {stats['two_labels_count']:5d} ({stats['two_labels_pct']:.1f}%)")
        print(f"  3+ labels: {stats['three_plus_labels_count']:5d} ({stats['three_plus_labels_pct']:.1f}%)")
        print(f"  Detailed distribution: {stats['detailed_distribution']}")
        print()

    # Create summary DataFrame
    summary_data = []
    for split_name in splits:
        stats = results[split_name]
        summary_data.append({
            'Split': split_name.capitalize(),
            '1 Label (%)': f"{stats['one_label_pct']:.1f}",
            '2 Labels (%)': f"{stats['two_labels_pct']:.1f}",
            '3+ Labels (%)': f"{stats['three_plus_labels_pct']:.1f}",
            'Total Samples': stats['total_samples']
        })

    summary_df = pd.DataFrame(summary_data)

    print("=" * 70)
    print("SUMMARY: Multi-Label Distribution Across Splits")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print()

    # Export to CSV
    output_file = 'multi_label_stats.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"Statistics exported to: {output_file}")

    # Print key finding for caption writing
    train_stats = results['train']
    total_multi_label_pct = train_stats['two_labels_pct'] + train_stats['three_plus_labels_pct']
    print(f"\nKEY FINDING for caption:")
    print(f"  {total_multi_label_pct:.1f}% of training samples contain multiple emotions")
    print(f"  (Breakdown: {train_stats['two_labels_pct']:.1f}% have 2 labels, {train_stats['three_plus_labels_pct']:.1f}% have 3+ labels)")


if __name__ == "__main__":
    main()
