"""
Split consistency validation for GoEmotions dataset.

This module validates that train/validation/test splits have consistent
distributions of labels and text lengths. Identifies potential distribution
drift that could affect model generalization.

Usage:
    python -m src.analysis.split_consistency
"""

import logging
import os
import csv
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
from scipy import stats
from dotenv import load_dotenv

from src.data.load_dataset import load_go_emotions, get_label_names

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Statistical test thresholds
SIGNIFICANCE_LEVEL = 0.05  # p-value threshold for statistical tests
KS_THRESHOLD = 0.1  # Maximum acceptable KS statistic for length distributions
CHI2_THRESHOLD = 0.1  # Maximum acceptable chi-square p-value for independence


# ============================================================================
# Label Distribution Analysis
# ============================================================================

def calculate_label_distribution_by_split(
    dataset,
    label_names: List[str]
) -> Dict[str, Dict[str, int]]:
    """
    Calculate label frequency for each split.

    Args:
        dataset: DatasetDict with train/validation/test splits
        label_names: List of emotion label names

    Returns:
        Dictionary mapping split name to label counts:
        {
            'train': {'admiration': 1234, 'amusement': 567, ...},
            'validation': {...},
            'test': {...}
        }
    """
    split_distributions = {}

    for split_name in ['train', 'validation', 'test']:
        if split_name not in dataset:
            continue

        label_counts = Counter()
        split = dataset[split_name]

        for sample in split:
            for label_idx in sample['labels']:
                emotion = label_names[label_idx]
                label_counts[emotion] += 1

        split_distributions[split_name] = dict(label_counts)

    return split_distributions


def chi_square_test_label_distribution(
    split_distributions: Dict[str, Dict[str, int]],
    label_names: List[str]
) -> Dict[str, any]:
    """
    Perform chi-square test to check if label distributions are consistent
    across splits.

    Args:
        split_distributions: Label counts per split
        label_names: List of all emotion labels

    Returns:
        Dictionary with test results:
        {
            'statistic': float,
            'p_value': float,
            'is_consistent': bool,
            'expected_frequencies': array,
            'observed_frequencies': array
        }
    """
    # Build contingency table: rows = splits, columns = labels
    splits = ['train', 'validation', 'test']
    observed = []

    for split_name in splits:
        if split_name not in split_distributions:
            continue

        counts = [split_distributions[split_name].get(label, 0)
                  for label in label_names]
        observed.append(counts)

    observed = np.array(observed)

    # Perform chi-square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)

    # Check if distributions are consistent
    is_consistent = p_value >= SIGNIFICANCE_LEVEL

    return {
        'statistic': float(chi2_stat),
        'p_value': float(p_value),
        'degrees_of_freedom': int(dof),
        'is_consistent': is_consistent,
        'expected_frequencies': expected.tolist(),
        'observed_frequencies': observed.tolist(),
        'threshold': SIGNIFICANCE_LEVEL
    }


def calculate_label_proportion_differences(
    split_distributions: Dict[str, Dict[str, int]],
    label_names: List[str]
) -> List[Dict[str, any]]:
    """
    Calculate per-label proportion differences across splits.

    Args:
        split_distributions: Label counts per split
        label_names: List of all emotion labels

    Returns:
        List of dictionaries with per-label analysis:
        [
            {
                'label': 'admiration',
                'train_count': 1234,
                'train_pct': 5.2,
                'val_count': 123,
                'val_pct': 5.1,
                'test_count': 234,
                'test_pct': 5.3,
                'max_diff_pct': 0.2
            },
            ...
        ]
    """
    results = []

    # Calculate totals per split
    totals = {}
    for split_name in ['train', 'validation', 'test']:
        if split_name in split_distributions:
            totals[split_name] = sum(split_distributions[split_name].values())

    for label in label_names:
        proportions = {}
        counts = {}

        for split_name in ['train', 'validation', 'test']:
            if split_name in split_distributions:
                count = split_distributions[split_name].get(label, 0)
                counts[split_name] = count
                proportions[split_name] = (count / totals[split_name] * 100) if totals[split_name] > 0 else 0

        # Calculate maximum difference in proportions
        if proportions:
            max_diff = max(proportions.values()) - min(proportions.values())
        else:
            max_diff = 0

        results.append({
            'label': label,
            'train_count': counts.get('train', 0),
            'train_pct': proportions.get('train', 0),
            'val_count': counts.get('validation', 0),
            'val_pct': proportions.get('validation', 0),
            'test_count': counts.get('test', 0),
            'test_pct': proportions.get('test', 0),
            'max_diff_pct': max_diff
        })

    # Sort by maximum difference (descending)
    results.sort(key=lambda x: x['max_diff_pct'], reverse=True)

    return results


# ============================================================================
# Text Length Distribution Analysis
# ============================================================================

def calculate_text_lengths_by_split(dataset) -> Dict[str, List[int]]:
    """
    Calculate character lengths for each split.

    Args:
        dataset: DatasetDict with train/validation/test splits

    Returns:
        Dictionary mapping split name to list of character lengths:
        {
            'train': [123, 45, 678, ...],
            'validation': [...],
            'test': [...]
        }
    """
    length_distributions = {}

    for split_name in ['train', 'validation', 'test']:
        if split_name not in dataset:
            continue

        lengths = []
        split = dataset[split_name]

        for sample in split:
            lengths.append(len(sample['text']))

        length_distributions[split_name] = lengths

    return length_distributions


def ks_test_text_lengths(
    length_distributions: Dict[str, List[int]]
) -> Dict[str, Dict[str, any]]:
    """
    Perform Kolmogorov-Smirnov tests to compare text length distributions
    across splits.

    Args:
        length_distributions: Character lengths per split

    Returns:
        Dictionary with test results for each pair:
        {
            'train_vs_val': {'statistic': 0.03, 'p_value': 0.25, 'is_consistent': True},
            'train_vs_test': {...},
            'val_vs_test': {...}
        }
    """
    results = {}
    comparisons = [
        ('train', 'validation', 'train_vs_val'),
        ('train', 'test', 'train_vs_test'),
        ('validation', 'test', 'val_vs_test')
    ]

    for split1, split2, key in comparisons:
        if split1 in length_distributions and split2 in length_distributions:
            data1 = length_distributions[split1]
            data2 = length_distributions[split2]

            ks_stat, p_value = stats.ks_2samp(data1, data2)

            results[key] = {
                'statistic': float(ks_stat),
                'p_value': float(p_value),
                'is_consistent': ks_stat <= KS_THRESHOLD,
                'threshold': KS_THRESHOLD
            }

    return results


def calculate_length_statistics_by_split(
    length_distributions: Dict[str, List[int]]
) -> List[Dict[str, any]]:
    """
    Calculate summary statistics for text lengths per split.

    Args:
        length_distributions: Character lengths per split

    Returns:
        List of dictionaries with per-split statistics:
        [
            {
                'split': 'train',
                'count': 43410,
                'mean': 104.5,
                'std': 42.3,
                'median': 98.0,
                'min': 5,
                'max': 547,
                'q25': 72.0,
                'q75': 132.0
            },
            ...
        ]
    """
    results = []

    for split_name in ['train', 'validation', 'test']:
        if split_name not in length_distributions:
            continue

        lengths = length_distributions[split_name]

        results.append({
            'split': split_name,
            'count': len(lengths),
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'median': float(np.median(lengths)),
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths)),
            'q25': float(np.percentile(lengths, 25)),
            'q75': float(np.percentile(lengths, 75))
        })

    return results


# ============================================================================
# Multi-Label Distribution Analysis
# ============================================================================

def calculate_multilabel_distribution_by_split(dataset) -> Dict[str, Dict[str, int]]:
    """
    Calculate multi-label distribution (1 label, 2 labels, 3+ labels) per split.

    Args:
        dataset: DatasetDict with train/validation/test splits

    Returns:
        Dictionary mapping split to label count distribution:
        {
            'train': {'1_label': 31234, '2_labels': 10123, '3plus_labels': 2053},
            'validation': {...},
            'test': {...}
        }
    """
    distributions = {}

    for split_name in ['train', 'validation', 'test']:
        if split_name not in dataset:
            continue

        counts = {'1_label': 0, '2_labels': 0, '3plus_labels': 0}
        split = dataset[split_name]

        for sample in split:
            num_labels = len(sample['labels'])

            if num_labels == 1:
                counts['1_label'] += 1
            elif num_labels == 2:
                counts['2_labels'] += 1
            else:
                counts['3plus_labels'] += 1

        distributions[split_name] = counts

    return distributions


def chi_square_test_multilabel_distribution(
    distributions: Dict[str, Dict[str, int]]
) -> Dict[str, any]:
    """
    Perform chi-square test to check if multi-label distributions are
    consistent across splits.

    Args:
        distributions: Multi-label counts per split

    Returns:
        Dictionary with test results
    """
    # Build contingency table
    splits = ['train', 'validation', 'test']
    categories = ['1_label', '2_labels', '3plus_labels']

    observed = []
    for split_name in splits:
        if split_name in distributions:
            counts = [distributions[split_name][cat] for cat in categories]
            observed.append(counts)

    observed = np.array(observed)

    # Perform chi-square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)

    is_consistent = p_value >= SIGNIFICANCE_LEVEL

    return {
        'statistic': float(chi2_stat),
        'p_value': float(p_value),
        'degrees_of_freedom': int(dof),
        'is_consistent': is_consistent,
        'threshold': SIGNIFICANCE_LEVEL
    }


# ============================================================================
# Export Functions
# ============================================================================

def export_label_proportion_analysis(
    results: List[Dict[str, any]],
    output_path: str
) -> str:
    """
    Export per-label proportion analysis to CSV.

    Args:
        results: List of per-label statistics
        output_path: Where to save the CSV file

    Returns:
        Absolute path to saved CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'label', 'train_count', 'train_pct', 'val_count', 'val_pct',
            'test_count', 'test_pct', 'max_diff_pct'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow({
                'label': row['label'],
                'train_count': row['train_count'],
                'train_pct': f"{row['train_pct']:.3f}",
                'val_count': row['val_count'],
                'val_pct': f"{row['val_pct']:.3f}",
                'test_count': row['test_count'],
                'test_pct': f"{row['test_pct']:.3f}",
                'max_diff_pct': f"{row['max_diff_pct']:.3f}"
            })

    return os.path.abspath(output_path)


def export_length_statistics(
    results: List[Dict[str, any]],
    output_path: str
) -> str:
    """
    Export text length statistics to CSV.

    Args:
        results: List of per-split length statistics
        output_path: Where to save the CSV file

    Returns:
        Absolute path to saved CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['split', 'count', 'mean', 'std', 'median', 'min', 'max', 'q25', 'q75']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow({
                'split': row['split'],
                'count': row['count'],
                'mean': f"{row['mean']:.2f}",
                'std': f"{row['std']:.2f}",
                'median': f"{row['median']:.2f}",
                'min': row['min'],
                'max': row['max'],
                'q25': f"{row['q25']:.2f}",
                'q75': f"{row['q75']:.2f}"
            })

    return os.path.abspath(output_path)


# ============================================================================
# Main Execution
# ============================================================================

def main() -> None:
    """
    Main entry point for split consistency validation.

    Loads dataset, performs statistical tests, and reports any distribution
    drift between train/validation/test splits.
    """
    # Load environment variables
    load_dotenv()

    logger.info("="*70)
    logger.info("GoEmotions Split Consistency Validation")
    logger.info("="*70)

    # Load dataset
    logger.info("Loading GoEmotions dataset...")
    dataset = load_go_emotions()
    label_names = get_label_names(dataset)
    logger.info("Dataset loaded successfully")

    # ========================================================================
    # 1. Label Distribution Analysis
    # ========================================================================

    logger.info("")
    logger.info("="*70)
    logger.info("1. LABEL DISTRIBUTION ANALYSIS")
    logger.info("="*70)

    logger.info("Calculating label distributions by split...")
    split_distributions = calculate_label_distribution_by_split(dataset, label_names)

    logger.info("Performing chi-square test for label distribution consistency...")
    chi2_result = chi_square_test_label_distribution(split_distributions, label_names)

    logger.info(f"Chi-square statistic: {chi2_result['statistic']:.4f}")
    logger.info(f"P-value: {chi2_result['p_value']:.6f}")
    logger.info(f"Degrees of freedom: {chi2_result['degrees_of_freedom']}")
    logger.info(f"Consistent: {chi2_result['is_consistent']} (threshold: {chi2_result['threshold']})")

    if chi2_result['is_consistent']:
        logger.info("PASS: Label distributions are statistically consistent across splits")
    else:
        logger.warning("WARNING: Label distributions show significant differences across splits")

    logger.info("Calculating per-label proportion differences...")
    label_proportions = calculate_label_proportion_differences(split_distributions, label_names)

    # Report labels with largest differences
    logger.info("")
    logger.info("Top 5 labels with largest proportion differences:")
    for i, result in enumerate(label_proportions[:5], 1):
        logger.info(f"  {i}. {result['label']}: max diff = {result['max_diff_pct']:.3f}%")
        logger.info(f"     Train: {result['train_pct']:.3f}%, Val: {result['val_pct']:.3f}%, Test: {result['test_pct']:.3f}%")

    # ========================================================================
    # 2. Text Length Distribution Analysis
    # ========================================================================

    logger.info("")
    logger.info("="*70)
    logger.info("2. TEXT LENGTH DISTRIBUTION ANALYSIS")
    logger.info("="*70)

    logger.info("Calculating text length distributions by split...")
    length_distributions = calculate_text_lengths_by_split(dataset)

    logger.info("Calculating length statistics...")
    length_stats = calculate_length_statistics_by_split(length_distributions)

    for stat in length_stats:
        logger.info(f"{stat['split'].capitalize()}: mean={stat['mean']:.2f}, "
                   f"std={stat['std']:.2f}, median={stat['median']:.2f}")

    logger.info("Performing Kolmogorov-Smirnov tests...")
    ks_results = ks_test_text_lengths(length_distributions)

    all_length_consistent = True
    for comparison, result in ks_results.items():
        logger.info(f"{comparison}: KS stat={result['statistic']:.4f}, "
                   f"p-value={result['p_value']:.6f}, "
                   f"consistent={result['is_consistent']}")
        if not result['is_consistent']:
            all_length_consistent = False

    if all_length_consistent:
        logger.info("PASS: Text length distributions are consistent across splits")
    else:
        logger.warning("WARNING: Text length distributions show differences across splits")

    # ========================================================================
    # 3. Multi-Label Distribution Analysis
    # ========================================================================

    logger.info("")
    logger.info("="*70)
    logger.info("3. MULTI-LABEL DISTRIBUTION ANALYSIS")
    logger.info("="*70)

    logger.info("Calculating multi-label distributions by split...")
    multilabel_dists = calculate_multilabel_distribution_by_split(dataset)

    for split_name, counts in multilabel_dists.items():
        total = sum(counts.values())
        pcts = {k: (v/total*100) for k, v in counts.items()}
        logger.info(f"{split_name.capitalize()}: "
                   f"1={pcts['1_label']:.1f}%, "
                   f"2={pcts['2_labels']:.1f}%, "
                   f"3+={pcts['3plus_labels']:.1f}%")

    logger.info("Performing chi-square test for multi-label distribution consistency...")
    multilabel_chi2 = chi_square_test_multilabel_distribution(multilabel_dists)

    logger.info(f"Chi-square statistic: {multilabel_chi2['statistic']:.4f}")
    logger.info(f"P-value: {multilabel_chi2['p_value']:.6f}")
    logger.info(f"Consistent: {multilabel_chi2['is_consistent']} (threshold: {multilabel_chi2['threshold']})")

    if multilabel_chi2['is_consistent']:
        logger.info("PASS: Multi-label distributions are consistent across splits")
    else:
        logger.warning("WARNING: Multi-label distributions show differences across splits")

    # ========================================================================
    # Export Results
    # ========================================================================

    logger.info("")
    logger.info("="*70)
    logger.info("EXPORTING RESULTS")
    logger.info("="*70)

    output_dir = os.getenv('OUTPUT_DIR', 'output')

    # Export label proportion analysis
    label_path = os.path.join(output_dir, 'stats', 'split_label_proportions.csv')
    label_saved = export_label_proportion_analysis(label_proportions, label_path)
    logger.info(f"Label proportions saved to: {label_saved}")

    # Export length statistics
    length_path = os.path.join(output_dir, 'stats', 'split_length_statistics.csv')
    length_saved = export_length_statistics(length_stats, length_path)
    logger.info(f"Length statistics saved to: {length_saved}")

    # ========================================================================
    # Final Summary
    # ========================================================================

    logger.info("")
    logger.info("="*70)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*70)

    tests_passed = 0
    tests_total = 3

    if chi2_result['is_consistent']:
        logger.info("PASS: Label distributions are consistent")
        tests_passed += 1
    else:
        logger.warning("FAIL: Label distributions show drift")

    if all_length_consistent:
        logger.info("PASS: Text length distributions are consistent")
        tests_passed += 1
    else:
        logger.warning("FAIL: Text length distributions show drift")

    if multilabel_chi2['is_consistent']:
        logger.info("PASS: Multi-label distributions are consistent")
        tests_passed += 1
    else:
        logger.warning("FAIL: Multi-label distributions show drift")

    logger.info("")
    logger.info(f"Result: {tests_passed}/{tests_total} consistency tests passed")

    if tests_passed == tests_total:
        logger.info("CONCLUSION: All splits have consistent distributions")
    else:
        logger.warning("CONCLUSION: Some distribution drift detected between splits")

    logger.info("="*70)

    print(f"\nLabel proportions saved to: {label_saved}")
    print(f"Length statistics saved to: {length_saved}")


if __name__ == "__main__":
    main()
