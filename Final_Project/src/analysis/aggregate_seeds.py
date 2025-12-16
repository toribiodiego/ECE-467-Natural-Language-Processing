"""
Aggregate Metrics Across Seeds

Aggregates test metrics from multiple training runs with different random seeds
and computes mean and standard deviation for robustness analysis.

Usage:
    python -m src.analysis.aggregate_seeds \
        --metrics-dir artifacts/stats/multiseed/ \
        --model-pattern distilbert \
        --output artifacts/stats/multiseed_summary.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_seed_metrics(metrics_dir: Path, model_pattern: str) -> List[Dict]:
    """Load metrics from all seed runs matching the model pattern.

    Args:
        metrics_dir: Directory containing seed metrics JSON files
        model_pattern: Pattern to match in model name from metadata (e.g., 'distilbert')

    Returns:
        List of metrics dictionaries from each seed run
    """
    # Find all JSON files (except README-like files)
    all_json_files = sorted(metrics_dir.glob("*.json"))

    if not all_json_files:
        raise FileNotFoundError(
            f"No JSON files found in {metrics_dir}"
        )

    logger.info(f"Scanning {len(all_json_files)} JSON files for model pattern '{model_pattern}'...")

    # Filter by model type in metadata
    all_metrics = []
    for metrics_file in all_json_files:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Check if this file matches the model pattern
        metadata = metrics.get('_metadata', {})
        model_name = metadata.get('model', '')

        if model_pattern.lower() in model_name.lower():
            logger.info(f"  - {metrics_file.name} (model: {model_name}, seed: {metadata.get('seed', 'unknown')})")
            all_metrics.append(metrics)

    if not all_metrics:
        raise FileNotFoundError(
            f"No metrics files found with model matching '{model_pattern}' in {metrics_dir}"
        )

    logger.info(f"\nLoaded {len(all_metrics)} seed runs for model pattern '{model_pattern}'")

    return all_metrics


def extract_test_metrics(seed_metrics_list: List[Dict]) -> pd.DataFrame:
    """Extract test metrics from all seed runs into a DataFrame.

    Args:
        seed_metrics_list: List of metrics dictionaries from each seed

    Returns:
        DataFrame with one row per seed, one column per metric
    """
    rows = []

    for metrics in seed_metrics_list:
        # Extract metadata
        metadata = metrics.get('_metadata', {})
        seed = metadata.get('seed', 'unknown')

        # Extract all test metrics (excluding metadata)
        row = {'seed': seed}
        for key, value in metrics.items():
            if key.startswith('test/') and isinstance(value, (int, float)):
                # Remove 'test/' prefix for cleaner column names
                metric_name = key.replace('test/', '')
                row[metric_name] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def compute_aggregate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and standard deviation across seeds.

    Args:
        df: DataFrame with one row per seed

    Returns:
        DataFrame with mean and std for each metric
    """
    # Get all metric columns (exclude 'seed')
    metric_cols = [col for col in df.columns if col != 'seed']

    aggregates = []
    for metric in metric_cols:
        values = df[metric].values
        aggregates.append({
            'metric': metric,
            'mean': np.mean(values),
            'std': np.std(values, ddof=1),  # Sample std with N-1 denominator
            'min': np.min(values),
            'max': np.max(values),
            'n_seeds': len(values)
        })

    agg_df = pd.DataFrame(aggregates)
    return agg_df


def format_metric_name(metric_name: str) -> str:
    """Format metric name for display.

    Args:
        metric_name: Raw metric name (e.g., 'auc_macro')

    Returns:
        Formatted metric name (e.g., 'AUC Macro')
    """
    # Convert underscores to spaces and title case
    formatted = metric_name.replace('_', ' ').title()

    # Special cases for common acronyms
    formatted = formatted.replace('Auc', 'AUC')
    formatted = formatted.replace('F1', 'F1')

    return formatted


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate test metrics across multiple seed runs'
    )
    parser.add_argument(
        '--metrics-dir',
        type=str,
        required=True,
        help='Directory containing seed metrics JSON files'
    )
    parser.add_argument(
        '--model-pattern',
        type=str,
        required=True,
        help='Pattern to match in filenames (e.g., "distilbert")'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV file'
    )

    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    # Load metrics from all seed runs
    logger.info("="*70)
    logger.info("Loading seed metrics...")
    logger.info("="*70)
    seed_metrics = load_seed_metrics(metrics_dir, args.model_pattern)

    # Extract test metrics into DataFrame
    logger.info("\nExtracting test metrics...")
    metrics_df = extract_test_metrics(seed_metrics)

    logger.info(f"\nLoaded metrics from {len(metrics_df)} seed runs:")
    logger.info(f"Seeds: {sorted(metrics_df['seed'].tolist())}")

    # Compute aggregate statistics
    logger.info("\nComputing mean and standard deviation...")
    agg_df = compute_aggregate_stats(metrics_df)

    # Format metric names
    agg_df['metric_formatted'] = agg_df['metric'].apply(format_metric_name)

    # Reorder columns for better readability
    agg_df = agg_df[['metric', 'metric_formatted', 'mean', 'std', 'min', 'max', 'n_seeds']]

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    logger.info(f"\nSaving aggregated metrics to: {args.output}")
    agg_df.to_csv(args.output, index=False, float_format='%.5f')

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("AGGREGATED METRICS SUMMARY")
    logger.info("="*70)

    # Print key metrics with formatting
    key_metrics = ['auc_macro', 'auc_micro', 'f1_macro', 'f1_micro',
                   'precision_macro', 'recall_macro']

    logger.info("\nKey Metrics (Mean ± Std):")
    for metric in key_metrics:
        if metric in agg_df['metric'].values:
            row = agg_df[agg_df['metric'] == metric].iloc[0]
            logger.info(f"  {row['metric_formatted']:20s}: "
                       f"{row['mean']:.5f} ± {row['std']:.5f} "
                       f"(range: {row['min']:.5f} - {row['max']:.5f})")

    # Print variance assessment
    logger.info("\nVariance Assessment:")
    auc_macro_row = agg_df[agg_df['metric'] == 'auc_macro'].iloc[0]
    auc_micro_row = agg_df[agg_df['metric'] == 'auc_micro'].iloc[0]

    auc_macro_cv = (auc_macro_row['std'] / auc_macro_row['mean']) * 100
    auc_micro_cv = (auc_micro_row['std'] / auc_micro_row['mean']) * 100

    logger.info(f"  AUC Macro CV: {auc_macro_cv:.2f}%")
    logger.info(f"  AUC Micro CV: {auc_micro_cv:.2f}%")

    if auc_macro_cv < 1.0 and auc_micro_cv < 1.0:
        logger.info("  -> Low variance across seeds (CV < 1%), stable training")
    elif auc_macro_cv < 2.0 and auc_micro_cv < 2.0:
        logger.info("  -> Moderate variance across seeds (CV < 2%), acceptable")
    else:
        logger.info("  -> High variance across seeds (CV >= 2%), may need more seeds")

    logger.info("\n" + "="*70)
    logger.info(f"Aggregated metrics saved to: {args.output}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
