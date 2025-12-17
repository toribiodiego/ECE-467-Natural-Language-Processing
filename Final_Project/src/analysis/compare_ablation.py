"""
Compare Ablation Results

Compares metrics between two model checkpoints to analyze ablation study results.
Computes performance deltas for overall metrics and per-class F1 scores.

Usage:
    python -m src.analysis.compare_ablation \
        --run1-dir artifacts/models/distilbert-neutral-on \
        --run2-dir artifacts/models/distilbert-neutral-off \
        --labels "With Neutral,Without Neutral" \
        --output artifacts/stats/neutral_ablation_summary.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_run_metrics(run_dir: Path) -> Tuple[Dict, pd.DataFrame]:
    """Load metrics and per-class metrics from a model checkpoint directory.

    Args:
        run_dir: Directory containing model checkpoint and metrics

    Returns:
        Tuple of (overall_metrics dict, per_class_metrics DataFrame)
    """
    # Load overall metrics from metrics.json
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Extract test metrics from test_metrics dict
    overall_metrics = {}
    if 'test_metrics' in metrics:
        test_metrics = metrics['test_metrics']
        for key, value in test_metrics.items():
            # Skip nested dicts like per_class_f1, confusion_summary
            if isinstance(value, (int, float)):
                overall_metrics[key] = value
    else:
        # Fallback: look for test_ prefixed keys
        for key, value in metrics.items():
            if key.startswith('test_'):
                metric_name = key.replace('test_', '')
                if isinstance(value, list) and len(value) > 0:
                    overall_metrics[metric_name] = value[-1]
                elif isinstance(value, (int, float)):
                    overall_metrics[metric_name] = value

    # Load per-class metrics from stats/ directory
    stats_dir = run_dir / "stats"
    if not stats_dir.exists():
        raise FileNotFoundError(f"Stats directory not found: {stats_dir}")

    # Find the per-class metrics CSV
    per_class_files = list(stats_dir.glob("per_class_metrics_*.csv"))
    if not per_class_files:
        raise FileNotFoundError(f"No per-class metrics found in {stats_dir}")

    per_class_df = pd.read_csv(per_class_files[0])

    # Load config to get num_labels
    config_file = run_dir / "config.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        overall_metrics['num_labels'] = config.get('num_labels', len(per_class_df))
    else:
        overall_metrics['num_labels'] = len(per_class_df)

    # Add class distribution stats
    overall_metrics['num_classes'] = len(per_class_df)
    overall_metrics['total_support'] = per_class_df['support'].sum()

    return overall_metrics, per_class_df


def compute_overall_comparison(metrics1: Dict, metrics2: Dict,
                                 label1: str, label2: str) -> pd.DataFrame:
    """Compare overall metrics between two runs.

    Args:
        metrics1: Metrics from first run
        metrics2: Metrics from second run
        label1: Label for first run
        label2: Label for second run

    Returns:
        DataFrame with side-by-side comparison and deltas
    """
    # Define metrics to compare (check both naming conventions)
    # Old convention: f1_macro, auc_macro, etc.
    # New convention: macro_f1, macro_precision, etc.
    possible_metrics = [
        ('auc_macro', 'macro_auc'), ('auc_micro', 'micro_auc'), ('auc', None),
        ('f1_macro', 'macro_f1'), ('f1_micro', 'micro_f1'),
        ('precision_macro', 'macro_precision'), ('precision_micro', 'micro_precision'),
        ('recall_macro', 'macro_recall'), ('recall_micro', 'micro_recall'),
        ('num_labels', None), ('num_classes', None), ('total_support', None)
    ]

    rows = []
    for metric_variants in possible_metrics:
        # Try each variant
        metric_name = None
        val1 = None
        val2 = None

        for variant in metric_variants:
            if variant is None:
                continue
            if variant in metrics1 and val1 is None:
                val1 = metrics1[variant]
                metric_name = variant
            if variant in metrics2 and val2 is None:
                val2 = metrics2[variant]
                if metric_name is None:
                    metric_name = variant

        if val1 is not None and val2 is not None and metric_name is not None:
            # Compute absolute and relative delta
            delta = val2 - val1

            # Compute percentage change for non-count metrics
            if metric_name not in ['num_labels', 'num_classes', 'total_support']:
                pct_change = (delta / val1) * 100 if val1 != 0 else 0
            else:
                pct_change = None

            rows.append({
                'metric': metric_name,
                label1: val1,
                label2: val2,
                'delta': delta,
                'pct_change': pct_change
            })

    df = pd.DataFrame(rows)
    return df


def compute_per_class_comparison(df1: pd.DataFrame, df2: pd.DataFrame,
                                   label1: str, label2: str) -> pd.DataFrame:
    """Compare per-class metrics between two runs.

    Args:
        df1: Per-class metrics from first run
        df2: Per-class metrics from second run
        label1: Label for first run
        label2: Label for second run

    Returns:
        DataFrame with per-class comparison
    """
    # Merge on emotion name (handle different number of classes)
    merged = df1.merge(df2, on='emotion', how='outer', suffixes=(f'_{label1}', f'_{label2}'))

    # Compute F1 delta
    f1_col1 = f'f1_score_{label1}'
    f1_col2 = f'f1_score_{label2}'

    merged['f1_delta'] = merged[f1_col2].fillna(0) - merged[f1_col1].fillna(0)

    # Compute support delta
    support_col1 = f'support_{label1}'
    support_col2 = f'support_{label2}'

    merged['support_delta'] = merged[support_col2].fillna(0) - merged[support_col1].fillna(0)

    # Select and reorder columns
    cols = ['emotion', f1_col1, f1_col2, 'f1_delta',
            support_col1, support_col2, 'support_delta']

    # Only keep columns that exist
    cols = [c for c in cols if c in merged.columns]

    result = merged[cols].copy()

    # Sort by absolute F1 delta (largest changes first)
    result = result.sort_values('f1_delta', key=abs, ascending=False)

    return result


def format_metric_name(metric_name: str) -> str:
    """Format metric name for display.

    Args:
        metric_name: Raw metric name (e.g., 'auc_macro')

    Returns:
        Formatted metric name (e.g., 'AUC Macro')
    """
    formatted = metric_name.replace('_', ' ').title()

    # Special cases for common acronyms
    formatted = formatted.replace('Auc', 'AUC')
    formatted = formatted.replace('F1', 'F1')

    return formatted


def main():
    parser = argparse.ArgumentParser(
        description='Compare metrics between two ablation study runs'
    )
    parser.add_argument(
        '--run1-dir',
        type=str,
        required=True,
        help='First model checkpoint directory'
    )
    parser.add_argument(
        '--run2-dir',
        type=str,
        required=True,
        help='Second model checkpoint directory'
    )
    parser.add_argument(
        '--labels',
        type=str,
        required=True,
        help='Comma-separated labels for the two runs (e.g., "With Neutral,Without Neutral")'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV file'
    )

    args = parser.parse_args()

    # Parse labels
    labels = [label.strip() for label in args.labels.split(',')]
    if len(labels) != 2:
        raise ValueError(f"Expected 2 labels, got {len(labels)}: {labels}")

    label1, label2 = labels

    # Load metrics from both runs
    logger.info("="*70)
    logger.info("LOADING ABLATION STUDY METRICS")
    logger.info("="*70)

    run1_dir = Path(args.run1_dir)
    run2_dir = Path(args.run2_dir)

    logger.info(f"\nRun 1: {label1}")
    logger.info(f"  Directory: {run1_dir}")
    metrics1, per_class1 = load_run_metrics(run1_dir)
    logger.info(f"  Num labels: {metrics1.get('num_labels', 'N/A')}")
    logger.info(f"  Num classes: {metrics1.get('num_classes', 'N/A')}")

    logger.info(f"\nRun 2: {label2}")
    logger.info(f"  Directory: {run2_dir}")
    metrics2, per_class2 = load_run_metrics(run2_dir)
    logger.info(f"  Num labels: {metrics2.get('num_labels', 'N/A')}")
    logger.info(f"  Num classes: {metrics2.get('num_classes', 'N/A')}")

    # Compare overall metrics
    logger.info("\n" + "="*70)
    logger.info("OVERALL METRICS COMPARISON")
    logger.info("="*70)

    overall_df = compute_overall_comparison(metrics1, metrics2, label1, label2)

    # Print key metrics
    logger.info(f"\n{'Metric':<25s} {label1:<15s} {label2:<15s} {'Delta':<12s} {'% Change':<12s}")
    logger.info("-"*80)

    for _, row in overall_df.iterrows():
        metric_formatted = format_metric_name(row['metric'])

        # Format values based on metric type
        if row['metric'] in ['num_labels', 'num_classes', 'total_support']:
            val1_str = f"{row[label1]:.0f}"
            val2_str = f"{row[label2]:.0f}"
            delta_str = f"{row['delta']:+.0f}"
            pct_str = "N/A"
        else:
            val1_str = f"{row[label1]:.5f}"
            val2_str = f"{row[label2]:.5f}"
            delta_str = f"{row['delta']:+.5f}"
            pct_str = f"{row['pct_change']:+.2f}%" if row['pct_change'] is not None else "N/A"

        logger.info(f"{metric_formatted:<25s} {val1_str:<15s} {val2_str:<15s} {delta_str:<12s} {pct_str:<12s}")

    # Compare per-class metrics
    logger.info("\n" + "="*70)
    logger.info("PER-CLASS F1 COMPARISON")
    logger.info("="*70)

    per_class_df = compute_per_class_comparison(per_class1, per_class2, label1, label2)

    logger.info(f"\nTop 10 Largest F1 Changes:")
    logger.info(f"{'Emotion':<20s} {label1+' F1':<12s} {label2+' F1':<12s} {'Delta':<12s}")
    logger.info("-"*60)

    for _, row in per_class_df.head(10).iterrows():
        f1_col1 = f'f1_score_{label1}'
        f1_col2 = f'f1_score_{label2}'

        val1 = row.get(f1_col1, 0)
        val2 = row.get(f1_col2, 0)
        delta = row['f1_delta']

        val1_str = f"{val1:.5f}" if not pd.isna(val1) else "N/A"
        val2_str = f"{val2:.5f}" if not pd.isna(val2) else "N/A"

        logger.info(f"{row['emotion']:<20s} {val1_str:<12s} {val2_str:<12s} {delta:+.5f}")

    # Create combined output with both comparisons
    logger.info("\n" + "="*70)
    logger.info("SAVING COMPARISON RESULTS")
    logger.info("="*70)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save overall metrics comparison
    overall_output = output_path.parent / (output_path.stem + "_overall.csv")
    overall_df.to_csv(overall_output, index=False, float_format='%.5f')
    logger.info(f"\nOverall metrics saved to: {overall_output}")

    # Save per-class comparison
    per_class_output = output_path.parent / (output_path.stem + "_per_class.csv")
    per_class_df.to_csv(per_class_output, index=False, float_format='%.5f')
    logger.info(f"Per-class metrics saved to: {per_class_output}")

    # Save combined summary
    with open(args.output, 'w') as f:
        f.write("# Ablation Study Comparison\n\n")
        f.write(f"## Run 1: {label1}\n")
        f.write(f"Directory: {run1_dir}\n")
        f.write(f"Num labels: {metrics1.get('num_labels', 'N/A')}\n\n")

        f.write(f"## Run 2: {label2}\n")
        f.write(f"Directory: {run2_dir}\n")
        f.write(f"Num labels: {metrics2.get('num_labels', 'N/A')}\n\n")

        f.write("## Overall Metrics\n")
        f.write(overall_df.to_csv(index=False))

        f.write("\n## Per-Class F1 Comparison (Top 10)\n")
        f.write(per_class_df.head(10).to_csv(index=False))

    logger.info(f"Combined summary saved to: {args.output}")

    # Print summary statistics
    logger.info("\n" + "="*70)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*70)

    # Helper function to safely get metric
    def get_metric_delta(metric_names):
        for name in metric_names:
            rows = overall_df[overall_df['metric'] == name]
            if not rows.empty:
                return rows['delta'].iloc[0]
        return None

    # AUC changes
    auc_macro_delta = get_metric_delta(['auc_macro', 'macro_auc'])
    auc_micro_delta = get_metric_delta(['auc_micro', 'micro_auc', 'auc'])

    if auc_macro_delta is not None:
        logger.info(f"\nAUC Macro change: {auc_macro_delta:+.5f}")
    if auc_micro_delta is not None:
        logger.info(f"AUC Micro change: {auc_micro_delta:+.5f}")

    # F1 changes
    f1_macro_delta = get_metric_delta(['f1_macro', 'macro_f1'])
    f1_micro_delta = get_metric_delta(['f1_micro', 'micro_f1'])

    if f1_macro_delta is not None:
        logger.info(f"F1 Macro change: {f1_macro_delta:+.5f}")
    if f1_micro_delta is not None:
        logger.info(f"F1 Micro change: {f1_micro_delta:+.5f}")

    # Class distribution changes
    num_labels_delta = get_metric_delta(['num_labels'])
    support_delta = get_metric_delta(['total_support'])

    if num_labels_delta is not None:
        logger.info(f"\nNum labels change: {num_labels_delta:+.0f}")
    if support_delta is not None:
        logger.info(f"Total support change: {support_delta:+.0f}")

    # Per-class insights
    logger.info(f"\nPer-class F1 changes:")
    logger.info(f"  Mean delta: {per_class_df['f1_delta'].mean():+.5f}")
    logger.info(f"  Median delta: {per_class_df['f1_delta'].median():+.5f}")
    logger.info(f"  Max improvement: {per_class_df['f1_delta'].max():+.5f}")
    logger.info(f"  Max degradation: {per_class_df['f1_delta'].min():+.5f}")

    logger.info("\n" + "="*70)
    logger.info("COMPARISON COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    main()
