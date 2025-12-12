"""
Metric Comparison Script

Compares test metrics between two trained models (e.g., RoBERTa-Large vs DistilBERT)
and generates a summary CSV with side-by-side metrics, deltas, and percentage differences.

Usage:
    python -m src.analysis.metric_comparison \
        --roberta-metrics artifacts/stats/test_metrics_roberta-large.json \
        --distilbert-metrics artifacts/stats/test_metrics_distilbert-base.json \
        --output artifacts/stats/metric_summary.csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file.

    Args:
        metrics_path: Path to metrics JSON file

    Returns:
        Dictionary containing metrics
    """
    logger.info(f"Loading metrics from: {metrics_path}")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def extract_test_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Extract key test metrics from metrics dictionary.

    Args:
        metrics: Full metrics dictionary

    Returns:
        Dictionary of key test metrics
    """
    test_metrics = metrics.get('test_metrics', {})

    return {
        'auc_micro': test_metrics.get('auc_micro', 0.0),
        'auc_macro': test_metrics.get('auc_macro', 0.0),
        'macro_f1': test_metrics.get('macro_f1', 0.0),
        'micro_f1': test_metrics.get('micro_f1', 0.0),
        'macro_precision': test_metrics.get('macro_precision', 0.0),
        'micro_precision': test_metrics.get('micro_precision', 0.0),
        'macro_recall': test_metrics.get('macro_recall', 0.0),
        'micro_recall': test_metrics.get('micro_recall', 0.0),
    }


def compare_metrics(roberta_metrics: Dict[str, float],
                   distilbert_metrics: Dict[str, float]) -> pd.DataFrame:
    """Compare metrics between two models.

    Args:
        roberta_metrics: RoBERTa test metrics
        distilbert_metrics: DistilBERT test metrics

    Returns:
        DataFrame with comparison results
    """
    comparison_data = []

    for metric_name in roberta_metrics.keys():
        roberta_value = roberta_metrics[metric_name]
        distilbert_value = distilbert_metrics[metric_name]

        # Calculate delta and percentage difference
        delta = distilbert_value - roberta_value
        pct_diff = (delta / roberta_value * 100) if roberta_value != 0 else 0.0

        comparison_data.append({
            'metric': metric_name,
            'roberta_large': roberta_value,
            'distilbert_base': distilbert_value,
            'delta': delta,
            'percent_difference': pct_diff
        })

    df = pd.DataFrame(comparison_data)
    return df


def format_metric_name(metric_name: str) -> str:
    """Format metric name for display.

    Args:
        metric_name: Raw metric name (e.g., 'auc_micro')

    Returns:
        Formatted metric name (e.g., 'AUC (micro)')
    """
    name_mapping = {
        'auc_micro': 'AUC (micro)',
        'auc_macro': 'AUC (macro)',
        'macro_f1': 'Macro F1',
        'micro_f1': 'Micro F1',
        'macro_precision': 'Macro Precision',
        'micro_precision': 'Micro Precision',
        'macro_recall': 'Macro Recall',
        'micro_recall': 'Micro Recall',
    }
    return name_mapping.get(metric_name, metric_name)


def main():
    parser = argparse.ArgumentParser(
        description='Compare test metrics between RoBERTa-Large and DistilBERT models'
    )
    parser.add_argument(
        '--roberta-metrics',
        type=str,
        required=True,
        help='Path to RoBERTa-Large test metrics JSON file'
    )
    parser.add_argument(
        '--distilbert-metrics',
        type=str,
        required=True,
        help='Path to DistilBERT test metrics JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV file'
    )

    args = parser.parse_args()

    # Load metrics from both models
    logger.info("Loading metrics from both models...")
    roberta_full = load_metrics(args.roberta_metrics)
    distilbert_full = load_metrics(args.distilbert_metrics)

    # Extract test metrics
    logger.info("Extracting test metrics...")
    roberta_test = extract_test_metrics(roberta_full)
    distilbert_test = extract_test_metrics(distilbert_full)

    # Compare metrics
    logger.info("Comparing metrics...")
    comparison_df = compare_metrics(roberta_test, distilbert_test)

    # Format metric names
    comparison_df['metric'] = comparison_df['metric'].apply(format_metric_name)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    logger.info(f"Saving comparison to: {args.output}")
    comparison_df.to_csv(args.output, index=False, float_format='%.4f')

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("METRIC COMPARISON SUMMARY")
    logger.info("="*70)
    logger.info(f"\n{comparison_df.to_string(index=False)}\n")
    logger.info("="*70)

    # Print key findings
    logger.info("\nKey Findings:")

    # AUC comparison
    auc_micro_row = comparison_df[comparison_df['metric'] == 'AUC (micro)'].iloc[0]
    logger.info(f"  • AUC (micro): DistilBERT achieves {auc_micro_row['distilbert_base']:.4f} "
                f"({auc_micro_row['percent_difference']:.1f}% vs RoBERTa)")

    # F1 comparison
    macro_f1_row = comparison_df[comparison_df['metric'] == 'Macro F1'].iloc[0]
    logger.info(f"  • Macro F1: DistilBERT achieves {macro_f1_row['distilbert_base']:.4f} "
                f"({macro_f1_row['percent_difference']:.1f}% vs RoBERTa)")

    # Overall assessment
    auc_gap = abs(auc_micro_row['percent_difference'])
    if auc_gap < 5:
        logger.info(f"  • DistilBERT maintains competitive performance (<5% AUC gap)")
    else:
        logger.info(f"  • DistilBERT shows {auc_gap:.1f}% AUC gap from RoBERTa")

    logger.info(f"\n✓ Comparison saved to: {args.output}")


if __name__ == '__main__':
    main()
