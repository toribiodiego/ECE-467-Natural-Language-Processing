"""
Statistical Significance Testing

Performs bootstrap significance testing to compare model performance between
a baseline model (single run) and a comparison model (multiple seed runs).

Usage:
    python -m src.analysis.significance_test \
        --model-a-metrics artifacts/stats/test_metrics_roberta-large.json \
        --model-b-metrics artifacts/stats/multiseed/seed*.json \
        --output artifacts/stats/significance_tests.csv \
        --n-bootstrap 10000
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from glob import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_metrics_file(filepath: Path) -> Dict:
    """Load metrics from a JSON file.

    Args:
        filepath: Path to metrics JSON file

    Returns:
        Dictionary containing metrics
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_test_metrics(metrics: Dict, source: str = 'roberta') -> Dict[str, float]:
    """Extract test metrics from metrics dictionary.

    Args:
        metrics: Full metrics dictionary
        source: Source type ('roberta' or 'distilbert') to handle different formats

    Returns:
        Dictionary of metric_name -> value
    """
    test_metrics = {}

    if source == 'roberta':
        # RoBERTa format: metrics.test_metrics.{metric_name}
        raw_metrics = metrics.get('test_metrics', {})
        test_metrics = {
            'auc_macro': raw_metrics.get('auc_macro', 0.0),
            'auc_micro': raw_metrics.get('auc_micro', 0.0),
            'f1_macro': raw_metrics.get('macro_f1', 0.0),
            'f1_micro': raw_metrics.get('micro_f1', 0.0),
            'precision_macro': raw_metrics.get('macro_precision', 0.0),
            'recall_macro': raw_metrics.get('macro_recall', 0.0),
        }
    else:
        # DistilBERT format: metrics.test/{metric_name}
        for key, value in metrics.items():
            if key.startswith('test/') and isinstance(value, (int, float)):
                # Remove 'test/' prefix
                metric_name = key.replace('test/', '')
                # Only keep key aggregate metrics
                if metric_name in ['auc_macro', 'auc_micro', 'f1_macro', 'f1_micro',
                                   'precision_macro', 'recall_macro']:
                    test_metrics[metric_name] = value

    return test_metrics


def load_model_a_metrics(filepath: str) -> Dict[str, float]:
    """Load metrics from Model A (baseline, single run).

    Args:
        filepath: Path to Model A metrics JSON file

    Returns:
        Dictionary of metric_name -> value
    """
    logger.info(f"Loading Model A (baseline) metrics from: {filepath}")
    metrics = load_metrics_file(Path(filepath))
    test_metrics = extract_test_metrics(metrics, source='roberta')
    logger.info(f"  Loaded {len(test_metrics)} metrics")
    return test_metrics


def load_model_b_metrics(pattern: str) -> List[Dict[str, float]]:
    """Load metrics from Model B (comparison, multiple seeds).

    Args:
        pattern: Glob pattern for Model B metrics files

    Returns:
        List of metric dictionaries, one per seed
    """
    logger.info(f"Loading Model B (comparison) metrics from pattern: {pattern}")

    # Expand glob pattern
    files = sorted(glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    logger.info(f"  Found {len(files)} seed runs")

    all_metrics = []
    for filepath in files:
        metrics = load_metrics_file(Path(filepath))
        test_metrics = extract_test_metrics(metrics, source='distilbert')
        all_metrics.append(test_metrics)
        logger.info(f"    - {Path(filepath).name}: {len(test_metrics)} metrics")

    return all_metrics


def bootstrap_mean(values: np.ndarray, n_bootstrap: int = 10000, seed: int = 42) -> np.ndarray:
    """Generate bootstrap samples of the mean.

    Args:
        values: Array of values to bootstrap
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        Array of bootstrap mean estimates
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    bootstrap_means = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Sample with replacement
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    return bootstrap_means


def compute_significance(model_a_value: float,
                         model_b_values: List[float],
                         n_bootstrap: int = 10000) -> Dict[str, float]:
    """Compute statistical significance metrics between two models.

    Args:
        model_a_value: Single metric value from Model A
        model_b_values: List of metric values from Model B (multiple seeds)
        n_bootstrap: Number of bootstrap iterations

    Returns:
        Dictionary with p-value, confidence intervals, and effect size
    """
    model_b_array = np.array(model_b_values)
    model_b_mean = np.mean(model_b_array)
    model_b_std = np.std(model_b_array, ddof=1)

    # Bootstrap Model B to get confidence intervals
    bootstrap_means = bootstrap_mean(model_b_array, n_bootstrap=n_bootstrap)

    # Compute 95% confidence interval
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    # Compute p-value: proportion of bootstrap samples where Model B >= Model A
    # (one-tailed test: is Model B significantly better than Model A?)
    p_value_greater = np.mean(bootstrap_means >= model_a_value)

    # Two-tailed p-value: is there a significant difference?
    p_value_twotail = 2 * min(p_value_greater, 1 - p_value_greater)

    # Effect size (Cohen's d approximation)
    # Using Model B std as pooled estimate (conservative)
    if model_b_std > 0:
        cohens_d = (model_b_mean - model_a_value) / model_b_std
    else:
        cohens_d = 0.0

    return {
        'model_a_value': model_a_value,
        'model_b_mean': model_b_mean,
        'model_b_std': model_b_std,
        'model_b_ci_lower': ci_lower,
        'model_b_ci_upper': ci_upper,
        'delta': model_b_mean - model_a_value,
        'percent_change': ((model_b_mean - model_a_value) / model_a_value * 100) if model_a_value != 0 else 0.0,
        'p_value_twotail': p_value_twotail,
        'p_value_greater': p_value_greater,
        'cohens_d': cohens_d,
        'n_seeds': len(model_b_values)
    }


def run_significance_tests(model_a_metrics: Dict[str, float],
                           model_b_metrics_list: List[Dict[str, float]],
                           n_bootstrap: int = 10000) -> pd.DataFrame:
    """Run significance tests for all metrics.

    Args:
        model_a_metrics: Metrics from Model A (single run)
        model_b_metrics_list: List of metrics from Model B (multiple seeds)
        n_bootstrap: Number of bootstrap iterations

    Returns:
        DataFrame with test results for each metric
    """
    # Get common metrics
    common_metrics = set(model_a_metrics.keys())

    # Verify all Model B runs have the same metrics
    for metrics in model_b_metrics_list:
        common_metrics &= set(metrics.keys())

    logger.info(f"\nTesting {len(common_metrics)} common metrics: {sorted(common_metrics)}")

    results = []
    for metric_name in sorted(common_metrics):
        logger.info(f"\n  Testing {metric_name}...")

        model_a_value = model_a_metrics[metric_name]
        model_b_values = [m[metric_name] for m in model_b_metrics_list]

        stats = compute_significance(model_a_value, model_b_values, n_bootstrap)
        stats['metric'] = metric_name
        results.append(stats)

        # Log summary
        logger.info(f"    Model A: {model_a_value:.5f}")
        logger.info(f"    Model B: {stats['model_b_mean']:.5f} ± {stats['model_b_std']:.5f}")
        logger.info(f"    95% CI: [{stats['model_b_ci_lower']:.5f}, {stats['model_b_ci_upper']:.5f}]")
        logger.info(f"    Delta: {stats['delta']:.5f} ({stats['percent_change']:.2f}%)")
        logger.info(f"    p-value (two-tailed): {stats['p_value_twotail']:.4f}")

    df = pd.DataFrame(results)

    # Reorder columns for readability
    column_order = [
        'metric', 'model_a_value', 'model_b_mean', 'model_b_std',
        'model_b_ci_lower', 'model_b_ci_upper', 'delta', 'percent_change',
        'p_value_twotail', 'p_value_greater', 'cohens_d', 'n_seeds'
    ]
    df = df[column_order]

    return df


def interpret_results(df: pd.DataFrame, alpha: float = 0.05):
    """Log interpretation of significance test results.

    Args:
        df: DataFrame with test results
        alpha: Significance level (default: 0.05)
    """
    logger.info("\n" + "="*70)
    logger.info("SIGNIFICANCE TEST INTERPRETATION")
    logger.info("="*70)

    logger.info(f"\nSignificance level: α = {alpha}")
    logger.info(f"Number of bootstrap samples: {df.iloc[0]['n_seeds']} seeds")

    significant_metrics = df[df['p_value_twotail'] < alpha]
    nonsignificant_metrics = df[df['p_value_twotail'] >= alpha]

    logger.info(f"\nStatistically significant differences ({len(significant_metrics)}):")
    if len(significant_metrics) > 0:
        for _, row in significant_metrics.iterrows():
            direction = "better" if row['delta'] > 0 else "worse"
            logger.info(f"  {row['metric']:20s}: Model B is {direction} "
                       f"(Δ = {row['delta']:+.5f}, p = {row['p_value_twotail']:.4f})")
    else:
        logger.info("  None")

    logger.info(f"\nNo significant difference ({len(nonsignificant_metrics)}):")
    if len(nonsignificant_metrics) > 0:
        for _, row in nonsignificant_metrics.iterrows():
            logger.info(f"  {row['metric']:20s}: p = {row['p_value_twotail']:.4f}")

    # Key metrics summary
    logger.info("\nKey Metrics Summary:")
    key_metrics = ['auc_macro', 'auc_micro', 'f1_macro', 'f1_micro']
    for metric in key_metrics:
        if metric in df['metric'].values:
            row = df[df['metric'] == metric].iloc[0]
            sig_marker = "*" if row['p_value_twotail'] < alpha else ""
            logger.info(f"  {metric:15s}: {row['percent_change']:+6.2f}% "
                       f"(p={row['p_value_twotail']:.4f}){sig_marker}")

    logger.info("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Perform bootstrap significance test between two models'
    )
    parser.add_argument(
        '--model-a-metrics',
        type=str,
        required=True,
        help='Path to Model A (baseline) metrics JSON file'
    )
    parser.add_argument(
        '--model-b-metrics',
        type=str,
        required=True,
        help='Glob pattern for Model B (comparison) metrics files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=10000,
        help='Number of bootstrap iterations (default: 10000)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("BOOTSTRAP SIGNIFICANCE TEST")
    logger.info("="*70)

    # Load metrics
    model_a_metrics = load_model_a_metrics(args.model_a_metrics)
    model_b_metrics_list = load_model_b_metrics(args.model_b_metrics)

    # Run significance tests
    logger.info(f"\nRunning bootstrap tests with {args.n_bootstrap} iterations...")
    results_df = run_significance_tests(
        model_a_metrics,
        model_b_metrics_list,
        n_bootstrap=args.n_bootstrap
    )

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save results
    logger.info(f"\nSaving results to: {args.output}")
    results_df.to_csv(args.output, index=False, float_format='%.6f')

    # Interpret results
    interpret_results(results_df, alpha=args.alpha)

    logger.info(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
