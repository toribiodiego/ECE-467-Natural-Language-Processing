"""
Robustness Visualization Script

Generates confidence interval plots comparing RoBERTa-Large (single run)
vs DistilBERT (mean ± std across 3 seeds) to visualize training stability
and performance trade-offs.

Usage:
    python -m src.analysis.visualize_robustness \
        --significance-csv artifacts/stats/significance_tests.csv \
        --output artifacts/stats/robustness_plots.png
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_significance_results(csv_path: str) -> pd.DataFrame:
    """Load significance test results.

    Args:
        csv_path: Path to significance_tests.csv

    Returns:
        DataFrame with significance test results
    """
    logger.info(f"Loading significance test results from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  Loaded {len(df)} metrics")
    return df


def format_metric_name(metric: str) -> str:
    """Format metric name for display.

    Args:
        metric: Raw metric name (e.g., 'auc_macro')

    Returns:
        Formatted metric name (e.g., 'AUC Macro')
    """
    name_mapping = {
        'auc_macro': 'AUC\nMacro',
        'auc_micro': 'AUC\nMicro',
        'f1_macro': 'F1\nMacro',
        'f1_micro': 'F1\nMicro',
        'precision_macro': 'Precision\nMacro',
        'recall_macro': 'Recall\nMacro',
    }
    return name_mapping.get(metric, metric.replace('_', '\n').title())


def create_robustness_plot(df: pd.DataFrame, output_path: str):
    """Create confidence interval comparison plot.

    Args:
        df: DataFrame with significance test results
        output_path: Path to save output figure
    """
    logger.info("Creating robustness visualization...")

    # Filter to key metrics only
    key_metrics = ['auc_macro', 'auc_micro', 'f1_macro', 'f1_micro',
                   'precision_macro', 'recall_macro']
    df_plot = df[df['metric'].isin(key_metrics)].copy()

    # Reorder by metric importance
    metric_order = ['auc_micro', 'auc_macro', 'f1_micro', 'f1_macro',
                    'precision_macro', 'recall_macro']
    df_plot['metric_cat'] = pd.Categorical(df_plot['metric'],
                                            categories=metric_order,
                                            ordered=True)
    df_plot = df_plot.sort_values('metric_cat')

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Color scheme
    roberta_color = '#0000FF'  # Blue for RoBERTa
    distilbert_color = '#00FF00'  # Green for DistilBERT
    ci_color = '#00AA00'  # Darker green for CI bars

    # Subplot 1: Absolute metric values with error bars
    x_pos = np.arange(len(df_plot))
    width = 0.35

    # RoBERTa bars (baseline, no error bars)
    ax1.bar(x_pos - width/2, df_plot['model_a_value'],
            width, label='RoBERTa-Large (single run)',
            color=roberta_color, alpha=0.8, edgecolor='black', linewidth=1)

    # DistilBERT bars with error bars (std dev)
    ax1.bar(x_pos + width/2, df_plot['model_b_mean'],
            width, yerr=df_plot['model_b_std'],
            label='DistilBERT (mean ± std, n=3)',
            color=distilbert_color, alpha=0.8, edgecolor='black', linewidth=1,
            error_kw={'elinewidth': 2, 'capsize': 5, 'capthick': 2,
                      'ecolor': ci_color})

    ax1.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Comparison with Confidence Intervals',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([format_metric_name(m) for m in df_plot['metric']],
                        fontsize=10)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.0)

    # Add significance markers
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        if row['p_value_twotail'] < 0.001:
            marker = '***'
        elif row['p_value_twotail'] < 0.01:
            marker = '**'
        elif row['p_value_twotail'] < 0.05:
            marker = '*'
        else:
            marker = 'ns'

        # Place marker above the taller bar
        y_pos = max(row['model_a_value'], row['model_b_mean']) + 0.05
        ax1.text(i, y_pos, marker, ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    # Subplot 2: Percentage change with confidence intervals
    # Calculate error bars in percentage terms
    percent_std = (df_plot['model_b_std'] / df_plot['model_a_value'] * 100).values

    ax2.bar(x_pos, df_plot['percent_change'],
            color=[distilbert_color if pc >= 0 else '#FF0000' for pc in df_plot['percent_change']],
            alpha=0.8, edgecolor='black', linewidth=1)

    # Add error bars
    ax2.errorbar(x_pos, df_plot['percent_change'], yerr=percent_std,
                 fmt='none', ecolor=ci_color, elinewidth=2,
                 capsize=5, capthick=2)

    # Add zero line
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

    ax2.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage Change (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Change: DistilBERT vs RoBERTa',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([format_metric_name(m) for m in df_plot['metric']],
                        fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add percentage labels on bars
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        y_offset = 2 if row['percent_change'] < 0 else -2
        va = 'top' if row['percent_change'] < 0 else 'bottom'
        ax2.text(i, row['percent_change'] + y_offset,
                f"{row['percent_change']:.1f}%",
                ha='center', va=va, fontsize=9, fontweight='bold')

    # Overall layout
    plt.suptitle('Multi-Seed Robustness Analysis: RoBERTa-Large vs DistilBERT',
                 fontsize=16, fontweight='bold', y=0.98)

    # Add footer with statistical note
    fig.text(0.5, 0.01,
             'Error bars show standard deviation across 3 random seeds. '
             'Significance: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05',
             ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"  Saved to: {output_path}")


def create_detailed_ci_plot(df: pd.DataFrame, output_path: str):
    """Create detailed confidence interval plot with bootstrap CIs.

    Args:
        df: DataFrame with significance test results
        output_path: Path to save output figure
    """
    logger.info("Creating detailed CI visualization...")

    # Filter to key metrics
    key_metrics = ['auc_macro', 'auc_micro', 'f1_macro', 'f1_micro']
    df_plot = df[df['metric'].isin(key_metrics)].copy()

    # Reorder
    metric_order = ['auc_micro', 'auc_macro', 'f1_micro', 'f1_macro']
    df_plot['metric_cat'] = pd.Categorical(df_plot['metric'],
                                            categories=metric_order,
                                            ordered=True)
    df_plot = df_plot.sort_values('metric_cat')

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors
    roberta_color = '#0000FF'
    distilbert_color = '#00FF00'
    ci_color = '#00AA00'

    y_pos = np.arange(len(df_plot))
    height = 0.3

    # Plot RoBERTa as points (no CI, single run)
    ax.scatter(df_plot['model_a_value'], y_pos - height/2,
               color=roberta_color, s=150, marker='D',
               label='RoBERTa-Large (single run)', zorder=3,
               edgecolors='black', linewidths=1.5)

    # Plot DistilBERT with 95% CI horizontal bars
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        # Mean point
        ax.scatter(row['model_b_mean'], i + height/2,
                   color=distilbert_color, s=150, marker='o',
                   zorder=3, edgecolors='black', linewidths=1.5)

        # 95% CI error bar
        ax.plot([row['model_b_ci_lower'], row['model_b_ci_upper']],
                [i + height/2, i + height/2],
                color=ci_color, linewidth=3, solid_capstyle='round',
                zorder=2)

        # CI caps
        cap_height = 0.1
        ax.plot([row['model_b_ci_lower'], row['model_b_ci_lower']],
                [i + height/2 - cap_height, i + height/2 + cap_height],
                color=ci_color, linewidth=2, zorder=2)
        ax.plot([row['model_b_ci_upper'], row['model_b_ci_upper']],
                [i + height/2 - cap_height, i + height/2 + cap_height],
                color=ci_color, linewidth=2, zorder=2)

    # Dummy plot for CI legend entry
    ax.plot([], [], color=ci_color, linewidth=3, solid_capstyle='round',
            label='DistilBERT (mean with 95% CI, n=3)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([format_metric_name(m).replace('\n', ' ')
                        for m in df_plot['metric']],
                       fontsize=12)
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance with Bootstrap Confidence Intervals',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1.0)

    # Invert y-axis so highest metric is on top
    ax.invert_yaxis()

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate robustness visualization with confidence intervals'
    )
    parser.add_argument(
        '--significance-csv',
        type=str,
        required=True,
        help='Path to significance_tests.csv'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output PNG file'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Also generate detailed CI plot'
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("ROBUSTNESS VISUALIZATION")
    logger.info("="*70)

    # Load data
    df = load_significance_results(args.significance_csv)

    # Create main plot
    create_robustness_plot(df, args.output)

    # Optionally create detailed CI plot
    if args.detailed:
        detailed_output = Path(args.output).parent / 'robustness_detailed_ci.png'
        create_detailed_ci_plot(df, str(detailed_output))

    logger.info("\n" + "="*70)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nPlot saved to: {args.output}")

    if args.detailed:
        logger.info(f"Detailed CI plot saved to: {detailed_output}")


if __name__ == '__main__':
    main()
