"""
Metric Visualization Script

Generates metric correlation plots and trade-off analysis visualizations
comparing RoBERTa-Large and DistilBERT performance.

Creates 4 separate figures showing:
- Metric comparison bars
- Percentage differences
- AUC vs F1 trade-offs
- Macro vs Micro metric comparison

Usage:
    python -m src.analysis.visualize_metrics \
        --metrics-csv artifacts/stats/metric_summary.csv \
        --output-dir artifacts/stats
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

# Use a clean, professional style
plt.style.use('seaborn-v0_8-darkgrid')


def load_metrics_csv(csv_path: str) -> pd.DataFrame:
    """Load metrics comparison CSV.

    Args:
        csv_path: Path to metric_summary.csv

    Returns:
        DataFrame with metric comparisons
    """
    logger.info(f"Loading metrics from: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def create_comparison_plots(df: pd.DataFrame, output_dir: str):
    """Create comprehensive metric comparison visualizations as separate files.

    Args:
        df: DataFrame with metric comparisons
        output_dir: Directory to save output figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Color scheme - pure colors
    roberta_color = '#0000FF'  # Pure Blue
    distilbert_color = '#00FF00'  # Pure Green
    decrease_color = '#FF0000'  # Pure Red
    increase_color = '#00FF00'  # Pure Green

    saved_files = []

    # Plot 1: Side-by-side metric comparison
    logger.info("Creating Plot 1: Metric Comparison...")
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    plot_metric_bars(ax1, df, roberta_color, distilbert_color)
    file1 = output_path / 'metric_comparison_bars.png'
    fig1.savefig(file1, dpi=300, bbox_inches='tight', facecolor='white')
    saved_files.append(file1)
    plt.close(fig1)
    logger.info(f"  ✓ Saved to: {file1}")

    # Plot 2: Percentage difference waterfall
    logger.info("Creating Plot 2: Percentage Differences...")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    plot_percentage_differences(ax2, df, decrease_color, increase_color)
    file2 = output_path / 'percentage_differences.png'
    fig2.savefig(file2, dpi=300, bbox_inches='tight', facecolor='white')
    saved_files.append(file2)
    plt.close(fig2)
    logger.info(f"  ✓ Saved to: {file2}")

    # Plot 3: AUC vs F1 scatter
    logger.info("Creating Plot 3: AUC vs F1 Trade-off...")
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    plot_auc_vs_f1(ax3, df, roberta_color, distilbert_color)
    file3 = output_path / 'auc_vs_f1_tradeoff.png'
    fig3.savefig(file3, dpi=300, bbox_inches='tight', facecolor='white')
    saved_files.append(file3)
    plt.close(fig3)
    logger.info(f"  ✓ Saved to: {file3}")

    # Plot 4: Macro vs Micro comparison
    logger.info("Creating Plot 4: Macro vs Micro Metrics...")
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    plot_macro_vs_micro(ax4, df, roberta_color, distilbert_color)
    file4 = output_path / 'macro_vs_micro.png'
    fig4.savefig(file4, dpi=300, bbox_inches='tight', facecolor='white')
    saved_files.append(file4)
    plt.close(fig4)
    logger.info(f"  ✓ Saved to: {file4}")

    return saved_files


def plot_metric_bars(ax, df, roberta_color, distilbert_color):
    """Plot side-by-side bar chart of all metrics."""
    metrics = df['metric'].values
    roberta_vals = df['roberta_large'].values
    distilbert_vals = df['distilbert_base'].values

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, roberta_vals, width,
                   label='RoBERTa-Large', color=roberta_color, alpha=0.8)
    bars2 = ax.bar(x + width/2, distilbert_vals, width,
                   label='DistilBERT', color=distilbert_color, alpha=0.8)

    ax.set_xlabel('Metric', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('A. Metric Comparison', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:  # Only label if bar is visible
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)


def plot_percentage_differences(ax, df, decrease_color, increase_color):
    """Plot percentage differences as horizontal bars."""
    metrics = df['metric'].values
    pct_diffs = df['percent_difference'].values

    # Sort by absolute percentage difference
    sorted_indices = np.argsort(np.abs(pct_diffs))[::-1]
    metrics_sorted = metrics[sorted_indices]
    pct_diffs_sorted = pct_diffs[sorted_indices]

    # Color bars based on positive/negative
    colors = [decrease_color if diff < 0 else increase_color for diff in pct_diffs_sorted]

    bars = ax.barh(metrics_sorted, pct_diffs_sorted, color=colors, alpha=0.8)

    ax.set_xlabel('Percentage Difference (%)', fontsize=11, fontweight='bold')
    ax.set_title('B. Relative Performance (DistilBERT vs RoBERTa)',
                fontsize=12, fontweight='bold', pad=10)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, pct_diffs_sorted)):
        ax.text(val, bar.get_y() + bar.get_height()/2,
               f' {val:.1f}%',
               va='center', ha='left' if val < 0 else 'right',
               fontsize=9, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=decrease_color, alpha=0.8, label='Decrease'),
        Patch(facecolor=increase_color, alpha=0.8, label='Increase')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)


def plot_auc_vs_f1(ax, df, roberta_color, distilbert_color):
    """Plot AUC vs F1 trade-off scatter plot."""
    # Extract AUC and F1 metrics
    auc_micro_roberta = df[df['metric'] == 'AUC (micro)']['roberta_large'].values[0]
    auc_micro_distilbert = df[df['metric'] == 'AUC (micro)']['distilbert_base'].values[0]

    f1_micro_roberta = df[df['metric'] == 'Micro F1']['roberta_large'].values[0]
    f1_micro_distilbert = df[df['metric'] == 'Micro F1']['distilbert_base'].values[0]

    # Plot points
    ax.scatter(auc_micro_roberta, f1_micro_roberta,
              s=300, c=roberta_color, alpha=0.7, edgecolors='black',
              linewidths=2, label='RoBERTa-Large', zorder=3)
    ax.scatter(auc_micro_distilbert, f1_micro_distilbert,
              s=300, c=distilbert_color, alpha=0.7, edgecolors='black',
              linewidths=2, label='DistilBERT', zorder=3)

    # Draw arrow between points
    ax.annotate('', xy=(auc_micro_distilbert, f1_micro_distilbert),
               xytext=(auc_micro_roberta, f1_micro_roberta),
               arrowprops=dict(arrowstyle='->', lw=2, alpha=0.5,
                             color='gray', linestyle='--'))

    # Add labels
    ax.text(auc_micro_roberta, f1_micro_roberta - 0.03,
           f'RoBERTa\n({auc_micro_roberta:.3f}, {f1_micro_roberta:.3f})',
           ha='center', va='top', fontsize=9, fontweight='bold')
    ax.text(auc_micro_distilbert, f1_micro_distilbert + 0.03,
           f'DistilBERT\n({auc_micro_distilbert:.3f}, {f1_micro_distilbert:.3f})',
           ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('AUC (micro)', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 Score (micro)', fontsize=11, fontweight='bold')
    ax.set_title('C. AUC vs F1 Trade-off', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9)

    # Set axis limits with padding
    ax.set_xlim(0.85, 0.92)
    ax.set_ylim(0.3, 0.45)


def plot_macro_vs_micro(ax, df, roberta_color, distilbert_color):
    """Plot macro vs micro metric comparison."""
    # Extract macro and micro metrics
    metric_types = ['AUC', 'F1', 'Precision', 'Recall']

    roberta_macro = []
    roberta_micro = []
    distilbert_macro = []
    distilbert_micro = []

    for metric_type in metric_types:
        # Get macro values
        macro_row = df[df['metric'].str.contains(f'Macro {metric_type}')]
        if len(macro_row) > 0:
            roberta_macro.append(macro_row['roberta_large'].values[0])
            distilbert_macro.append(macro_row['distilbert_base'].values[0])
        else:
            # For AUC, the pattern is different
            macro_row = df[df['metric'] == f'{metric_type} (macro)']
            roberta_macro.append(macro_row['roberta_large'].values[0])
            distilbert_macro.append(macro_row['distilbert_base'].values[0])

        # Get micro values
        micro_row = df[df['metric'].str.contains(f'Micro {metric_type}')]
        if len(micro_row) > 0:
            roberta_micro.append(micro_row['roberta_large'].values[0])
            distilbert_micro.append(micro_row['distilbert_base'].values[0])
        else:
            # For AUC, the pattern is different
            micro_row = df[df['metric'] == f'{metric_type} (micro)']
            roberta_micro.append(micro_row['roberta_large'].values[0])
            distilbert_micro.append(micro_row['distilbert_base'].values[0])

    x = np.arange(len(metric_types))
    width = 0.2

    # Plot bars
    ax.bar(x - 1.5*width, roberta_macro, width,
           label='RoBERTa Macro', color=roberta_color, alpha=0.9)
    ax.bar(x - 0.5*width, roberta_micro, width,
           label='RoBERTa Micro', color=roberta_color, alpha=0.5)
    ax.bar(x + 0.5*width, distilbert_macro, width,
           label='DistilBERT Macro', color=distilbert_color, alpha=0.9)
    ax.bar(x + 1.5*width, distilbert_micro, width,
           label='DistilBERT Micro', color=distilbert_color, alpha=0.5)

    ax.set_xlabel('Metric Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('D. Macro vs Micro Metrics', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_types)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description='Generate metric correlation plots and trade-off analysis'
    )
    parser.add_argument(
        '--metrics-csv',
        type=str,
        required=True,
        help='Path to metric summary CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save output PNG files'
    )

    args = parser.parse_args()

    # Load metrics
    df = load_metrics_csv(args.metrics_csv)

    # Create visualizations
    logger.info("Generating metric correlation plots...")
    saved_files = create_comparison_plots(df, args.output_dir)

    logger.info("\n" + "="*70)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nGenerated {len(saved_files)} plots:")
    for i, file_path in enumerate(saved_files, 1):
        logger.info(f"  {i}. {file_path.name}")


if __name__ == '__main__':
    main()
