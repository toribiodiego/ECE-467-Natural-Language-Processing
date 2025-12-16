"""
Per-Emotion F1 Visualization Script

Generates a bar chart showing F1 scores for all 28 emotions, ranked by performance.
Helps identify which emotions the model classifies well vs. poorly.

Usage:
    python -m src.analysis.visualize_per_emotion_f1 \
        --input artifacts/stats/per_emotion_scores.csv \
        --output artifacts/stats/per_emotion_f1_chart.png
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


def load_per_emotion_scores(csv_path: str) -> pd.DataFrame:
    """Load per-emotion scores from CSV.

    Args:
        csv_path: Path to per_emotion_scores.csv

    Returns:
        DataFrame with per-emotion metrics
    """
    logger.info(f"Loading per-emotion scores from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} emotions")
    return df


def create_f1_bar_chart(df: pd.DataFrame, output_path: str):
    """Create bar chart of F1 scores for all emotions.

    Args:
        df: DataFrame with per-emotion scores (must be sorted by F1)
        output_path: Path to save output PNG file
    """
    logger.info("Creating F1 score bar chart...")

    # Filter out emotions with F1=0.000 for clearer visualization
    df_filtered = df[df['f1'] > 0.0].copy()
    num_excluded = len(df) - len(df_filtered)
    logger.info(f"Filtering visualization to {len(df_filtered)} emotions with F1 > 0 (excluding {num_excluded} with F1=0.000)")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data (already sorted by F1 in the CSV)
    emotions = df_filtered['emotion'].values
    f1_scores = df_filtered['f1'].values
    support = df_filtered['support'].values

    # Create positions for bars
    y_pos = np.arange(len(emotions))

    # Color gradient: green for high F1, red for low F1
    # Normalize F1 scores to [0, 1] range for color mapping
    max_f1 = f1_scores.max()
    if max_f1 > 0:
        normalized_f1 = f1_scores / max_f1
    else:
        normalized_f1 = f1_scores

    # Create color map (red to yellow to green)
    colors = plt.cm.RdYlGn(normalized_f1)

    # Create horizontal bar chart
    bars = ax.barh(y_pos, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(emotions, fontsize=9)
    ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_title('Per-Emotion F1 Scores (Sorted by Performance)',
                fontsize=14, fontweight='bold', pad=15)

    # Add F1 score labels on bars
    for i, (bar, f1, sup) in enumerate(zip(bars, f1_scores, support)):
        width = bar.get_width()
        # Position label inside bar if it's long enough, otherwise outside
        if width > 0.15:
            label_x = width - 0.02
            ha = 'right'
        else:
            label_x = width + 0.01
            ha = 'left'

        # Show F1 score only (support counts available in table)
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
               f'{f1:.3f}',
               ha=ha, va='center', fontsize=9, fontweight='bold')

    # Add vertical line at 0.5 for reference
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='F1=0.5')

    # Set x-axis limits
    ax.set_xlim(0, 1.0)

    # Add grid
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend in top right
    ax.legend(loc='upper right', framealpha=0.9)

    # Add colorbar to show F1 score scale
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=plt.cm.RdYlGn, norm=Normalize(vmin=0, vmax=max_f1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('F1 Score', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Chart saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate bar chart visualization of per-emotion F1 scores'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to per_emotion_scores.csv'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output PNG file'
    )

    args = parser.parse_args()

    # Load scores
    df = load_per_emotion_scores(args.input)

    # Create chart
    create_f1_bar_chart(df, args.output)

    logger.info("\n" + "="*70)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nBar chart saved to: {args.output}")
    logger.info(f"\nTop 5 emotions by F1:")
    for _, row in df.head(5).iterrows():
        logger.info(f"  {row['rank']:2d}. {row['emotion']:15s} F1={row['f1']:.3f} (support={int(row['support'])})")
    logger.info(f"\nBottom 5 emotions by F1:")
    for _, row in df.tail(5).iterrows():
        logger.info(f"  {row['rank']:2d}. {row['emotion']:15s} F1={row['f1']:.3f} (support={int(row['support'])})")


if __name__ == '__main__':
    main()
