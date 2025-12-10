"""
Class distribution visualization for GoEmotions dataset.

This module creates visualizations showing emotion class frequencies and
multi-label distributions. Dataset is cached automatically by HuggingFace,
so subsequent runs are fast (<5 seconds) for iterative design work.

Usage:
    python -m src.visualization.class_distribution
"""

import logging
import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dotenv import load_dotenv

from src.data.load_dataset import load_go_emotions, get_label_names

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION - Modify these variables to iterate on visualization design
# ============================================================================
#
# This section contains all configurable parameters for the visualization.
# Simply modify the values below and rerun the script to see changes.
# The script uses cached dataset (no redownload), so iteration is fast (<10s).
#
# Usage:
#   1. Edit configuration values below
#   2. Run: python -m src.visualization.class_distribution
#   3. View output at: output/figures/class_distribution[_style].png
#   4. Repeat to refine visualization
# ============================================================================

# Figure settings
FIGURE_WIDTH = 14        # Width in inches (default: 14)
                         # Larger values give more space for emotion labels
                         # Recommended range: 10-20 inches

FIGURE_HEIGHT = 8        # Height in inches (default: 8)
                         # Taller figures emphasize bar heights
                         # Recommended range: 6-12 inches

DPI = 300                # Resolution for saved figure (default: 300)
                         # 300 DPI is publication quality
                         # Use 150 for drafts, 600 for print

# Data filtering
INCLUDE_NEUTRAL = True   # Include 'neutral' emotion in visualization
                         # Set to False to exclude (neutral is overrepresented)
                         # True: shows all 28 emotions
                         # False: shows 27 emotions (excludes neutral)

# Bar style
BAR_STYLE = 'basic'      # Visualization style (default: 'basic')
                         # Options:
                         #   'basic': Simple frequency bars, no multi-label breakdown
                         #   'stacked': Stacked segments showing 1/2/3+ label breakdown
                         #   'overlaid': Overlapping transparent bars for comparison
                         # Files saved as: class_distribution_[style].png

# Color scheme
COLOR_SCHEME = 'default' # Color palette for multi-label categories
                         # Options:
                         #   'default': Blue/Purple/Orange (high contrast)
                         #   'colorblind': Colorblind-friendly palette
                         #   'sequential': Single-hue sequential scale
                         # Note: Only applies to 'stacked' and 'overlaid' styles

# Output
OUTPUT_FILENAME = 'class_distribution.png'
                         # Base filename for saved figure
                         # Actual name: class_distribution[_style].png
                         # Location: output/figures/

EXPORT_PER_EMOTION_STATS = True
                         # Export per-emotion multi-label breakdown to CSV
                         # True: saves to output/stats/per_emotion_multilabel.csv
                         # False: skips CSV export (visualization only)


# ============================================================================
# Helper Functions
# ============================================================================

def get_color_scheme(scheme: str = 'default') -> Tuple[str, str, str]:
    """
    Get color palette for multi-label visualization.

    Args:
        scheme: Color scheme name ('default', 'colorblind', 'sequential')

    Returns:
        Tuple of (color_1label, color_2labels, color_3plus)
    """
    schemes = {
        'default': {
            '1_label': '#2E86AB',      # Blue
            '2_labels': '#A23B72',     # Purple
            '3plus_labels': '#F18F01'  # Orange
        },
        'colorblind': {
            '1_label': '#0173B2',      # Blue (colorblind safe)
            '2_labels': '#DE8F05',     # Orange (colorblind safe)
            '3plus_labels': '#029E73'  # Green (colorblind safe)
        },
        'sequential': {
            '1_label': '#084594',      # Dark blue
            '2_labels': '#4292C6',     # Medium blue
            '3plus_labels': '#9ECAE1'  # Light blue
        }
    }

    if scheme not in schemes:
        logger.warning(f"Unknown color scheme '{scheme}', using 'default'")
        scheme = 'default'

    colors = schemes[scheme]
    return (colors['1_label'], colors['2_labels'], colors['3plus_labels'])


def export_per_emotion_breakdown(
    multilabel_breakdown: Dict[str, Dict[str, int]],
    emotion_frequencies: Dict[str, int],
    output_path: str
) -> str:
    """
    Export per-emotion multi-label breakdown to CSV file.

    Args:
        multilabel_breakdown: Dict mapping emotion to label count breakdown
        emotion_frequencies: Dict mapping emotion to total frequency
        output_path: Where to save the CSV file

    Returns:
        Absolute path to saved CSV file
    """
    # Sort emotions by total frequency (descending)
    sorted_emotions = sorted(
        emotion_frequencies.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Prepare CSV data
    rows = []
    for emotion, total_freq in sorted_emotions:
        breakdown = multilabel_breakdown[emotion]
        one_label = breakdown['1_label']
        two_labels = breakdown['2_labels']
        three_plus = breakdown['3plus_labels']

        # Calculate percentages
        pct_one = (one_label / total_freq * 100) if total_freq > 0 else 0
        pct_two = (two_labels / total_freq * 100) if total_freq > 0 else 0
        pct_three = (three_plus / total_freq * 100) if total_freq > 0 else 0

        rows.append({
            'emotion': emotion,
            'total_frequency': total_freq,
            '1_label_count': one_label,
            '1_label_pct': f'{pct_one:.1f}',
            '2_labels_count': two_labels,
            '2_labels_pct': f'{pct_two:.1f}',
            '3plus_labels_count': three_plus,
            '3plus_labels_pct': f'{pct_three:.1f}'
        })

    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'emotion', 'total_frequency',
            '1_label_count', '1_label_pct',
            '2_labels_count', '2_labels_pct',
            '3plus_labels_count', '3plus_labels_pct'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return os.path.abspath(output_path)


# ============================================================================
# Data Processing Functions
# ============================================================================

def calculate_emotion_frequencies(
    dataset,
    label_names: List[str],
    include_neutral: bool = True
) -> Dict[str, int]:
    """
    Calculate frequency of each emotion across all splits.

    Args:
        dataset: DatasetDict with train/validation/test splits
        label_names: List of emotion label names
        include_neutral: Whether to include neutral emotion in counts

    Returns:
        Dictionary mapping emotion name to frequency count
    """
    emotion_counts = Counter()

    # Combine all splits
    for split_name in ['train', 'validation', 'test']:
        if split_name not in dataset:
            continue

        split = dataset[split_name]

        for sample in split:
            # sample['labels'] is a list of label indices
            for label_idx in sample['labels']:
                emotion = label_names[label_idx]

                # Skip neutral if configured
                if not include_neutral and emotion == 'neutral':
                    continue

                emotion_counts[emotion] += 1

    return dict(emotion_counts)


def calculate_per_emotion_multilabel_breakdown(
    dataset,
    label_names: List[str],
    include_neutral: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    For each emotion, count how many samples have 1 label vs 2 vs 3+.

    This enables visualization of whether certain emotions tend to appear
    alone or co-occur with other emotions.

    Args:
        dataset: DatasetDict with train/validation/test splits
        label_names: List of emotion label names
        include_neutral: Whether to include neutral emotion

    Returns:
        Dictionary mapping emotion name to breakdown:
        {
            'emotion_name': {
                '1_label': count,
                '2_labels': count,
                '3plus_labels': count
            }
        }
    """
    # Initialize breakdown for each emotion
    breakdown = {
        emotion: {'1_label': 0, '2_labels': 0, '3plus_labels': 0}
        for emotion in label_names
        if include_neutral or emotion != 'neutral'
    }

    # Combine all splits
    for split_name in ['train', 'validation', 'test']:
        if split_name not in dataset:
            continue

        split = dataset[split_name]

        for sample in split:
            labels = sample['labels']
            num_labels = len(labels)

            # Categorize by number of labels
            if num_labels == 1:
                category = '1_label'
            elif num_labels == 2:
                category = '2_labels'
            else:  # 3+
                category = '3plus_labels'

            # Increment for each emotion in this sample
            for label_idx in labels:
                emotion = label_names[label_idx]

                if not include_neutral and emotion == 'neutral':
                    continue

                if emotion in breakdown:
                    breakdown[emotion][category] += 1

    return breakdown


# ============================================================================
# Visualization Functions
# ============================================================================

def create_basic_bar_chart(
    emotion_frequencies: Dict[str, int],
    output_path: str,
    figsize: Tuple[float, float] = (14, 8),
    dpi: int = 300
) -> str:
    """
    Create basic bar chart of emotion frequencies.

    Args:
        emotion_frequencies: Dict mapping emotion to frequency
        output_path: Where to save the figure
        figsize: Figure dimensions (width, height) in inches
        dpi: Resolution for saved figure

    Returns:
        Absolute path to saved figure
    """
    # Sort emotions by frequency (descending)
    sorted_emotions = sorted(
        emotion_frequencies.items(),
        key=lambda x: x[1],
        reverse=True
    )

    emotions = [e[0] for e in sorted_emotions]
    frequencies = [e[1] for e in sorted_emotions]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create bar chart
    bars = ax.bar(range(len(emotions)), frequencies, color='steelblue', alpha=0.8)

    # Customize appearance
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('GoEmotions: Emotion Class Distribution', fontsize=14, fontweight='bold')

    # Set x-axis labels
    ax.set_xticks(range(len(emotions)))
    ax.set_xticklabels(emotions, rotation=45, ha='right')

    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return os.path.abspath(output_path)


def create_stacked_bar_chart(
    emotion_frequencies: Dict[str, int],
    multilabel_breakdown: Dict[str, Dict[str, int]],
    output_path: str,
    figsize: Tuple[float, float] = (14, 8),
    dpi: int = 300,
    color_scheme: str = 'default'
) -> str:
    """
    Create stacked bar chart showing multi-label breakdown per emotion.

    Each bar shows the total frequency of an emotion, with stacked segments
    indicating how many samples have 1 label, 2 labels, or 3+ labels.

    Args:
        emotion_frequencies: Dict mapping emotion to total frequency
        multilabel_breakdown: Dict mapping emotion to label count breakdown
        output_path: Where to save the figure
        figsize: Figure dimensions (width, height) in inches
        dpi: Resolution for saved figure
        color_scheme: Color scheme to use ('default', 'colorblind', 'sequential')

    Returns:
        Absolute path to saved figure
    """
    # Sort emotions by total frequency (descending)
    sorted_emotions = sorted(
        emotion_frequencies.items(),
        key=lambda x: x[1],
        reverse=True
    )

    emotions = [e[0] for e in sorted_emotions]

    # Extract breakdown data in same order
    one_label = [multilabel_breakdown[e]['1_label'] for e in emotions]
    two_labels = [multilabel_breakdown[e]['2_labels'] for e in emotions]
    three_plus = [multilabel_breakdown[e]['3plus_labels'] for e in emotions]

    # Calculate total sample counts for legend
    total_one_label = sum(one_label)
    total_two_labels = sum(two_labels)
    total_three_plus = sum(three_plus)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Get colors from scheme
    color_1, color_2, color_3 = get_color_scheme(color_scheme)

    # Create stacked bars with sample counts in labels
    x_pos = np.arange(len(emotions))
    width = 0.8

    bars1 = ax.bar(x_pos, one_label, width,
                   label=f'1 label (n={total_one_label:,})',
                   color=color_1, alpha=0.9)
    bars2 = ax.bar(x_pos, two_labels, width, bottom=one_label,
                   label=f'2 labels (n={total_two_labels:,})',
                   color=color_2, alpha=0.9)
    bars3 = ax.bar(x_pos, three_plus, width,
                   bottom=np.array(one_label) + np.array(two_labels),
                   label=f'3+ labels (n={total_three_plus:,})',
                   color=color_3, alpha=0.9)

    # Customize appearance
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('GoEmotions: Emotion Distribution with Multi-Label Breakdown',
                 fontsize=14, fontweight='bold')

    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(emotions, rotation=45, ha='right')

    # Add legend
    ax.legend(loc='upper right', framealpha=0.95)

    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Tight layout
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return os.path.abspath(output_path)


def create_overlaid_bar_chart(
    emotion_frequencies: Dict[str, int],
    multilabel_breakdown: Dict[str, Dict[str, int]],
    output_path: str,
    figsize: Tuple[float, float] = (14, 8),
    dpi: int = 300,
    color_scheme: str = 'default'
) -> str:
    """
    Create overlaid bar chart showing multi-label breakdown per emotion.

    Each emotion has three semi-transparent overlapping bars showing counts
    for 1-label, 2-label, and 3+ label samples.

    Args:
        emotion_frequencies: Dict mapping emotion to total frequency
        multilabel_breakdown: Dict mapping emotion to label count breakdown
        output_path: Where to save the figure
        figsize: Figure dimensions (width, height) in inches
        dpi: Resolution for saved figure
        color_scheme: Color scheme to use ('default', 'colorblind', 'sequential')

    Returns:
        Absolute path to saved figure
    """
    # Sort emotions by total frequency (descending)
    sorted_emotions = sorted(
        emotion_frequencies.items(),
        key=lambda x: x[1],
        reverse=True
    )

    emotions = [e[0] for e in sorted_emotions]

    # Extract breakdown data in same order
    one_label = [multilabel_breakdown[e]['1_label'] for e in emotions]
    two_labels = [multilabel_breakdown[e]['2_labels'] for e in emotions]
    three_plus = [multilabel_breakdown[e]['3plus_labels'] for e in emotions]

    # Calculate total sample counts for legend
    total_one_label = sum(one_label)
    total_two_labels = sum(two_labels)
    total_three_plus = sum(three_plus)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Get colors from scheme
    color_1, color_2, color_3 = get_color_scheme(color_scheme)

    # Create overlaid bars with different widths for visual separation
    x_pos = np.arange(len(emotions))
    width_base = 0.8

    # Widest bar (1 label) at back
    bars1 = ax.bar(x_pos, one_label, width_base,
                   label=f'1 label (n={total_one_label:,})',
                   color=color_1, alpha=0.5, zorder=1)

    # Medium bar (2 labels) in middle
    bars2 = ax.bar(x_pos, two_labels, width_base * 0.7,
                   label=f'2 labels (n={total_two_labels:,})',
                   color=color_2, alpha=0.6, zorder=2)

    # Narrowest bar (3+ labels) at front
    bars3 = ax.bar(x_pos, three_plus, width_base * 0.5,
                   label=f'3+ labels (n={total_three_plus:,})',
                   color=color_3, alpha=0.7, zorder=3)

    # Customize appearance
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('GoEmotions: Emotion Distribution with Multi-Label Breakdown (Overlaid)',
                 fontsize=14, fontweight='bold')

    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(emotions, rotation=45, ha='right')

    # Add legend
    ax.legend(loc='upper right', framealpha=0.95)

    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Tight layout
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return os.path.abspath(output_path)


# ============================================================================
# Main Execution
# ============================================================================

def main() -> None:
    """
    Main entry point for class distribution visualization.

    Loads dataset (with caching), calculates statistics, and generates
    visualization based on configuration settings at top of file.
    """
    # Load environment variables
    load_dotenv()

    logger.info("="*70)
    logger.info("GoEmotions Class Distribution Visualization")
    logger.info("="*70)

    # Load dataset (cached after first run)
    logger.info("Loading GoEmotions dataset (cached if previously downloaded)...")
    dataset = load_go_emotions()
    logger.info("Dataset loaded successfully")

    # Get label names
    label_names = get_label_names(dataset)
    logger.info(f"Found {len(label_names)} emotion labels")

    # Calculate emotion frequencies
    logger.info("Calculating emotion frequencies...")
    emotion_frequencies = calculate_emotion_frequencies(
        dataset,
        label_names,
        include_neutral=INCLUDE_NEUTRAL
    )

    num_emotions = len(emotion_frequencies)
    total_samples = sum(emotion_frequencies.values())
    logger.info(f"Calculated frequencies for {num_emotions} emotions")
    logger.info(f"Total emotion occurrences: {total_samples:,}")

    # Calculate per-emotion multi-label breakdown
    logger.info("Calculating per-emotion multi-label breakdowns...")
    multilabel_breakdown = calculate_per_emotion_multilabel_breakdown(
        dataset,
        label_names,
        include_neutral=INCLUDE_NEUTRAL
    )
    logger.info("Multi-label breakdown calculated")

    # Get output directory
    output_dir = os.getenv('OUTPUT_DIR', 'output')

    # Export per-emotion breakdown if configured
    if EXPORT_PER_EMOTION_STATS:
        logger.info("Exporting per-emotion multi-label breakdown to CSV...")
        csv_path = os.path.join(output_dir, 'stats', 'per_emotion_multilabel.csv')
        csv_saved_path = export_per_emotion_breakdown(
            multilabel_breakdown,
            emotion_frequencies,
            csv_path
        )
        logger.info(f"Per-emotion breakdown saved to: {csv_saved_path}")

    # Generate visualization based on BAR_STYLE
    logger.info(f"Generating visualization (style: {BAR_STYLE})...")

    # Modify filename based on bar style and neutral inclusion for comparison
    base_name = OUTPUT_FILENAME.replace('.png', '')

    # Add style suffix if not basic
    if BAR_STYLE != 'basic':
        filename_parts = [base_name, BAR_STYLE]
    else:
        filename_parts = [base_name]

    # Add neutral exclusion suffix if applicable
    if not INCLUDE_NEUTRAL:
        filename_parts.append('no_neutral')

    output_filename = '_'.join(filename_parts) + '.png'
    output_path = os.path.join(output_dir, 'figures', output_filename)

    if BAR_STYLE == 'basic':
        saved_path = create_basic_bar_chart(
            emotion_frequencies,
            output_path,
            figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
            dpi=DPI
        )
    elif BAR_STYLE == 'stacked':
        saved_path = create_stacked_bar_chart(
            emotion_frequencies,
            multilabel_breakdown,
            output_path,
            figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
            dpi=DPI,
            color_scheme=COLOR_SCHEME
        )
    elif BAR_STYLE == 'overlaid':
        saved_path = create_overlaid_bar_chart(
            emotion_frequencies,
            multilabel_breakdown,
            output_path,
            figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
            dpi=DPI,
            color_scheme=COLOR_SCHEME
        )
    else:
        logger.warning(f"Unknown bar style '{BAR_STYLE}', using 'basic'")
        saved_path = create_basic_bar_chart(
            emotion_frequencies,
            output_path,
            figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
            dpi=DPI
        )

    # Report success
    logger.info("="*70)
    logger.info("Visualization complete!")
    logger.info("="*70)
    logger.info(f"Figure saved to: {saved_path}")
    logger.info(f"Configuration used:")
    logger.info(f"  - Include neutral: {INCLUDE_NEUTRAL}")
    logger.info(f"  - Bar style: {BAR_STYLE}")
    if BAR_STYLE in ['stacked', 'overlaid']:
        logger.info(f"  - Color scheme: {COLOR_SCHEME}")
    logger.info(f"  - Figure size: {FIGURE_WIDTH}x{FIGURE_HEIGHT} inches")
    logger.info(f"  - DPI: {DPI}")
    logger.info("="*70)

    print(f"\nFigure saved to: {saved_path}")


if __name__ == "__main__":
    main()
