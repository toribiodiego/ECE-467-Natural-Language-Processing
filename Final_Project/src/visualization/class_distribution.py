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
# The script uses cached dataset (no redownload), so iteration is fast (~10s).
#
# Quick Start:
#   1. Edit configuration values below
#   2. Run: python -m src.visualization.class_distribution
#   3. View output at: output/figures/class_distribution_stacked[_no_neutral].png
#   4. Repeat to refine visualization
#
# Performance:
#   - First run: ~30-40 seconds (downloads dataset)
#   - Subsequent runs: ~8-10 seconds (uses cached dataset)
# ============================================================================

# Figure Dimensions
# -----------------
# Controls the size of the output image. Larger dimensions provide more space
# for labels and details but increase file size.

FIGURE_WIDTH = 14        # Width in inches (default: 14)
                         # Effect: Wider figures give more horizontal space for
                         #         emotion labels on x-axis, reducing overlap
                         # Recommended: 10-20 inches
                         # Example: 14 inches works well for 28 emotions

FIGURE_HEIGHT = 8        # Height in inches (default: 8)
                         # Effect: Taller figures emphasize bar height differences
                         #         and provide more vertical space for y-axis scale
                         # Recommended: 6-12 inches
                         # Example: 8 inches provides good aspect ratio at 14" width

DPI = 300                # Resolution for saved figure (default: 300)
                         # Effect: Higher DPI = sharper image, larger file size
                         # Options:
                         #   150 DPI: Draft quality, smaller files (~150KB)
                         #   300 DPI: Publication quality, medium files (~250KB)
                         #   600 DPI: Print quality, larger files (~500KB+)
                         # Note: DPI does not affect displayed size, only quality

# Data Filtering
# --------------
# Controls which emotions are included in the visualization.

INCLUDE_NEUTRAL = True   # Include 'neutral' emotion in visualization
                         # Effect on output:
                         #   True: Shows all 28 emotions (17,772 neutral samples)
                         #         Y-axis scales to ~18,000, compressing smaller emotions
                         #         File: class_distribution_stacked.png
                         #   False: Shows 27 emotions (excludes neutral)
                         #          Y-axis scales to ~5,000, better visibility of patterns
                         #          File: class_distribution_stacked_no_neutral.png
                         # Recommendation: Create both versions for comparison
                         # Context: Neutral is 3.5x larger than next emotion (admiration)
                         # See docs/replication.md for detailed neutral handling rationale

# Visualization Style
# -------------------
# Controls how multi-label breakdown is displayed.

BAR_STYLE = 'stacked'    # Visualization style (default: 'stacked')
                         # Effect: Shows each emotion as a stacked bar with segments for
                         #         1-label (blue), 2-labels (orange), 3+ labels (green)
                         # Note: Only 'stacked' style is currently supported
                         #       (basic and overlaid styles have been deprecated)

# Color Scheme
# ------------
# Controls the color palette used for multi-label categories.

COLOR_SCHEME = 'default' # Color palette for multi-label categories
                         # Effect on output:
                         #   'default': Blue (#0000ff), Green (#00ff00), Red (#ff0000)
                         #              Pure RGB colors, high visibility when scaled down
                         #   'colorblind': Blue/Orange/Green optimized for colorblindness
                         #                 (Okabe-Ito palette, safe for deuteranopia)
                         #   'sequential': Dark/Medium/Light blue gradient
                         #                 Single-hue scale, good for grayscale printing
                         # Recommendation: Use 'colorblind' for accessibility

# Output Configuration
# --------------------
# Controls file naming and additional exports.

OUTPUT_FILENAME = 'class_distribution.png'
                         # Base filename for saved figure
                         # Effect: Final filename includes modifiers:
                         #   - Style: Always appends '_stacked'
                         #   - Neutral: Appends '_no_neutral' when INCLUDE_NEUTRAL=False
                         # Examples:
                         #   'class_distribution.png' → 'class_distribution_stacked.png'
                         #   (no neutral) → 'class_distribution_stacked_no_neutral.png'
                         # Location: output/figures/

EXPORT_PER_EMOTION_STATS = True
                         # Export per-emotion multi-label breakdown to CSV
                         # Effect:
                         #   True: Creates output/stats/per_emotion_multilabel.csv
                         #         Contains counts and percentages for each emotion
                         #         Adds ~0.1 seconds to execution time
                         #   False: Skips CSV export (slightly faster iteration)
                         # CSV columns: emotion, total_frequency, 1_label_count,
                         #              1_label_pct, 2_labels_count, 2_labels_pct,
                         #              3plus_labels_count, 3plus_labels_pct

EXPORT_SUPPORTS_BY_SPLIT = True
                         # Export per-emotion support counts by split to CSV
                         # Effect:
                         #   True: Creates output/stats/per_emotion_supports_by_split.csv
                         #         Contains sample counts for each emotion per split
                         #         Useful for per-class metric joins and split analysis
                         #         Adds ~0.2 seconds to execution time
                         #   False: Skips CSV export (slightly faster iteration)
                         # CSV columns: emotion, train_count, val_count, test_count,
                         #              total_count


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
            '1_label': '#0000ff',      # Pure blue (red=0, green=0, blue=max)
            '2_labels': '#00ff00',     # Pure green (red=0, green=max, blue=0)
            '3plus_labels': '#ff0000'  # Pure red (red=max, green=0, blue=0)
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


def calculate_per_emotion_supports_by_split(
    dataset,
    label_names: List[str],
    include_neutral: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    Calculate per-emotion support counts for each dataset split.

    Args:
        dataset: DatasetDict with train/validation/test splits
        label_names: List of emotion label names
        include_neutral: Whether to include neutral emotion

    Returns:
        Dictionary mapping emotion name to split counts:
        {
            'emotion_name': {
                'train': count,
                'validation': count,
                'test': count,
                'total': count
            }
        }
    """
    # Initialize support counts for each emotion
    supports = {
        emotion: {'train': 0, 'validation': 0, 'test': 0, 'total': 0}
        for emotion in label_names
        if include_neutral or emotion != 'neutral'
    }

    # Count occurrences per split
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

                supports[emotion][split_name] += 1
                supports[emotion]['total'] += 1

    return supports


def export_per_emotion_supports_by_split(
    supports_by_split: Dict[str, Dict[str, int]],
    output_path: str
) -> str:
    """
    Export per-emotion support counts by split to CSV file.

    Args:
        supports_by_split: Dict mapping emotion to split counts
        output_path: Where to save the CSV file

    Returns:
        Absolute path to saved CSV file
    """
    # Sort emotions by total count (descending)
    sorted_emotions = sorted(
        supports_by_split.items(),
        key=lambda x: x[1]['total'],
        reverse=True
    )

    # Prepare CSV data
    rows = []
    for emotion, counts in sorted_emotions:
        rows.append({
            'emotion': emotion,
            'train_count': counts['train'],
            'val_count': counts['validation'],
            'test_count': counts['test'],
            'total_count': counts['total']
        })

    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['emotion', 'train_count', 'val_count', 'test_count', 'total_count']
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
) -> Tuple[str, Dict[str, int]]:
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
        Tuple of (absolute path to saved figure, sample counts dict)
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

    # Create stacked bars with clean labels (counts printed to console)
    x_pos = np.arange(len(emotions))
    width = 0.8

    bars1 = ax.bar(x_pos, one_label, width,
                   label='1 label',
                   color=color_1, alpha=0.9)
    bars2 = ax.bar(x_pos, two_labels, width, bottom=one_label,
                   label='2 labels',
                   color=color_2, alpha=0.9)
    bars3 = ax.bar(x_pos, three_plus, width,
                   bottom=np.array(one_label) + np.array(two_labels),
                   label='3+ labels',
                   color=color_3, alpha=0.9)

    # Customize appearance
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('GoEmotions: Multi-Label Distribution',
                 fontsize=14, fontweight='bold')

    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(emotions, rotation=45, ha='right')

    # Add legend with larger font
    ax.legend(loc='upper right', framealpha=0.95, fontsize=11)

    # Add grid for readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Tight layout
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    # Return path and sample counts for documentation
    sample_counts = {
        '1_label': total_one_label,
        '2_labels': total_two_labels,
        '3plus_labels': total_three_plus
    }

    return os.path.abspath(output_path), sample_counts


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

    # Export per-emotion supports by split if configured
    if EXPORT_SUPPORTS_BY_SPLIT:
        logger.info("Calculating per-emotion support counts by split...")
        supports_by_split = calculate_per_emotion_supports_by_split(
            dataset,
            label_names,
            include_neutral=INCLUDE_NEUTRAL
        )
        logger.info("Support counts calculated")

        logger.info("Exporting per-emotion supports by split to CSV...")
        csv_path = os.path.join(output_dir, 'stats', 'per_emotion_supports_by_split.csv')
        csv_saved_path = export_per_emotion_supports_by_split(
            supports_by_split,
            csv_path
        )
        logger.info(f"Per-emotion supports by split saved to: {csv_saved_path}")

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

    # Generate visualization (only stacked style supported)
    sample_counts = None
    if BAR_STYLE == 'stacked':
        saved_path, sample_counts = create_stacked_bar_chart(
            emotion_frequencies,
            multilabel_breakdown,
            output_path,
            figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
            dpi=DPI,
            color_scheme=COLOR_SCHEME
        )
    else:
        # Fallback to stacked if invalid style specified
        if BAR_STYLE != 'stacked':
            logger.warning(f"Only 'stacked' bar style is supported, ignoring '{BAR_STYLE}'")
        saved_path, sample_counts = create_stacked_bar_chart(
            emotion_frequencies,
            multilabel_breakdown,
            output_path,
            figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
            dpi=DPI,
            color_scheme=COLOR_SCHEME
        )

    # Calculate emotion ranking for documentation
    sorted_emotions = sorted(emotion_frequencies.items(), key=lambda x: x[1], reverse=True)
    largest_emotion = sorted_emotions[0]
    second_largest = sorted_emotions[1]

    # Report success
    logger.info("="*70)
    logger.info("Visualization complete!")
    logger.info("="*70)
    logger.info(f"Figure saved to: {saved_path}")
    logger.info(f"Configuration used:")
    logger.info(f"  - Include neutral: {INCLUDE_NEUTRAL}")
    logger.info(f"  - Bar style: stacked")
    logger.info(f"  - Color scheme: {COLOR_SCHEME}")
    logger.info(f"  - Figure size: {FIGURE_WIDTH}x{FIGURE_HEIGHT} inches")
    logger.info(f"  - DPI: {DPI}")
    logger.info("")
    logger.info("Multi-label sample counts (for documentation):")
    logger.info(f"  - 1 label samples: {sample_counts['1_label']:,}")
    logger.info(f"  - 2 labels samples: {sample_counts['2_labels']:,}")
    logger.info(f"  - 3+ labels samples: {sample_counts['3plus_labels']:,}")
    logger.info(f"  - Total: {sum(sample_counts.values()):,}")
    logger.info("")
    logger.info("Emotion class distribution (for documentation):")
    logger.info(f"  - Largest emotion: '{largest_emotion[0]}' with {largest_emotion[1]:,} samples")
    logger.info(f"  - Second largest: '{second_largest[0]}' with {second_largest[1]:,} samples")
    if largest_emotion[0] == 'neutral':
        ratio = largest_emotion[1] / second_largest[1]
        logger.info(f"  - Ratio: neutral is {ratio:.1f}x larger than {second_largest[0]}")
        logger.info(f"  - Note: Neutral's dominance ({largest_emotion[1]:,} samples) causes y-axis")
        logger.info(f"    scaling issues, making smaller emotions difficult to interpret.")
        logger.info(f"    Visualizations without neutral enable better examination of")
        logger.info(f"    multi-label patterns in the remaining {len(emotion_frequencies)-1} emotions.")
    logger.info("="*70)

    print(f"\nFigure saved to: {saved_path}")


if __name__ == "__main__":
    main()
