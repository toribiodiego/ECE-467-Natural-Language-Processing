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

# Figure settings
FIGURE_WIDTH = 14        # Width in inches
FIGURE_HEIGHT = 8        # Height in inches
DPI = 300                # Resolution for saved figure

# Data filtering
INCLUDE_NEUTRAL = True   # Set to False to exclude 'neutral' emotion

# Bar style
BAR_STYLE = 'basic'      # Options: 'basic', 'stacked', 'overlaid'
                         # 'basic': simple frequency bars
                         # 'stacked': show 1-label/2-label/3+ stacked within each bar
                         # 'overlaid': semi-transparent overlapping layers

# Color scheme
COLOR_SCHEME = 'default' # Options: 'default', 'colorblind', 'sequential'

# Output
OUTPUT_FILENAME = 'class_distribution.png'


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

    # Calculate per-emotion multi-label breakdown (for future use)
    logger.info("Calculating per-emotion multi-label breakdowns...")
    multilabel_breakdown = calculate_per_emotion_multilabel_breakdown(
        dataset,
        label_names,
        include_neutral=INCLUDE_NEUTRAL
    )
    logger.info("Multi-label breakdown calculated")

    # Generate visualization based on BAR_STYLE
    logger.info(f"Generating visualization (style: {BAR_STYLE})...")

    output_dir = os.getenv('OUTPUT_DIR', 'output')
    output_path = os.path.join(output_dir, 'figures', OUTPUT_FILENAME)

    if BAR_STYLE == 'basic':
        saved_path = create_basic_bar_chart(
            emotion_frequencies,
            output_path,
            figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
            dpi=DPI
        )
    else:
        # Stacked and overlaid styles will be implemented in later subtasks
        logger.warning(f"Bar style '{BAR_STYLE}' not yet implemented, using 'basic'")
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
    logger.info(f"  - Figure size: {FIGURE_WIDTH}x{FIGURE_HEIGHT} inches")
    logger.info(f"  - DPI: {DPI}")
    logger.info("="*70)

    print(f"\nFigure saved to: {saved_path}")


if __name__ == "__main__":
    main()
