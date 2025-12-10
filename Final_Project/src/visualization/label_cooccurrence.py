"""
Label co-occurrence matrix generation for GoEmotions dataset.

This module calculates how frequently emotion labels appear together in
multi-label samples, producing a co-occurrence matrix and optional heatmap
visualization. Helps identify emotion correlations and inform modeling decisions.

Usage:
    python -m src.visualization.label_cooccurrence
"""

import logging
import os
import csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from src.data.load_dataset import load_go_emotions, get_label_names

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

INCLUDE_NEUTRAL = True   # Include 'neutral' emotion in co-occurrence analysis
GENERATE_HEATMAP = True  # Generate heatmap visualization
HEATMAP_DPI = 300        # Resolution for heatmap figure
HEATMAP_FIGSIZE = (16, 14)  # Figure size for heatmap


# ============================================================================
# Co-occurrence Calculation Functions
# ============================================================================

def calculate_cooccurrence_matrix(
    dataset,
    label_names: List[str],
    include_neutral: bool = True
) -> np.ndarray:
    """
    Calculate label co-occurrence matrix across all dataset splits.

    For each pair of labels (i, j), counts how many samples contain both
    labels simultaneously. The diagonal contains self-occurrence counts
    (how many times each label appears overall).

    Args:
        dataset: DatasetDict with train/validation/test splits
        label_names: List of emotion label names
        include_neutral: Whether to include neutral emotion

    Returns:
        Co-occurrence matrix as numpy array of shape (n_labels, n_labels)
    """
    # Filter label names if needed
    if not include_neutral and 'neutral' in label_names:
        filtered_names = [name for name in label_names if name != 'neutral']
        label_idx_map = {name: i for i, name in enumerate(filtered_names)}
        n_labels = len(filtered_names)
    else:
        filtered_names = label_names
        label_idx_map = {name: i for i, name in enumerate(label_names)}
        n_labels = len(label_names)

    # Initialize co-occurrence matrix
    cooccurrence = np.zeros((n_labels, n_labels), dtype=int)

    # Process all splits
    for split_name in ['train', 'validation', 'test']:
        if split_name not in dataset:
            continue

        split = dataset[split_name]

        for sample in split:
            # Get label names for this sample
            label_indices = sample['labels']
            sample_labels = [label_names[idx] for idx in label_indices]

            # Filter out neutral if needed
            if not include_neutral:
                sample_labels = [lbl for lbl in sample_labels if lbl != 'neutral']

            # Skip if no labels remain after filtering
            if not sample_labels:
                continue

            # Update co-occurrence counts for all label pairs
            for i, label_i in enumerate(sample_labels):
                for j, label_j in enumerate(sample_labels):
                    idx_i = label_idx_map[label_i]
                    idx_j = label_idx_map[label_j]
                    cooccurrence[idx_i, idx_j] += 1

    return cooccurrence


def export_cooccurrence_csv(
    cooccurrence_matrix: np.ndarray,
    label_names: List[str],
    output_path: str,
    include_neutral: bool = True
) -> str:
    """
    Export co-occurrence matrix to CSV file.

    CSV format:
    - First row: header with label names
    - Subsequent rows: each label's co-occurrence counts with all labels

    Args:
        cooccurrence_matrix: Co-occurrence matrix
        label_names: List of emotion label names
        output_path: Where to save the CSV file
        include_neutral: Whether neutral was included

    Returns:
        Absolute path to saved CSV file
    """
    # Filter label names if needed
    if not include_neutral and 'neutral' in label_names:
        filtered_names = [name for name in label_names if name != 'neutral']
    else:
        filtered_names = label_names

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header row
        writer.writerow(['label'] + filtered_names)

        # Data rows
        for i, label in enumerate(filtered_names):
            row = [label] + cooccurrence_matrix[i, :].tolist()
            writer.writerow(row)

    return os.path.abspath(output_path)


def create_cooccurrence_heatmap(
    cooccurrence_matrix: np.ndarray,
    label_names: List[str],
    output_path: str,
    figsize: Tuple[float, float] = (16, 14),
    dpi: int = 300,
    include_neutral: bool = True
) -> str:
    """
    Create heatmap visualization of label co-occurrence matrix.

    Args:
        cooccurrence_matrix: Co-occurrence matrix
        label_names: List of emotion label names
        output_path: Where to save the figure
        figsize: Figure dimensions (width, height) in inches
        dpi: Resolution for saved figure
        include_neutral: Whether neutral was included

    Returns:
        Absolute path to saved figure
    """
    # Filter label names if needed
    if not include_neutral and 'neutral' in label_names:
        filtered_names = [name for name in label_names if name != 'neutral']
    else:
        filtered_names = label_names

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Create heatmap with seaborn
    sns.heatmap(
        cooccurrence_matrix,
        xticklabels=filtered_names,
        yticklabels=filtered_names,
        cmap='YlOrRd',
        fmt='d',
        cbar_kws={'label': 'Co-occurrence Count'},
        ax=ax,
        square=True,
        linewidths=0.5,
        linecolor='lightgray'
    )

    # Customize appearance
    ax.set_xlabel('Emotion', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Emotion', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Label Co-occurrence Matrix', fontsize=16, fontweight='bold', pad=15)

    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

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
    Main entry point for label co-occurrence analysis.

    Loads dataset, calculates co-occurrence matrix, and exports CSV and
    optional heatmap visualization.
    """
    # Load environment variables
    load_dotenv()

    logger.info("="*70)
    logger.info("GoEmotions Label Co-occurrence Analysis")
    logger.info("="*70)

    # Load dataset (cached after first run)
    logger.info("Loading GoEmotions dataset (cached if previously downloaded)...")
    dataset = load_go_emotions()
    logger.info("Dataset loaded successfully")

    # Get label names
    label_names = get_label_names(dataset)
    logger.info(f"Found {len(label_names)} emotion labels")

    # Calculate co-occurrence matrix
    logger.info("Calculating label co-occurrence matrix...")
    cooccurrence_matrix = calculate_cooccurrence_matrix(
        dataset,
        label_names,
        include_neutral=INCLUDE_NEUTRAL
    )
    logger.info("Co-occurrence matrix calculated")

    # Get output directory
    output_dir = os.getenv('OUTPUT_DIR', 'output')

    # Export to CSV
    logger.info("Exporting co-occurrence matrix to CSV...")
    csv_path = os.path.join(output_dir, 'stats', 'label_cooccurrence.csv')
    csv_saved_path = export_cooccurrence_csv(
        cooccurrence_matrix,
        label_names,
        csv_path,
        include_neutral=INCLUDE_NEUTRAL
    )
    logger.info(f"Co-occurrence matrix saved to: {csv_saved_path}")

    # Generate heatmap if configured
    if GENERATE_HEATMAP:
        logger.info("Generating co-occurrence heatmap...")

        # Determine filename based on neutral inclusion
        if INCLUDE_NEUTRAL:
            heatmap_filename = '02_label_cooccurrence.png'
        else:
            heatmap_filename = '03_label_cooccurrence_no_neutral.png'

        heatmap_path = os.path.join(output_dir, 'figures', heatmap_filename)
        heatmap_saved_path = create_cooccurrence_heatmap(
            cooccurrence_matrix,
            label_names,
            heatmap_path,
            figsize=HEATMAP_FIGSIZE,
            dpi=HEATMAP_DPI,
            include_neutral=INCLUDE_NEUTRAL
        )
        logger.info(f"Heatmap saved to: {heatmap_saved_path}")

    # Calculate and report statistics
    filtered_labels = label_names if INCLUDE_NEUTRAL else [l for l in label_names if l != 'neutral']
    n_labels = len(filtered_labels)

    # Diagonal contains self-occurrence counts
    total_samples_with_labels = cooccurrence_matrix.diagonal().sum()

    # Off-diagonal contains co-occurrences
    off_diagonal_mask = ~np.eye(n_labels, dtype=bool)
    total_cooccurrences = cooccurrence_matrix[off_diagonal_mask].sum() // 2  # Divide by 2 because matrix is symmetric

    # Find most common co-occurrence pairs
    max_cooccurrence = 0
    max_pair = None
    for i in range(n_labels):
        for j in range(i + 1, n_labels):  # Only upper triangle
            count = cooccurrence_matrix[i, j]
            if count > max_cooccurrence:
                max_cooccurrence = count
                max_pair = (filtered_labels[i], filtered_labels[j])

    # Report results
    logger.info("="*70)
    logger.info("Co-occurrence analysis complete!")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  - Include neutral: {INCLUDE_NEUTRAL}")
    logger.info(f"  - Number of labels: {n_labels}")
    logger.info(f"  - Generate heatmap: {GENERATE_HEATMAP}")
    logger.info("")
    logger.info("Statistics:")
    logger.info(f"  - Total label occurrences: {total_samples_with_labels:,}")
    logger.info(f"  - Total co-occurrences: {total_cooccurrences:,}")
    if max_pair:
        logger.info(f"  - Most common pair: '{max_pair[0]}' + '{max_pair[1]}' ({max_cooccurrence:,} times)")
    logger.info("="*70)

    print(f"\nCo-occurrence matrix saved to: {csv_saved_path}")
    if GENERATE_HEATMAP:
        print(f"Heatmap saved to: {heatmap_saved_path}")


if __name__ == "__main__":
    main()
