"""
Text length distribution analysis for GoEmotions dataset.

This module analyzes comment length distributions across dataset splits,
calculating both character counts and token counts. Exports summary
statistics and histogram visualizations to help inform max_seq_length
choices for model training.

Usage:
    python -m src.analysis.text_length_analysis
"""

import logging
import os
import csv
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.data.load_dataset import load_go_emotions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Tokenizer to use for token counting
# Using RoBERTa tokenizer as it's one of the primary models for this project
TOKENIZER_NAME = "roberta-base"

# Histogram configuration
GENERATE_HISTOGRAMS = True
HISTOGRAM_DPI = 300
HISTOGRAM_FIGSIZE = (14, 5)  # Width x Height for single row of 3 plots
CHAR_BINS = 50  # Number of bins for character length histogram
TOKEN_BINS = 50  # Number of bins for token length histogram


# ============================================================================
# Length Calculation Functions
# ============================================================================

def calculate_length_statistics(
    dataset,
    tokenizer,
    split_name: str
) -> Tuple[Dict[str, float], List[int], List[int]]:
    """
    Calculate text length statistics for a dataset split.

    Args:
        dataset: Dataset split to analyze
        tokenizer: HuggingFace tokenizer for token counting
        split_name: Name of the split (for logging)

    Returns:
        Tuple of (statistics dict, character counts list, token counts list)
    """
    char_lengths = []
    token_lengths = []

    logger.info(f"Processing {split_name} split ({len(dataset)} samples)...")

    for i, sample in enumerate(dataset):
        text = sample['text']

        # Character count
        char_count = len(text)
        char_lengths.append(char_count)

        # Token count (using tokenizer)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_count = len(tokens)
        token_lengths.append(token_count)

        # Progress logging
        if (i + 1) % 10000 == 0:
            logger.info(f"  Processed {i + 1}/{len(dataset)} samples...")

    # Calculate statistics
    stats = {
        'split': split_name,
        'num_samples': len(dataset),

        # Character statistics
        'char_min': int(np.min(char_lengths)),
        'char_max': int(np.max(char_lengths)),
        'char_mean': float(np.mean(char_lengths)),
        'char_median': float(np.median(char_lengths)),
        'char_std': float(np.std(char_lengths)),
        'char_q25': float(np.percentile(char_lengths, 25)),
        'char_q75': float(np.percentile(char_lengths, 75)),
        'char_q90': float(np.percentile(char_lengths, 90)),
        'char_q95': float(np.percentile(char_lengths, 95)),
        'char_q99': float(np.percentile(char_lengths, 99)),

        # Token statistics
        'token_min': int(np.min(token_lengths)),
        'token_max': int(np.max(token_lengths)),
        'token_mean': float(np.mean(token_lengths)),
        'token_median': float(np.median(token_lengths)),
        'token_std': float(np.std(token_lengths)),
        'token_q25': float(np.percentile(token_lengths, 25)),
        'token_q75': float(np.percentile(token_lengths, 75)),
        'token_q90': float(np.percentile(token_lengths, 90)),
        'token_q95': float(np.percentile(token_lengths, 95)),
        'token_q99': float(np.percentile(token_lengths, 99)),
    }

    return stats, char_lengths, token_lengths


def calculate_truncation_rates(
    token_lengths: List[int],
    max_lengths: List[int] = [128, 256, 512]
) -> Dict[int, float]:
    """
    Calculate what percentage of samples would be truncated at various max lengths.

    Args:
        token_lengths: List of token counts
        max_lengths: List of max_seq_length values to test

    Returns:
        Dictionary mapping max_length to truncation percentage
    """
    total_samples = len(token_lengths)
    truncation_rates = {}

    for max_len in max_lengths:
        truncated = sum(1 for length in token_lengths if length > max_len)
        truncation_rates[max_len] = (truncated / total_samples) * 100

    return truncation_rates


# ============================================================================
# Export Functions
# ============================================================================

def export_length_statistics(
    all_stats: List[Dict[str, float]],
    output_path: str
) -> str:
    """
    Export length statistics to CSV file.

    Args:
        all_stats: List of statistics dictionaries (one per split)
        output_path: Where to save the CSV file

    Returns:
        Absolute path to saved CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if not all_stats:
            return os.path.abspath(output_path)

        fieldnames = list(all_stats[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)

    return os.path.abspath(output_path)


def export_truncation_analysis(
    split_names: List[str],
    all_token_lengths: Dict[str, List[int]],
    output_path: str,
    max_lengths: List[int] = [128, 256, 512]
) -> str:
    """
    Export truncation rate analysis to CSV file.

    Args:
        split_names: List of split names
        all_token_lengths: Dictionary mapping split name to token lengths
        output_path: Where to save the CSV file
        max_lengths: List of max_seq_length values to analyze

    Returns:
        Absolute path to saved CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows = []
    for split_name in split_names:
        token_lengths = all_token_lengths[split_name]
        truncation_rates = calculate_truncation_rates(token_lengths, max_lengths)

        for max_len, trunc_rate in truncation_rates.items():
            rows.append({
                'split': split_name,
                'max_seq_length': max_len,
                'truncation_rate_pct': f'{trunc_rate:.2f}',
                'samples_truncated': int(len(token_lengths) * trunc_rate / 100),
                'total_samples': len(token_lengths)
            })

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['split', 'max_seq_length', 'truncation_rate_pct',
                     'samples_truncated', 'total_samples']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return os.path.abspath(output_path)


# ============================================================================
# Visualization Functions
# ============================================================================

def create_character_histogram(
    split_names: List[str],
    all_char_lengths: Dict[str, List[int]],
    output_path: str,
    figsize: Tuple[float, float] = (14, 5),
    dpi: int = 300,
    char_bins: int = 50
) -> str:
    """
    Create histogram visualization of character length distributions.

    Args:
        split_names: List of split names
        all_char_lengths: Dictionary mapping split to character lengths
        output_path: Where to save the figure
        figsize: Figure dimensions (width, height) in inches
        dpi: Resolution for saved figure
        char_bins: Number of bins for character histograms

    Returns:
        Absolute path to saved figure
    """
    # Create figure with 1 row x 3 columns (train, val, test)
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green

    # Find maximum character length across all splits for unified x-axis
    max_char_length = max(max(lengths) for lengths in all_char_lengths.values())

    # Plot character length distributions
    for i, split_name in enumerate(split_names):
        ax = axes[i]
        char_lengths = all_char_lengths[split_name]

        ax.hist(char_lengths, bins=char_bins, color=colors[i],
                alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Character Count', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{split_name.capitalize()}',
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Set unified x-axis limit
        ax.set_xlim(0, max_char_length)

    # Overall title
    fig.suptitle('Character Length Distributions by Split',
                fontsize=16, fontweight='bold', y=1.02)

    # Tight layout
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return os.path.abspath(output_path)


def create_token_histogram(
    split_names: List[str],
    all_token_lengths: Dict[str, List[int]],
    output_path: str,
    figsize: Tuple[float, float] = (14, 5),
    dpi: int = 300,
    token_bins: int = 50
) -> str:
    """
    Create histogram visualization of token length distributions.

    Args:
        split_names: List of split names
        all_token_lengths: Dictionary mapping split to token lengths
        output_path: Where to save the figure
        figsize: Figure dimensions (width, height) in inches
        dpi: Resolution for saved figure
        token_bins: Number of bins for token histograms

    Returns:
        Absolute path to saved figure
    """
    # Create figure with 1 row x 3 columns (train, val, test)
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green

    # Find maximum token length across all splits for unified x-axis
    max_token_length = max(max(lengths) for lengths in all_token_lengths.values())

    # Plot token length distributions
    for i, split_name in enumerate(split_names):
        ax = axes[i]
        token_lengths = all_token_lengths[split_name]

        ax.hist(token_lengths, bins=token_bins, color=colors[i],
                alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Token Count', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{split_name.capitalize()}',
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Set unified x-axis limit
        ax.set_xlim(0, max_token_length)

        # Add vertical lines for common max_seq_length values
        for max_len, line_color, label in [(128, '#ff0000', '128'),
                                            (256, '#ff8c00', '256'),
                                            (512, '#008000', '512')]:
            ax.axvline(x=max_len, color=line_color, linestyle='--',
                      linewidth=2, alpha=0.8, label=f'max={label}')

        if i == 2:  # Only add legend to rightmost plot
            ax.legend(loc='upper right', fontsize=9)

    # Overall title
    fig.suptitle('Token Length Distributions by Split',
                fontsize=16, fontweight='bold', y=1.02)

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
    Main entry point for text length analysis.

    Loads dataset, calculates length statistics, and exports CSV and
    histogram visualizations.
    """
    # Load environment variables
    load_dotenv()

    logger.info("="*70)
    logger.info("GoEmotions Text Length Distribution Analysis")
    logger.info("="*70)

    # Load dataset
    logger.info("Loading GoEmotions dataset...")
    dataset = load_go_emotions()
    logger.info("Dataset loaded successfully")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    logger.info("Tokenizer loaded successfully")

    # Calculate statistics for each split
    split_names = ['train', 'validation', 'test']
    all_stats = []
    all_char_lengths = {}
    all_token_lengths = {}

    for split_name in split_names:
        stats, char_lengths, token_lengths = calculate_length_statistics(
            dataset[split_name],
            tokenizer,
            split_name
        )
        all_stats.append(stats)
        all_char_lengths[split_name] = char_lengths
        all_token_lengths[split_name] = token_lengths

    # Get output directory
    output_dir = os.getenv('OUTPUT_DIR', 'output')

    # Export summary statistics
    logger.info("Exporting length statistics to CSV...")
    stats_path = os.path.join(output_dir, 'stats', 'text_length_statistics.csv')
    stats_saved_path = export_length_statistics(all_stats, stats_path)
    logger.info(f"Statistics saved to: {stats_saved_path}")

    # Export truncation analysis
    logger.info("Exporting truncation analysis to CSV...")
    trunc_path = os.path.join(output_dir, 'stats', 'truncation_analysis.csv')
    trunc_saved_path = export_truncation_analysis(
        split_names,
        all_token_lengths,
        trunc_path
    )
    logger.info(f"Truncation analysis saved to: {trunc_saved_path}")

    # Generate histograms if configured
    if GENERATE_HISTOGRAMS:
        logger.info("Generating character length distribution histogram...")
        char_histogram_path = os.path.join(output_dir, 'figures', '04_character_length_distributions.png')
        char_histogram_saved_path = create_character_histogram(
            split_names,
            all_char_lengths,
            char_histogram_path,
            figsize=HISTOGRAM_FIGSIZE,
            dpi=HISTOGRAM_DPI,
            char_bins=CHAR_BINS
        )
        logger.info(f"Character histogram saved to: {char_histogram_saved_path}")

        logger.info("Generating token length distribution histogram...")
        token_histogram_path = os.path.join(output_dir, 'figures', '05_token_length_distributions.png')
        token_histogram_saved_path = create_token_histogram(
            split_names,
            all_token_lengths,
            token_histogram_path,
            figsize=HISTOGRAM_FIGSIZE,
            dpi=HISTOGRAM_DPI,
            token_bins=TOKEN_BINS
        )
        logger.info(f"Token histogram saved to: {token_histogram_saved_path}")

    # Report summary
    logger.info("="*70)
    logger.info("Text length analysis complete!")
    logger.info("="*70)
    logger.info(f"Tokenizer used: {TOKENIZER_NAME}")
    logger.info("")

    # Print key statistics
    for stats in all_stats:
        logger.info(f"{stats['split'].capitalize()} split:")
        logger.info(f"  Character length: mean={stats['char_mean']:.1f}, "
                   f"median={stats['char_median']:.0f}, "
                   f"95th={stats['char_q95']:.0f}")
        logger.info(f"  Token length:     mean={stats['token_mean']:.1f}, "
                   f"median={stats['token_median']:.0f}, "
                   f"95th={stats['token_q95']:.0f}")

    # Print truncation rates
    logger.info("")
    logger.info("Truncation rates at common max_seq_length values:")
    for split_name in split_names:
        token_lengths = all_token_lengths[split_name]
        trunc_rates = calculate_truncation_rates(token_lengths)
        logger.info(f"  {split_name.capitalize()}: "
                   f"128={trunc_rates[128]:.2f}%, "
                   f"256={trunc_rates[256]:.2f}%, "
                   f"512={trunc_rates[512]:.2f}%")

    logger.info("="*70)

    print(f"\nStatistics saved to: {stats_saved_path}")
    print(f"Truncation analysis saved to: {trunc_saved_path}")
    if GENERATE_HISTOGRAMS:
        print(f"Character histogram saved to: {char_histogram_saved_path}")
        print(f"Token histogram saved to: {token_histogram_saved_path}")


if __name__ == "__main__":
    main()
