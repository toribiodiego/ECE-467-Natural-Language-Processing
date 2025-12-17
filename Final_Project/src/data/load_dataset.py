"""
Dataset loading utilities for GoEmotions emotion classification.

This module provides functions to load the GoEmotions dataset from
HuggingFace Hub with proper error handling and validation.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, DatasetDict

# Configure logging
logger = logging.getLogger(__name__)


def load_go_emotions(
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """
    Load the GoEmotions dataset from HuggingFace Hub.

    The GoEmotions dataset contains ~58k Reddit comments labeled with 27 emotion
    categories plus neutral. It's a multi-label classification dataset where
    samples can have multiple emotions.

    Args:
        cache_dir: Optional directory to cache the downloaded dataset.
                  If None, uses HuggingFace's default cache location.

    Returns:
        DatasetDict with 'train', 'validation', and 'test' splits.
        Each split contains 'text' and 'labels' fields.

    Raises:
        FileNotFoundError: If the dataset cannot be found on HuggingFace Hub.
        ConnectionError: If there's a network issue downloading the dataset.
        RuntimeError: If the dataset structure is invalid or corrupted.

    Example:
        >>> dataset = load_go_emotions()
        >>> print(f"Train size: {len(dataset['train'])}")
        >>> print(f"Validation size: {len(dataset['validation'])}")
        >>> print(f"Test size: {len(dataset['test'])}")
    """
    try:
        logger.info("Loading GoEmotions dataset from HuggingFace Hub...")

        dataset = load_dataset(
            "go_emotions",
            cache_dir=cache_dir
        )

        # Validate dataset structure
        if not isinstance(dataset, DatasetDict):
            raise RuntimeError(
                f"Expected DatasetDict but got {type(dataset).__name__}"
            )

        required_splits = {'train', 'validation', 'test'}
        available_splits = set(dataset.keys())

        if not required_splits.issubset(available_splits):
            missing = required_splits - available_splits
            raise RuntimeError(
                f"Dataset missing required splits: {missing}. "
                f"Available splits: {available_splits}"
            )

        # Validate required columns
        required_columns = {'text', 'labels'}
        for split_name in required_splits:
            split_columns = set(dataset[split_name].column_names)
            if not required_columns.issubset(split_columns):
                missing = required_columns - split_columns
                raise RuntimeError(
                    f"Split '{split_name}' missing required columns: {missing}. "
                    f"Available columns: {split_columns}"
                )

        # Log dataset statistics
        logger.info("GoEmotions dataset loaded successfully")
        logger.info(f"  Train samples: {len(dataset['train']):,}")
        logger.info(f"  Validation samples: {len(dataset['validation']):,}")
        logger.info(f"  Test samples: {len(dataset['test']):,}")

        # Log label information
        if hasattr(dataset['train'].features['labels'], 'feature'):
            label_names = dataset['train'].features['labels'].feature.names
            logger.info(f"  Number of labels: {len(label_names)}")
            logger.debug(f"  Label names: {label_names}")

        return dataset

    except FileNotFoundError as e:
        logger.error("GoEmotions dataset not found on HuggingFace Hub")
        raise FileNotFoundError(
            "Could not find 'go_emotions' dataset on HuggingFace Hub. "
            "Ensure you have an internet connection and the dataset name is correct."
        ) from e

    except ConnectionError as e:
        logger.error("Network error while downloading dataset")
        raise ConnectionError(
            "Failed to download GoEmotions dataset. "
            "Check your internet connection and try again."
        ) from e

    except Exception as e:
        logger.error(f"Unexpected error loading dataset: {e}")
        raise RuntimeError(
            f"Failed to load GoEmotions dataset: {e}"
        ) from e


def get_label_names(dataset: DatasetDict) -> List[str]:
    """
    Extract emotion label names from the dataset.

    Args:
        dataset: DatasetDict returned from load_go_emotions()

    Returns:
        List of emotion label names (e.g., ['admiration', 'amusement', ...])

    Example:
        >>> dataset = load_go_emotions()
        >>> labels = get_label_names(dataset)
        >>> print(f"Number of emotions: {len(labels)}")
        >>> print(f"First 5 emotions: {labels[:5]}")
    """
    if 'train' not in dataset:
        raise ValueError("Dataset must contain 'train' split")

    if 'labels' not in dataset['train'].features:
        raise ValueError("Dataset must contain 'labels' feature")

    label_feature = dataset['train'].features['labels']

    if hasattr(label_feature, 'feature') and hasattr(label_feature.feature, 'names'):
        return label_feature.feature.names
    else:
        raise RuntimeError(
            "Could not extract label names from dataset. "
            "Dataset structure may have changed."
        )


def get_dataset_statistics(dataset: DatasetDict) -> Dict[str, Dict[str, int]]:
    """
    Calculate basic statistics for each dataset split including label distribution.

    Args:
        dataset: DatasetDict returned from load_go_emotions()

    Returns:
        Dictionary mapping split names to statistics:
        {
            'train': {'num_samples': 43410, 'num_labels': 28},
            'validation': {'num_samples': 5426, 'num_labels': 28},
            'test': {'num_samples': 5427, 'num_labels': 28},
            'label_distribution': [1234, 5678, ...]  # Count per label in train set
        }

    Example:
        >>> dataset = load_go_emotions()
        >>> stats = get_dataset_statistics(dataset)
        >>> print(f"Total samples: {sum(s['num_samples'] for s in stats.values())}")
    """
    statistics = {}

    label_names = get_label_names(dataset)
    num_labels = len(label_names)

    for split_name in ['train', 'validation', 'test']:
        if split_name in dataset:
            statistics[split_name] = {
                'num_samples': len(dataset[split_name]),
                'num_labels': num_labels
            }

    # Compute label distribution from training set for class weighting
    if 'train' in dataset:
        label_counts = [0] * num_labels
        for sample in dataset['train']:
            for label_idx in sample['labels']:
                label_counts[label_idx] += 1
        statistics['label_distribution'] = label_counts

    return statistics


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    print("Loading GoEmotions dataset...")
    dataset = load_go_emotions()

    print("\nDataset splits:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data):,} samples")

    print("\nEmotion labels:")
    labels = get_label_names(dataset)
    print(f"  Total: {len(labels)} emotions")
    print(f"  Labels: {', '.join(labels[:10])}...")

    print("\nDataset statistics:")
    stats = get_dataset_statistics(dataset)
    for split_name, split_stats in stats.items():
        print(f"  {split_name}: {split_stats}")
