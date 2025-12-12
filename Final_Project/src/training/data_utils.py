"""
Data loading and tokenization utilities for GoEmotions training.

This module provides reusable functions for loading the GoEmotions dataset,
tokenizing texts, and creating PyTorch DataLoaders with proper seed control
and token coverage statistics.
"""

import logging
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer

from src.data.load_dataset import load_go_emotions, get_label_names


# Configure logging
logger = logging.getLogger(__name__)


class GoEmotionsDataset(TorchDataset):
    """
    PyTorch Dataset wrapper for GoEmotions with tokenized inputs.

    Converts HuggingFace dataset samples to PyTorch tensors suitable for
    multi-label classification training.
    """

    def __init__(
        self,
        encodings: Dict[str, List],
        labels: List[List[int]],
        num_labels: int,
        texts: Optional[List[str]] = None
    ):
        """
        Initialize dataset.

        Args:
            encodings: Dictionary with 'input_ids' and 'attention_mask'
            labels: List of label lists (multi-label format)
            num_labels: Total number of possible labels
            texts: Optional list of original text strings (for prediction export)
        """
        self.encodings = encodings
        self.labels = labels
        self.num_labels = num_labels
        self.texts = texts

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with input_ids, attention_mask, and labels tensors
        """
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx])
        }

        # Convert multi-label to binary vector
        label_vector = torch.zeros(self.num_labels)
        for label_idx in self.labels[idx]:
            label_vector[label_idx] = 1.0

        item['labels'] = label_vector
        return item


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_with_limits(
    max_train_samples: int = None,
    max_eval_samples: int = None
) -> Tuple[DatasetDict, List[str], int]:
    """
    Load GoEmotions dataset with optional sample limiting for testing.

    Args:
        max_train_samples: Maximum number of training samples (None for all)
        max_eval_samples: Maximum number of eval samples (None for all)

    Returns:
        Tuple of (dataset, label_names, num_labels):
        - dataset: DatasetDict with train/validation/test splits
        - label_names: List of emotion label names
        - num_labels: Number of emotion labels (28 for GoEmotions)

    Raises:
        RuntimeError: If dataset loading fails
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Loading Dataset")
    logger.info("=" * 70)

    try:
        # Load dataset from HuggingFace Hub
        dataset = load_go_emotions()

        # Get label information
        label_names = get_label_names(dataset)
        num_labels = len(label_names)

        logger.info(f"Loaded {num_labels} emotion labels")
        logger.debug(f"Labels: {', '.join(label_names)}")

        # Apply sample limiting for testing if specified
        if max_train_samples is not None:
            original_size = len(dataset['train'])
            dataset['train'] = dataset['train'].select(range(min(max_train_samples, original_size)))
            logger.warning(f"Limited training samples: {original_size:,} → {len(dataset['train']):,}")

        if max_eval_samples is not None:
            original_val_size = len(dataset['validation'])
            dataset['validation'] = dataset['validation'].select(range(min(max_eval_samples, original_val_size)))
            logger.warning(f"Limited validation samples: {original_val_size:,} → {len(dataset['validation']):,}")

            original_test_size = len(dataset['test'])
            dataset['test'] = dataset['test'].select(range(min(max_eval_samples, original_test_size)))
            logger.warning(f"Limited test samples: {original_test_size:,} → {len(dataset['test']):,}")

        # Print final dataset statistics
        logger.info("")
        logger.info("Final dataset sizes:")
        logger.info(f"  Train:      {len(dataset['train']):,} samples")
        logger.info(f"  Validation: {len(dataset['validation']):,} samples")
        logger.info(f"  Test:       {len(dataset['test']):,} samples")
        logger.info(f"  Total:      {len(dataset['train']) + len(dataset['validation']) + len(dataset['test']):,} samples")

        logger.info("=" * 70)

        return dataset, label_names, num_labels

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise RuntimeError(f"Dataset loading failed: {e}") from e


def compute_token_coverage_stats(
    encodings: Dict[str, List],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    split_name: str = "dataset"
) -> Dict[str, float]:
    """
    Compute token coverage statistics for tokenized data.

    Args:
        encodings: Tokenizer output with 'input_ids' and 'attention_mask'
        tokenizer: Tokenizer used for encoding
        max_seq_length: Maximum sequence length
        split_name: Name of the dataset split for logging

    Returns:
        Dictionary with coverage statistics:
        - truncation_rate: Percentage of sequences that were truncated
        - avg_token_length: Average number of tokens per sequence
        - max_token_length: Maximum token length found
    """
    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']

    # Calculate actual lengths (excluding padding)
    actual_lengths = [sum(mask) for mask in attention_masks]

    # Count truncated sequences (those that hit max length)
    truncated_count = sum(1 for length in actual_lengths if length == max_seq_length)
    truncation_rate = (truncated_count / len(actual_lengths)) * 100 if actual_lengths else 0

    # Calculate statistics
    avg_length = np.mean(actual_lengths) if actual_lengths else 0
    max_length = max(actual_lengths) if actual_lengths else 0

    stats = {
        'truncation_rate': truncation_rate,
        'avg_token_length': avg_length,
        'max_token_length': max_length
    }

    logger.debug(f"{split_name} tokenization stats:")
    logger.debug(f"  Truncation rate: {truncation_rate:.2f}%")
    logger.debug(f"  Avg token length: {avg_length:.1f}")
    logger.debug(f"  Max token length: {max_length}")

    return stats


def create_dataloaders(
    dataset: DatasetDict,
    num_labels: int,
    model_name_or_path: str,
    batch_size: int = 16,
    max_seq_length: int = 128,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer]:
    """
    Tokenize dataset and create DataLoaders for training.

    Args:
        dataset: DatasetDict with train/validation/test splits
        num_labels: Number of emotion labels
        model_name_or_path: HuggingFace model identifier for tokenizer
        batch_size: Batch size for DataLoaders
        max_seq_length: Maximum sequence length for tokenization
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader, tokenizer)

    Raises:
        RuntimeError: If tokenization fails
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Tokenization and Preprocessing")
    logger.info("=" * 70)

    try:
        # Set seed for reproducibility
        set_seed(seed)

        # Load tokenizer for the specified model
        logger.info(f"Loading tokenizer for {model_name_or_path}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

        # Tokenize each split
        logger.info(f"Tokenizing texts (max_length={max_seq_length})...")

        def tokenize_split(split_data, split_name: str):
            """Helper to tokenize a dataset split."""
            # Convert Dataset columns to lists
            texts = list(split_data['text'])
            labels = list(split_data['labels'])

            # Tokenize texts
            encodings = tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=max_seq_length,
                return_tensors=None  # Return lists, not tensors
            )

            # Compute token coverage statistics
            compute_token_coverage_stats(encodings, tokenizer, max_seq_length, split_name)

            return encodings, labels

        # Tokenize all splits
        train_encodings, train_labels = tokenize_split(dataset['train'], "Train")
        val_encodings, val_labels = tokenize_split(dataset['validation'], "Validation")
        test_encodings, test_labels = tokenize_split(dataset['test'], "Test")

        logger.info(f"  Train: {len(train_labels):,} samples tokenized")
        logger.info(f"  Validation: {len(val_labels):,} samples tokenized")
        logger.info(f"  Test: {len(test_labels):,} samples tokenized")

        # Extract texts from raw datasets for prediction export
        train_texts = dataset['train']['text']
        val_texts = dataset['validation']['text']
        test_texts = dataset['test']['text']

        # Create PyTorch datasets
        logger.info("Creating PyTorch datasets...")
        train_dataset = GoEmotionsDataset(train_encodings, train_labels, num_labels, texts=train_texts)
        val_dataset = GoEmotionsDataset(val_encodings, val_labels, num_labels, texts=val_texts)
        test_dataset = GoEmotionsDataset(test_encodings, test_labels, num_labels, texts=test_texts)

        # Create DataLoaders with seed control
        logger.info(f"Creating DataLoaders (batch_size={batch_size}, seed={seed})...")

        # Create generator for reproducible shuffling
        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for compatibility
            generator=generator
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        logger.info(f"  Train batches: {len(train_loader):,}")
        logger.info(f"  Validation batches: {len(val_loader):,}")
        logger.info(f"  Test batches: {len(test_loader):,}")

        # Show sample batch info
        sample_batch = next(iter(train_loader))
        logger.info("")
        logger.info("Sample batch shapes:")
        logger.info(f"  input_ids: {sample_batch['input_ids'].shape}")
        logger.info(f"  attention_mask: {sample_batch['attention_mask'].shape}")
        logger.info(f"  labels: {sample_batch['labels'].shape}")

        logger.info("=" * 70)

        return train_loader, val_loader, test_loader, tokenizer

    except Exception as e:
        logger.error(f"Failed to tokenize and create dataloaders: {e}")
        raise RuntimeError(f"Tokenization failed: {e}") from e
