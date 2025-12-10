#!/usr/bin/env python3
"""
Training script for GoEmotions emotion classification models.

This script provides a command-line interface for training transformer models
on the GoEmotions dataset with configurable hyperparameters and optional
Weights & Biases integration.

Usage:
    python -m src.training.train --model roberta-large --lr 2e-5 --batch-size 16 --epochs 4
    python -m src.training.train --model distilbert-base --lr 3e-5 --batch-size 32 --no-wandb
"""

import argparse
import logging
import sys
from typing import Dict, Any, Tuple, List
from datasets import DatasetDict

from src.data.load_dataset import load_go_emotions, get_label_names, get_dataset_statistics


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for training configuration.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Train emotion classification models on GoEmotions dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['roberta-large', 'distilbert-base', 'bert-base'],
        help='Model architecture to train'
    )
    model_group.add_argument(
        '--max-seq-length',
        type=int,
        default=128,
        help='Maximum sequence length for tokenization'
    )

    # Training hyperparameters
    training_group = parser.add_argument_group('Training Hyperparameters')
    training_group.add_argument(
        '--lr',
        '--learning-rate',
        type=float,
        default=2e-5,
        dest='learning_rate',
        help='Learning rate for optimizer'
    )
    training_group.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size'
    )
    training_group.add_argument(
        '--epochs',
        type=int,
        default=4,
        help='Number of training epochs'
    )
    training_group.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout probability'
    )
    training_group.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay for optimizer'
    )
    training_group.add_argument(
        '--warmup-steps',
        type=int,
        default=500,
        help='Number of warmup steps for learning rate scheduler'
    )
    training_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    # Testing and debugging
    testing_group = parser.add_argument_group('Testing and Debugging')
    testing_group.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='Maximum epochs for testing (overrides --epochs for quick validation runs)'
    )
    testing_group.add_argument(
        '--max-train-samples',
        type=int,
        default=None,
        help='Maximum training samples for testing (limits dataset size for quick runs)'
    )
    testing_group.add_argument(
        '--max-eval-samples',
        type=int,
        default=None,
        help='Maximum evaluation samples for testing'
    )

    # Weights & Biases configuration
    wandb_group = parser.add_argument_group('Weights & Biases')
    wandb_group.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging (for local testing)'
    )
    wandb_group.add_argument(
        '--wandb-project',
        type=str,
        default='goemotions-emotion-classification',
        help='W&B project name'
    )
    wandb_group.add_argument(
        '--wandb-entity',
        type=str,
        default=None,
        help='W&B entity (username or team name)'
    )
    wandb_group.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='W&B run name (auto-generated if not specified)'
    )

    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/models',
        help='Directory to save model checkpoints'
    )
    output_group.add_argument(
        '--save-steps',
        type=int,
        default=None,
        help='Save checkpoint every N steps (default: save only best model)'
    )

    args = parser.parse_args()

    # Apply max-epochs override if specified
    if args.max_epochs is not None:
        logger.info(f"Overriding --epochs ({args.epochs}) with --max-epochs ({args.max_epochs})")
        args.epochs = args.max_epochs

    return args


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate parsed arguments for consistency and feasibility.

    Args:
        args: Parsed command line arguments

    Raises:
        ValueError: If arguments are invalid or inconsistent
    """
    # Validate learning rate
    if args.learning_rate <= 0 or args.learning_rate > 1:
        raise ValueError(f"Learning rate must be in (0, 1], got {args.learning_rate}")

    # Validate batch size
    if args.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {args.batch_size}")

    # Validate epochs
    if args.epochs <= 0:
        raise ValueError(f"Epochs must be positive, got {args.epochs}")

    # Validate dropout
    if args.dropout < 0 or args.dropout >= 1:
        raise ValueError(f"Dropout must be in [0, 1), got {args.dropout}")

    # Validate max sequence length
    if args.max_seq_length <= 0 or args.max_seq_length > 512:
        raise ValueError(f"Max sequence length must be in (0, 512], got {args.max_seq_length}")

    logger.info("Arguments validated successfully")


def get_model_defaults(model_name: str) -> Dict[str, Any]:
    """
    Get recommended default hyperparameters for specific models.

    Args:
        model_name: Model architecture name

    Returns:
        Dictionary of recommended hyperparameter defaults
    """
    defaults = {
        'roberta-large': {
            'learning_rate': 2e-5,
            'batch_size': 16,
            'epochs': 4,
            'dropout': 0.1,
            'description': 'RoBERTa-Large (355M params) - Best performance model'
        },
        'distilbert-base': {
            'learning_rate': 3e-5,
            'batch_size': 32,
            'epochs': 4,
            'dropout': 0.1,
            'description': 'DistilBERT (66M params) - Best efficiency model'
        },
        'bert-base': {
            'learning_rate': 2e-5,
            'batch_size': 32,
            'epochs': 4,
            'dropout': 0.1,
            'description': 'BERT-base (110M params) - Baseline model'
        }
    }

    return defaults.get(model_name, {})


def print_configuration(args: argparse.Namespace) -> None:
    """
    Print training configuration summary.

    Args:
        args: Parsed command line arguments
    """
    defaults = get_model_defaults(args.model)

    logger.info("=" * 70)
    logger.info("Training Configuration")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    if defaults:
        logger.info(f"  {defaults.get('description', '')}")
    logger.info("")
    logger.info("Hyperparameters:")
    logger.info(f"  Learning rate:    {args.learning_rate}")
    logger.info(f"  Batch size:       {args.batch_size}")
    logger.info(f"  Epochs:           {args.epochs}")
    logger.info(f"  Dropout:          {args.dropout}")
    logger.info(f"  Weight decay:     {args.weight_decay}")
    logger.info(f"  Warmup steps:     {args.warmup_steps}")
    logger.info(f"  Max seq length:   {args.max_seq_length}")
    logger.info(f"  Random seed:      {args.seed}")
    logger.info("")
    logger.info("Output:")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  W&B enabled:      {not args.no_wandb}")
    if not args.no_wandb:
        logger.info(f"  W&B project:      {args.wandb_project}")
        if args.wandb_entity:
            logger.info(f"  W&B entity:       {args.wandb_entity}")
        if args.run_name:
            logger.info(f"  Run name:         {args.run_name}")

    # Show warnings for non-default values
    if defaults:
        if args.learning_rate != defaults.get('learning_rate'):
            logger.warning(f"Learning rate differs from recommended {defaults['learning_rate']}")
        if args.batch_size != defaults.get('batch_size'):
            logger.warning(f"Batch size differs from recommended {defaults['batch_size']}")

    # Show testing mode indicators
    if args.max_epochs is not None:
        logger.warning("Running in TEST MODE with --max-epochs")
    if args.max_train_samples is not None:
        logger.warning(f"Limiting training to {args.max_train_samples} samples (TEST MODE)")
    if args.max_eval_samples is not None:
        logger.warning(f"Limiting evaluation to {args.max_eval_samples} samples (TEST MODE)")

    logger.info("=" * 70)


# ============================================================================
# Data Loading
# ============================================================================

def load_data(args: argparse.Namespace) -> Tuple[DatasetDict, List[str], int]:
    """
    Load GoEmotions dataset with optional sample limiting for testing.

    Args:
        args: Parsed command line arguments

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
        if args.max_train_samples is not None:
            original_size = len(dataset['train'])
            dataset['train'] = dataset['train'].select(range(min(args.max_train_samples, original_size)))
            logger.warning(f"Limited training samples: {original_size:,} → {len(dataset['train']):,}")

        if args.max_eval_samples is not None:
            original_val_size = len(dataset['validation'])
            dataset['validation'] = dataset['validation'].select(range(min(args.max_eval_samples, original_val_size)))
            logger.warning(f"Limited validation samples: {original_val_size:,} → {len(dataset['validation']):,}")

            original_test_size = len(dataset['test'])
            dataset['test'] = dataset['test'].select(range(min(args.max_eval_samples, original_test_size)))
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


# ============================================================================
# Main Function
# ============================================================================

def main() -> None:
    """
    Main entry point for training script.

    This function will be expanded in subsequent tasks to include:
    - Tokenization and preprocessing
    - Model initialization
    - Training loop
    - Evaluation
    - Checkpoint saving
    - W&B logging
    """
    try:
        # Parse and validate arguments
        args = parse_args()
        validate_args(args)

        # Print configuration
        print_configuration(args)

        # Load dataset
        dataset, label_names, num_labels = load_data(args)

        # Placeholder for remaining training implementation
        logger.info("")
        logger.info("Training pipeline will be implemented in subsequent tasks:")
        logger.info("  ✓ Data loading")
        logger.info("  - Tokenization and preprocessing")
        logger.info("  - Model initialization")
        logger.info("  - Training loop with optimization")
        logger.info("  - Evaluation and metrics calculation")
        logger.info("  - Checkpoint saving")
        logger.info("  - W&B logging and artifact upload")
        logger.info("")
        logger.info(f"Dataset ready: {num_labels} labels, {len(label_names)} emotions")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
