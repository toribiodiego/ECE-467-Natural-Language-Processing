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
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import torch
import torch.nn as nn
import numpy as np
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.data.load_dataset import load_go_emotions, get_label_names, get_dataset_statistics


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Model Configuration
# ============================================================================

def get_model_name_or_path(model_name: str) -> str:
    """
    Map CLI model names to HuggingFace model identifiers.

    Args:
        model_name: Model name from CLI (e.g., 'roberta-large')

    Returns:
        HuggingFace model identifier (e.g., 'roberta-large')
    """
    model_mapping = {
        'roberta-large': 'roberta-large',
        'distilbert-base': 'distilbert-base-uncased',
        'bert-base': 'bert-base-uncased'
    }

    return model_mapping.get(model_name, model_name)


# ============================================================================
# Dataset and DataLoader
# ============================================================================

class GoEmotionsDataset(TorchDataset):
    """
    PyTorch Dataset wrapper for GoEmotions with tokenized inputs.

    Converts HuggingFace dataset samples to PyTorch tensors suitable for
    multi-label classification training.
    """

    def __init__(self, encodings: Dict[str, List], labels: List[List[int]], num_labels: int):
        """
        Initialize dataset.

        Args:
            encodings: Dictionary with 'input_ids' and 'attention_mask'
            labels: List of label lists (multi-label format)
            num_labels: Total number of possible labels
        """
        self.encodings = encodings
        self.labels = labels
        self.num_labels = num_labels

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


def tokenize_and_create_dataloaders(
    dataset: DatasetDict,
    num_labels: int,
    args: argparse.Namespace
) -> Tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer]:
    """
    Tokenize dataset and create DataLoaders for training.

    Args:
        dataset: DatasetDict with train/validation/test splits
        num_labels: Number of emotion labels
        args: Parsed command line arguments

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
        # Load tokenizer for the specified model
        model_name_or_path = get_model_name_or_path(args.model)
        logger.info(f"Loading tokenizer for {model_name_or_path}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

        # Tokenize each split
        logger.info(f"Tokenizing texts (max_length={args.max_seq_length})...")

        def tokenize_split(split_data):
            """Helper to tokenize a dataset split."""
            texts = split_data['text']
            labels = split_data['labels']

            # Tokenize texts
            encodings = tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=args.max_seq_length,
                return_tensors=None  # Return lists, not tensors
            )

            return encodings, labels

        # Tokenize all splits
        train_encodings, train_labels = tokenize_split(dataset['train'])
        val_encodings, val_labels = tokenize_split(dataset['validation'])
        test_encodings, test_labels = tokenize_split(dataset['test'])

        logger.info(f"  Train: {len(train_labels):,} samples tokenized")
        logger.info(f"  Validation: {len(val_labels):,} samples tokenized")
        logger.info(f"  Test: {len(test_labels):,} samples tokenized")

        # Create PyTorch datasets
        logger.info("Creating PyTorch datasets...")
        train_dataset = GoEmotionsDataset(train_encodings, train_labels, num_labels)
        val_dataset = GoEmotionsDataset(val_encodings, val_labels, num_labels)
        test_dataset = GoEmotionsDataset(test_encodings, test_labels, num_labels)

        # Create DataLoaders
        logger.info(f"Creating DataLoaders (batch_size={args.batch_size})...")

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
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


# ============================================================================
# Model Initialization
# ============================================================================

class MultiLabelClassificationModel(nn.Module):
    """
    Multi-label classification model with pretrained transformer backbone.

    Adds a classification head on top of pretrained transformer models
    (RoBERTa, BERT, DistilBERT) for multi-label emotion classification.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        dropout: float = 0.1
    ):
        """
        Initialize multi-label classification model.

        Args:
            model_name_or_path: HuggingFace model identifier
            num_labels: Number of labels for classification (28 for GoEmotions)
            dropout: Dropout probability for classification head
        """
        super().__init__()

        # Load pretrained transformer model
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.transformer = AutoModel.from_pretrained(model_name_or_path)

        # Get hidden size from model configuration
        self.hidden_size = self.config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        # Store configuration
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            labels: Multi-label targets [batch_size, num_labels] (optional)

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout and classification layer
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Binary cross-entropy with logits for multi-label classification
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return {
            'logits': logits,
            'loss': loss
        }


def initialize_model(
    model_name: str,
    num_labels: int,
    args: argparse.Namespace
) -> MultiLabelClassificationModel:
    """
    Initialize pretrained model for multi-label classification.

    Args:
        model_name: CLI model name (e.g., 'roberta-large')
        num_labels: Number of emotion labels
        args: Parsed command line arguments

    Returns:
        Initialized model ready for training

    Raises:
        RuntimeError: If model initialization fails
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Model Initialization")
    logger.info("=" * 70)

    try:
        # Get HuggingFace model path
        model_name_or_path = get_model_name_or_path(model_name)
        logger.info(f"Loading pretrained model: {model_name_or_path}")

        # Initialize model
        model = MultiLabelClassificationModel(
            model_name_or_path=model_name_or_path,
            num_labels=num_labels,
            dropout=args.dropout
        )

        logger.info(f"Model loaded: {model.__class__.__name__}")
        logger.info(f"  Backbone: {model_name_or_path}")
        logger.info(f"  Hidden size: {model.hidden_size}")
        logger.info(f"  Number of labels: {model.num_labels}")
        logger.info(f"  Dropout: {args.dropout}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        logger.info(f"  Device: {device}")

        logger.info("=" * 70)

        return model

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise RuntimeError(f"Model initialization failed: {e}") from e


# ============================================================================
# Training Loop
# ============================================================================

def train_model(
    model: MultiLabelClassificationModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Train the model for the specified number of epochs.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        args: Parsed command line arguments
        device: Device to train on (cuda or cpu)

    Returns:
        Dictionary with training history:
        {
            'train_loss': [...],
            'val_loss': [...],
            'learning_rates': [...]
        }
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Training")
    logger.info("=" * 70)

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Calculate total training steps
    total_steps = len(train_loader) * args.epochs

    # Initialize learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    logger.info(f"Optimizer: Adam (lr={args.learning_rate}, weight_decay={args.weight_decay})")
    logger.info(f"Scheduler: Linear with warmup (warmup_steps={args.warmup_steps})")
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Steps per epoch: {len(train_loader)}")
    logger.info("")

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'learning_rates': [],
        'epoch_times': []
    }

    # Track training start time
    training_start_time = time.time()

    # Training loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()

        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info("-" * 70)

        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()
            scheduler.step()

            # Track metrics
            train_loss += loss.item()
            train_steps += 1

            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })

        # Calculate average training loss
        avg_train_loss = train_loss / train_steps
        current_lr = scheduler.get_last_lr()[0]

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        val_logits = []
        val_labels = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")

            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                outputs = model(**batch)
                loss = outputs['loss']
                logits = outputs['logits']

                # Track metrics
                val_loss += loss.item()
                val_steps += 1

                # Collect logits and labels for AUC calculation
                val_logits.append(logits.cpu())
                val_labels.append(batch['labels'].cpu())

                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate average validation loss
        avg_val_loss = val_loss / val_steps

        # Compute validation AUC
        val_logits_all = torch.cat(val_logits, dim=0).numpy()
        val_labels_all = torch.cat(val_labels, dim=0).numpy()
        val_probs = 1 / (1 + np.exp(-val_logits_all))

        try:
            val_auc = roc_auc_score(val_labels_all, val_probs, average='micro')
        except ValueError:
            val_auc = 0.0

        # Calculate epoch time and estimate remaining time
        epoch_time = time.time() - epoch_start_time
        avg_epoch_time = (time.time() - training_start_time) / (epoch + 1)
        remaining_epochs = args.epochs - (epoch + 1)
        estimated_time_remaining = avg_epoch_time * remaining_epochs

        # Format time remaining
        if estimated_time_remaining > 0:
            hours = int(estimated_time_remaining // 3600)
            minutes = int((estimated_time_remaining % 3600) // 60)
            seconds = int(estimated_time_remaining % 60)
            time_remaining_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            time_remaining_str = "00:00:00"

        # Log epoch summary
        logger.info(f"  Train Loss:     {avg_train_loss:.4f}")
        logger.info(f"  Val Loss:       {avg_val_loss:.4f}")
        logger.info(f"  Val AUC:        {val_auc:.4f}")
        logger.info(f"  Learning Rate:  {current_lr:.2e}")
        logger.info(f"  Epoch Time:     {epoch_time:.1f}s")
        if remaining_epochs > 0:
            logger.info(f"  Time Remaining: ~{time_remaining_str}")
        logger.info("")

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append(epoch_time)

    # Calculate total training time
    total_training_time = time.time() - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)

    logger.info("=" * 70)
    logger.info("Training Complete")
    logger.info("=" * 70)
    logger.info(f"Total Time:       {hours:02d}:{minutes:02d}:{seconds:02d}")
    logger.info(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final Val Loss:   {history['val_loss'][-1]:.4f}")
    logger.info(f"Final Val AUC:    {history['val_auc'][-1]:.4f}")
    logger.info(f"Best Val Loss:    {min(history['val_loss']):.4f} (epoch {history['val_loss'].index(min(history['val_loss'])) + 1})")
    logger.info(f"Best Val AUC:     {max(history['val_auc']):.4f} (epoch {history['val_auc'].index(max(history['val_auc'])) + 1})")
    logger.info("=" * 70)

    return history


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(
    model: MultiLabelClassificationModel,
    dataloader: DataLoader,
    device: torch.device,
    label_names: List[str]
) -> Dict[str, Any]:
    """
    Evaluate the model and compute comprehensive metrics.

    Computes:
    - AUC score (micro-averaged across all labels)
    - Macro and micro F1, precision, recall
    - Per-class F1 scores for all emotion labels

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on (cuda or cpu)
        label_names: List of label names for per-class metrics

    Returns:
        Dictionary containing:
        {
            'auc': float,
            'macro_f1': float,
            'micro_f1': float,
            'macro_precision': float,
            'micro_precision': float,
            'macro_recall': float,
            'micro_recall': float,
            'per_class_f1': Dict[str, float]
        }
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Evaluation")
    logger.info("=" * 70)

    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            logits = outputs['logits']

            # Collect predictions and labels
            all_logits.append(logits.cpu())
            all_labels.append(batch['labels'].cpu())

    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Convert to numpy
    all_logits_np = all_logits.numpy()
    all_labels_np = all_labels.numpy()

    # Apply sigmoid to get probabilities
    all_probs = 1 / (1 + np.exp(-all_logits_np))

    # Compute binary predictions (threshold = 0.5)
    all_preds = (all_probs >= 0.5).astype(int)

    logger.info(f"Number of samples: {len(all_labels_np)}")
    logger.info(f"Number of labels: {len(label_names)}")
    logger.info("")

    # Compute AUC score (micro-averaged)
    try:
        auc = roc_auc_score(all_labels_np, all_probs, average='micro')
        logger.info(f"AUC (micro): {auc:.4f}")
    except ValueError as e:
        logger.warning(f"Could not compute AUC: {e}")
        auc = 0.0

    # Compute macro and micro averaged metrics
    macro_f1 = f1_score(all_labels_np, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels_np, all_preds, average='micro', zero_division=0)
    macro_precision = precision_score(all_labels_np, all_preds, average='macro', zero_division=0)
    micro_precision = precision_score(all_labels_np, all_preds, average='micro', zero_division=0)
    macro_recall = recall_score(all_labels_np, all_preds, average='macro', zero_division=0)
    micro_recall = recall_score(all_labels_np, all_preds, average='micro', zero_division=0)

    logger.info("")
    logger.info("Aggregate Metrics:")
    logger.info(f"  Macro F1:        {macro_f1:.4f}")
    logger.info(f"  Micro F1:        {micro_f1:.4f}")
    logger.info(f"  Macro Precision: {macro_precision:.4f}")
    logger.info(f"  Micro Precision: {micro_precision:.4f}")
    logger.info(f"  Macro Recall:    {macro_recall:.4f}")
    logger.info(f"  Micro Recall:    {micro_recall:.4f}")

    # Compute per-class F1 scores
    per_class_f1_scores = f1_score(all_labels_np, all_preds, average=None, zero_division=0)
    per_class_f1 = {label_names[i]: float(per_class_f1_scores[i]) for i in range(len(label_names))}

    logger.info("")
    logger.info("Per-Class F1 Scores:")
    for label, score in sorted(per_class_f1.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {label:20s}: {score:.4f}")

    logger.info("=" * 70)

    # Return metrics dictionary
    metrics = {
        'auc': float(auc),
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'macro_precision': float(macro_precision),
        'micro_precision': float(micro_precision),
        'macro_recall': float(macro_recall),
        'micro_recall': float(micro_recall),
        'per_class_f1': per_class_f1
    }

    return metrics


# ============================================================================
# Checkpoint Saving
# ============================================================================

def save_checkpoint(
    model: MultiLabelClassificationModel,
    tokenizer: AutoTokenizer,
    model_name: str,
    metrics: Optional[Dict[str, Any]] = None,
    checkpoint_dir: str = "artifacts/models"
) -> str:
    """
    Save model checkpoint with config, weights, and tokenizer.

    Creates a timestamped directory under artifacts/models/ and saves:
    - Model weights (pytorch_model.bin)
    - Model config (config.json)
    - Tokenizer files
    - Training metrics if provided (metrics.json)

    Args:
        model: Trained model to save
        tokenizer: Tokenizer used for training
        model_name: Base model name (e.g., 'roberta-large', 'distilbert-base')
        metrics: Optional dictionary of training/evaluation metrics
        checkpoint_dir: Base directory for checkpoints (default: artifacts/models)

    Returns:
        Path to the saved checkpoint directory
    """
    # Create timestamped checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = Path(checkpoint_dir) / f"{model_name}-{timestamp}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Saving Checkpoint")
    logger.info("=" * 70)
    logger.info(f"Checkpoint directory: {checkpoint_path}")

    # Save model weights
    model_file = checkpoint_path / "pytorch_model.bin"
    torch.save(model.state_dict(), model_file)
    logger.info(f"  ✓ Saved model weights: {model_file.name}")

    # Save model config
    config_dict = {
        'model_type': model_name,
        'hidden_size': model.config.hidden_size,
        'num_labels': model.classifier.out_features,
        'dropout': model.dropout.p,
        'transformer_config': model.config.to_dict()
    }
    config_file = checkpoint_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"  ✓ Saved model config: {config_file.name}")

    # Save tokenizer
    tokenizer.save_pretrained(checkpoint_path)
    logger.info(f"  ✓ Saved tokenizer files")

    # Save metrics if provided
    if metrics is not None:
        metrics_file = checkpoint_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"  ✓ Saved metrics: {metrics_file.name}")

    logger.info("")
    logger.info(f"Checkpoint saved successfully to: {checkpoint_path}")
    logger.info("=" * 70)

    return str(checkpoint_path)


# ============================================================================
# Main Function
# ============================================================================

def main() -> None:
    """
    Main entry point for training script.

    This function will be expanded in subsequent tasks to include:
    - Evaluation metrics (AUC, F1, precision, recall)
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

        # Tokenize and create dataloaders
        train_loader, val_loader, test_loader, tokenizer = tokenize_and_create_dataloaders(
            dataset, num_labels, args
        )

        # Initialize model
        model = initialize_model(args.model, num_labels, args)

        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Train model
        history = train_model(model, train_loader, val_loader, args, device)

        # Evaluate model on test set
        test_metrics = evaluate_model(model, test_loader, device, label_names)

        # Save checkpoint with training history and test metrics
        checkpoint_metrics = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'best_val_loss': float(min(history['val_loss'])),
            'best_epoch': int(history['val_loss'].index(min(history['val_loss'])) + 1),
            'test_metrics': test_metrics
        }
        checkpoint_path = save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            metrics=checkpoint_metrics
        )

        # Placeholder for remaining implementation
        logger.info("")
        logger.info("Training pipeline status:")
        logger.info("  ✓ Data loading")
        logger.info("  ✓ Tokenization and preprocessing")
        logger.info("  ✓ Model initialization")
        logger.info("  ✓ Training loop with optimization")
        logger.info("  ✓ Evaluation metrics (AUC, F1, precision, recall)")
        logger.info("  ✓ Checkpoint saving")
        logger.info("  - W&B logging and artifact upload")
        logger.info("")
        logger.info(f"Training complete. Best val loss: {min(history['val_loss']):.4f}")
        logger.info(f"Test metrics: AUC={test_metrics['auc']:.4f}, Macro F1={test_metrics['macro_f1']:.4f}, Micro F1={test_metrics['micro_f1']:.4f}")
        logger.info(f"Checkpoint saved to: {checkpoint_path}")

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
