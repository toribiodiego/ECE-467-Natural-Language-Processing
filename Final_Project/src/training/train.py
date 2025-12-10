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
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.data.load_dataset import get_dataset_statistics
from src.training.data_utils import load_dataset_with_limits, create_dataloaders
from src.training.metrics_utils import evaluate_with_threshold, log_evaluation_results
from src.training.loss_utils import get_loss_function, compute_class_weights, TrainingCostTracker
from src.training.wandb_utils import (
    init_wandb,
    log_training_metrics,
    log_evaluation_metrics,
    log_artifact_checkpoint,
    finish_wandb
)


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

    # Loss configuration
    loss_group = parser.add_argument_group('Loss Configuration')
    loss_group.add_argument(
        '--loss-type',
        type=str,
        default='bce',
        choices=['bce', 'weighted_bce', 'focal'],
        help='Loss function type'
    )
    loss_group.add_argument(
        '--class-weight-method',
        type=str,
        default='inverse',
        choices=['inverse', 'sqrt_inverse', 'effective_samples', 'none'],
        help='Method for computing class weights (used with weighted_bce)'
    )
    loss_group.add_argument(
        '--focal-alpha',
        type=float,
        default=0.25,
        help='Alpha parameter for focal loss'
    )
    loss_group.add_argument(
        '--focal-gamma',
        type=float,
        default=2.0,
        help='Gamma parameter for focal loss'
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
        dropout: float = 0.1,
        loss_fn: Optional[nn.Module] = None
    ):
        """
        Initialize multi-label classification model.

        Args:
            model_name_or_path: HuggingFace model identifier
            num_labels: Number of labels for classification (28 for GoEmotions)
            dropout: Dropout probability for classification head
            loss_fn: Custom loss function (defaults to BCEWithLogitsLoss)
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

        # Loss function
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()

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
            loss = self.loss_fn(logits, labels)

        return {
            'logits': logits,
            'loss': loss
        }


def initialize_model(
    model_name: str,
    num_labels: int,
    args: argparse.Namespace,
    loss_fn: Optional[nn.Module] = None
) -> MultiLabelClassificationModel:
    """
    Initialize pretrained model for multi-label classification.

    Args:
        model_name: CLI model name (e.g., 'roberta-large')
        num_labels: Number of emotion labels
        args: Parsed command line arguments
        loss_fn: Optional custom loss function

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
            dropout=args.dropout,
            loss_fn=loss_fn
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
    device: torch.device,
    use_wandb: bool = False
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

    # Initialize cost tracker
    cost_tracker = TrainingCostTracker()

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
        train_losses = []  # Track individual batch losses for std calculation
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
            train_losses.append(loss.item())
            train_steps += 1

            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })

        # Calculate average training loss and std
        avg_train_loss = train_loss / train_steps
        train_loss_std = np.std(train_losses) if train_losses else 0.0
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

        # Track training costs
        num_samples = len(train_loader.dataset)
        cost_tracker.log_epoch(epoch_time, num_samples, logger_instance=logger)

        # Log metrics to W&B
        samples_per_sec = num_samples / epoch_time if epoch_time > 0 else 0
        log_training_metrics(
            epoch=epoch,
            train_loss=avg_train_loss,
            train_loss_std=train_loss_std,
            val_loss=avg_val_loss,
            val_auc=val_auc,
            learning_rate=current_lr,
            epoch_time=epoch_time,
            samples_per_sec=samples_per_sec,
            use_wandb=use_wandb
        )

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

    # Log cost summary
    cost_stats = cost_tracker.log_summary(logger_instance=logger)
    history['cost_stats'] = cost_stats

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

    logger.info(f"Number of samples: {len(all_labels_np)}")
    logger.info(f"Number of labels: {len(label_names)}")

    # Use metrics utilities for comprehensive evaluation
    results = evaluate_with_threshold(
        y_true=all_labels_np,
        y_probs=all_probs,
        label_names=label_names,
        threshold=0.5,
        compute_pr_curves_flag=False,
        compute_confusion=True
    )

    # Log results using the utility function
    log_evaluation_results(results, logger_instance=logger)

    # Return metrics in backward-compatible format
    metrics = {
        'auc': results.get('auc_micro', 0.0),
        'auc_micro': results.get('auc_micro', 0.0),
        'auc_macro': results.get('auc_macro', 0.0),
        'macro_f1': results.get('f1_macro', 0.0),
        'micro_f1': results.get('f1_micro', 0.0),
        'macro_precision': results.get('precision_macro', 0.0),
        'micro_precision': results.get('precision_micro', 0.0),
        'macro_recall': results.get('recall_macro', 0.0),
        'micro_recall': results.get('recall_micro', 0.0),
        'per_class_f1': results.get('per_class_f1', {}),
        'per_class_precision': results.get('per_class_precision', {}),
        'per_class_recall': results.get('per_class_recall', {}),
        'confusion_summary': results.get('confusion_summary', {})
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
        dataset, label_names, num_labels = load_dataset_with_limits(
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples
        )

        # Tokenize and create dataloaders
        model_name_or_path = get_model_name_or_path(args.model)
        train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
            dataset=dataset,
            num_labels=num_labels,
            model_name_or_path=model_name_or_path,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            seed=args.seed
        )

        # Get device first (needed for loss function)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Configure loss function
        class_weights = None
        if args.loss_type == 'weighted_bce' and args.class_weight_method != 'none':
            # Compute class weights from dataset statistics
            stats = get_dataset_statistics(dataset)
            label_counts = stats['label_distribution']
            total_samples = len(dataset['train'])
            class_weights = compute_class_weights(
                label_counts=label_counts,
                total_samples=total_samples,
                method=args.class_weight_method
            )

        loss_fn = get_loss_function(
            loss_type=args.loss_type,
            class_weights=class_weights,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            device=device
        )

        # Initialize model with configured loss
        model = initialize_model(args.model, num_labels, args, loss_fn=loss_fn)

        # Count model parameters
        model_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {model_params:,}")

        # Initialize W&B if enabled
        wandb_run = init_wandb(
            args=args,
            model_name=args.model,
            num_labels=num_labels,
            train_size=len(dataset['train']),
            val_size=len(dataset['validation']),
            test_size=len(dataset['test']),
            model_params=model_params
        )
        use_wandb = wandb_run is not None

        # Train model
        history = train_model(model, train_loader, val_loader, args, device, use_wandb=use_wandb)

        # Evaluate model on test set
        test_metrics = evaluate_model(model, test_loader, device, label_names)

        # Calculate best epoch and total training time
        best_epoch = int(history['val_auc'].index(max(history['val_auc'])) + 1)
        total_training_time = sum(history['epoch_times'])

        # Log evaluation metrics to W&B
        log_evaluation_metrics(
            test_results=test_metrics,
            label_names=label_names,
            total_training_time=total_training_time,
            best_epoch=best_epoch,
            final_epoch=args.epochs,
            use_wandb=use_wandb
        )

        # Save checkpoint with training history and test metrics
        checkpoint_metrics = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'best_val_loss': float(min(history['val_loss'])),
            'best_epoch': best_epoch,
            'test_metrics': test_metrics
        }
        checkpoint_path = save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            model_name=args.model,
            metrics=checkpoint_metrics,
            checkpoint_dir=args.output_dir
        )

        # Upload checkpoint as W&B artifact
        if use_wandb:
            hyperparameters = {
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'dropout': args.dropout,
                'max_seq_length': args.max_seq_length,
                'loss_type': args.loss_type,
                'weight_decay': args.weight_decay,
                'warmup_steps': args.warmup_steps,
                'seed': args.seed
            }

            log_artifact_checkpoint(
                checkpoint_dir=checkpoint_path,
                model_name=args.model,
                final_test_auc=test_metrics.get('auc_micro', 0.0),
                final_test_f1=test_metrics.get('macro_f1', 0.0),
                best_epoch=best_epoch,
                training_time_hours=total_training_time / 3600.0,
                hyperparameters=hyperparameters,
                random_seed=args.seed,
                use_wandb=use_wandb
            )

        # Final status
        logger.info("")
        logger.info("Training pipeline status:")
        logger.info("  ✓ Data loading")
        logger.info("  ✓ Tokenization and preprocessing")
        logger.info("  ✓ Model initialization")
        logger.info("  ✓ Training loop with optimization")
        logger.info("  ✓ Evaluation metrics (AUC, F1, precision, recall)")
        logger.info("  ✓ Checkpoint saving")
        logger.info(f"  {'✓' if use_wandb else '-'} W&B logging and artifact upload")
        logger.info("")
        logger.info(f"Training complete. Best val AUC: {max(history['val_auc']):.4f} (epoch {best_epoch})")
        logger.info(f"Test metrics: AUC={test_metrics['auc']:.4f}, Macro F1={test_metrics['macro_f1']:.4f}, Micro F1={test_metrics['micro_f1']:.4f}")
        logger.info(f"Checkpoint saved to: {checkpoint_path}")

        # Finish W&B run
        finish_wandb(use_wandb=use_wandb)

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
