"""
Weights & Biases logging utilities for GoEmotions training.

This module provides reusable functions for W&B initialization, metric logging,
and artifact management following the specification in docs/w_and_b_guide.md.
"""

import logging
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import transformers

logger = logging.getLogger(__name__)

# Optional W&B import - gracefully handle if not installed
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed - W&B logging will be disabled")


def init_wandb(
    args: Any,
    model_name: str,
    num_labels: int,
    train_size: int,
    val_size: int,
    test_size: int,
    model_params: Optional[int] = None
) -> Optional[Any]:
    """
    Initialize Weights & Biases run with config and system info.

    Args:
        args: Argparse namespace with training configuration
        model_name: Name of the model architecture
        num_labels: Number of emotion labels
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        model_params: Number of model parameters (optional)

    Returns:
        wandb run object if enabled, None otherwise
    """
    if args.no_wandb or not WANDB_AVAILABLE:
        logger.info("W&B logging disabled")
        return None

    # Generate run name if not provided
    run_name = args.run_name
    if run_name is None:
        # Simplify model name: distilbert-base -> distilbert, keep size variants
        display_model = model_name.replace('-base', '') if 'distilbert' in model_name.lower() else model_name
        # Use MM-DD-YYYY format for clarity
        timestamp = datetime.now().strftime("%m-%d-%Y-%H%M%S")
        run_name = f"{display_model}-{timestamp}"

    # Initialize W&B
    logger.info(f"Initializing W&B run: {run_name}")
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            # Model Configuration
            'model_name': model_name,
            'model_params': model_params,
            'num_labels': num_labels,

            # Training Hyperparameters
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'dropout': args.dropout,
            'max_seq_length': args.max_seq_length,
            'warmup_steps': args.warmup_steps,
            'weight_decay': args.weight_decay,
            'random_seed': args.seed,

            # Loss Configuration
            'loss_type': args.loss_type,
            'class_weight_method': args.class_weight_method if args.loss_type == 'weighted_bce' else None,
            'focal_alpha': args.focal_alpha if args.loss_type == 'focal' else None,
            'focal_gamma': args.focal_gamma if args.loss_type == 'focal' else None,

            # Dataset Configuration
            'dataset_name': 'go_emotions',
            'train_samples': train_size,
            'val_samples': val_size,
            'test_samples': test_size,
        }
    )

    # Log system information
    system_info = get_system_info()
    wandb.config.update(system_info)

    logger.info(f"W&B run initialized: {wandb.run.url}")
    return run


def get_system_info() -> Dict[str, Any]:
    """
    Gather system and environment information for logging.

    Returns:
        Dictionary with system details
    """
    info = {
        'system/platform': platform.platform(),
        'system/python_version': platform.python_version(),
        'system/pytorch_version': torch.__version__,
        'system/transformers_version': transformers.__version__,
        'system/cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info.update({
            'system/cuda_version': torch.version.cuda,
            'system/gpu_count': torch.cuda.device_count(),
            'system/gpu_name': torch.cuda.get_device_name(0)
        })
    else:
        info.update({
            'system/cuda_version': None,
            'system/gpu_count': 0,
            'system/gpu_name': None
        })

    return info


def log_training_metrics(
    epoch: int,
    train_loss: float,
    train_loss_std: float,
    val_loss: float,
    val_auc: float,
    learning_rate: float,
    epoch_time: float,
    samples_per_sec: float,
    val_f1_micro: Optional[float] = None,
    val_f1_macro: Optional[float] = None,
    grad_norm: Optional[float] = None,
    train_auc: Optional[float] = None,
    use_wandb: bool = True
) -> None:
    """
    Log training metrics for a single epoch to W&B.

    Args:
        epoch: Current epoch number
        train_loss: Average training loss
        train_loss_std: Standard deviation of training loss
        val_loss: Validation loss
        val_auc: Validation AUC score
        learning_rate: Current learning rate
        epoch_time: Time taken for epoch in seconds
        samples_per_sec: Training throughput
        val_f1_micro: Validation F1 micro score (optional)
        val_f1_macro: Validation F1 macro score (optional)
        grad_norm: Gradient norm (optional)
        train_auc: Training AUC score (optional, expensive to compute)
        use_wandb: Whether to log to W&B
    """
    if not use_wandb or not WANDB_AVAILABLE or wandb.run is None:
        return

    # Use alphabetically ordered prefixes: A_train, B_val, C_test
    metrics = {
        'A_train/loss': train_loss,
        'A_train/loss_std': train_loss_std,
        'A_train/learning_rate': learning_rate,
        'A_train/epoch_time': epoch_time,
        'A_train/samples_per_sec': samples_per_sec,
        'B_val/loss': val_loss,
        'B_val/auc': val_auc,
    }

    if val_f1_micro is not None:
        metrics['B_val/f1_micro'] = val_f1_micro

    if val_f1_macro is not None:
        metrics['B_val/f1_macro'] = val_f1_macro

    if grad_norm is not None:
        metrics['A_train/grad_norm'] = grad_norm

    if train_auc is not None:
        metrics['A_train/auc'] = train_auc

    wandb.log(metrics, step=epoch)


def log_evaluation_metrics(
    test_results: Dict[str, Any],
    label_names: List[str],
    total_training_time: float,
    best_epoch: int,
    final_epoch: int,
    use_wandb: bool = True
) -> None:
    """
    Log final evaluation metrics to W&B.

    High-level metrics (AUC, macro F1/precision/recall) are logged as charts for
    visual comparison across runs. Per-class metrics and metadata are saved to
    run summary for table-based filtering without creating charts.

    Args:
        test_results: Dictionary with evaluation results from evaluate_with_threshold()
        label_names: List of emotion label names
        total_training_time: Total training time in seconds
        best_epoch: Epoch with best validation performance
        final_epoch: Final epoch number
        use_wandb: Whether to log to W&B
    """
    if not use_wandb or not WANDB_AVAILABLE or wandb.run is None:
        return

    # High-level test metrics - log as charts for visual comparison
    # Use C_test prefix to ensure test metrics appear after A_train and B_val
    chart_metrics = {
        'C_test/auc_micro': test_results.get('auc_micro', 0.0),
        'C_test/auc_macro': test_results.get('auc_macro', 0.0),
        'C_test/f1_macro': test_results.get('f1_macro', 0.0),
        'C_test/f1_micro': test_results.get('f1_micro', 0.0),
        'C_test/precision_macro': test_results.get('precision_macro', 0.0),
        'C_test/recall_macro': test_results.get('recall_macro', 0.0),
    }
    wandb.log(chart_metrics)

    # Metadata and per-class metrics - save to summary (table view, no charts)
    summary_metrics = {
        'best_epoch': best_epoch,
        'final_epoch': final_epoch,
        'total_training_time': total_training_time,
    }

    # Per-class metrics - accessible in runs table for filtering/sorting
    if 'per_class_f1' in test_results:
        for emotion in label_names:
            summary_metrics[f'test/f1_{emotion}'] = test_results['per_class_f1'].get(emotion, 0.0)
            summary_metrics[f'test/precision_{emotion}'] = test_results['per_class_precision'].get(emotion, 0.0)
            summary_metrics[f'test/recall_{emotion}'] = test_results['per_class_recall'].get(emotion, 0.0)

    # Support values - dataset properties, useful for filtering runs
    if 'confusion_summary' in test_results:
        for emotion in label_names:
            conf = test_results['confusion_summary'].get(emotion, {})
            summary_metrics[f'test/support_{emotion}'] = conf.get('support', 0)

    wandb.summary.update(summary_metrics)
    logger.info(f"Logged {len(chart_metrics)} test metrics to charts, {len(summary_metrics)} to summary")


def log_artifact_checkpoint(
    checkpoint_dir: str,
    model_name: str,
    final_test_auc: float,
    final_test_f1: float,
    best_epoch: int,
    training_time_hours: float,
    hyperparameters: Dict[str, Any],
    random_seed: int,
    use_wandb: bool = True
) -> None:
    """
    Upload model checkpoint as W&B artifact.

    Args:
        checkpoint_dir: Path to checkpoint directory
        model_name: Name of the model
        final_test_auc: Final test AUC score
        final_test_f1: Final test F1 score
        best_epoch: Best epoch number
        training_time_hours: Training time in hours
        hyperparameters: Dictionary of hyperparameters
        random_seed: Random seed used
        use_wandb: Whether to log to W&B
    """
    if not use_wandb or not WANDB_AVAILABLE or wandb.run is None:
        return

    logger.info(f"Uploading checkpoint artifact from {checkpoint_dir}")

    artifact = wandb.Artifact(
        name=f'{model_name}-checkpoint-{wandb.run.id}',
        type='model',
        description=f'Best {model_name} checkpoint based on validation AUC',
        metadata={
            'model_name': model_name,
            'final_test_auc': final_test_auc,
            'final_test_f1': final_test_f1,
            'best_epoch': best_epoch,
            'training_time_hours': training_time_hours,
            'hyperparameters': hyperparameters,
            'random_seed': random_seed
        }
    )

    artifact.add_dir(checkpoint_dir)
    wandb.log_artifact(artifact)
    logger.info(f"Checkpoint artifact uploaded: {artifact.name}")


def log_artifact_predictions(
    predictions_path: str,
    artifact_type: str,
    model_name: str,
    split_name: str,
    use_wandb: bool = True
) -> None:
    """
    Upload predictions CSV as W&B artifact.

    Args:
        predictions_path: Path to predictions CSV file
        artifact_type: Type of artifact ('val-predictions' or 'test-predictions')
        model_name: Name of the model
        split_name: Name of the split ('validation' or 'test')
        use_wandb: Whether to log to W&B
    """
    if not use_wandb or not WANDB_AVAILABLE or wandb.run is None:
        return

    if not os.path.exists(predictions_path):
        logger.warning(f"Predictions file not found: {predictions_path}")
        return

    logger.info(f"Uploading {split_name} predictions artifact")

    artifact = wandb.Artifact(
        name=f'{model_name}-{artifact_type}-{wandb.run.id}',
        type='dataset',
        description=f'{split_name.capitalize()} set predictions with probabilities'
    )

    artifact.add_file(predictions_path)
    wandb.log_artifact(artifact)
    logger.info(f"Predictions artifact uploaded: {artifact.name}")


def log_artifact_metrics(
    metrics_path: str,
    model_name: str,
    use_wandb: bool = True
) -> None:
    """
    Upload per-class metrics CSV as W&B artifact.

    Args:
        metrics_path: Path to per-class metrics CSV file
        model_name: Name of the model
        use_wandb: Whether to log to W&B
    """
    if not use_wandb or not WANDB_AVAILABLE or wandb.run is None:
        return

    if not os.path.exists(metrics_path):
        logger.warning(f"Metrics file not found: {metrics_path}")
        return

    logger.info(f"Uploading per-class metrics artifact")

    artifact = wandb.Artifact(
        name=f'{model_name}-per-class-metrics-{wandb.run.id}',
        type='dataset',
        description='Per-class performance metrics'
    )

    artifact.add_file(metrics_path)
    wandb.log_artifact(artifact)
    logger.info(f"Metrics artifact uploaded: {artifact.name}")


def finish_wandb(use_wandb: bool = True) -> None:
    """
    Finish W&B run and sync any pending data.

    Args:
        use_wandb: Whether W&B is enabled
    """
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        logger.info("Finishing W&B run")
        wandb.finish()
