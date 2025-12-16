#!/usr/bin/env python3
"""
Export per-class metrics from a saved model checkpoint.

This script loads a trained model checkpoint and computes per-class (per-emotion)
precision, recall, and F1 scores on the test set, exporting them to CSV. Useful
for analyzing which emotions the model handles well vs poorly without needing
to first export full predictions.

Usage:
    python -m src.analysis.export_per_class_metrics \
        --checkpoint artifacts/models/roberta/roberta-large-20251212-211010 \
        --output artifacts/stats/per_class_metrics.csv

    python -m src.analysis.export_per_class_metrics \
        --checkpoint artifacts/models/distilbert/distilbert-base-20251212-225748 \
        --output artifacts/stats/per_class_metrics.csv \
        --threshold 0.3
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support

from src.data.load_dataset import load_go_emotions, get_label_names
from src.training.train import MultiLabelClassificationModel
from src.training.data_utils import GoEmotionsDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint(
    checkpoint_dir: Path,
    device: torch.device
) -> tuple[MultiLabelClassificationModel, AutoTokenizer, Dict[str, Any]]:
    """
    Load model checkpoint, tokenizer, and metadata.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer, metrics_dict)
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    logger.info(f"Loading checkpoint from: {checkpoint_dir}")

    # Load metrics to get model configuration
    metrics_file = checkpoint_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        logger.info(f"Loaded metrics: {list(metrics.keys())}")
    else:
        logger.warning("No metrics.json found in checkpoint")
        metrics = {}

    # Load config to determine model architecture
    config_file = checkpoint_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"No config.json found in {checkpoint_dir}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Get original model name from config
    original_model_name = config.get('_name_or_path', None)
    if original_model_name is None:
        transformer_config = config.get('transformer_config', {})
        original_model_name = transformer_config.get('_name_or_path', 'roberta-large')

    logger.info(f"Original model: {original_model_name}")

    # Load tokenizer from original model
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    logger.info(f"Loaded tokenizer from {original_model_name}")

    # Initialize model with same architecture
    num_labels = 28  # GoEmotions has 28 emotion labels
    dropout = config.get('dropout', 0.1)

    model = MultiLabelClassificationModel(
        model_name_or_path=original_model_name,
        num_labels=num_labels,
        dropout=dropout
    )

    # Load trained weights from checkpoint
    model_file = checkpoint_dir / "pytorch_model.bin"
    if not model_file.exists():
        raise FileNotFoundError(f"No pytorch_model.bin found in {checkpoint_dir}")

    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    logger.info(f"✓ Model loaded successfully")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer, metrics


def generate_predictions(
    model: MultiLabelClassificationModel,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions for a dataset.

    Args:
        model: Trained model
        dataloader: DataLoader for dataset
        device: Device to run inference on
        threshold: Classification threshold

    Returns:
        Tuple of (predicted_binary, true_labels)
    """
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Generating predictions")

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

    # Apply threshold to get binary predictions
    all_preds = (all_probs >= threshold).astype(int)

    return all_preds, all_labels_np


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str]
) -> pd.DataFrame:
    """
    Compute precision, recall, and F1 for each class (emotion).

    Args:
        y_true: Binary matrix of true labels (n_samples, n_classes)
        y_pred: Binary matrix of predicted labels (n_samples, n_classes)
        label_names: List of label names

    Returns:
        DataFrame with per-class metrics
    """
    logger.info("Computing per-class metrics...")

    # Compute precision, recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Create results dataframe
    results = []
    for i, label in enumerate(label_names):
        results.append({
            'emotion': label,
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': int(support[i])
        })

    df_results = pd.DataFrame(results)

    # Sort by F1 score descending
    df_results = df_results.sort_values('f1', ascending=False).reset_index(drop=True)

    # Add rank column
    df_results.insert(0, 'rank', range(1, len(df_results) + 1))

    logger.info("✓ Metrics computed successfully")
    return df_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Export per-class metrics from a saved model checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint directory (e.g., artifacts/models/roberta-large-20251212-211010)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV file for per-class metrics'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold for predictions'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=128,
        help='Maximum sequence length for tokenization'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'validation', 'test'],
        help='Dataset split to evaluate on'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    model, tokenizer, metrics = load_checkpoint(checkpoint_path, device)

    # Load dataset
    logger.info(f"Loading GoEmotions {args.split} set...")
    dataset = load_go_emotions()
    label_names = get_label_names(dataset)
    logger.info(f"Labels: {len(label_names)} emotions")

    # Create dataloader
    split_data = dataset[args.split]
    split_dataset = GoEmotionsDataset(
        texts=split_data['text'],
        labels=split_data['labels'],
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    split_loader = DataLoader(
        split_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Evaluating on {args.split} set")
    logger.info("=" * 70)
    logger.info(f"Dataset size: {len(split_dataset)} samples")
    logger.info(f"Threshold: {args.threshold}")

    # Generate predictions and compute metrics
    y_pred, y_true = generate_predictions(
        model, split_loader, device, threshold=args.threshold
    )

    logger.info(f"Generated predictions: {y_pred.shape}")
    logger.info(f"True positives (total): {y_true.sum()}")
    logger.info(f"Predicted positives (total): {y_pred.sum()}")

    # Compute per-class metrics
    df_metrics = compute_per_class_metrics(y_true, y_pred, label_names)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(output_path, index=False)

    # Display summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("PER-CLASS METRICS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\nTop 5 Emotions (by F1 score):")
    logger.info("\n" + df_metrics.head(5).to_string(index=False))
    logger.info(f"\nBottom 5 Emotions (by F1 score):")
    logger.info("\n" + df_metrics.tail(5).to_string(index=False))
    logger.info("")
    logger.info(f"✓ Results saved to: {output_path}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
