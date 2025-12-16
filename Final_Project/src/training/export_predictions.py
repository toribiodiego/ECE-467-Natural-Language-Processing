#!/usr/bin/env python3
"""
Export predictions from a saved model checkpoint.

This script loads a trained model checkpoint and regenerates validation and test
predictions. Useful when predictions were not saved during training or need to be
regenerated with different thresholds.

Usage:
    python -m src.training.export_predictions \
        --checkpoint artifacts/models/roberta-large-20251212-211010 \
        --output artifacts/predictions

    python -m src.training.export_predictions \
        --checkpoint artifacts/models/distilbert-base-20251212-225748 \
        --output artifacts/predictions \
        --threshold 0.3
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from transformers import AutoTokenizer
from datetime import datetime

from src.data.load_dataset import load_go_emotions, get_label_names
from src.training.train import MultiLabelClassificationModel, save_predictions_to_csv
from src.training.data_utils import create_dataloaders

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

    # Determine model type from config
    model_type = config.get('model_type', 'roberta')

    # Get original model name from config
    original_model_name = config.get('_name_or_path', None)
    if original_model_name is None:
        # Fallback to transformer_config if present
        transformer_config = config.get('transformer_config', {})
        original_model_name = transformer_config.get('_name_or_path', 'roberta-large')

    logger.info(f"Model type: {model_type}")
    logger.info(f"Original model: {original_model_name}")

    # Load tokenizer from original model (not checkpoint, to avoid tokenizer.json corruption)
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    logger.info(f"Loaded tokenizer from {original_model_name}")

    # Initialize model with same architecture from original model
    # GoEmotions has 28 emotion labels
    num_labels = 28
    dropout = config.get('dropout', 0.1)

    model = MultiLabelClassificationModel(
        model_name_or_path=original_model_name,  # Use original model for architecture
        num_labels=num_labels,
        dropout=dropout
    )

    # Load trained weights from checkpoint
    model_file = checkpoint_dir / "pytorch_model.bin"
    if not model_file.exists():
        raise FileNotFoundError(f"No pytorch_model.bin found in {checkpoint_dir}")

    state_dict = torch.load(model_file, map_location=device)
    # Load with strict=False to handle missing buffers like position_ids
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    logger.info(f"✓ Model loaded successfully")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer, metrics


def generate_predictions(
    model: MultiLabelClassificationModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate predictions for a dataset.

    Args:
        model: Trained model
        dataloader: DataLoader for dataset
        device: Device to run inference on

    Returns:
        Tuple of (predicted_probs, true_labels, texts)
    """
    from tqdm import tqdm

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

    # Extract texts from dataset
    all_texts = None
    if hasattr(dataloader.dataset, 'texts') and dataloader.dataset.texts is not None:
        all_texts = list(dataloader.dataset.texts)
    else:
        logger.warning("No texts found in dataset, using placeholder")
        all_texts = [f"sample_{i}" for i in range(len(all_labels_np))]

    return all_probs, all_labels_np, all_texts


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Export predictions from a saved model checkpoint',
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
        default='artifacts/predictions',
        help='Output directory for prediction CSVs'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for converting probabilities to binary predictions'
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
        '--splits',
        nargs='+',
        default=['validation', 'test'],
        choices=['train', 'validation', 'test'],
        help='Which dataset splits to export predictions for'
    )

    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name for output filenames (auto-detected from checkpoint if not provided)'
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

    # Determine model name from checkpoint path if not provided
    if args.model_name:
        model_name = args.model_name
    else:
        # Extract model name from checkpoint directory
        # e.g., "roberta-large-20251212-211010" -> "roberta-large"
        checkpoint_dir_name = checkpoint_path.name
        parts = checkpoint_dir_name.rsplit('-', 2)  # Split off timestamp
        model_name = parts[0] if len(parts) > 1 else checkpoint_dir_name

    logger.info(f"Model name: {model_name}")

    # Load dataset
    logger.info("Loading GoEmotions dataset...")
    dataset = load_go_emotions()
    label_names = get_label_names(dataset)
    logger.info(f"Labels: {len(label_names)} emotions")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split_name in args.splits:
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Processing {split_name} split")
        logger.info("=" * 70)

        # Create dataloader
        from src.training.data_utils import GoEmotionsDataset
        from torch.utils.data import DataLoader

        split_data = dataset[split_name]
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

        logger.info(f"Dataset size: {len(split_dataset)} samples")

        # Generate predictions
        pred_probs, true_labels, texts = generate_predictions(
            model, split_loader, device
        )

        logger.info(f"Generated predictions: {pred_probs.shape}")

        # Save predictions to CSV
        # Map validation -> val for consistency
        csv_split_name = 'val' if split_name == 'validation' else split_name

        predictions_path = save_predictions_to_csv(
            predictions_dir=output_dir,
            model_name=model_name,
            split_name=csv_split_name,
            texts=texts,
            true_labels=true_labels,
            pred_probs=pred_probs,
            label_names=label_names,
            threshold=args.threshold
        )

        logger.info(f"✓ Saved predictions to: {predictions_path}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Export complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
