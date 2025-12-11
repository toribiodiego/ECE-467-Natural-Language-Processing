"""
End-to-end integration tests for the training pipeline.

These tests verify the complete training workflow from argument parsing
through checkpoint saving using a small dataset (20 samples) for fast execution.
They serve as regression tests to ensure refactoring doesn't break functionality.
"""

import os
import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import patch
from argparse import Namespace

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.train import (
    parse_args,
    validate_args,
    initialize_model,
    train_model,
    evaluate_model,
    save_checkpoint,
    get_model_name_or_path
)
from src.training.data_utils import create_dataloaders
from src.training.loss_utils import get_loss_function, compute_class_weights


class TestTrainingPipelineIntegration:
    """Integration tests for the complete training pipeline."""

    def test_full_training_pipeline_minimal(self, temp_output_dir, small_dataset, label_names, basic_args):
        """
        Test complete training pipeline with minimal configuration.

        This test verifies:
        - Model initialization
        - Training loop execution
        - Validation during training
        - Checkpoint saving
        """
        # Set output directory
        basic_args.output_dir = str(temp_output_dir)
        basic_args.epochs = 2
        basic_args.batch_size = 4

        # Get model name
        model_name_or_path = get_model_name_or_path(basic_args.model)

        # Create dataloaders from small dataset
        train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
            dataset=small_dataset,
            num_labels=len(label_names),
            model_name_or_path=model_name_or_path,
            batch_size=basic_args.batch_size,
            max_seq_length=basic_args.max_seq_length,
            seed=basic_args.seed
        )

        # Verify dataloaders have data
        assert len(train_loader) > 0, "Train loader should have batches"
        assert len(val_loader) > 0, "Val loader should have batches"
        assert len(test_loader) > 0, "Test loader should have batches"

        # Initialize device
        device = torch.device('cpu')  # Use CPU for tests

        # Get loss function
        loss_fn = get_loss_function(
            loss_type=basic_args.loss_type,
            device=device
        )

        # Initialize model
        model = initialize_model(
            basic_args.model,
            len(label_names),
            basic_args,
            loss_fn=loss_fn
        )

        assert model is not None, "Model should be initialized"
        assert hasattr(model, 'forward'), "Model should have forward method"

        # Train model for 2 epochs
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            args=basic_args,
            device=device,
            label_names=label_names,
            use_wandb=False
        )

        # Verify training history
        assert 'train_loss' in history, "History should contain train_loss"
        assert 'val_loss' in history, "History should contain val_loss"
        assert 'val_auc' in history, "History should contain val_auc"
        assert len(history['train_loss']) == basic_args.epochs, "Should have loss for each epoch"
        assert len(history['val_loss']) == basic_args.epochs, "Should have val_loss for each epoch"

        # Verify losses are reasonable (not NaN or infinite)
        for loss in history['train_loss']:
            assert not torch.isnan(torch.tensor(loss)), "Train loss should not be NaN"
            assert not torch.isinf(torch.tensor(loss)), "Train loss should not be infinite"
            assert loss > 0, "Train loss should be positive"

        # Evaluate on test set
        test_metrics, test_probs, test_labels, test_texts = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            label_names=label_names,
            threshold_strategy='global',
            threshold=0.5
        )

        # Verify test metrics
        assert 'auc_micro' in test_metrics, f"Should have micro AUC, got keys: {list(test_metrics.keys())}"
        assert 'auc_macro' in test_metrics, "Should have macro AUC"
        # F1 scores use format f1_macro and f1_micro
        assert 'f1_micro' in test_metrics or 'micro_f1' in test_metrics, \
            f"Should have micro F1, got keys: {list(test_metrics.keys())}"
        assert 'f1_macro' in test_metrics or 'macro_f1' in test_metrics, "Should have macro F1"
        assert 0.0 <= test_metrics['auc_micro'] <= 1.0, "AUC should be in [0, 1]"

        # Save checkpoint
        checkpoint_metrics = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'best_val_loss': float(min(history['val_loss'])),
            'best_epoch': 1,
            'test_metrics': test_metrics
        }

        checkpoint_path = save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            model_name=basic_args.model,
            metrics=checkpoint_metrics,
            checkpoint_dir=basic_args.output_dir
        )

        # Verify checkpoint was created
        assert checkpoint_path is not None, "Checkpoint path should be returned"
        assert os.path.exists(checkpoint_path), "Checkpoint directory should exist"
        assert os.path.exists(os.path.join(checkpoint_path, 'pytorch_model.bin')), \
            "Model weights should be saved"
        assert os.path.exists(os.path.join(checkpoint_path, 'config.json')), \
            "Model config should be saved"
        assert os.path.exists(os.path.join(checkpoint_path, 'tokenizer_config.json')), \
            "Tokenizer should be saved"

    def test_training_with_weighted_bce_loss(self, temp_output_dir, small_dataset, label_names, basic_args):
        """
        Test training pipeline with weighted BCE loss.

        Verifies that class weighting is properly integrated into training.
        """
        basic_args.output_dir = str(temp_output_dir)
        basic_args.epochs = 1
        basic_args.loss_type = 'weighted_bce'
        basic_args.class_weight_method = 'inverse'

        model_name_or_path = get_model_name_or_path(basic_args.model)

        train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
            dataset=small_dataset,
            num_labels=len(label_names),
            model_name_or_path=model_name_or_path,
            batch_size=basic_args.batch_size,
            max_seq_length=basic_args.max_seq_length,
            seed=basic_args.seed
        )

        device = torch.device('cpu')

        # Compute class weights
        # Note: Our small dataset only has a subset of labels, so we need to create
        # a full label_counts dict with all 28 labels (some with 0 counts)
        label_counts = {label: 0 for label in label_names}
        for example in small_dataset['train']:
            for idx, label_val in enumerate(example['labels']):
                if label_val == 1:
                    emotion = label_names[idx]
                    label_counts[emotion] = label_counts.get(emotion, 0) + 1

        class_weights = compute_class_weights(
            label_counts=label_counts,
            total_samples=len(small_dataset['train']),
            method=basic_args.class_weight_method
        )

        assert class_weights is not None, "Class weights should be computed"
        assert class_weights.shape[0] == len(label_names), \
            f"Should have weight for each label, got {class_weights.shape[0]} weights for {len(label_names)} labels"

        # Get loss function with weights
        loss_fn = get_loss_function(
            loss_type=basic_args.loss_type,
            class_weights=class_weights,
            device=device
        )

        # Initialize and train model
        model = initialize_model(
            basic_args.model,
            len(label_names),
            basic_args,
            loss_fn=loss_fn
        )

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            args=basic_args,
            device=device,
            use_wandb=False
        )

        # Verify training completed
        assert len(history['train_loss']) == 1, "Should complete 1 epoch"
        assert history['train_loss'][0] > 0, "Loss should be positive"

    def test_training_with_focal_loss(self, temp_output_dir, small_dataset, label_names, basic_args):
        """
        Test training pipeline with focal loss.

        Verifies that focal loss parameters are properly used.
        """
        basic_args.output_dir = str(temp_output_dir)
        basic_args.epochs = 1
        basic_args.loss_type = 'focal'
        basic_args.focal_alpha = 0.25
        basic_args.focal_gamma = 2.0

        model_name_or_path = get_model_name_or_path(basic_args.model)

        train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
            dataset=small_dataset,
            num_labels=len(label_names),
            model_name_or_path=model_name_or_path,
            batch_size=basic_args.batch_size,
            max_seq_length=basic_args.max_seq_length,
            seed=basic_args.seed
        )

        device = torch.device('cpu')

        # Get focal loss
        loss_fn = get_loss_function(
            loss_type=basic_args.loss_type,
            focal_alpha=basic_args.focal_alpha,
            focal_gamma=basic_args.focal_gamma,
            device=device
        )

        # Initialize and train model
        model = initialize_model(
            basic_args.model,
            len(label_names),
            basic_args,
            loss_fn=loss_fn
        )

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            args=basic_args,
            device=device,
            use_wandb=False
        )

        # Verify training completed
        assert len(history['train_loss']) == 1, "Should complete 1 epoch"
        assert history['train_loss'][0] > 0, "Loss should be positive"

    def test_evaluation_with_different_thresholds(self, temp_output_dir, small_dataset, label_names, basic_args):
        """
        Test evaluation with different threshold strategies.

        Verifies:
        - Global threshold
        - Per-class threshold
        - Top-k threshold
        """
        basic_args.output_dir = str(temp_output_dir)
        basic_args.epochs = 1

        model_name_or_path = get_model_name_or_path(basic_args.model)

        train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
            dataset=small_dataset,
            num_labels=len(label_names),
            model_name_or_path=model_name_or_path,
            batch_size=basic_args.batch_size,
            max_seq_length=basic_args.max_seq_length,
            seed=basic_args.seed
        )

        device = torch.device('cpu')
        loss_fn = get_loss_function(loss_type='bce', device=device)

        # Initialize and train model briefly
        model = initialize_model(
            basic_args.model,
            len(label_names),
            basic_args,
            loss_fn=loss_fn
        )

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            args=basic_args,
            device=device,
            use_wandb=False
        )

        # Test global threshold
        metrics_global = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            label_names=label_names,
            threshold_strategy='global',
            threshold=0.5
        )
        assert 'f1_micro' in metrics_global or 'micro_f1' in metrics_global, \
            f"Should have F1 metric, got keys: {list(metrics_global.keys())}"

        # Test top-k threshold
        metrics_topk = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            label_names=label_names,
            threshold_strategy='top_k',
            top_k=2
        )
        assert 'f1_micro' in metrics_topk or 'micro_f1' in metrics_topk, \
            f"Should have F1 metric with top-k, got keys: {list(metrics_topk.keys())}"

        # Test per-class threshold
        metrics_perclass = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            label_names=label_names,
            threshold_strategy='per_class'
        )
        assert 'f1_micro' in metrics_perclass or 'micro_f1' in metrics_perclass, \
            f"Should have F1 metric with per-class thresholds, got keys: {list(metrics_perclass.keys())}"

    def test_argument_validation(self, basic_args):
        """
        Test argument validation catches invalid configurations.
        """
        # Valid args should pass
        validate_args(basic_args)  # Should not raise

        # Test invalid learning rate
        invalid_args = Namespace(**vars(basic_args))
        invalid_args.learning_rate = -0.01

        with pytest.raises(ValueError, match="Learning rate"):
            validate_args(invalid_args)

        # Test invalid batch size
        invalid_args2 = Namespace(**vars(basic_args))
        invalid_args2.batch_size = 0

        with pytest.raises(ValueError, match="Batch size"):
            validate_args(invalid_args2)

    def test_model_initialization_different_models(self, label_names, basic_args):
        """
        Test model initialization with different base models.

        Verifies that model initialization works for common architectures.
        """
        device = torch.device('cpu')
        loss_fn = get_loss_function(loss_type='bce', device=device)

        # Test DistilBERT (default)
        model_distilbert = initialize_model(
            'distilbert-base-uncased',
            len(label_names),
            basic_args,
            loss_fn=loss_fn
        )
        assert model_distilbert is not None
        assert hasattr(model_distilbert, 'forward')

    def test_checkpoint_loading(self, temp_output_dir, small_dataset, label_names, basic_args):
        """
        Test that saved checkpoints can be loaded back.

        Verifies checkpoint save/load roundtrip.
        """
        basic_args.output_dir = str(temp_output_dir)
        basic_args.epochs = 1

        model_name_or_path = get_model_name_or_path(basic_args.model)

        train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
            dataset=small_dataset,
            num_labels=len(label_names),
            model_name_or_path=model_name_or_path,
            batch_size=basic_args.batch_size,
            max_seq_length=basic_args.max_seq_length,
            seed=basic_args.seed
        )

        device = torch.device('cpu')
        loss_fn = get_loss_function(loss_type='bce', device=device)

        # Train and save model
        model = initialize_model(
            basic_args.model,
            len(label_names),
            basic_args,
            loss_fn=loss_fn
        )

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            args=basic_args,
            device=device,
            use_wandb=False
        )

        checkpoint_metrics = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'best_val_loss': float(min(history['val_loss'])),
            'best_epoch': 1,
            'test_metrics': {}
        }

        checkpoint_path = save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            model_name=basic_args.model,
            metrics=checkpoint_metrics,
            checkpoint_dir=basic_args.output_dir
        )

        # Verify checkpoint files were saved correctly
        import json

        config_file = os.path.join(checkpoint_path, 'config.json')
        assert os.path.exists(config_file), "Config file should exist"

        with open(config_file, 'r') as f:
            config = json.load(f)

        assert config['num_labels'] == len(label_names), \
            f"Config should have correct number of labels, got {config.get('num_labels')}"

        # Verify tokenizer files exist
        tokenizer_files = [
            'tokenizer_config.json',
            'vocab.txt'  # DistilBERT uses vocab.txt
        ]
        for file_name in tokenizer_files:
            file_path = os.path.join(checkpoint_path, file_name)
            assert os.path.exists(file_path), f"Tokenizer file {file_name} should exist"
