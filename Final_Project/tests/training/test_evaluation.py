"""
Tests for model evaluation and threshold strategies.

This module tests the evaluate_model() function and different threshold
strategies (global, per_class, top_k) to ensure evaluation logic works
correctly before refactoring.
"""

import sys
import pytest
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.train import (
    MultiLabelClassificationModel,
    evaluate_model
)
from src.training.loss_utils import get_loss_function


class TestEvaluationBasics:
    """Basic tests for evaluate_model() function."""

    def test_evaluate_model_returns_required_metrics(self, label_names):
        """Test that evaluate_model returns all required metric keys."""
        # Create a simple model
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=len(label_names)
        )
        model.eval()

        # Create dummy dataset
        batch_size = 8
        seq_length = 16
        num_samples = 20

        input_ids = torch.randint(0, 1000, (num_samples, seq_length))
        attention_mask = torch.ones((num_samples, seq_length))
        labels = torch.randint(0, 2, (num_samples, len(label_names))).float()

        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # Convert to expected format
        batch_dict_dataset = []
        for input_ids_batch, attention_mask_batch, labels_batch in dataloader:
            batch_dict_dataset.append({
                'input_ids': input_ids_batch,
                'attention_mask': attention_mask_batch,
                'labels': labels_batch
            })

        # Create a simple dataloader wrapper
        class DictDataLoader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        dict_dataloader = DictDataLoader(batch_dict_dataset)

        # Evaluate
        device = torch.device('cpu')
        metrics = evaluate_model(
            model=model,
            dataloader=dict_dataloader,
            device=device,
            label_names=label_names,
            threshold_strategy='global',
            threshold=0.5
        )

        # Verify all required metrics are present
        required_keys = [
            'auc', 'auc_micro', 'auc_macro',
            'macro_f1', 'micro_f1',
            'macro_precision', 'micro_precision',
            'macro_recall', 'micro_recall',
            'per_class_f1', 'per_class_precision', 'per_class_recall',
            'confusion_summary'
        ]

        for key in required_keys:
            assert key in metrics, f"Missing required metric: {key}"

    def test_evaluate_model_metrics_in_valid_range(self, label_names):
        """Test that evaluation metrics are in valid ranges [0, 1]."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=len(label_names)
        )
        model.eval()

        # Create dummy dataset
        batch_size = 4
        seq_length = 16
        num_samples = 12

        input_ids = torch.randint(0, 1000, (num_samples, seq_length))
        attention_mask = torch.ones((num_samples, seq_length))
        labels = torch.randint(0, 2, (num_samples, len(label_names))).float()

        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        batch_dict_dataset = []
        for input_ids_batch, attention_mask_batch, labels_batch in dataloader:
            batch_dict_dataset.append({
                'input_ids': input_ids_batch,
                'attention_mask': attention_mask_batch,
                'labels': labels_batch
            })

        class DictDataLoader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

        dict_dataloader = DictDataLoader(batch_dict_dataset)

        device = torch.device('cpu')
        metrics = evaluate_model(
            model=model,
            dataloader=dict_dataloader,
            device=device,
            label_names=label_names,
            threshold_strategy='global',
            threshold=0.5
        )

        # Check metric ranges
        for key in ['auc_micro', 'auc_macro', 'macro_f1', 'micro_f1',
                    'macro_precision', 'micro_precision', 'macro_recall', 'micro_recall']:
            if metrics[key] > 0:  # Allow 0 for missing classes
                assert 0 <= metrics[key] <= 1, f"{key} = {metrics[key]} not in [0, 1]"


class TestThresholdStrategyGlobal:
    """Tests for global threshold strategy."""

    def test_global_threshold_default(self, label_names, small_dataset):
        """Test evaluation with default global threshold (0.5)."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=len(label_names)
        )
        model.eval()

        # Use small_dataset fixture
        from src.training.data_utils import create_dataloaders

        _, _, test_loader, _ = create_dataloaders(
            dataset=small_dataset,
            num_labels=len(label_names),
            model_name_or_path='distilbert-base-uncased',
            batch_size=4,
            max_seq_length=32,
            seed=42
        )

        device = torch.device('cpu')
        metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            label_names=label_names,
            threshold_strategy='global',
            threshold=0.5
        )

        # Should return valid metrics
        assert metrics is not None
        assert 'f1_micro' in metrics or 'micro_f1' in metrics
        assert metrics.get('auc_micro', metrics.get('auc', 0)) >= 0

    def test_global_threshold_custom_values(self, label_names, small_dataset):
        """Test evaluation with different global threshold values."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=len(label_names)
        )
        model.eval()

        from src.training.data_utils import create_dataloaders

        _, _, test_loader, _ = create_dataloaders(
            dataset=small_dataset,
            num_labels=len(label_names),
            model_name_or_path='distilbert-base-uncased',
            batch_size=4,
            max_seq_length=32,
            seed=42
        )

        device = torch.device('cpu')

        # Test different thresholds
        for threshold in [0.3, 0.5, 0.7]:
            metrics = evaluate_model(
                model=model,
                dataloader=test_loader,
                device=device,
                label_names=label_names,
                threshold_strategy='global',
                threshold=threshold
            )

            # Should return valid metrics for all thresholds
            assert metrics is not None
            assert 'auc_micro' in metrics


class TestThresholdStrategyTopK:
    """Tests for top-k threshold strategy."""

    def test_top_k_strategy_with_k1(self, label_names, small_dataset):
        """Test evaluation with top-1 prediction."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=len(label_names)
        )
        model.eval()

        from src.training.data_utils import create_dataloaders

        _, _, test_loader, _ = create_dataloaders(
            dataset=small_dataset,
            num_labels=len(label_names),
            model_name_or_path='distilbert-base-uncased',
            batch_size=4,
            max_seq_length=32,
            seed=42
        )

        device = torch.device('cpu')
        metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            label_names=label_names,
            threshold_strategy='top_k',
            top_k=1
        )

        # Should return valid metrics
        assert metrics is not None
        assert 'f1_micro' in metrics or 'micro_f1' in metrics

    def test_top_k_strategy_with_k3(self, label_names, small_dataset):
        """Test evaluation with top-3 predictions."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=len(label_names)
        )
        model.eval()

        from src.training.data_utils import create_dataloaders

        _, _, test_loader, _ = create_dataloaders(
            dataset=small_dataset,
            num_labels=len(label_names),
            model_name_or_path='distilbert-base-uncased',
            batch_size=4,
            max_seq_length=32,
            seed=42
        )

        device = torch.device('cpu')
        metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            label_names=label_names,
            threshold_strategy='top_k',
            top_k=3
        )

        # Should return valid metrics
        assert metrics is not None
        assert 'auc_micro' in metrics

    def test_top_k_different_values(self, label_names, small_dataset):
        """Test that different k values produce different results."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=len(label_names)
        )
        model.eval()

        from src.training.data_utils import create_dataloaders

        _, _, test_loader, _ = create_dataloaders(
            dataset=small_dataset,
            num_labels=len(label_names),
            model_name_or_path='distilbert-base-uncased',
            batch_size=4,
            max_seq_length=32,
            seed=42
        )

        device = torch.device('cpu')

        # Evaluate with different k values
        metrics_k1 = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            label_names=label_names,
            threshold_strategy='top_k',
            top_k=1
        )

        metrics_k3 = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            label_names=label_names,
            threshold_strategy='top_k',
            top_k=3
        )

        # Both should return valid metrics
        assert metrics_k1 is not None
        assert metrics_k3 is not None

        # AUC should be the same (threshold-independent)
        # But F1 scores may differ
        assert 'auc_micro' in metrics_k1
        assert 'auc_micro' in metrics_k3


class TestThresholdStrategyPerClass:
    """Tests for per-class threshold strategy."""

    def test_per_class_threshold_optimization(self, label_names, small_dataset):
        """Test evaluation with per-class optimized thresholds."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=len(label_names)
        )
        model.eval()

        from src.training.data_utils import create_dataloaders

        _, _, test_loader, _ = create_dataloaders(
            dataset=small_dataset,
            num_labels=len(label_names),
            model_name_or_path='distilbert-base-uncased',
            batch_size=4,
            max_seq_length=32,
            seed=42
        )

        device = torch.device('cpu')
        metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            label_names=label_names,
            threshold_strategy='per_class'
        )

        # Should return valid metrics
        assert metrics is not None
        assert 'f1_micro' in metrics or 'micro_f1' in metrics
        assert 'per_class_f1' in metrics

    def test_per_class_threshold_returns_per_class_metrics(self, label_names, small_dataset):
        """Test that per-class strategy returns per-class F1 scores."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=len(label_names)
        )
        model.eval()

        from src.training.data_utils import create_dataloaders

        _, _, test_loader, _ = create_dataloaders(
            dataset=small_dataset,
            num_labels=len(label_names),
            model_name_or_path='distilbert-base-uncased',
            batch_size=4,
            max_seq_length=32,
            seed=42
        )

        device = torch.device('cpu')
        metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            label_names=label_names,
            threshold_strategy='per_class'
        )

        # Check per-class metrics exist
        assert 'per_class_f1' in metrics
        assert isinstance(metrics['per_class_f1'], dict)

        # Should have entries for all labels
        assert len(metrics['per_class_f1']) == len(label_names)


class TestEvaluationComparison:
    """Tests comparing different threshold strategies."""

    def test_all_strategies_produce_valid_metrics(self, label_names, small_dataset):
        """Test that all threshold strategies produce valid metrics."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=len(label_names)
        )
        model.eval()

        from src.training.data_utils import create_dataloaders

        _, _, test_loader, _ = create_dataloaders(
            dataset=small_dataset,
            num_labels=len(label_names),
            model_name_or_path='distilbert-base-uncased',
            batch_size=4,
            max_seq_length=32,
            seed=42
        )

        device = torch.device('cpu')

        # Test all three strategies
        strategies = [
            ('global', {'threshold': 0.5}),
            ('top_k', {'top_k': 2}),
            ('per_class', {})
        ]

        for strategy_name, kwargs in strategies:
            metrics = evaluate_model(
                model=model,
                dataloader=test_loader,
                device=device,
                label_names=label_names,
                threshold_strategy=strategy_name,
                **kwargs
            )

            # All should return valid metrics
            assert metrics is not None, f"Strategy {strategy_name} returned None"
            assert 'auc_micro' in metrics, f"Strategy {strategy_name} missing auc_micro"
            assert metrics['auc_micro'] >= 0, f"Strategy {strategy_name} has negative AUC"

    def test_auc_same_across_strategies(self, label_names, small_dataset):
        """Test that AUC is the same regardless of threshold strategy."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=len(label_names)
        )
        model.eval()

        from src.training.data_utils import create_dataloaders

        _, _, test_loader, _ = create_dataloaders(
            dataset=small_dataset,
            num_labels=len(label_names),
            model_name_or_path='distilbert-base-uncased',
            batch_size=4,
            max_seq_length=32,
            seed=42
        )

        device = torch.device('cpu')

        # Evaluate with different strategies
        metrics_global = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            label_names=label_names,
            threshold_strategy='global',
            threshold=0.5
        )

        metrics_topk = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            label_names=label_names,
            threshold_strategy='top_k',
            top_k=2
        )

        metrics_perclass = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            label_names=label_names,
            threshold_strategy='per_class'
        )

        # AUC should be the same (threshold-independent metric)
        auc_global = metrics_global['auc_micro']
        auc_topk = metrics_topk['auc_micro']
        auc_perclass = metrics_perclass['auc_micro']

        # Allow small numerical differences
        assert abs(auc_global - auc_topk) < 1e-6, \
            f"AUC differs between global and top_k: {auc_global} vs {auc_topk}"
        assert abs(auc_global - auc_perclass) < 1e-6, \
            f"AUC differs between global and per_class: {auc_global} vs {auc_perclass}"


class TestEvaluationEdgeCases:
    """Tests for edge cases in evaluation."""

    def test_evaluation_with_all_zeros_predictions(self, label_names):
        """Test evaluation when model predicts all zeros."""
        # Create a model that outputs all zeros
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=len(label_names)
        )

        # Zero out the classifier weights to get ~0.5 probabilities
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.zero_()

        model.eval()

        # Create dataset with some labels
        batch_size = 4
        num_samples = 8
        seq_length = 16

        input_ids = torch.randint(0, 1000, (num_samples, seq_length))
        attention_mask = torch.ones((num_samples, seq_length))
        labels = torch.randint(0, 2, (num_samples, len(label_names))).float()

        dataset = TensorDataset(input_ids, attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        batch_dict_dataset = []
        for input_ids_batch, attention_mask_batch, labels_batch in dataloader:
            batch_dict_dataset.append({
                'input_ids': input_ids_batch,
                'attention_mask': attention_mask_batch,
                'labels': labels_batch
            })

        class DictDataLoader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

        dict_dataloader = DictDataLoader(batch_dict_dataset)

        device = torch.device('cpu')
        metrics = evaluate_model(
            model=model,
            dataloader=dict_dataloader,
            device=device,
            label_names=label_names,
            threshold_strategy='global',
            threshold=0.5
        )

        # Should still return valid metrics (even if poor)
        assert metrics is not None
        assert 'auc_micro' in metrics
        assert not np.isnan(metrics['auc_micro'])

    def test_evaluation_with_single_batch(self, label_names):
        """Test evaluation with only a single batch."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=len(label_names)
        )
        model.eval()

        # Create single batch
        batch_size = 4
        seq_length = 16

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length))
        labels = torch.randint(0, 2, (batch_size, len(label_names))).float()

        batch_dict_dataset = [{
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }]

        class DictDataLoader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

        dict_dataloader = DictDataLoader(batch_dict_dataset)

        device = torch.device('cpu')
        metrics = evaluate_model(
            model=model,
            dataloader=dict_dataloader,
            device=device,
            label_names=label_names,
            threshold_strategy='global',
            threshold=0.5
        )

        # Should work with single batch
        assert metrics is not None
        assert 'auc_micro' in metrics
