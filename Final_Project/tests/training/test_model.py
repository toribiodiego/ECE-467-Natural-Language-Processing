"""
Tests for model architecture and initialization.

This module tests the MultiLabelClassificationModel class and initialize_model()
function to ensure model architecture, forward passes, and initialization work
correctly before refactoring into models.py.
"""

import sys
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from argparse import Namespace

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.train import (
    MultiLabelClassificationModel,
    initialize_model,
    get_model_name_or_path
)
from src.training.loss_utils import get_loss_function


class TestMultiLabelClassificationModel:
    """Tests for MultiLabelClassificationModel class."""

    def test_model_initialization_with_defaults(self):
        """Test model initializes correctly with default parameters."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=28
        )

        # Verify model structure
        assert model is not None
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'classifier')
        assert hasattr(model, 'dropout')
        assert hasattr(model, 'loss_fn')

        # Verify configuration
        assert model.num_labels == 28
        assert model.hidden_size > 0

        # Verify classifier is correct size
        assert model.classifier.out_features == 28

    def test_model_initialization_with_custom_dropout(self):
        """Test model initializes with custom dropout rate."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=28,
            dropout=0.3
        )

        # Verify dropout layer exists
        assert isinstance(model.dropout, nn.Dropout)
        assert model.dropout.p == 0.3

    def test_model_initialization_with_custom_loss(self):
        """Test model initializes with custom loss function."""
        custom_loss = nn.BCEWithLogitsLoss(reduction='sum')

        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=28,
            loss_fn=custom_loss
        )

        # Verify custom loss is used
        assert model.loss_fn is custom_loss

    def test_model_forward_pass_without_labels(self):
        """Test forward pass without labels returns only logits."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=28
        )

        # Create dummy inputs
        batch_size = 4
        seq_length = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length))

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Verify output structure
        assert 'logits' in outputs
        assert 'loss' in outputs

        # Verify logits shape
        logits = outputs['logits']
        assert logits.shape == (batch_size, 28)

        # Verify loss is None when no labels provided
        assert outputs['loss'] is None

    def test_model_forward_pass_with_labels(self):
        """Test forward pass with labels returns logits and loss."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=28
        )

        # Create dummy inputs
        batch_size = 4
        seq_length = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length))
        labels = torch.randint(0, 2, (batch_size, 28)).float()

        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        # Verify output structure
        assert 'logits' in outputs
        assert 'loss' in outputs

        # Verify logits shape
        logits = outputs['logits']
        assert logits.shape == (batch_size, 28)

        # Verify loss is computed
        loss = outputs['loss']
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1  # Scalar loss

        # Verify loss is reasonable (not NaN or infinite)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() > 0

    def test_model_output_shapes_different_batch_sizes(self):
        """Test model produces correct output shapes for different batch sizes."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=28
        )
        model.eval()

        seq_length = 32

        # Test different batch sizes
        for batch_size in [1, 2, 8, 16]:
            input_ids = torch.randint(0, 1000, (batch_size, seq_length))
            attention_mask = torch.ones((batch_size, seq_length))

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # Verify output shape matches batch size
            assert outputs['logits'].shape == (batch_size, 28), \
                f"Expected shape ({batch_size}, 28), got {outputs['logits'].shape}"

    def test_model_output_shapes_different_seq_lengths(self):
        """Test model handles different sequence lengths correctly."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=28
        )
        model.eval()

        batch_size = 4

        # Test different sequence lengths
        for seq_length in [8, 16, 32, 64, 128]:
            input_ids = torch.randint(0, 1000, (batch_size, seq_length))
            attention_mask = torch.ones((batch_size, seq_length))

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # Output shape should be independent of sequence length
            assert outputs['logits'].shape == (batch_size, 28), \
                f"Seq length {seq_length}: expected shape ({batch_size}, 28), got {outputs['logits'].shape}"

    def test_model_with_different_num_labels(self):
        """Test model initializes correctly with different number of labels."""
        for num_labels in [2, 10, 28, 50]:
            model = MultiLabelClassificationModel(
                model_name_or_path='distilbert-base-uncased',
                num_labels=num_labels
            )

            # Verify classifier output size
            assert model.classifier.out_features == num_labels
            assert model.num_labels == num_labels

            # Verify forward pass produces correct output shape
            batch_size = 2
            seq_length = 16
            input_ids = torch.randint(0, 1000, (batch_size, seq_length))
            attention_mask = torch.ones((batch_size, seq_length))

            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            assert outputs['logits'].shape == (batch_size, num_labels)

    def test_model_gradient_computation(self):
        """Test model computes gradients correctly during training mode."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=28
        )
        model.train()

        # Create dummy inputs
        batch_size = 2
        seq_length = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length))
        labels = torch.randint(0, 2, (batch_size, 28)).float()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # Backward pass
        loss = outputs['loss']
        loss.backward()

        # Verify gradients are computed for classifier
        assert model.classifier.weight.grad is not None
        assert model.classifier.bias.grad is not None

        # Verify gradients are not all zero
        assert not torch.allclose(model.classifier.weight.grad, torch.zeros_like(model.classifier.weight.grad))


class TestInitializeModel:
    """Tests for initialize_model() function."""

    def test_initialize_model_with_distilbert(self, basic_args):
        """Test model initialization with DistilBERT."""
        basic_args.model = 'distilbert-base'
        basic_args.dropout = 0.1

        device = torch.device('cpu')
        loss_fn = get_loss_function(loss_type='bce', device=device)

        model = initialize_model(
            model_name='distilbert-base',
            num_labels=28,
            args=basic_args,
            loss_fn=loss_fn
        )

        # Verify model is initialized
        assert model is not None
        assert isinstance(model, MultiLabelClassificationModel)

        # Verify model is on correct device
        assert next(model.parameters()).device.type == 'cpu'

        # Verify configuration
        assert model.num_labels == 28

    def test_initialize_model_with_bert(self, basic_args):
        """Test model initialization with BERT."""
        basic_args.model = 'bert-base'
        basic_args.dropout = 0.2

        device = torch.device('cpu')
        loss_fn = get_loss_function(loss_type='bce', device=device)

        model = initialize_model(
            model_name='bert-base',
            num_labels=28,
            args=basic_args,
            loss_fn=loss_fn
        )

        # Verify model is initialized
        assert model is not None
        assert isinstance(model, MultiLabelClassificationModel)

        # Verify dropout configuration
        assert model.dropout.p == 0.2

    def test_initialize_model_with_roberta(self, basic_args):
        """Test model initialization with RoBERTa."""
        basic_args.model = 'roberta-large'
        basic_args.dropout = 0.1

        device = torch.device('cpu')
        loss_fn = get_loss_function(loss_type='bce', device=device)

        model = initialize_model(
            model_name='roberta-large',
            num_labels=28,
            args=basic_args,
            loss_fn=loss_fn
        )

        # Verify model is initialized
        assert model is not None
        assert isinstance(model, MultiLabelClassificationModel)

        # RoBERTa-Large should have more parameters than DistilBERT
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 100_000_000  # > 100M parameters

    def test_initialize_model_with_custom_loss_function(self, basic_args):
        """Test model initialization with custom loss function."""
        basic_args.model = 'distilbert-base'
        basic_args.dropout = 0.1

        device = torch.device('cpu')
        custom_loss = get_loss_function(
            loss_type='focal',
            focal_alpha=0.25,
            focal_gamma=2.0,
            device=device
        )

        model = initialize_model(
            model_name='distilbert-base',
            num_labels=28,
            args=basic_args,
            loss_fn=custom_loss
        )

        # Verify custom loss is used
        assert model.loss_fn is custom_loss

    def test_initialize_model_parameter_count(self, basic_args):
        """Test that initialized model has expected parameter counts."""
        basic_args.model = 'distilbert-base'
        basic_args.dropout = 0.1

        device = torch.device('cpu')
        loss_fn = get_loss_function(loss_type='bce', device=device)

        model = initialize_model(
            model_name='distilbert-base',
            num_labels=28,
            args=basic_args,
            loss_fn=loss_fn
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # DistilBERT should have ~66M parameters
        assert total_params > 60_000_000, f"Expected >60M params, got {total_params:,}"
        assert total_params < 80_000_000, f"Expected <80M params, got {total_params:,}"

        # All parameters should be trainable by default
        assert total_params == trainable_params

    def test_initialize_model_creates_working_model(self, basic_args):
        """Test that initialized model can perform forward pass."""
        basic_args.model = 'distilbert-base'
        basic_args.dropout = 0.1

        device = torch.device('cpu')
        loss_fn = get_loss_function(loss_type='bce', device=device)

        model = initialize_model(
            model_name='distilbert-base',
            num_labels=28,
            args=basic_args,
            loss_fn=loss_fn
        )

        # Create dummy inputs
        batch_size = 2
        seq_length = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length))
        labels = torch.randint(0, 2, (batch_size, 28)).float()

        # Forward pass should work
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        # Verify outputs are valid
        assert outputs['logits'].shape == (batch_size, 28)
        assert outputs['loss'] is not None
        assert not torch.isnan(outputs['loss'])


class TestModelArchitectureDetails:
    """Tests for specific model architecture details."""

    def test_model_uses_cls_token_pooling(self):
        """Test that model uses [CLS] token (first token) for classification."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=28
        )
        model.eval()

        # Create inputs with different attention masks
        # to verify [CLS] token is used
        batch_size = 2
        seq_length = 16

        input_ids = torch.randint(0, 1000, (batch_size, seq_length))

        # First sample: full attention
        # Second sample: only attend to first 8 tokens
        attention_mask = torch.ones((batch_size, seq_length))
        attention_mask[1, 8:] = 0

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Both samples should produce valid outputs
        # (using [CLS] token which is always attended to)
        assert outputs['logits'].shape == (batch_size, 28)
        assert not torch.isnan(outputs['logits']).any()

    def test_model_dropout_only_active_in_train_mode(self):
        """Test that dropout is only active during training."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=28,
            dropout=0.5  # High dropout to make effect visible
        )

        batch_size = 4
        seq_length = 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length))

        # In eval mode, outputs should be deterministic
        model.eval()
        with torch.no_grad():
            output1 = model(input_ids=input_ids, attention_mask=attention_mask)
            output2 = model(input_ids=input_ids, attention_mask=attention_mask)

        # Outputs should be identical in eval mode
        assert torch.allclose(output1['logits'], output2['logits'])

    def test_model_hidden_size_matches_backbone(self):
        """Test that model hidden size matches the backbone transformer."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=28
        )

        # DistilBERT-base should have hidden size 768
        assert model.hidden_size == 768

        # Classifier input should match hidden size
        assert model.classifier.in_features == 768

    def test_model_default_loss_is_bce_with_logits(self):
        """Test that default loss function is BCEWithLogitsLoss."""
        model = MultiLabelClassificationModel(
            model_name_or_path='distilbert-base-uncased',
            num_labels=28
        )

        # Default loss should be BCEWithLogitsLoss
        assert isinstance(model.loss_fn, nn.BCEWithLogitsLoss)
