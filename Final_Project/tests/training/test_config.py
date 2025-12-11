"""
Tests for CLI configuration and argument parsing.

This module tests all configuration-related functions including argument parsing,
validation, model mapping, and default hyperparameters. These tests ensure the
configuration layer works correctly before refactoring into config.py.
"""

import sys
import pytest
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.train import (
    parse_args,
    validate_args,
    get_model_name_or_path,
    get_model_defaults
)


class TestArgumentParsing:
    """Tests for parse_args() function."""

    def test_parse_args_with_defaults(self):
        """Test that parse_args returns valid defaults when minimal args provided."""
        with patch('sys.argv', ['train.py', '--model', 'distilbert-base']):
            args = parse_args()

        # Verify core attributes exist
        assert hasattr(args, 'model')
        assert hasattr(args, 'learning_rate')
        assert hasattr(args, 'batch_size')
        assert hasattr(args, 'epochs')
        assert hasattr(args, 'seed')

        # Verify model is set
        assert args.model == 'distilbert-base'

        # Verify default values are reasonable
        assert args.batch_size > 0
        assert args.epochs > 0
        assert 0 < args.learning_rate < 1
        assert args.seed >= 0

    def test_parse_args_with_custom_model(self):
        """Test parsing with custom model argument."""
        with patch('sys.argv', ['train.py', '--model', 'roberta-large']):
            args = parse_args()

        assert args.model == 'roberta-large'

    def test_parse_args_with_custom_training_params(self):
        """Test parsing with custom training hyperparameters."""
        with patch('sys.argv', [
            'train.py',
            '--model', 'distilbert-base',
            '--lr', '1e-5',
            '--batch-size', '16',
            '--epochs', '5',
            '--dropout', '0.2'
        ]):
            args = parse_args()

        assert args.learning_rate == 1e-5
        assert args.batch_size == 16
        assert args.epochs == 5
        assert args.dropout == 0.2

    def test_parse_args_with_loss_configuration(self):
        """Test parsing loss function configuration."""
        with patch('sys.argv', [
            'train.py',
            '--model', 'bert-base',
            '--loss-type', 'focal',
            '--focal-alpha', '0.5',
            '--focal-gamma', '3.0'
        ]):
            args = parse_args()

        assert args.loss_type == 'focal'
        assert args.focal_alpha == 0.5
        assert args.focal_gamma == 3.0

    def test_parse_args_with_weighted_bce(self):
        """Test parsing weighted BCE loss configuration."""
        with patch('sys.argv', [
            'train.py',
            '--model', 'distilbert-base',
            '--loss-type', 'weighted_bce',
            '--class-weight-method', 'sqrt_inverse'
        ]):
            args = parse_args()

        assert args.loss_type == 'weighted_bce'
        assert args.class_weight_method == 'sqrt_inverse'

    def test_parse_args_with_threshold_strategy(self):
        """Test parsing threshold strategy configuration."""
        # Test global threshold
        with patch('sys.argv', [
            'train.py',
            '--model', 'bert-base',
            '--threshold-strategy', 'global',
            '--threshold', '0.6'
        ]):
            args = parse_args()

        assert args.threshold_strategy == 'global'
        assert args.threshold == 0.6

        # Test top-k threshold
        with patch('sys.argv', [
            'train.py',
            '--model', 'bert-base',
            '--threshold-strategy', 'top_k',
            '--top-k', '3'
        ]):
            args = parse_args()

        assert args.threshold_strategy == 'top_k'
        assert args.top_k == 3

        # Test per-class threshold
        with patch('sys.argv', [
            'train.py',
            '--model', 'bert-base',
            '--threshold-strategy', 'per_class'
        ]):
            args = parse_args()

        assert args.threshold_strategy == 'per_class'

    def test_parse_args_with_preprocessing_flags(self):
        """Test parsing preprocessing configuration flags."""
        with patch('sys.argv', [
            'train.py',
            '--model', 'distilbert-base',
            '--lowercase',
            '--remove-urls',
            '--remove-emojis'
        ]):
            args = parse_args()

        assert args.lowercase is True
        assert args.remove_urls is True
        assert args.remove_emojis is True

    def test_parse_args_with_wandb_config(self):
        """Test parsing W&B configuration."""
        with patch('sys.argv', [
            'train.py',
            '--model', 'bert-base',
            '--wandb-project', 'my-project',
            '--wandb-entity', 'my-team',
            '--run-name', 'test-run'
        ]):
            args = parse_args()

        assert args.wandb_project == 'my-project'
        assert args.wandb_entity == 'my-team'
        assert args.run_name == 'test-run'

    def test_parse_args_with_no_wandb(self):
        """Test disabling W&B logging."""
        with patch('sys.argv', ['train.py', '--model', 'distilbert-base', '--no-wandb']):
            args = parse_args()

        assert args.no_wandb is True

    def test_parse_args_with_output_dir(self):
        """Test parsing output directory configuration."""
        with patch('sys.argv', ['train.py', '--model', 'bert-base', '--output-dir', '/tmp/models']):
            args = parse_args()

        assert args.output_dir == '/tmp/models'

    def test_parse_args_with_seed(self):
        """Test parsing random seed."""
        with patch('sys.argv', ['train.py', '--model', 'roberta-large', '--seed', '123']):
            args = parse_args()

        assert args.seed == 123


class TestArgumentValidation:
    """Tests for validate_args() function."""

    def test_validate_args_with_valid_config(self):
        """Test that validation passes with valid arguments."""
        args = Namespace(
            learning_rate=2e-5,
            batch_size=32,
            epochs=4,
            dropout=0.1,
            max_seq_length=128
        )

        # Should not raise any exception
        validate_args(args)

    def test_validate_args_rejects_invalid_learning_rate(self):
        """Test validation rejects invalid learning rates."""
        # Negative learning rate
        args = Namespace(
            learning_rate=-0.01,
            batch_size=32,
            epochs=4,
            dropout=0.1,
            max_seq_length=128
        )

        with pytest.raises(ValueError, match="Learning rate"):
            validate_args(args)

        # Learning rate > 1
        args.learning_rate = 1.5

        with pytest.raises(ValueError, match="Learning rate"):
            validate_args(args)

    def test_validate_args_rejects_invalid_batch_size(self):
        """Test validation rejects invalid batch sizes."""
        args = Namespace(
            learning_rate=2e-5,
            batch_size=0,
            epochs=4,
            dropout=0.1,
            max_seq_length=128
        )

        with pytest.raises(ValueError, match="Batch size"):
            validate_args(args)

        args.batch_size = -5

        with pytest.raises(ValueError, match="Batch size"):
            validate_args(args)

    def test_validate_args_rejects_invalid_epochs(self):
        """Test validation rejects invalid epoch counts."""
        args = Namespace(
            learning_rate=2e-5,
            batch_size=32,
            epochs=0,
            dropout=0.1,
            max_seq_length=128
        )

        with pytest.raises(ValueError, match="Epochs"):
            validate_args(args)

        args.epochs = -1

        with pytest.raises(ValueError, match="Epochs"):
            validate_args(args)

    def test_validate_args_rejects_invalid_dropout(self):
        """Test validation rejects invalid dropout values."""
        # Negative dropout
        args = Namespace(
            learning_rate=2e-5,
            batch_size=32,
            epochs=4,
            dropout=-0.1,
            max_seq_length=128
        )

        with pytest.raises(ValueError, match="Dropout"):
            validate_args(args)

        # Dropout >= 1
        args.dropout = 1.0

        with pytest.raises(ValueError, match="Dropout"):
            validate_args(args)

    def test_validate_args_rejects_invalid_max_seq_length(self):
        """Test validation rejects invalid max sequence lengths."""
        # Zero or negative
        args = Namespace(
            learning_rate=2e-5,
            batch_size=32,
            epochs=4,
            dropout=0.1,
            max_seq_length=0
        )

        with pytest.raises(ValueError, match="Max sequence length"):
            validate_args(args)

        # Too large (> 512)
        args.max_seq_length = 1024

        with pytest.raises(ValueError, match="Max sequence length"):
            validate_args(args)

    def test_validate_args_accepts_edge_cases(self):
        """Test validation accepts valid edge cases."""
        # Minimum valid learning rate
        args = Namespace(
            learning_rate=1e-10,
            batch_size=1,
            epochs=1,
            dropout=0.0,
            max_seq_length=1
        )

        validate_args(args)  # Should not raise

        # Maximum valid learning rate
        args.learning_rate = 1.0

        validate_args(args)  # Should not raise

        # Maximum valid max_seq_length
        args.max_seq_length = 512

        validate_args(args)  # Should not raise

        # Maximum valid dropout (just under 1.0)
        args.dropout = 0.99

        validate_args(args)  # Should not raise


class TestModelMapping:
    """Tests for get_model_name_or_path() function."""

    def test_get_model_name_for_roberta_large(self):
        """Test model name mapping for RoBERTa-Large."""
        result = get_model_name_or_path('roberta-large')
        assert result == 'roberta-large'

    def test_get_model_name_for_distilbert(self):
        """Test model name mapping for DistilBERT."""
        result = get_model_name_or_path('distilbert-base')
        assert result == 'distilbert-base-uncased'

    def test_get_model_name_for_bert(self):
        """Test model name mapping for BERT."""
        result = get_model_name_or_path('bert-base')
        assert result == 'bert-base-uncased'

    def test_get_model_name_passthrough_for_unknown(self):
        """Test that unknown model names are passed through unchanged."""
        # Direct HuggingFace model ID
        result = get_model_name_or_path('google/electra-base-discriminator')
        assert result == 'google/electra-base-discriminator'

        # Custom path
        result = get_model_name_or_path('/path/to/local/model')
        assert result == '/path/to/local/model'

        # Another HF model
        result = get_model_name_or_path('albert-base-v2')
        assert result == 'albert-base-v2'


class TestModelDefaults:
    """Tests for get_model_defaults() function."""

    def test_get_defaults_for_roberta_large(self):
        """Test getting default hyperparameters for RoBERTa-Large."""
        defaults = get_model_defaults('roberta-large')

        assert defaults is not None
        assert 'learning_rate' in defaults
        assert 'batch_size' in defaults
        assert 'epochs' in defaults
        assert 'dropout' in defaults
        assert 'description' in defaults

        # Verify reasonable values
        assert defaults['learning_rate'] > 0
        assert defaults['batch_size'] > 0
        assert defaults['epochs'] > 0
        assert 0 <= defaults['dropout'] < 1

    def test_get_defaults_for_distilbert(self):
        """Test getting default hyperparameters for DistilBERT."""
        defaults = get_model_defaults('distilbert-base')

        assert defaults is not None
        assert defaults['learning_rate'] > 0
        assert defaults['batch_size'] > 0
        assert defaults['epochs'] > 0
        assert 0 <= defaults['dropout'] < 1
        assert 'DistilBERT' in defaults['description']

    def test_get_defaults_for_bert(self):
        """Test getting default hyperparameters for BERT."""
        defaults = get_model_defaults('bert-base')

        assert defaults is not None
        assert defaults['learning_rate'] > 0
        assert defaults['batch_size'] > 0
        assert defaults['epochs'] > 0
        assert 0 <= defaults['dropout'] < 1
        assert 'BERT' in defaults['description']

    def test_get_defaults_for_unknown_model(self):
        """Test that unknown models return empty dict."""
        defaults = get_model_defaults('unknown-model')

        assert defaults == {}

    def test_defaults_differ_by_model(self):
        """Test that different models have different defaults."""
        roberta_defaults = get_model_defaults('roberta-large')
        distilbert_defaults = get_model_defaults('distilbert-base')

        # At least one parameter should differ
        # (RoBERTa-Large is typically trained with smaller batch sizes due to size)
        assert (
            roberta_defaults['batch_size'] != distilbert_defaults['batch_size'] or
            roberta_defaults['learning_rate'] != distilbert_defaults['learning_rate']
        )


class TestConfigurationIntegration:
    """Integration tests for configuration workflow."""

    def test_full_config_workflow_with_defaults(self):
        """Test complete configuration workflow with minimal arguments."""
        with patch('sys.argv', ['train.py', '--model', 'distilbert-base']):
            args = parse_args()

        # Validate should pass
        validate_args(args)

        # Model mapping should work
        model_path = get_model_name_or_path(args.model)
        assert model_path is not None

        # Should be able to get defaults for the model
        defaults = get_model_defaults(args.model)
        assert defaults != {}, "DistilBERT should have predefined defaults"

    def test_full_config_workflow_with_custom_args(self):
        """Test complete configuration workflow with custom arguments."""
        with patch('sys.argv', [
            'train.py',
            '--model', 'roberta-large',
            '--lr', '1e-5',
            '--batch-size', '8',
            '--epochs', '3',
            '--loss-type', 'focal',
            '--threshold-strategy', 'per_class'
        ]):
            args = parse_args()

        # Validate should pass
        validate_args(args)

        # Verify all custom values were parsed
        assert args.model == 'roberta-large'
        assert args.learning_rate == 1e-5
        assert args.batch_size == 8
        assert args.epochs == 3
        assert args.loss_type == 'focal'
        assert args.threshold_strategy == 'per_class'

        # Model mapping should work
        model_path = get_model_name_or_path(args.model)
        assert model_path == 'roberta-large'

        # Should have defaults for roberta-large
        defaults = get_model_defaults(args.model)
        assert defaults != {}

    def test_config_workflow_catches_invalid_args(self):
        """Test that validation catches invalid configurations."""
        with patch('sys.argv', [
            'train.py',
            '--model', 'bert-base',
            '--lr', '5.0',  # Invalid: > 1.0
            '--batch-size', '32'
        ]):
            args = parse_args()

        # Parse should succeed, but validate should fail
        with pytest.raises(ValueError, match="Learning rate"):
            validate_args(args)
