"""
Pytest fixtures for training integration tests.

This module provides reusable fixtures for testing the training pipeline,
including small datasets, temporary directories, and mock configurations.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import torch
from datasets import Dataset


@pytest.fixture
def temp_output_dir():
    """
    Provide a temporary directory for test outputs.

    Automatically cleaned up after test completion.

    Yields:
        Path: Temporary directory path
    """
    temp_dir = tempfile.mkdtemp(prefix="test_train_")
    yield Path(temp_dir)
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def small_dataset():
    """
    Provide a minimal GoEmotions-style dataset for fast testing.

    Returns a dataset with 20 samples (10 train, 5 val, 5 test) to enable
    quick integration tests without loading the full dataset.

    Returns:
        Dict[str, Dataset]: Dictionary with 'train', 'validation', 'test' splits
    """
    # Create minimal sample data
    train_data = {
        'text': [
            'I am so happy today!',
            'This makes me very angry.',
            'I feel sad about this.',
            'What a surprise!',
            'I am grateful for your help.',
            'This is confusing.',
            'I love this!',
            'I am worried about the results.',
            'This is amusing.',
            'I feel neutral about it.',
        ],
        'labels': [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # joy
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # anger
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # sadness
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # surprise
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # gratitude
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # confusion
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # love/joy
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # nervousness
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # amusement
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # neutral
        ]
    }

    val_data = {
        'text': [
            'I am excited!',
            'This is disappointing.',
            'I feel afraid.',
            'I am proud of this.',
            'This is annoying.',
        ],
        'labels': [
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # excitement
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # disappointment
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # fear
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # pride
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # annoyance
        ]
    }

    test_data = {
        'text': [
            'I am feeling optimistic.',
            'This makes me feel embarrassed.',
            'I am curious about this.',
            'I feel relieved.',
            'This is disgusting.',
        ],
        'labels': [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # optimism
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # embarrassment
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # curiosity
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # relief
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # disgust
        ]
    }

    return {
        'train': Dataset.from_dict(train_data),
        'validation': Dataset.from_dict(val_data),
        'test': Dataset.from_dict(test_data)
    }


@pytest.fixture
def label_names():
    """
    Provide the 28 GoEmotions label names.

    Returns:
        List[str]: List of emotion label names
    """
    return [
        'anger', 'amusement', 'confusion', 'disappointment', 'excitement',
        'fear', 'embarrassment', 'gratitude', 'disgust', 'desire',
        'caring', 'annoyance', 'approval', 'nervousness', 'neutral',
        'pride', 'optimism', 'sadness', 'relief', 'realization',
        'joy', 'surprise', 'admiration', 'curiosity', 'disapproval',
        'remorse', 'grief', 'love'
    ]


@pytest.fixture
def basic_args():
    """
    Provide a basic argument namespace for testing.

    Returns a minimal configuration that can be extended by individual tests.

    Returns:
        argparse.Namespace: Basic training arguments
    """
    from argparse import Namespace

    return Namespace(
        # Model
        model='distilbert-base-uncased',
        max_seq_length=128,
        dropout=0.1,

        # Training
        batch_size=4,
        epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=50,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,

        # Loss
        loss_type='bce',
        class_weight_method='inverse',
        focal_alpha=0.25,
        focal_gamma=2.0,

        # Threshold
        threshold_strategy='global',
        threshold=0.5,
        top_k=1,

        # Preprocessing
        lowercase=False,
        remove_urls=False,
        remove_emojis=False,

        # System
        seed=42,
        device='cpu',
        num_workers=0,

        # W&B
        no_wandb=True,
        wandb_project='test-goemotions',
        wandb_entity=None,
        run_name=None,

        # Output
        save_model=False,
        save_predictions=False,
        output_dir=None  # Set by test using temp_output_dir
    )


@pytest.fixture
def mock_model_config():
    """
    Provide a mock model configuration dictionary.

    Returns:
        Dict[str, Any]: Model configuration
    """
    return {
        'model_name': 'distilbert-base-uncased',
        'num_labels': 28,
        'dropout': 0.1,
        'max_seq_length': 128
    }


@pytest.fixture(autouse=True)
def set_random_seed():
    """
    Set random seed before each test for reproducibility.

    This fixture runs automatically for all tests.
    """
    import random
    import numpy as np

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
