"""
Loss function utilities for multi-label classification.

This module provides configurable loss functions including standard BCE,
class-weighted BCE, and focal loss for handling class imbalance and
hard negative mining in multi-label emotion classification.
"""

import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# Configure logging
logger = logging.getLogger(__name__)


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross-Entropy loss with per-class weights.

    Useful for addressing class imbalance in multi-label classification
    by assigning higher weights to minority classes.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        """
        Initialize weighted BCE loss.

        Args:
            pos_weight: Weight for positive examples per class [num_classes]
                       Higher values increase recall, lower values increase precision
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            logits: Predicted logits [batch_size, num_classes]
            targets: Ground truth binary labels [batch_size, num_classes]

        Returns:
            Scalar loss value
        """
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.

    Focal loss focuses training on hard examples by down-weighting
    easy examples, helping the model learn difficult cases better.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize focal loss.

        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Focusing parameter (gamma >= 0). Higher values focus more on hard examples
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Predicted logits [batch_size, num_classes]
            targets: Ground truth binary labels [batch_size, num_classes]

        Returns:
            Scalar loss value
        """
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none'
        )

        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute p_t (probability of correct class)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Combine all components
        focal_loss = alpha_t * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(
    label_counts: Dict[str, int],
    total_samples: int,
    method: str = 'inverse',
    smoothing: float = 1.0
) -> torch.Tensor:
    """
    Compute per-class weights for handling class imbalance.

    Args:
        label_counts: Dictionary mapping label names to positive sample counts
        total_samples: Total number of samples in dataset
        method: Weighting method ('inverse', 'sqrt_inverse', 'effective_samples')
        smoothing: Smoothing factor to prevent extreme weights

    Returns:
        Tensor of weights [num_classes]
    """
    num_classes = len(label_counts)
    weights = torch.ones(num_classes)

    label_names = sorted(label_counts.keys())

    for i, label in enumerate(label_names):
        pos_count = label_counts[label]
        neg_count = total_samples - pos_count

        if pos_count == 0:
            weights[i] = 1.0
            continue

        if method == 'inverse':
            # Inverse frequency: weight = neg_count / pos_count
            weights[i] = (neg_count + smoothing) / (pos_count + smoothing)
        elif method == 'sqrt_inverse':
            # Square root of inverse frequency (less aggressive)
            weights[i] = ((neg_count + smoothing) / (pos_count + smoothing)) ** 0.5
        elif method == 'effective_samples':
            # Effective number of samples method
            beta = 0.9999
            effective_num = 1.0 - beta ** pos_count
            weights[i] = (1.0 - beta) / effective_num
        else:
            weights[i] = 1.0

    return weights


def get_loss_function(
    loss_type: str = 'bce',
    class_weights: Optional[torch.Tensor] = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Factory function to create loss function based on configuration.

    Args:
        loss_type: Type of loss ('bce', 'weighted_bce', 'focal')
        class_weights: Optional per-class weights for weighted BCE
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        device: Device to place weights on

    Returns:
        Loss function module
    """
    if device is None:
        device = torch.device('cpu')

    if loss_type == 'bce':
        logger.info("Using standard Binary Cross-Entropy loss")
        return nn.BCEWithLogitsLoss()

    elif loss_type == 'weighted_bce':
        if class_weights is not None:
            class_weights = class_weights.to(device)
            logger.info(f"Using weighted BCE loss with {len(class_weights)} class weights")
            logger.info(f"  Weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")
        else:
            logger.warning("Weighted BCE requested but no weights provided, using standard BCE")
        return WeightedBCEWithLogitsLoss(pos_weight=class_weights)

    elif loss_type == 'focal':
        logger.info(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    else:
        logger.warning(f"Unknown loss type '{loss_type}', defaulting to BCE")
        return nn.BCEWithLogitsLoss()


class TrainingCostTracker:
    """
    Track and log training costs including time and computational metrics.
    """

    def __init__(self):
        """Initialize cost tracker."""
        self.epoch_times = []
        self.total_time = 0.0
        self.total_samples_trained = 0

    def log_epoch(
        self,
        epoch_time: float,
        num_samples: int,
        logger_instance: Optional[logging.Logger] = None
    ) -> None:
        """
        Log metrics for a completed epoch.

        Args:
            epoch_time: Time taken for epoch in seconds
            num_samples: Number of samples trained this epoch
            logger_instance: Logger to use
        """
        log = logger_instance or logger

        self.epoch_times.append(epoch_time)
        self.total_time += epoch_time
        self.total_samples_trained += num_samples

        # Compute metrics
        avg_time_per_epoch = sum(self.epoch_times) / len(self.epoch_times)
        samples_per_second = num_samples / epoch_time if epoch_time > 0 else 0

        log.debug(f"Epoch time: {epoch_time:.1f}s "
                 f"({samples_per_second:.1f} samples/sec)")

    def log_summary(
        self,
        logger_instance: Optional[logging.Logger] = None,
        cost_per_gpu_hour: float = 0.0
    ) -> Dict[str, Any]:
        """
        Log summary of training costs.

        Args:
            logger_instance: Logger to use
            cost_per_gpu_hour: Cost per GPU hour for cost estimation

        Returns:
            Dictionary with cost statistics
        """
        log = logger_instance or logger

        if not self.epoch_times:
            return {}

        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        total_hours = self.total_time / 3600.0

        stats = {
            'total_training_time_seconds': self.total_time,
            'total_training_time_hours': total_hours,
            'average_epoch_time_seconds': avg_epoch_time,
            'total_epochs': len(self.epoch_times),
            'total_samples_trained': self.total_samples_trained,
            'samples_per_second': self.total_samples_trained / self.total_time if self.total_time > 0 else 0
        }

        if cost_per_gpu_hour > 0:
            estimated_cost = total_hours * cost_per_gpu_hour
            stats['estimated_cost_usd'] = estimated_cost
            log.info(f"Estimated training cost: ${estimated_cost:.2f} "
                    f"(${cost_per_gpu_hour:.2f}/GPU-hour × {total_hours:.2f} hours)")

        log.info("")
        log.info("Training Cost Summary:")
        log.info(f"  Total time: {total_hours:.3f} hours ({self.total_time:.1f} seconds)")
        log.info(f"  Epochs: {len(self.epoch_times)}")
        log.info(f"  Avg epoch time: {avg_epoch_time:.1f} seconds")
        log.info(f"  Total samples: {self.total_samples_trained:,}")
        log.info(f"  Throughput: {stats['samples_per_second']:.1f} samples/sec")

        return stats
