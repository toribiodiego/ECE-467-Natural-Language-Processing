#!/usr/bin/env python3
"""
Validate Local Artifact Integrity

This script validates that saved prediction CSVs, per-class metrics, and figures
are properly formatted, can be loaded, and have correct schemas. It also documents
the relative paths for downstream scripts.

Usage:
    python -m tests.validate_artifacts
    python -m tests.validate_artifacts --verbose
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArtifactValidator:
    """Validates local artifacts for integrity and schema correctness."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.validations_passed = 0
        self.validations_failed = 0

    def validate_prediction_csv(self, csv_path: Path) -> bool:
        """
        Validate a prediction CSV file.

        Expected columns:
        - text: Input text
        - true_labels: Ground truth emotions (comma-separated)
        - pred_labels: Predicted emotions (comma-separated)
        - pred_prob_{emotion}: Probability for each emotion

        Args:
            csv_path: Path to prediction CSV

        Returns:
            True if valid, False otherwise
        """
        try:
            df = pd.read_csv(csv_path)

            # Check required columns
            required_cols = ['text', 'true_labels', 'pred_labels']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.errors.append(f"{csv_path.name}: Missing columns: {missing_cols}")
                return False

            # Check for probability columns (should have pred_prob_* columns)
            prob_cols = [col for col in df.columns if col.startswith('pred_prob_')]
            if len(prob_cols) == 0:
                self.warnings.append(f"{csv_path.name}: No probability columns found")

            # Validate data types
            if not df['text'].dtype == object:
                self.errors.append(f"{csv_path.name}: 'text' column should be string type")
                return False

            # Check for empty dataframe
            if len(df) == 0:
                self.errors.append(f"{csv_path.name}: CSV is empty")
                return False

            # Check probability ranges
            for prob_col in prob_cols:
                if df[prob_col].dtype not in [np.float64, np.float32]:
                    self.errors.append(f"{csv_path.name}: {prob_col} should be float type")
                    return False
                if df[prob_col].min() < 0 or df[prob_col].max() > 1:
                    self.errors.append(f"{csv_path.name}: {prob_col} values out of [0,1] range")
                    return False

            if self.verbose:
                logger.info(f"  ✓ {csv_path.name}: {len(df)} rows, {len(df.columns)} columns")
                logger.info(f"    Columns: {', '.join(df.columns[:5])}...")
                logger.info(f"    Probability columns: {len(prob_cols)}")

            return True

        except Exception as e:
            self.errors.append(f"{csv_path.name}: Failed to load - {str(e)}")
            return False

    def validate_per_class_metrics_csv(self, csv_path: Path) -> bool:
        """
        Validate a per-class metrics CSV file.

        Expected columns:
        - rank: Ranking by F1 score
        - emotion: Emotion label name
        - precision: Precision for this emotion
        - recall: Recall for this emotion
        - f1: F1 score for this emotion
        - support: Number of samples

        Args:
            csv_path: Path to per-class metrics CSV

        Returns:
            True if valid, False otherwise
        """
        try:
            df = pd.read_csv(csv_path)

            # Check required columns (flexible to handle different formats)
            possible_required = [
                ['rank', 'emotion', 'precision', 'recall', 'f1', 'support'],
                ['emotion', 'f1_score', 'precision', 'recall', 'support'],  # Alternative format
            ]

            valid_format = False
            for required_cols in possible_required:
                if all(col in df.columns for col in required_cols):
                    valid_format = True
                    break

            if not valid_format:
                self.errors.append(f"{csv_path.name}: Missing required columns. Expected one of: {possible_required}")
                return False

            # Check for empty dataframe
            if len(df) == 0:
                self.errors.append(f"{csv_path.name}: CSV is empty")
                return False

            # Validate metric ranges (0 to 1)
            metric_cols = ['precision', 'recall', 'f1'] if 'f1' in df.columns else ['precision', 'recall', 'f1_score']
            for col in metric_cols:
                if col in df.columns:
                    if df[col].min() < 0 or df[col].max() > 1:
                        self.errors.append(f"{csv_path.name}: {col} values out of [0,1] range")
                        return False

            if self.verbose:
                logger.info(f"  ✓ {csv_path.name}: {len(df)} emotions, {len(df.columns)} columns")
                logger.info(f"    Columns: {', '.join(df.columns)}")

            return True

        except Exception as e:
            self.errors.append(f"{csv_path.name}: Failed to load - {str(e)}")
            return False

    def validate_json_metrics(self, json_path: Path) -> bool:
        """
        Validate a JSON metrics file.

        Args:
            json_path: Path to JSON metrics file

        Returns:
            True if valid, False otherwise
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Check it's a dictionary
            if not isinstance(data, dict):
                self.errors.append(f"{json_path.name}: JSON should be a dictionary")
                return False

            # Check for common metric keys
            if len(data) == 0:
                self.errors.append(f"{json_path.name}: JSON is empty")
                return False

            if self.verbose:
                logger.info(f"  ✓ {json_path.name}: {len(data)} keys")
                logger.info(f"    Keys: {', '.join(list(data.keys())[:5])}...")

            return True

        except json.JSONDecodeError as e:
            self.errors.append(f"{json_path.name}: Invalid JSON - {str(e)}")
            return False
        except Exception as e:
            self.errors.append(f"{json_path.name}: Failed to load - {str(e)}")
            return False

    def validate_figure(self, fig_path: Path) -> bool:
        """
        Validate a figure file exists and has reasonable size.

        Args:
            fig_path: Path to figure file

        Returns:
            True if valid, False otherwise
        """
        try:
            if not fig_path.exists():
                self.errors.append(f"{fig_path.name}: File does not exist")
                return False

            file_size = fig_path.stat().st_size
            # Check file size (should be at least 1KB, less than 10MB)
            if file_size < 1024:
                self.warnings.append(f"{fig_path.name}: File very small ({file_size} bytes)")
            elif file_size > 10 * 1024 * 1024:
                self.warnings.append(f"{fig_path.name}: File very large ({file_size / 1024 / 1024:.1f} MB)")

            if self.verbose:
                logger.info(f"  ✓ {fig_path.name}: {file_size / 1024:.1f} KB")

            return True

        except Exception as e:
            self.errors.append(f"{fig_path.name}: Failed to validate - {str(e)}")
            return False

    def run_validation(self) -> bool:
        """
        Run all validations.

        Returns:
            True if all validations pass, False otherwise
        """
        logger.info("=" * 70)
        logger.info("ARTIFACT INTEGRITY VALIDATION")
        logger.info("=" * 70)

        # Validate predictions
        logger.info("\n1. Validating Prediction CSVs (artifacts/predictions/)")
        logger.info("-" * 70)
        predictions_dir = Path("artifacts/predictions")
        if predictions_dir.exists():
            prediction_files = list(predictions_dir.glob("*.csv"))
            if prediction_files:
                for pred_file in sorted(prediction_files)[:5]:  # Validate first 5
                    if self.validate_prediction_csv(pred_file):
                        self.validations_passed += 1
                    else:
                        self.validations_failed += 1
                if len(prediction_files) > 5:
                    logger.info(f"  ... and {len(prediction_files) - 5} more files (not shown)")
            else:
                self.warnings.append("No prediction CSV files found")
        else:
            self.warnings.append("Predictions directory does not exist")

        # Validate per-class metrics
        logger.info("\n2. Validating Per-Class Metrics (artifacts/stats/)")
        logger.info("-" * 70)
        stats_dir = Path("artifacts/stats")
        if stats_dir.exists():
            # Check for per-class metrics
            per_class_files = list(stats_dir.glob("per_class_metrics_*.csv")) + \
                             list(stats_dir.glob("per_emotion_scores*.csv"))
            if per_class_files:
                for metrics_file in sorted(per_class_files):
                    if self.validate_per_class_metrics_csv(metrics_file):
                        self.validations_passed += 1
                    else:
                        self.validations_failed += 1
            else:
                self.warnings.append("No per-class metrics CSV files found")

            # Check for JSON metrics
            json_files = list(stats_dir.glob("*.json"))
            if json_files:
                for json_file in sorted(json_files):
                    if self.validate_json_metrics(json_file):
                        self.validations_passed += 1
                    else:
                        self.validations_failed += 1
            else:
                self.warnings.append("No JSON metrics files found")
        else:
            self.warnings.append("Stats directory does not exist")

        # Validate figures
        logger.info("\n3. Validating Figures (output/figures/)")
        logger.info("-" * 70)
        figures_dir = Path("output/figures")
        if figures_dir.exists():
            figure_files = list(figures_dir.glob("*.png"))
            if figure_files:
                for fig_file in sorted(figure_files):
                    if self.validate_figure(fig_file):
                        self.validations_passed += 1
                    else:
                        self.validations_failed += 1
            else:
                self.warnings.append("No PNG figure files found")
        else:
            self.warnings.append("Figures directory does not exist")

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Validations Passed: {self.validations_passed}")
        logger.info(f"Validations Failed: {self.validations_failed}")
        logger.info(f"Warnings: {len(self.warnings)}")
        logger.info(f"Errors: {len(self.errors)}")

        if self.warnings:
            logger.info("\nWarnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        if self.errors:
            logger.info("\nErrors:")
            for error in self.errors:
                logger.error(f"  - {error}")

        # Document relative paths
        logger.info("\n" + "=" * 70)
        logger.info("ARTIFACT PATHS FOR DOWNSTREAM SCRIPTS")
        logger.info("=" * 70)
        logger.info("Predictions:")
        logger.info("  artifacts/predictions/test_predictions_{model}_{timestamp}.csv")
        logger.info("  artifacts/predictions/val_epoch{N}_predictions_{model}_{timestamp}.csv")
        logger.info("\nPer-Class Metrics:")
        logger.info("  artifacts/stats/per_class_metrics_{model}_{timestamp}.csv")
        logger.info("  artifacts/stats/per_emotion_scores.csv")
        logger.info("\nTest Metrics (JSON):")
        logger.info("  artifacts/stats/test_metrics_{model}.json")
        logger.info("\nFigures:")
        logger.info("  output/figures/00_class_distribution.png")
        logger.info("  output/figures/07_per_emotion_f1_scores.png")
        logger.info("  ... (see output/figures/ for all visualizations)")
        logger.info("=" * 70)

        return self.validations_failed == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate local artifact integrity'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed validation information'
    )

    args = parser.parse_args()

    validator = ArtifactValidator(verbose=args.verbose)
    success = validator.run_validation()

    if success:
        logger.info("\n✓ All validations passed!")
        sys.exit(0)
    else:
        logger.error("\n✗ Some validations failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
