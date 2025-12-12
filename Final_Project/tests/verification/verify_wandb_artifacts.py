#!/usr/bin/env python3
"""
Verification script for W&B artifacts uploaded during training.

This script verifies that all expected artifacts were successfully uploaded
to Weights & Biases for a training run.

Usage:
    python tests/verification/verify_wandb_artifacts.py <run_path>

Example:
    python tests/verification/verify_wandb_artifacts.py Cooper-Union/GoEmotions_Classification/a71b9ddo
"""

import sys
import wandb


def verify_wandb_artifacts(run_path: str) -> bool:
    """
    Verify all expected artifacts were uploaded to W&B.

    Args:
        run_path: W&B run path in format "entity/project/run_id"

    Returns:
        True if all expected artifacts are present, False otherwise
    """
    print("=" * 70)
    print("W&B Artifact Verification")
    print("=" * 70)

    # Initialize API
    api = wandb.Api()

    # Get the run
    try:
        run = api.run(run_path)
    except Exception as e:
        print(f"Error: Failed to fetch run {run_path}")
        print(f"  {e}")
        return False

    print(f"\nRun: {run.name}")
    print(f"ID: {run.id}")
    print(f"State: {run.state}")
    print(f"URL: {run.url}")

    # Define expected artifacts
    expected_artifacts = {
        'checkpoint': {
            'patterns': ['pytorch_model.bin', 'config.json', 'metrics.json'],
            'description': 'Model checkpoint files'
        },
        'validation_predictions': {
            'patterns': ['val_epoch10_predictions'],
            'description': 'Final validation predictions CSV'
        },
        'test_predictions': {
            'patterns': ['test_predictions'],
            'description': 'Test predictions CSV'
        },
        'per_class_metrics': {
            'patterns': ['per_class_metrics'],
            'description': 'Per-class metrics CSV'
        }
    }

    # Get all files
    files = list(run.files())
    file_names = [f.name for f in files]

    print("\n" + "=" * 70)
    print("Verification Results")
    print("=" * 70)

    all_found = True

    for artifact_name, artifact_info in expected_artifacts.items():
        patterns = artifact_info['patterns']
        description = artifact_info['description']

        # Check if any pattern matches
        found_files = []
        for pattern in patterns:
            matches = [f for f in file_names if pattern in f]
            found_files.extend(matches)

        if found_files:
            print(f"\n✓ {artifact_name.upper()}: FOUND")
            print(f"  Description: {description}")
            print(f"  Files:")
            for file_name in found_files:
                # Find the file object to get size
                file_obj = next(f for f in files if f.name == file_name)
                size_mb = file_obj.size / (1024 * 1024)
                print(f"    - {file_name} ({size_mb:.2f} MB)")
        else:
            print(f"\n✗ {artifact_name.upper()}: MISSING")
            print(f"  Description: {description}")
            print(f"  Expected patterns: {patterns}")
            all_found = False

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total files uploaded: {len(files)}")
    print(f"Total size: {sum(f.size for f in files) / (1024 * 1024):.2f} MB")

    if all_found:
        print("\n✓ All expected artifacts verified successfully!")
        return True
    else:
        print("\n✗ Some expected artifacts are missing!")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_wandb_artifacts.py <run_path>")
        print("Example: python verify_wandb_artifacts.py Cooper-Union/GoEmotions_Classification/a71b9ddo")
        sys.exit(1)

    run_path = sys.argv[1]
    success = verify_wandb_artifacts(run_path)
    sys.exit(0 if success else 1)
