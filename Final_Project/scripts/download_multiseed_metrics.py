#!/usr/bin/env python3
"""
Download test metrics from multi-seed DistilBERT runs for robustness analysis.

Downloads metrics from 3 DistilBERT runs (seeds 13, 23, 0) and stores them
locally for aggregation and statistical significance testing.
"""

import json
import os
from pathlib import Path

import wandb


def download_run_metrics(entity: str, project: str, run_id: str, output_path: Path):
    """Download test metrics from a W&B run.

    Args:
        entity: W&B entity (username or team)
        project: W&B project name
        run_id: W&B run ID
        output_path: Path to save metrics JSON file
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Extract test metrics from run summary
    metrics = {}
    for key, value in run.summary.items():
        if key.startswith('test/'):
            metrics[key] = value

    # Add run metadata
    metrics['_metadata'] = {
        'run_id': run_id,
        'run_name': run.name,
        'seed': run.config.get('seed', None),
        'model': run.config.get('model', None),
        'best_epoch': run.summary.get('best_epoch', None),
    }

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Downloaded metrics from {run_id} ({run.name})")
    print(f"  Seed: {metrics['_metadata']['seed']}")
    print(f"  AUC Macro: {metrics.get('test/auc_macro', 'N/A')}")
    print(f"  AUC Micro: {metrics.get('test/auc_micro', 'N/A')}")
    print(f"  Saved to: {output_path}")
    print()


def main():
    """Download metrics from all multi-seed runs."""
    entity = "Cooper-Union"
    project = "GoEmotions_Classification"
    output_dir = Path("artifacts/stats/multiseed")

    # DistilBERT multi-seed runs
    distilbert_runs = [
        ("hcx0hper", "seed13"),  # Seed 13
        ("xglzn8sy", "seed23"),  # Seed 23
        # Seed 0 run ID needs to be found from W&B
    ]

    print("=" * 70)
    print("Downloading Multi-Seed Test Metrics")
    print("=" * 70)
    print()

    # Find seed 0 run by querying recent runs
    print("Finding seed 0 run...")
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}",
                    filters={"config.seed": 0, "config.model": "distilbert-base"},
                    order="-created_at")

    if runs:
        seed0_run = runs[0]
        distilbert_runs.append((seed0_run.id, "seed0"))
        print(f"✓ Found seed 0 run: {seed0_run.id} ({seed0_run.name})")
        print()
    else:
        print("⚠ Warning: Seed 0 run not found, skipping...")
        print()

    # Download metrics for each run
    for run_id, filename_prefix in distilbert_runs:
        output_path = output_dir / f"{filename_prefix}_metrics.json"
        try:
            download_run_metrics(entity, project, run_id, output_path)
        except Exception as e:
            print(f"✗ Error downloading {run_id}: {e}")
            print()

    print("=" * 70)
    print("Download Complete")
    print("=" * 70)
    print(f"Metrics saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
