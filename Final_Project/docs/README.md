# Project Documentation

Docs for the GoEmotions emotion classification project. Use this as the landing page to find the right guide, reference, or result without digging through files. For a project overview and reports, see `../README.md` and `output/submission/`.

## Start Here
- Reproduce everything end-to-end: `guides/replication.md`
- Train on a GPU machine: `guides/gpu_training.md`
- W&B integration and local artifact access: `guides/wandbguide.md`
- Just want the metrics: `results/model_performance.md` and `../output/ablation_studies/README.md`
- Explore the analysis and training notebooks: `notebooks/Dataset_Analysis.ipynb`, `notebooks/Final_Project.ipynb`

## Tasks → Where to go

| Task | Doc(s) | Code entry points |
| --- | --- | --- |
| Run full pipeline on a fresh machine | `guides/replication.md` | `python -m src.training.train ...`; data helpers in `src/data/load_dataset.py` |
| Train on a remote/local GPU with W&B | `guides/gpu_training.md`, `guides/wandbguide.md` | `src/training/train.py`, `src/training/wandb_utils.py` |
| Access artifacts locally (no W&B) | `guides/wandbguide.md`, `artifacts/README.md` | Export scripts in `src/training/`, `src/analysis/` |
| Inspect dataset stats/figures | `reference/dataset_analysis.md` | `notebooks/Dataset_Analysis.ipynb`, plots via `src/analysis/*.py` |
| Compare model performance and ablations | `results/model_performance.md`, `../output/ablation_studies/README.md` | Metrics export in `src/analysis/metric_comparison.py` |
| Understand design choices | `reference/design_decisions.md` | Training/config rationale referenced from `src/training/*` |

## Common Commands

```bash
# Standard training run with W&B logging
python -m src.training.train --model roberta-large --lr 2e-5 --batch-size 16 --epochs 4 --wandb-project GoEmotions_Classification

# Quick CPU sanity check (limits data, disables W&B)
python -m src.training.train --model distilbert-base --epochs 1 --max-train-samples 500 --max-eval-samples 200 --no-wandb
```

W&B setup and artifact access: see `guides/wandbguide.md` (covers both local and W&B paths).

## Where results live
- Final model metrics and tables: `results/model_performance.md`
- Ablation summaries and links to runs: `../output/ablation_studies/README.md`
- Slides/report: `../output/submission/Final_Project_Presentation.pdf`, `../output/submission/Final_Project_Report.pdf`

## Maintaining these docs
- When training configs change, update `guides/replication.md`, `guides/gpu_training.md`, and thresholds/metrics in `results/model_performance.md`.
- When data processing or figures change, refresh `reference/dataset_analysis.md` and any generated plots.
- When adding W&B artifacts or file layouts, update `tools/wandb/README.md` and `tools/wandb/file_organization.md`.

## Quick Navigation

```
docs/
├── guides/        # Step-by-step how-to guides
├── reference/     # Technical reference and design docs
├── results/       # Model performance and findings
└── tools/         # Tool-specific documentation
```
