# Project Documentation

This directory contains comprehensive documentation for the GoEmotions emotion classification project.

## Documentation Files

### Core Documentation

- **`replication.md`** - Step-by-step guide for reproducing all analysis, training, and visualization
  - Environment setup and dependencies
  - Running data analysis scripts
  - Model training procedures
  - Figure generation
  - Troubleshooting common issues

- **`dataset_analysis.md`** - Comprehensive dataset statistics and characteristics
  - Dataset overview and splits
  - Multi-label distribution analysis
  - Per-emotion statistics and imbalance
  - CSV export specifications
  - Data file reference

- **`design_decisions.md`** - Design choices and rationales
  - Data preprocessing decisions (neutral handling, etc.)
  - Visualization design choices (colors, sizing)
  - Model architecture selection
  - Training configuration and hyperparameters
  - Evaluation metrics and threshold strategies

- **`model_performance.md`** - Final trained model metrics and comparison
  - RoBERTa-Large and DistilBERT performance
  - Model comparison table
  - Per-emotion performance breakdown
  - Threshold selection results
  - W&B run references and checkpoint locations

- **`wandb/`** - Weights & Biases integration documentation
  - **`README.md`** - Quick start and overview
  - **`file_organization.md`** - Files tab organization and structure
  - **`downloading_files.md`** - Download files from W&B (UI and API)
  - **`metrics_guide.md`** - Metrics logging and interpretation

### Ablation Studies

- **`../ablation_studies/README.md`** - Results and analysis of ablation experiments
  - Quick comparison table of all ablations
  - Individual study results (neutral, loss weighting, thresholds, etc.)
  - W&B artifact links
  - Cross-references to design decisions and model performance

### Future Documentation

The following documentation files may be added as needed:

- **`deployment.md`** - Production deployment guide (if applicable)

## Documentation Organization

Documentation is organized by purpose:

- **Procedures** (`replication.md`) - How to run things
- **Data** (`dataset_analysis.md`) - What the data looks like
- **Decisions** (`design_decisions.md`) - Why we made certain choices
- **Results** (`ablation_studies/`, `model_performance.md`) - What we found

This modular structure makes it easier to:
- Find specific information quickly
- Update individual topics without affecting others
- Maintain consistency as the project grows

## Documentation Standards

All documentation should be:
- **Practical** - Focus on actionable steps and commands
- **Complete** - Include prerequisites, setup, execution, and troubleshooting
- **Tested** - Verify instructions work on a clean environment
- **Maintained** - Update when workflows or code structure changes
- **Modular** - Keep related content together, cross-reference when needed

## Contributing to Documentation

When making changes that affect workflows:
1. Identify which documentation file(s) need updating
2. Update relevant documentation files
3. Test instructions from scratch if possible
4. Include examples and expected outputs
5. Add troubleshooting for common issues
6. Update cross-references if adding new sections
