# Project Documentation

This directory contains comprehensive documentation for the GoEmotions emotion classification project.

## Quick Navigation

```
docs/
├── guides/        # Step-by-step how-to guides
├── reference/     # Technical reference and design docs
├── results/       # Model performance and findings
└── tools/         # Tool-specific documentation
```

## Documentation by Category

### Guides

How-to instructions for running experiments and reproducing results.

- **`guides/replication.md`** - Complete replication guide
  - Environment setup and dependencies
  - Running data analysis scripts
  - Model training procedures
  - Figure generation
  - Troubleshooting common issues

### Reference

Technical reference material and design documentation.

- **`reference/dataset_analysis.md`** - Dataset statistics and characteristics
  - Dataset overview and splits
  - Multi-label distribution analysis
  - Per-emotion statistics and imbalance
  - CSV export specifications
  - Data file reference

- **`reference/design_decisions.md`** - Design choices and rationales
  - Data preprocessing decisions (neutral handling, etc.)
  - Visualization design choices (colors, sizing)
  - Model architecture selection
  - Training configuration and hyperparameters
  - Evaluation metrics and threshold strategies

### Results

Performance metrics and experimental findings.

- **`results/model_performance.md`** - Final trained model metrics
  - RoBERTa-Large and DistilBERT performance
  - Model comparison table
  - Per-emotion performance breakdown
  - Threshold selection results
  - W&B run references and checkpoint locations

- **`../ablation_studies/README.md`** - Ablation study results
  - Quick comparison table of all ablations
  - Individual study results (neutral, loss weighting, thresholds, etc.)
  - W&B artifact links
  - Cross-references to design decisions

### Tools

Tool-specific integration documentation.

- **`tools/wandb/`** - Weights & Biases integration
  - **`README.md`** - Quick start and overview
  - **`file_organization.md`** - Files tab organization
  - **`downloading_files.md`** - Download files (UI and API)
  - **`metrics_guide.md`** - Metrics logging and interpretation

## Documentation Organization

Documentation follows a clear categorization pattern:

- **guides/** - Procedural instructions for running tasks
- **reference/** - Technical specs and design rationale
- **results/** - Experimental outcomes and findings
- **tools/** - Tool integration and workflows

This structure makes it easier to:
- Navigate to the right documentation quickly
- Understand the purpose of each document
- Maintain and update documentation independently
- Scale documentation as the project grows

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
