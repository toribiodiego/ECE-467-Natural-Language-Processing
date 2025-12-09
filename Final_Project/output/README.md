# Output Directory

This directory contains generated analysis results and visualizations.

## Subdirectories

- `figures/` - Generated plots and visualizations (PNG files)
- `stats/` - Statistical analysis results (CSV files)
- `reports/` - Analysis summaries and reports (text/markdown files)

## Usage

Scripts in `src/` will automatically write their outputs to these directories.
All files in this directory are gitignored to avoid committing large generated files.

## Regenerating Outputs

To regenerate all outputs, run the analysis scripts from the project root:

```bash
# Activate virtual environment
source venv/bin/activate

# Run analysis scripts
python -m src.data.multilabel_stats
python -m src.visualization.class_distribution
# ... etc
```
