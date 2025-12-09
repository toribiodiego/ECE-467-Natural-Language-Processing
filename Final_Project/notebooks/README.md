# Notebooks Directory

This directory contains Jupyter notebooks for exploratory data analysis, model training experiments, and prototyping.

## Notebooks

### Dataset_Analysis.ipynb
Exploratory data analysis of the GoEmotions dataset including:
- Dataset split statistics (train/validation/test sizes)
- Class distribution analysis across 28 emotion labels
- Multi-label distribution calculation
- Token length analysis
- Example text samples per emotion

**Purpose:** Understand dataset characteristics and inform preprocessing decisions.

### Final_Project.ipynb
Complete model training pipeline with hyperparameter sweeps:
- Model training for multiple transformer architectures (BERT, RoBERTa, DistilBERT)
- Weights & Biases integration for experiment tracking
- Hyperparameter optimization using random search
- Model evaluation with AUC metrics
- Best model checkpoint saving

**Purpose:** Train and evaluate emotion classification models, identify best performing architecture.

### calculate_multilabel_stats.py
Standalone script for calculating multi-label distribution statistics (originally created for Task 04, now superseded by `src/data/multilabel_stats.py` in Phase 0 refactoring).

## Running Notebooks

### Local Execution
```bash
# Activate virtual environment
source venv/bin/activate

# Launch Jupyter
jupyter notebook

# Open desired notebook in browser
```

### Google Colab Execution (Recommended for Training)
1. Upload notebook to Google Drive
2. Open with Google Colab
3. Enable GPU runtime: Runtime → Change runtime type → GPU
4. Run cells sequentially

**Note:** Model training requires GPU access. Use Google Colab with GPU runtime for faster training.

## Notebook vs. Production Code

- **Notebooks** (`notebooks/`): Exploratory analysis, prototyping, one-off experiments
- **Scripts** (`src/`): Reusable, tested, production-ready modules

Code that needs to be rerun frequently or integrated into workflows should be extracted from notebooks into `src/` modules.
