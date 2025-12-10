# GoEmotions Emotion Classification - Replication Guide

This guide provides step-by-step instructions for reproducing all analysis, model training, and visualizations from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Analysis](#data-analysis)
4. [Model Training](#model-training)
5. [Figure Generation](#figure-generation)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Operating System:** macOS, Linux, or Windows (WSL recommended)
- **Python:** 3.8, 3.9, or 3.10 (3.10 recommended)
- **Disk Space:** Minimum 5GB free space
  - ~500MB for dependencies
  - ~2GB for cached datasets
  - ~2GB for model checkpoints (if training locally)
- **RAM:** 8GB minimum, 16GB recommended
- **Internet Connection:** Required for downloading datasets and models

### Hardware Requirements

**For Data Analysis Only:**
- CPU-only setup is sufficient
- 8GB RAM minimum

**For Model Training:**
- **GPU Required:** NVIDIA GPU with CUDA support (12GB+ VRAM recommended)
- **Alternative:** Use Google Colab (free GPU available)
- 16GB+ system RAM recommended

### Software Dependencies

All Python dependencies are specified in `requirements.txt` and installed automatically by `setup.sh`. Key packages include:

- `datasets==2.14.5` - HuggingFace datasets for GoEmotions
- `transformers==4.30.2` - Transformer models (BERT, RoBERTa, DistilBERT)
- `torch==2.0.1` - PyTorch for model training
- `pandas==2.0.3` - Data manipulation
- `matplotlib==3.7.2` - Visualization
- `jupyter==1.0.0` - Notebook environment

---

## Environment Setup

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Final_Project
```

### Step 2: Configure Environment Variables

The `.env` file controls paths and Python version. A default `.env` is created automatically, but you can customize it:

```bash
# View current configuration
cat .env

# Optional: Edit to customize paths or Python version
# Example .env contents:
# VENV_DIR=venv
# PYTHON_CMD=python3.10
# ARTIFACTS_DIR=artifacts
# OUTPUT_DIR=output
```

**Key Variables:**
- `PYTHON_CMD`: Python executable to use (default: `python3`, recommended: `python3.10`)
- `VENV_DIR`: Virtual environment directory (default: `venv`)
- `OUTPUT_DIR`: Output directory for generated files (default: `output`)
- `ARTIFACTS_DIR`: Directory for large files like model checkpoints (default: `artifacts`)

### Step 3: Run Automated Setup

The `setup.sh` script automates environment creation and dependency installation:

```bash
./setup.sh
```

**What setup.sh does:**
1. Loads configuration from `.env`
2. Validates Python version (requires 3.8+)
3. Creates virtual environment using specified Python version
4. Upgrades pip to latest version
5. Installs all dependencies from `requirements.txt`
6. Validates installation of critical packages
7. Creates `activate.sh` helper script

**Expected Output:**
```
[INFO] Loading environment variables from .env
[INFO] Starting environment setup for GoEmotions Project
[INFO] Virtual environment directory: venv
[INFO] Checking Python version...
[INFO] Found Python 3.10.17
[INFO] Creating virtual environment in venv...
[INFO] Virtual environment created successfully
[INFO] Installing dependencies from requirements.txt...
[INFO] All required packages installed successfully ✓
```

### Step 4: Activate Virtual Environment

```bash
source venv/bin/activate

# Or use the helper script:
source activate.sh
```

**Verify activation:**
```bash
which python
# Should show: /path/to/Final_Project/venv/bin/python

python --version
# Should show: Python 3.10.x
```

### Step 5: Verify Installation

Test that critical packages are importable:

```bash
python -c "import datasets, transformers, torch, pandas, matplotlib; print('All imports successful')"
```

**Expected output:** `All imports successful`

Test dataset loading module:

```bash
python -c "from src.data.load_dataset import load_go_emotions; print('Module loaded successfully')"
```

---

## Data Analysis

### Multi-Label Distribution Statistics

Calculate statistics about how many samples contain 1, 2, or 3+ emotion labels.

#### Using the Standalone Module

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Run the multilabel statistics module
python -m src.data.multilabel_stats
```

**What this does:**
1. Downloads GoEmotions dataset from HuggingFace Hub (~3.5MB)
2. Caches dataset locally (in `~/.cache/huggingface/datasets`)
3. Calculates label distribution for train/validation/test splits
4. Prints summary statistics to console
5. Exports CSV to `output/stats/multi_label_stats.csv`

**Expected Output:**

```
Loading GoEmotions dataset...
[Progress bars for downloading/processing]

Calculating multi-label distribution for each split...

Train split:
  Total samples: 43410
  1 label:  36308 (83.6%)
  2 labels:  6541 (15.1%)
  3+ labels:   561 (1.3%)

Validation split:
  Total samples: 5426
  1 label:   4548 (83.8%)
  2 labels:   809 (14.9%)
  3+ labels:    69 (1.3%)

Test split:
  Total samples: 5427
  1 label:   4590 (84.6%)
  2 labels:   774 (14.3%)
  3+ labels:    63 (1.2%)

======================================================================
SUMMARY: Multi-Label Distribution Across Splits
======================================================================
     Split 1 Label (%) 2 Labels (%) 3+ Labels (%)  Total Samples
     Train        83.6         15.1           1.3          43410
Validation        83.8         14.9           1.3           5426
      Test        84.6         14.3           1.2           5427

KEY FINDING for caption:
  16.4% of training samples contain multiple emotions

Statistics exported to: output/stats/multi_label_stats.csv
```

#### Output Files

**Location:** `output/stats/multi_label_stats.csv`

**Format:**
```csv
Split,1 Label (%),2 Labels (%),3+ Labels (%),Total Samples
Train,83.6,15.1,1.3,43410
Validation,83.8,14.9,1.3,5426
Test,84.6,14.3,1.2,5427
```

**For detailed analysis and interpretation of these statistics, see `dataset_analysis.md#multi-label-distribution`.**

#### Using the Notebook (Alternative)

You can also run the original notebook script:

```bash
cd notebooks
python calculate_multilabel_stats.py
```

This produces the same results but saves to `notebooks/multi_label_stats.csv` instead.

### Class Distribution Visualizations

Generate publication-ready stacked bar charts showing emotion frequencies and multi-label breakdown patterns.

#### Running the Visualization Module

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Run the class distribution visualization module
python -m src.visualization.class_distribution
```

**What this does:**
1. Downloads GoEmotions dataset from HuggingFace Hub (if not cached)
2. Calculates emotion frequencies and multi-label distributions
3. Generates two visualization variants (with and without neutral emotion)
4. Exports comprehensive statistics to CSV files

**Expected Output:**

The module generates:

**Visualizations:**
- `output/figures/class_distribution_stacked.png` - All 28 emotions including neutral
- `output/figures/class_distribution_stacked_no_neutral.png` - 27 emotions excluding neutral

**CSV Exports:**
- `output/stats/per_emotion_multilabel.csv` - Per-emotion multi-label breakdown
- `output/stats/per_emotion_supports_by_split.csv` - Per-emotion counts by train/val/test split

**Performance:**
- First run: ~30-40 seconds (downloads dataset)
- Subsequent runs: ~8-10 seconds (uses cached dataset)

**For detailed analysis of the dataset statistics and CSV file specifications, see `dataset_analysis.md`.**

**For the rationale behind dual visualizations (with/without neutral), see `design_decisions.md`.**

### Dataset Statistics

To load the dataset and inspect its structure:

```python
from src.data.load_dataset import load_go_emotions, get_label_names, get_dataset_statistics

# Load dataset
dataset = load_go_emotions()

# Get emotion labels
labels = get_label_names(dataset)
print(f"Number of emotions: {len(labels)}")
print(f"Emotions: {labels}")

# Get statistics
stats = get_dataset_statistics(dataset)
print(stats)
```

**Expected Output:**
```python
Number of emotions: 28
Emotions: ['admiration', 'amusement', 'anger', 'annoyance', ...]

{
    'train': {'num_samples': 43410, 'num_labels': 28},
    'validation': {'num_samples': 5426, 'num_labels': 28},
    'test': {'num_samples': 5427, 'num_labels': 28}
}
```

---

## Model Training

### Training Environment Options

#### Option 1: Google Colab (Recommended for Free GPU)

1. Open `notebooks/Final_Project.ipynb` in Google Colab
2. Runtime → Change runtime type → GPU (T4 or better)
3. Run all cells sequentially

**Advantages:**
- Free GPU access
- No local setup required
- Persistent storage in Google Drive

#### Option 2: Local Training (Requires GPU)

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Open Jupyter notebook
jupyter notebook notebooks/Final_Project.ipynb
```

**GPU Requirements:**
- CUDA-compatible NVIDIA GPU
- 12GB+ VRAM for RoBERTa-Large
- 6GB+ VRAM for DistilBERT

### Training Process

The `Final_Project.ipynb` notebook contains the complete training pipeline:

1. **Data Loading:** Load GoEmotions dataset from HuggingFace Hub
2. **Preprocessing:** Tokenize text using model-specific tokenizer
3. **Model Configuration:**
   - Architecture selection (BERT, RoBERTa, DistilBERT)
   - Hyperparameter tuning (learning rate, batch size, dropout)
4. **Training:**
   - Multi-label classification with BCEWithLogitsLoss
   - Adam optimizer
   - Learning rate scheduling
5. **Evaluation:**
   - AUC score calculation
   - Per-emotion F1 scores
   - Confusion matrix analysis
6. **Checkpoint Saving:**
   - Best model saved to `artifacts/models/`
   - Optional: Upload to Weights & Biases

### Hyperparameters

**RoBERTa-Large (Best Performance):**
```python
{
    'learning_rate': 2e-5,
    'batch_size': 16,
    'num_epochs': 3,
    'dropout': 0.1,
    'max_length': 128
}
```

**DistilBERT (Best Efficiency):**
```python
{
    'learning_rate': 3e-5,
    'batch_size': 32,
    'num_epochs': 4,
    'dropout': 0.1,
    'max_length': 128
}
```

### Expected Performance

**RoBERTa-Large:**
- Test AUC: ~95.7%
- Training time: ~2-3 hours on T4 GPU
- Model size: ~1.4GB

**DistilBERT:**
- Test AUC: ~94.8%
- Training time: ~1-2 hours on T4 GPU
- Model size: ~260MB

### Saving Model Checkpoints

Models are saved automatically to `artifacts/models/`:

```python
# In notebook:
model.save_pretrained('artifacts/models/roberta-large-best')
tokenizer.save_pretrained('artifacts/models/roberta-large-best')
```

**Directory structure:**
```
artifacts/
└── models/
    ├── roberta-large-best/
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   └── tokenizer files
    └── distilbert-base-best/
        ├── config.json
        ├── pytorch_model.bin
        └── tokenizer files
```

### Loading Saved Models

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    'artifacts/models/roberta-large-best'
)
tokenizer = AutoTokenizer.from_pretrained(
    'artifacts/models/roberta-large-best'
)
```

---

## Figure Generation

### Class Distribution Figure

**Status:** To be implemented in Phase 3 (Task 13)

This figure will show:
- Main plot: 27 emotion frequencies sorted by frequency
- Inset: Multi-label distribution (1 label, 2 labels, 3+ labels)

**Expected output location:** `output/figures/01-class-distribution-enhanced.png`

### Per-Emotion Performance Figure

**Status:** To be implemented in Phase 3 (Task 14)

This figure will show:
- Top 5 best performing emotions by F1 score
- Bottom 5 worst performing emotions by F1 score

**Expected output location:** `output/figures/02-per-emotion-performance.png`

### Generating Figures

```bash
# To be implemented
python -m src.visualization.generate_figures

# Or from notebook:
jupyter notebook notebooks/Final_Project.ipynb
# Run cells in "Visualization" section
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Python Version Incompatibility

**Error:**
```
ERROR: No matching distribution found for torch==2.0.1
```

**Cause:** Python 3.11+ is not compatible with torch==2.0.1

**Solution:**
```bash
# Update .env to use Python 3.10
echo "PYTHON_CMD=python3.10" >> .env

# Remove existing venv and rerun setup
rm -rf venv
./setup.sh
```

#### 2. PyArrow Compatibility Error

**Error:**
```
AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'
```

**Cause:** Incompatible pyarrow version with datasets library

**Solution:**
This is already fixed in `requirements.txt` with `pyarrow==12.0.1`. If you still encounter this:

```bash
source venv/bin/activate
pip install pyarrow==12.0.1
```

#### 3. Dataset Download Failure

**Error:**
```
ConnectionError: Failed to download GoEmotions dataset
```

**Causes:**
- No internet connection
- HuggingFace Hub is down
- Firewall blocking requests

**Solutions:**

1. **Check internet connection:**
   ```bash
   ping huggingface.co
   ```

2. **Clear cache and retry:**
   ```bash
   rm -rf ~/.cache/huggingface/datasets/go_emotions
   python -m src.data.multilabel_stats
   ```

3. **Use cached dataset:** If you've downloaded before, dataset should be cached locally

4. **Manual download:** Download from https://huggingface.co/datasets/go_emotions

#### 4. CUDA/GPU Not Available

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```python
   # In notebook
   batch_size = 8  # Instead of 16 or 32
   ```

2. **Use gradient accumulation:**
   ```python
   gradient_accumulation_steps = 2
   effective_batch_size = batch_size * gradient_accumulation_steps
   ```

3. **Use smaller model:**
   - Switch from RoBERTa-Large to DistilBERT
   - DistilBERT uses 75% less memory

4. **Use Google Colab:**
   - Free GPU available
   - No local setup required

#### 5. Module Not Found Errors

**Error:**
```
ModuleNotFoundError: No module named 'src'
```

**Cause:** Running from wrong directory or virtual environment not activated

**Solution:**
```bash
# Ensure you're in project root
cd /path/to/Final_Project

# Activate virtual environment
source venv/bin/activate

# Verify you're in the right directory
pwd
# Should show: /path/to/Final_Project

# Run module
python -m src.data.multilabel_stats
```

#### 6. Permission Denied on setup.sh

**Error:**
```
bash: ./setup.sh: Permission denied
```

**Solution:**
```bash
chmod +x setup.sh
./setup.sh
```

#### 7. Output Directory Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'output/stats/'
```

**Cause:** Output directories are gitignored and may not exist

**Solution:**
The code creates directories automatically via `os.makedirs(os.path.dirname(output_path), exist_ok=True)`. If this fails:

```bash
mkdir -p output/stats output/figures output/reports
```

### Getting Help

If you encounter issues not covered here:

1. **Check logs:** Look for detailed error messages in console output
2. **Verify environment:**
   ```bash
   python --version
   pip list | grep -E "datasets|transformers|torch"
   ```
3. **Search issues:** Check GitHub issues for similar problems
4. **Open issue:** Include full error trace and environment details

---

## Additional Resources

### Project Structure

```
Final_Project/
├── src/                    # Source code modules
│   ├── data/              # Data loading and preprocessing
│   ├── analysis/          # Model evaluation and metrics
│   └── visualization/     # Figure generation
├── notebooks/             # Jupyter notebooks for exploration
├── docs/                  # Documentation
├── output/                # Generated outputs (gitignored)
│   ├── figures/          # Generated visualizations
│   ├── stats/            # Calculated statistics
│   └── reports/          # Analysis reports
├── artifacts/             # Large files (gitignored)
│   └── models/           # Trained model checkpoints
├── requirements.txt       # Python dependencies
├── setup.sh              # Automated setup script
└── .env                  # Environment configuration (gitignored)
```

### Dataset Information

**GoEmotions Dataset:**
- **Source:** https://huggingface.co/datasets/go_emotions
- **Size:** ~58K Reddit comments
- **Labels:** 27 emotion categories + neutral (28 total)
- **Type:** Multi-label classification
- **Splits:**
  - Train: 43,410 samples
  - Validation: 5,426 samples
  - Test: 5,427 samples

**Emotion Categories:**
admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

### Useful Commands

```bash
# Activate environment
source venv/bin/activate

# Deactivate environment
deactivate

# Update dependencies
pip install -r requirements.txt --upgrade

# Run multilabel statistics
python -m src.data.multilabel_stats

# Start Jupyter
jupyter notebook

# Check Python path
which python

# List installed packages
pip list

# Check disk usage
du -sh output/ artifacts/ venv/
```

---

**Last Updated:** December 2024

**Maintainer:** ECE-467 Final Project Team

**License:** See repository LICENSE file
