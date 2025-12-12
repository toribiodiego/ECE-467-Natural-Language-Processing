# GPU Training Guide

This guide covers setting up and running model training on a remote GPU instance.

## Prerequisites

- Access to a GPU instance (AWS, GCP, Lambda Labs, etc.)
- SSH access to the instance
- Weights & Biases account and API key

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/toribiodiego/ECE-467-Natural-Language-Processing.git
cd ECE-467-Natural-Language-Processing/Final_Project

# 2. Run setup script
./setup.sh

# 3. Set W&B API key
set -a
source .env
set +a

# 4. Activate environment
source venv/bin/activate

# 5. Start training
python -m src.training.train --model distilbert-base --epochs 5
```

## Detailed Setup Instructions

### Step 1: Clone Repository

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/toribiodiego/ECE-467-Natural-Language-Processing.git
cd ECE-467-Natural-Language-Processing/Final_Project
```

**Verify you're in the correct directory:**
```bash
ls
# Should show: README.md, setup.sh, src/, tests/, docs/, etc.
```

### Step 2: Run Setup Script

The setup script creates a virtual environment, installs dependencies, and downloads the dataset.

```bash
chmod +x setup.sh
./setup.sh
```

**What this does:**
- Creates Python virtual environment in `venv/`
- Installs PyTorch with CUDA support
- Installs transformers, datasets, scikit-learn, wandb, etc.
- Downloads GoEmotions dataset from Hugging Face
- Verifies GPU availability

**Expected output:**
```
Creating virtual environment...
Installing dependencies...
Downloading GoEmotions dataset...
Setup complete!

GPU available: True
GPU count: 1
GPU name: NVIDIA A100-SXM4-40GB
```

**Troubleshooting:**
- If GPU not detected, check CUDA installation: `nvidia-smi`
- If dependencies fail, ensure Python 3.8+ is installed: `python3 --version`

### Step 3: Configure Weights & Biases

Create a `.env` file with your W&B API key:

```bash
cat > .env << 'EOF'
WANDB_API_KEY=your_api_key_here
EOF
```

**Get your API key:**
1. Go to https://wandb.ai/settings
2. Scroll to "API keys"
3. Copy your key

**Load environment variables:**

```bash
set -a
source .env
set +a
```

**What this does:**
- `set -a` - Automatically export all variables
- `source .env` - Load variables from .env file
- `set +a` - Disable automatic export

**Verify W&B is configured:**
```bash
wandb login
# Should show: "Already logged in to Weights & Biases"
```

**Alternative method (without .env file):**
```bash
wandb login
# Paste your API key when prompted
```

### Step 4: Activate Virtual Environment

```bash
source venv/bin/activate
```

Your prompt should now show `(venv)` prefix.

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.0.1+cu118
CUDA available: True
```

### Step 5: Start Training

## Task 03 — RoBERTa-Large Production Training

**Exact command for production RoBERTa-Large training:**

```bash
python -m src.training.train \
  --model roberta-large \
  --lr 2e-5 \
  --batch-size 16 \
  --epochs 10 \
  --dropout 0.1
```

**Resources Required:**
- GPU Memory: ~12-16 GB
- Training Time: ~12-16 hours (10 epochs on full dataset)
- Expected Test AUC: ~0.65-0.70

**What This Does:**
1. Trains RoBERTa-Large (355M parameters) on full GoEmotions dataset
2. Automatically saves best checkpoint based on validation AUC (not final epoch)
3. Exports predictions, metrics, and per-class performance to CSVs
4. Logs all metrics and artifacts to W&B
5. Saves checkpoint to `artifacts/models/roberta-large-*`

**W&B Tags (Optional):**
Add these tags in W&B UI after run starts: `model:roberta-large`, `purpose:production`

---

## Task 04 — DistilBERT Efficiency Baseline Training

**Exact command for DistilBERT efficiency baseline:**

```bash
python -m src.training.train \
  --model distilbert-base \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 10 \
  --dropout 0.1
```

**Resources Required:**
- GPU Memory: ~4-6 GB
- Training Time: ~4-6 hours (10 epochs on full dataset)
- Expected Test AUC: ~0.60-0.65

**What This Does:**
1. Trains DistilBERT (66M parameters) as efficiency baseline
2. Automatically saves best checkpoint based on validation AUC
3. Exports predictions, metrics, and per-class performance to CSVs
4. Logs all metrics and artifacts to W&B
5. Saves checkpoint to `artifacts/models/distilbert-base-*`

**W&B Tags (Optional):**
Add these tags in W&B UI after run starts: `model:distilbert-base`, `purpose:efficiency`
