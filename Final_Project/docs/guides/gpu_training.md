# GPU Training Guide

This guide covers setting up and running model training on a remote GPU instance.

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

### Step 3: Configure Weights & Biases

**Get your W&B API key:**
1. Go to https://wandb.ai/settings
2. Copy your API key from the "API keys" section

**Create a `.env` file with your W&B API key:**

```bash
cat > .env << 'EOF'
WANDB_API_KEY=your_wandb_api_key_here
EOF
```

**Important:** Replace `your_wandb_api_key_here` with your actual API key from the W&B settings page.


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

## RoBERTa-Large Production Training

**Exact command for production RoBERTa-Large training:**

```bash
python -m src.training.train \
  --model roberta-large \
  --lr 2e-5 \
  --batch-size 16 \
  --epochs 10 \
  --dropout 0.1
```

---

## DistilBERT Efficiency Baseline Training

**Exact command for DistilBERT efficiency baseline:**

```bash
python -m src.training.train \
  --model distilbert-base \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 10 \
  --dropout 0.1
```

---

## DistilBERT Multi-Seed Robustness Training (Task 08)

**Objective:** Train DistilBERT with 3 different random seeds to quantify variance for statistical significance testing.

**Why DistilBERT?** 6.6x faster training than RoBERTa-Large (3 × 0.31 hrs = ~1 hour vs 3 × 2.05 hrs = ~6 hours), while providing statistically valid variance estimates.

### Seed 42

```bash
python -m src.training.train \
  --model distilbert-base-uncased \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 10 \
  --dropout 0.1 \
  --seed 42 \
  --wandb-project GoEmotions_Classification \
  --wandb-tags robustness,seed42 \
  --output-dir artifacts/models/distilbert-seed42
```

**Expected training time:** ~20 minutes

### Seed 43

```bash
python -m src.training.train \
  --model distilbert-base-uncased \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 10 \
  --dropout 0.1 \
  --seed 43 \
  --wandb-project GoEmotions_Classification \
  --wandb-tags robustness,seed43 \
  --output-dir artifacts/models/distilbert-seed43
```

**Expected training time:** ~20 minutes

### Seed 44

```bash
python -m src.training.train \
  --model distilbert-base-uncased \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 10 \
  --dropout 0.1 \
  --seed 44 \
  --wandb-project GoEmotions_Classification \
  --wandb-tags robustness,seed44 \
  --output-dir artifacts/models/distilbert-seed44
```

**Expected training time:** ~20 minutes

**Total time for all 3 seeds:** ~1 hour

**Next steps after training:**
1. Download test metrics from all 3 runs from W&B
2. Aggregate metrics to compute mean ± std
3. Perform statistical significance tests comparing RoBERTa vs DistilBERT

