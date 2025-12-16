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

## Colab Auto-Disconnect Setup

If you're running training on Google Colab, set up the auto-disconnect watcher to prevent wasted compute units after training completes.

### How It Works

1. The training script creates a flag file (`/content/__DISCONNECT__`) when training finishes
2. A notebook watcher cell monitors this flag file
3. When detected, the watcher automatically disconnects the runtime using `runtime.unassign()`

### Setup Instructions

**Create a new code cell at the TOP of your Colab notebook** and run it before starting training:

```python
# Colab Runtime Auto-Disconnect Watcher
# Run this cell FIRST, then run your training command in a separate cell

import time
import os
from pathlib import Path

FLAG_FILE = "/content/__DISCONNECT__"

print("Starting Colab runtime watcher...")
print(f"Monitoring flag file: {FLAG_FILE}")
print("Training can now be started in another cell.")
print("-" * 60)

def watch_for_disconnect():
    """Monitor flag file and disconnect runtime when detected."""
    while True:
        if os.path.exists(FLAG_FILE):
            print("\n" + "=" * 60)
            print("DISCONNECT FLAG DETECTED!")
            print("=" * 60)
            print(f"Flag file found at: {FLAG_FILE}")

            # Read timestamp from flag file
            try:
                with open(FLAG_FILE, 'r') as f:
                    content = f.read().strip()
                    print(f"Content: {content}")
            except Exception as e:
                print(f"Could not read flag file: {e}")

            print("\nDisconnecting Colab runtime in 5 seconds...")
            print("This prevents wasted compute units after training completes.")
            time.sleep(5)

            # Disconnect the runtime
            from google.colab import runtime
            runtime.unassign()
            break

        # Check every 30 seconds
        time.sleep(30)

# Start watching (this will run indefinitely until flag is detected)
watch_for_disconnect()
```

**Usage:**
1. Run the watcher cell above FIRST
2. In a SEPARATE cell, run your training command with `--colab` flag
3. The watcher will automatically disconnect the runtime when training completes

**Example training command** (run in separate cell):
```bash
python -m src.training.train \
  --model distilbert-base \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 2 \
  --dropout 0.1 \
  --colab \
  --output-dir artifacts/models/distilbert-test
```

---

## Production Training Commands

**Important:** All commands below include the `--colab` flag for automatic runtime disconnect. This flag works together with the watcher cell (see "Colab Auto-Disconnect Setup" above):

1. **Watcher cell** monitors `/content/__DISCONNECT__` file
2. **Training command** with `--colab` flag creates the file when training completes
3. **Automatic disconnect** happens when watcher detects the flag file

If you're **NOT** using Colab, you can safely omit the `--colab` flag - the script will detect non-Colab environments and skip disconnect logic.

---

## RoBERTa-Large Production Training

**Exact command for production RoBERTa-Large training:**

```bash
python -m src.training.train \
  --model roberta-large \
  --lr 2e-5 \
  --batch-size 16 \
  --epochs 10 \
  --dropout 0.1 \
  --colab
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
  --dropout 0.1 \
  --colab
```

---

## Quick Test Training (Verify Auto-Disconnect)

**Objective:** Run a fast training test to verify the Colab auto-disconnect feature works correctly before starting longer training jobs.

**Why run this first:**
- Completes in ~3-5 minutes
- Verifies watcher cell detects the flag file
- Confirms runtime disconnects automatically
- No wasted compute if setup is incorrect

**Prerequisites:**
1. Watcher cell is running (see "Colab Auto-Disconnect Setup" above)
2. Virtual environment is activated
3. W&B is configured

**Test command** (copy-paste ready):

```bash
python -m src.training.train \
  --model distilbert-base \
  --lr 3e-5 \
  --batch-size 32 \
  --max-epochs 2 \
  --max-train-samples 100 \
  --max-eval-samples 50 \
  --dropout 0.1 \
  --colab \
  --output-dir artifacts/models/distilbert-test
```

**What to expect:**
1. Training starts and runs for 2 epochs on 100 samples (~2-3 minutes)
2. Training completes and logs "Training complete" message
3. Script creates flag file at `/content/__DISCONNECT__`
4. Watcher cell detects the flag and prints "DISCONNECT FLAG DETECTED!"
5. Runtime disconnects automatically after 5 second countdown

**If disconnect doesn't happen:**
- Check watcher cell is still running (not stopped/errored)
- Verify flag file exists: `!ls -la /content/__DISCONNECT__`
- Check watcher cell output for errors
- Ensure `--colab` flag is included in training command

---

## DistilBERT Multi-Seed Robustness Training (Task 08)

**Objective:** Train DistilBERT with 3 different random seeds to quantify variance for statistical significance testing.

### Seed 13

```bash
python -m src.training.train \
  --model distilbert-base \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 10 \
  --dropout 0.1 \
  --seed 13 \
  --wandb-project GoEmotions_Classification \
  --colab \
  --output-dir artifacts/models/distilbert-seed13
```

### Seed 23

```bash
python -m src.training.train \
  --model distilbert-base \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 10 \
  --dropout 0.1 \
  --seed 23 \
  --wandb-project GoEmotions_Classification \
  --colab \
  --output-dir artifacts/models/distilbert-seed23
```

### Seed 0

```bash
python -m src.training.train \
  --model distilbert-base \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 10 \
  --dropout 0.1 \
  --seed 0 \
  --wandb-project GoEmotions_Classification \
  --colab \
  --output-dir artifacts/models/distilbert-seed0
```

---

## Loss Function Experiments

**Objective:** Compare different loss functions to improve rare-label performance.

### Baseline BCE Loss

```bash
python -m src.training.train \
  --model distilbert-base \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 10 \
  --dropout 0.1 \
  --loss-type bce \
  --wandb-project GoEmotions_Classification \
  --colab \
  --output-dir artifacts/models/distilbert-loss-baseline
```

### Weighted BCE Loss

```bash
python -m src.training.train \
  --model distilbert-base \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 10 \
  --dropout 0.1 \
  --loss-type weighted-bce \
  --wandb-project GoEmotions_Classification \
  --colab \
  --output-dir artifacts/models/distilbert-loss-weighted
```

### Focal Loss

```bash
python -m src.training.train \
  --model distilbert-base \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 10 \
  --dropout 0.1 \
  --loss-type focal \
  --focal-gamma 2.0 \
  --wandb-project GoEmotions_Classification \
  --colab \
  --output-dir artifacts/models/distilbert-loss-focal
```

---

## Sequence Length Experiments

**Objective:** Optimize max_seq_length for the best performance/efficiency trade-off.

### max_seq_length=128

```bash
python -m src.training.train \
  --model distilbert-base \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 10 \
  --dropout 0.1 \
  --max-seq-length 128 \
  --wandb-project GoEmotions_Classification \
  --colab \
  --output-dir artifacts/models/distilbert-seq128
```

### max_seq_length=256

```bash
python -m src.training.train \
  --model distilbert-base \
  --lr 3e-5 \
  --batch-size 32 \
  --epochs 10 \
  --dropout 0.1 \
  --max-seq-length 256 \
  --wandb-project GoEmotions_Classification \
  --colab \
  --output-dir artifacts/models/distilbert-seq256
```

### max_seq_length=512

```bash
python -m src.training.train \
  --model distilbert-base \
  --lr 3e-5 \
  --batch-size 16 \
  --epochs 10 \
  --dropout 0.1 \
  --max-seq-length 512 \
  --wandb-project GoEmotions_Classification \
  --colab \
  --output-dir artifacts/models/distilbert-seq512
```

**Note:** Reduced batch size to 16 to fit in GPU memory.

