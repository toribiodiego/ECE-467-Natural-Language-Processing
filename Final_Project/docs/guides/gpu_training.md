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

Run training with your desired configuration:

```bash
python -m src.training.train \
  --model distilbert-base \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 5e-5 \
  --wandb-project GoEmotions_Classification
```

**Training will:**
1. Initialize model and move to GPU
2. Load GoEmotions dataset
3. Train for specified epochs
4. Save checkpoints to `artifacts/models/`
5. Log metrics to W&B
6. Export predictions and metrics to CSVs
7. Upload all outputs to W&B Files tab

## Common Training Configurations

### DistilBERT (Fast, Less Memory)

```bash
python -m src.training.train \
  --model distilbert-base \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 5e-5
```

**Resources:**
- GPU Memory: ~4-6 GB
- Training Time: ~2-3 hours (5 epochs on full dataset)
- Expected Test AUC: ~0.60-0.65

### RoBERTa-Base (Balanced)

```bash
python -m src.training.train \
  --model roberta-base \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 3e-5
```

**Resources:**
- GPU Memory: ~6-8 GB
- Training Time: ~3-4 hours (5 epochs)
- Expected Test AUC: ~0.62-0.67

### RoBERTa-Large (Best Performance)

```bash
python -m src.training.train \
  --model roberta-large \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 2e-5
```

**Resources:**
- GPU Memory: ~12-16 GB
- Training Time: ~6-8 hours (5 epochs)
- Expected Test AUC: ~0.65-0.70

## Running in Background

To keep training running after disconnecting from SSH:

### Option 1: tmux (Recommended)

```bash
# Start tmux session
tmux new -s training

# Inside tmux, start training
source venv/bin/activate
set -a && source .env && set +a
python -m src.training.train --model distilbert-base --epochs 5

# Detach from tmux: Ctrl+B, then D

# Reattach later
tmux attach -t training
```

### Option 2: nohup

```bash
nohup python -m src.training.train \
  --model distilbert-base \
  --epochs 5 \
  > training.log 2>&1 &

# Check progress
tail -f training.log

# Check process
ps aux | grep train
```

### Option 3: screen

```bash
# Start screen session
screen -S training

# Inside screen, start training
source venv/bin/activate
set -a && source .env && set +a
python -m src.training.train --model distilbert-base --epochs 5

# Detach from screen: Ctrl+A, then D

# Reattach later
screen -r training
```

## Monitoring Training

### Check GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Single check
nvidia-smi
```

### Check W&B Dashboard

1. Go to https://wandb.ai
2. Navigate to `GoEmotions_Classification` project
3. Click on your run
4. View real-time metrics in Charts tab

### Check Local Logs

```bash
# If using nohup
tail -f training.log

# If using tmux/screen
# Just reattach to the session
```

## Retrieving Results

All outputs are automatically uploaded to W&B Files tab. You can:

1. **Download from W&B UI:**
   - Go to run page
   - Click "Files" tab
   - Download checkpoint, predictions, or metrics

2. **Download via Python API:**
   ```python
   import wandb

   api = wandb.Api()
   run = api.run('username/GoEmotions_Classification/run_id')

   # Download all files
   for file in run.files():
       file.download(root='./downloads')
   ```

3. **Copy from GPU instance:**
   ```bash
   # From your local machine
   scp -r user@gpu-instance:/path/to/Final_Project/artifacts ./
   ```

See [downloading_files.md](../tools/wandb/downloading_files.md) for detailed download instructions.

## Troubleshooting

### CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
Reduce batch size:
```bash
python -m src.training.train --model distilbert-base --batch-size 8
```

Or use gradient accumulation:
```bash
python -m src.training.train --model distilbert-base --batch-size 8 --gradient-accumulation-steps 2
```

### W&B Login Failed

**Error:**
```
wandb: ERROR Unable to authenticate
```

**Solution:**
```bash
# Method 1: Re-login
wandb login
# Paste your API key

# Method 2: Check .env file
cat .env
# Verify WANDB_API_KEY is correct

# Method 3: Export directly
export WANDB_API_KEY=your_key_here
```

### Dataset Download Failed

**Error:**
```
ConnectionError: Couldn't reach the Hugging Face Hub
```

**Solution:**
```bash
# Retry setup
./setup.sh

# Or manually install datasets
pip install datasets
python -c "from datasets import load_dataset; load_dataset('google-research-datasets/go_emotions', 'simplified')"
```

### ImportError: No module named 'transformers'

**Error:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Training Crashes Mid-Run

**Recovery:**

Training automatically saves checkpoints. If interrupted, you can resume from the latest checkpoint (feature not yet implemented, but checkpoints are saved).

Current best practice:
1. Check W&B to see which epoch completed
2. Restart training with remaining epochs
3. All completed epochs have been logged to W&B

## Advanced Configuration

### Custom Hyperparameters

```bash
python -m src.training.train \
  --model roberta-base \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --warmup-steps 500 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --threshold 0.3 \
  --early-stopping-patience 3
```

### Subset Training (Testing)

```bash
# Quick test with small dataset
python -m src.training.train \
  --model distilbert-base \
  --epochs 2 \
  --max-train-samples 1000 \
  --max-eval-samples 500 \
  --batch-size 8
```

### Multiple Runs in Sequence

```bash
#!/bin/bash
# train_all.sh

source venv/bin/activate
set -a && source .env && set +a

# Train DistilBERT
python -m src.training.train --model distilbert-base --epochs 5

# Train RoBERTa-Base
python -m src.training.train --model roberta-base --epochs 5

# Train RoBERTa-Large
python -m src.training.train --model roberta-large --epochs 5 --batch-size 8
```

Run with:
```bash
chmod +x train_all.sh
nohup ./train_all.sh > all_training.log 2>&1 &
```

## Performance Tips

1. **Use larger batch sizes** when GPU memory allows (faster training)
2. **Use mixed precision** for faster training (automatically enabled if supported)
3. **Monitor GPU utilization** with `nvidia-smi` - aim for 90%+ usage
4. **Use gradient accumulation** if memory-constrained
5. **Enable cudnn benchmarking** (already enabled in training script)

## Cost Estimation

Approximate costs for common GPU instances (as of 2025):

| GPU Type | $/hour | DistilBERT (5 epochs) | RoBERTa-Large (5 epochs) |
|----------|--------|----------------------|--------------------------|
| NVIDIA T4 | $0.35 | ~$1.05 (3 hours) | ~$3.50 (10 hours) |
| NVIDIA V100 | $2.50 | ~$5.00 (2 hours) | ~$15.00 (6 hours) |
| NVIDIA A100 | $3.00 | ~$6.00 (2 hours) | ~$18.00 (6 hours) |

## Next Steps

After training completes:

1. **Review results in W&B:**
   - Compare metrics across runs
   - Identify best checkpoint
   - Download predictions for analysis

2. **Run ablation studies:**
   - See `../ablation_studies/README.md`
   - Compare different configurations
   - Analyze per-emotion performance

3. **Download artifacts:**
   - See [downloading_files.md](../tools/wandb/downloading_files.md)
   - Retrieve checkpoints and predictions
   - Perform local analysis

## Additional Resources

- **Replication Guide:** [replication.md](replication.md)
- **W&B Integration:** [../tools/wandb/README.md](../tools/wandb/README.md)
- **Model Performance:** [../results/model_performance.md](../results/model_performance.md)
- **Design Decisions:** [../reference/design_decisions.md](../reference/design_decisions.md)
