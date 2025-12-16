# GPU Training Guide

Run training on a remote or local GPU. For full end-to-end replication (including CPU smoke tests), see `docs/guides/replication.md`.

## Table of Contents
1. [Clone & prerequisites](#clone--prerequisites)
2. [Setup & dependencies](#setup--dependencies)
3. [Configure Weights & Biases](#configure-weights--biases)
4. [Activate venv & verify GPU](#activate-venv--verify-gpu)
5. [Run training commands](#run-training-commands)

## Clone & prerequisites

```bash
git clone https://github.com/toribiodiego/ECE-467-Natural-Language-Processing.git
cd ECE-467-Natural-Language-Processing/Final_Project
```

Verify you're in the project root (should contain `README.md`, `setup.sh`, `src/`, `docs/`, etc.).

## Setup & dependencies

Run the automated setup (same as replication guide):

```bash
chmod +x setup.sh
./setup.sh
```

This creates `venv/`, installs requirements, and caches the GoEmotions dataset if possible. For detailed steps, see `docs/guides/replication.md#environment-setup`.

## Configure Weights & Biases

Create `.env` with your W&B API key (and any entity/project overrides if desired):

```bash
cat > .env << 'EOF'
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=GoEmotions_Classification
# WANDB_ENTITY=your_team_or_username   # optional
EOF
```

Load the env vars before training:

```bash
set -a
source .env
set +a
```

See `docs/tools/wandb/README.md` for login, artifact handling, and metrics conventions.

## Activate venv & verify GPU

```bash
source venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expect `CUDA available: True` on a GPU machine.

## Run training commands

Recommended runs (use `--wandb-project`/`--wandb-entity` as needed):

- **RoBERTa-Large (best performance):**
  ```bash
  python -m src.training.train \
    --model roberta-large \
    --lr 2e-5 \
    --batch-size 16 \
    --epochs 4 \
    --dropout 0.1
  ```

- **DistilBERT (best efficiency):**
  ```bash
  python -m src.training.train \
    --model distilbert-base \
    --lr 3e-5 \
    --batch-size 32 \
    --epochs 4 \
    --dropout 0.1
  ```

Thresholding options (multi-label):  
`--threshold-strategy global --threshold 0.5` (default), `per_class` (search per label), or `top_k` with `--top-k`. Match the strategy used in `docs/results/model_performance.md` when comparing results.
