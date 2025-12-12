#!/usr/bin/env bash
#
# Environment Setup Script for GoEmotions Project
#
# This script automates the creation of a Python virtual environment,
# dependency installation, and validation of the setup.
#
# Usage:
#   ./setup.sh
#
# Requirements:
#   - Python 3.8 or higher
#   - pip

set -e  # Exit on error

# Set defaults
VENV_DIR=${VENV_DIR:-venv}
PYTHON_CMD=${PYTHON_CMD:-python3}

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR_VERSION" -lt 3 ] || ([ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -lt 8 ]); then
    echo "Error: Python 3.8 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "$VENV_DIR" ]; then
    rm -rf "$VENV_DIR"
fi

# Temporarily disable exit-on-error for venv creation
set +e
$PYTHON_CMD -m venv "$VENV_DIR" 2>/dev/null
VENV_STATUS=$?
set -e

if [ $VENV_STATUS -ne 0 ]; then
    # Fallback for Colab environments
    $PYTHON_CMD -m venv "$VENV_DIR" --without-pip
    source "$VENV_DIR/bin/activate"
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    $PYTHON_CMD /tmp/get-pip.py --quiet
    rm -f /tmp/get-pip.py
    pip install --upgrade pip setuptools wheel --quiet
else
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel --quiet
fi

# Install dependencies with progress bar
echo "Installing dependencies..."
if [ -f requirements.txt ]; then
    # Simple spinner while installing (since accurate progress is hard to track)
    pip install -r requirements.txt > /tmp/pip_install.log 2>&1 &
    PIP_PID=$!

    # Show spinner while pip is running
    spin='-\|/'
    i=0
    while kill -0 $PIP_PID 2>/dev/null; do
        i=$(( (i+1) %4 ))
        printf "\r  Installing packages... ${spin:$i:1}"
        sleep 0.1
    done

    # Wait for pip to finish and check exit status
    wait $PIP_PID
    if [ $? -eq 0 ]; then
        printf "\r  Installing packages... Done!\n"
    else
        printf "\r  Installing packages... Failed!\n"
        cat /tmp/pip_install.log
        exit 1
    fi
else
    echo "Error: requirements.txt not found"
    exit 1
fi

# Download dataset
echo "Downloading GoEmotions dataset..."
python -c "
from datasets import load_dataset
import sys
try:
    dataset = load_dataset('google-research-datasets/go_emotions', 'simplified')
    print('  Dataset downloaded and cached successfully')
except Exception as e:
    print(f'  Warning: Could not download dataset: {e}', file=sys.stderr)
" 2>&1 | grep -v "^Downloading" | grep -v "^Generating"

echo ""
echo "Setup complete!"
echo ""

# Check GPU availability
python -c "
import torch
print(f'GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('GPU count: 0')
    print('GPU name: CPU only')
"

echo ""
echo "Virtual environment is active. To deactivate, run: deactivate"
