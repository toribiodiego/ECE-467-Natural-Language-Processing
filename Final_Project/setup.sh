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
set -u  # Exit on undefined variable

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Load environment variables from .env file
if [ -f .env ]; then
    log_info "Loading environment variables from .env"
    export $(grep -v '^#' .env | xargs)
else
    log_warn ".env file not found. Creating from .env.example"
    if [ -f .env.example ]; then
        cp .env.example .env
        export $(grep -v '^#' .env | xargs)
        log_info "Created .env from template. You can customize it if needed."
    else
        log_error ".env.example not found. Cannot proceed."
        exit 1
    fi
fi

# Set defaults if not specified in .env
VENV_DIR=${VENV_DIR:-venv}
PYTHON_CMD=${PYTHON_CMD:-python3}

log_info "Starting environment setup for GoEmotions Project"
log_info "Virtual environment directory: $VENV_DIR"

# Check Python version
log_info "Checking Python version..."
if ! command -v $PYTHON_CMD &> /dev/null; then
    log_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
log_info "Found Python $PYTHON_VERSION"

# Check if version is >= 3.8
MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$MAJOR_VERSION" -lt 3 ] || ([ "$MAJOR_VERSION" -eq 3 ] && [ "$MINOR_VERSION" -lt 8 ]); then
    log_error "Python 3.8 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    log_warn "Virtual environment already exists at $VENV_DIR"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        log_info "Skipping virtual environment creation"
        SKIP_VENV_CREATION=true
    fi
fi

if [ "${SKIP_VENV_CREATION:-false}" != "true" ]; then
    log_info "Creating virtual environment in $VENV_DIR..."
    $PYTHON_CMD -m venv "$VENV_DIR"

    if [ $? -ne 0 ]; then
        log_warn "Standard venv creation failed, trying without pip (common in Colab)..."
        $PYTHON_CMD -m venv "$VENV_DIR" --without-pip

        if [ $? -ne 0 ]; then
            log_error "Failed to create virtual environment"
            exit 1
        fi

        log_info "Virtual environment created without pip"
        NEED_PIP_INSTALL=true
    else
        log_info "Virtual environment created successfully"
        NEED_PIP_INSTALL=false
    fi
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install pip if needed
if [ "${NEED_PIP_INSTALL:-false}" = "true" ]; then
    log_info "Installing pip via get-pip.py..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    $PYTHON_CMD /tmp/get-pip.py
    rm -f /tmp/get-pip.py
    log_info "Pip installed successfully"
fi

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
if [ -f requirements.txt ]; then
    log_info "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt

    if [ $? -ne 0 ]; then
        log_error "Failed to install dependencies"
        exit 1
    fi

    log_info "Dependencies installed successfully"
else
    log_warn "requirements.txt not found. Skipping dependency installation."
    log_warn "You will need to create requirements.txt and run: pip install -r requirements.txt"
fi

# Validate installation
log_info "Validating installation..."

# Check critical packages
REQUIRED_PACKAGES=("datasets" "transformers" "torch" "sklearn" "pandas" "matplotlib" "numpy")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import ${package//-/_}" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    log_info "All required packages installed successfully ✓"
else
    log_warn "Some packages could not be validated: ${MISSING_PACKAGES[*]}"
    log_warn "This might be expected if requirements.txt doesn't exist yet"
fi

# Print summary
echo
log_info "=========================================="
log_info "Setup completed successfully!"
log_info "=========================================="
echo
echo "To activate the virtual environment:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "To deactivate:"
echo "  deactivate"
echo
echo "To run analysis scripts:"
echo "  source $VENV_DIR/bin/activate"
echo "  python -m src.data.multilabel_stats"
echo

# Create activation helper script
cat > activate.sh << 'ACTIVATION_SCRIPT'
#!/usr/bin/env bash
# Quick activation helper
source venv/bin/activate
echo "Virtual environment activated!"
ACTIVATION_SCRIPT

chmod +x activate.sh
log_info "Created activate.sh helper script for quick activation"
