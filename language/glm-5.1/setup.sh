#!/bin/bash
# Setup script for GLM-5.1 MLPerf environment using uv
# Automatically detects the backend and sets up the virtual environment

set -e

BACKEND=${MLPERF_BACKEND:-"openai"}
VENV_DIR=".venv_${BACKEND}"

echo "========================================="
echo "GLM-5.1 MLPerf Setup (uv)"
echo "========================================="
echo "Backend: ${BACKEND}"
echo "Virtual environment: ${VENV_DIR}"
echo "========================================="

# Check if uv is installed
if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. Install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  # or: pip install uv"
    exit 1
fi

# Create virtual environment with uv (respects .python-version)
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment with uv..."
    uv venv "${VENV_DIR}"
fi

# Install dependencies with uv
echo "Installing dependencies..."
uv pip install --python "${VENV_DIR}/bin/python" \
    openai httpx pandas numpy tqdm

# Install tokenizer dependencies (optional, disabled by default)
# Uncomment the following line if you need transformers/torch for tokenization:
# uv pip install --python "${VENV_DIR}/bin/python" transformers torch

# Install MLPerf LoadGen if available (optional dependency)
echo ""
echo "Installing MLPerf LoadGen (optional)..."
if uv pip install --python "${VENV_DIR}/bin/python" mlperf-loadgen 2>/dev/null; then
    echo "MLPerf LoadGen installed successfully"
else
    echo "Warning: mlperf_loadgen not available via pip."
    echo "If needed, install it manually from the MLPerf inference repository:"
    echo "  pip install <path-to-inference>/loadgen"
fi

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Or use uv directly:"
echo "  uv run --python ${VENV_DIR}/bin/python run_eval.py ..."
echo ""
echo "Run evaluation:"
echo "  python run_eval.py --input-file <dataset.pkl>"
echo ""
echo "Run MLPerf benchmark:"
echo "  python run_mlperf.py --mode offline --input-file <dataset.pkl>"
echo "========================================="
