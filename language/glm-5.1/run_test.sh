#!/bin/bash
# Run evaluation script for GLM-5.1
# Usage: ./run_test.sh [--async]

set -e

cd "$(dirname "$0")"

# --- Configuration ---
export MLPERF_BACKEND="openai"
export OPENAI_API_BASE="https://api.deepseek.com"
export OPENAI_API_KEY="YourKey"
export OPENAI_MODEL="deepseek-v4-flash"
# --------------------

echo "========================================="
echo "GLM-5.1 Evaluation Test"
echo "========================================="
echo "Backend:    $MLPERF_BACKEND"
echo "API Base:   $OPENAI_API_BASE"
echo "Model:      $OPENAI_MODEL"
echo "Mode:       $([ "$1" = "--async" ] && echo Async || echo Sync)"
echo "========================================="

# Generate test dataset if not exists
if [ ! -f "test_input.pkl" ]; then
    echo "Generating test dataset..."
    uv run python gen_test_dataset.py
    echo ""
fi

# Run evaluation
echo "Starting evaluation..."
if [ "$1" = "--async" ]; then
    uv run python run_eval.py --input-file test_input.pkl --async
else
    uv run python run_eval.py --input-file test_input.pkl
fi

echo ""
echo "Done!"
