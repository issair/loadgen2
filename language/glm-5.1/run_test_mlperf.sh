#!/bin/bash
# Run MLPerf benchmark test for GLM-5.1
# Usage: ./run_test_mlperf.sh [--accuracy]

set -e

cd "$(dirname "$0")"

# --- Configuration ---
export MLPERF_BACKEND="openai"
export OPENAI_API_BASE="http://10.188.128.12:30100/api/ai-gateway/v1"
# export OPENAI_API_BASE="https://api.deepseek.com"
# export OPENAI_API_BASE="http://120.52.65.181:5000/v1"
# export OPENAI_API_BASE="http://127.0.0.1:21000/v1"
export OPENAI_API_KEY="1762841707187z7PvZvDGVtEmuRVsGhEqEiS"
export OPENAI_MODEL="GLM-4.5V"
export MLPERF_MAX_OSL=256

# Fix libstdc++ version issue - bypass anaconda's old libstdc++.so.6
# mlperf_loadgen C++ extension requires GLIBCXX_3.4.30+

MLPERF_CONF="../../mlperf.conf"
USER_CONF="mlperf/user.conf"

echo "========================================="
echo "GLM-5.1 MLPerf Benchmark Test"
echo "========================================="
echo "Backend:      $MLPERF_BACKEND"
echo "API Base:     $OPENAI_API_BASE"
echo "Model:        $OPENAI_MODEL"
echo "MLPerf conf:  $MLPERF_CONF"
echo "User conf:    $USER_CONF"
echo "Accuracy:     $([ "x$1" = "x--accuracy" ] && echo Yes || echo No)"
echo "========================================="
export HF_ENDPOINT="https://hf-mirror.com"

# Generate test dataset if not exists
# if [ ! -f "test_input.pkl" ]; then
#     echo "Generating test dataset..."
#     uv run python gen_test_dataset.py
#     echo ""
# fi

# Run MLPerf benchmark
echo "Starting MLPerf benchmark..."
if [ "x$1" = "x--accuracy" ]; then
    uv run python run_mlperf.py \
        --input-file test_input.pkl \
        --mode server \
        --mlperf-conf "$MLPERF_CONF" \
        --user-conf "$USER_CONF" \
        --accuracy
else
    uv run python run_mlperf.py \
        --input-file preprocess.json \
        --mode server \
        --mlperf-conf "$MLPERF_CONF" \
        --user-conf "$USER_CONF"
fi

echo ""
echo "Done!"
