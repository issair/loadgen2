#!/bin/bash
# Run SessionSUT benchmark test for GLM-5.1 (Path B)
# Usage: ./run_test_session.sh [--accuracy]
#
# Differences from run_test_mlperf.sh:
#   - Uses SessionSUT instead of ServerSUT/OfflineSUT
#   - Self-managed scheduling with N concurrent slots
#   - Per-VU Poisson pacing via LargeModelSession
#   - No dependency_replay — SUT handles chains internally

set -e

cd "$(dirname "$0")"

# ── Configuration ────────────────────────────────────────────────────────────
export MLPERF_BACKEND="openai"
export OPENAI_API_BASE=""
export OPENAI_API_KEY="YourKeys"
export OPENAI_MODEL="Mock-Model"

export OPENAI_API_BASE="http://localhost:21000"
export OPENAI_API_KEY="Mock-Model"
export OPENAI_MODEL="Mock-Model"

# export OPENAI_API_BASE="http://10.188.128.12:30100/api/ai-gateway/v1"
# export OPENAI_API_BASE="https://api.deepseek.com/v1"
# export OPENAI_API_BASE="http://120.52.65.181:5000/v1"
export OPENAI_API_BASE="http://127.0.0.1:21000/v1"
export OPENAI_API_KEY="Your-Key"
export OPENAI_MODEL="Mock-Model"
export MLPERF_MAX_OSL=66
# LLM_MODEL=deepseek-chat
# LLM_BINDING_HOST=https://api.deepseek.com
# export MLPERF_MAX_OSL=2

# mlperf_loadgen C++ extension requires GLIBCXX_3.4.30+
MLPERF_CONF="../../mlperf.conf"

# ── Data ─────────────────────────────────────────────────────────────────────
# Point to the preprocessed conversation JSON file or directory.
# This file/dir should contain multi-turn conversations in the format
# expected by process_dataset() (see data_tools/split_andmapping.py).
INPUT_DATA="/home/ldx/mlperf/inference/download/codex_swebenchpro"
# INPUT_DATA="data/conversations/"

# ── SessionSUT settings ──────────────────────────────────────────────────────
MAX_TRIGGER=1         # max trigger (= max concurrent VU turns)
POISSON_LAM=7000         # Poisson lambda for inter-turn intervals (unit: --time-unit)
POISSON_SEED=420000       # RNG seed for reproducible Poisson intervals
TIME_UNIT="s"         # "s" for seconds, "ms" for milliseconds
#MAX_TRACE=400         # uncomment to limit virtual users, e.g. MAX_TRACE=100

# ── Logging ──────────────────────────────────────────────────────────────────
echo "========================================="
echo "GLM-5.1 SessionSUT Benchmark (Path B)"
echo "========================================="
echo "Backend:       $MLPERF_BACKEND"
echo "API Base:      $OPENAI_API_BASE"
echo "Model:         $OPENAI_MODEL"
echo "Input data:    $INPUT_DATA"
echo "Slots:         $MAX_TRIGGER"
echo "Poisson lam:   $POISSON_LAM"
echo "Poisson seed:  $POISSON_SEED"
echo "Time unit:     $TIME_UNIT"
echo "MLPerf conf:   $MLPERF_CONF"
echo "User conf:     $USER_CONF"
echo "Accuracy:      $([ "x$1" = "x--accuracy" ] && echo Yes || echo No)"
echo "========================================="

export HF_ENDPOINT="https://hf-mirror.com"

# Save script argument before set -- overwrites positional parameters.
_accuracy_flag="$1"

# ── Build command ────────────────────────────────────────────────────────────
set -- \
    uv run python run_session_mlperf.py \
    --input "$INPUT_DATA" \
    --output-dir "traces_mlperf_results" \
    --max-trigger "$MAX_TRIGGER" \
    --poisson-lam "$POISSON_LAM" \
    --poisson-seed "$POISSON_SEED" \
    --time-unit "$TIME_UNIT" \
    --mlperf-conf "$MLPERF_CONF" 

if [ -n "$USER_CONF" ] && [ -f "$USER_CONF" ]; then
    set -- "$@" --user-conf "$USER_CONF"
fi

if [ -n "${MAX_TRACE:-}" ]; then
    set -- "$@" --max-trace "$MAX_TRACE"
fi

if [ "x${_accuracy_flag}" = "x--accuracy" ]; then
    set -- "$@" --accuracy
fi

# ── Run ──────────────────────────────────────────────────────────────────────
echo "Starting SessionSUT benchmark..."
echo "$@"
echo ""

"$@"

echo ""
echo "Done! Results in traces_mlperf_results/"
