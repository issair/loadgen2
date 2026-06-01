# MLPerf Inference GLM-5.1 Reference Implementation

## Overview

This is the MLPerf inference reference implementation for **GLM-5.1** using an **OpenAI-compatible API backend**.

The backend connects to an externally deployed OpenAI-compatible API server (e.g., vLLM, SGLang, or the official GLM API service) and does **not** manage the server lifecycle. This allows you to test any model served via an OpenAI-compatible API.

## Architecture

```
glm-5.1/
├── backends/
│   ├── __init__.py           # Backend module init
│   ├── base_backend.py       # Abstract base class
│   ├── openai_backend.py     # OpenAI-compatible API backend
│   └── utils.py              # Backend utilities
├── mlperf/
│   ├── __init__.py           # MLPerf module init
│   ├── base_sut.py           # Base SUT class
│   ├── offline_sut.py        # Offline scenario SUT
│   ├── server_sut.py         # Server scenario SUT (streaming)
│   ├── qsl.py                # QuerySampleLibrary
│   ├── user.conf             # MLPerf user configuration
│   └── utils.py              # MLPerf dataset utilities
├── utils/
│   ├── __init__.py           # Utils module init
│   ├── backend_registry.py   # Backend configuration registry
│   ├── data_utils.py         # Dataset loading/saving
│   ├── error_handling.py     # Error handling utilities
│   ├── runner_utils.py       # Runner argument parsers
│   ├── tokenization.py       # Tokenizer utilities
│   └── validation.py         # Input validation
├── run_mlperf.py             # MLPerf benchmark runner
├── run_eval.py               # Evaluation runner
├── eval_accuracy.py          # Accuracy evaluation
└── README.md                 # This file
```

## Prerequisites

- Python 3.10+
- `mlperf_loadgen` installed
- `openai` Python package
- An OpenAI-compatible API endpoint (e.g., vLLM server, SGLang server, GLM API)

## Setup

```bash
# Install dependencies
pip install openai httpx pandas numpy tqdm transformers mlperf_loadgen
```

## Configuration

Set the backend environment variable:

```bash
export MLPERF_BACKEND=openai
```

Configure the API endpoint by setting environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
# Or use the GLM-specific env var:
export GLM_API_KEY="your-api-key"
```

The default configuration in `utils/backend_registry.py`:
- `base_url`: `http://localhost:8000/v1`
- `model`: `glm-5-1`
- `max_tokens`: 8192
- `temperature`: 0.0
- `max_concurrent_requests`: 64

Override these by setting environment variables or modifying the registry.

## Usage

### Evaluation Mode

```bash
# Sync evaluation
python run_eval.py --input-file <dataset.pkl>

# Async evaluation
python run_eval.py --input-file <dataset.pkl> --async

# Limit number of samples
python run_eval.py --input-file <dataset.pkl> --num-samples 100
```

### MLPerf Benchmark Mode

```bash
# Offline scenario (performance)
python run_mlperf.py --mode offline --input-file <dataset.pkl>

# Server scenario
python run_mlperf.py --mode server --input-file <dataset.pkl>

# Accuracy mode
python run_mlperf.py --mode offline --accuracy --input-file <dataset.pkl>
```

### Accuracy Evaluation

```bash
# Standalone accuracy evaluation
python eval_accuracy.py --input-file results.pkl
```

## Backend Configuration

To use a different API endpoint or model, either:

1. **Environment variables** (at runtime):
   ```bash
   export OPENAI_API_BASE="https://your-api-endpoint/v1"
   export OPENAI_API_KEY="your-key"
   ```

2. **Modify the registry** in `utils/backend_registry.py`:
   ```python
   BACKEND_REGISTRY['openai']['config']['base_url'] = 'https://your-api-endpoint/v1'
   BACKEND_REGISTRY['openai']['config']['model'] = 'your-model-name'
   ```

## Backend Support Matrix

| Backend | `run_eval.py` | `run_mlperf.py --mode=offline` | `run_mlperf.py --mode=server` |
|---------|---------------|-------------------------------|------------------------------|
| openai  | x             | x                             | x (streaming)                |
