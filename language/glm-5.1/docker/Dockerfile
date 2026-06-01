# =============================================================================
#  Dockerfile — GLM-5.1 MLPerf Inference (with MLPerf LoadGen C++ bindings)
#
#  Build context MUST be the `inference/` root so that both
#    language/glm-5.1/   (this project)
#    loadgen/            (mlcommons-loadgen local dependency)
#  are visible.
#
#  Usage:
#    cd <path-to-inference>/
#    docker build -f language/glm-5.1/docker/Dockerfile -t glm-5.1-mlperf .
#
#  Run:
#    docker run --rm -it \
#        -v /path/to/data.json:/data/data.json \
#        glm-5.1-mlperf run_session_mlperf.py --input /data/data.json
# =============================================================================

FROM python:3.12-slim

# ------------------------------------------------------------------
#  1. System build tools (needed for loadgen C++ extension)
# ------------------------------------------------------------------
# 使用清华 TUNA 镜像加速 apt 下载
RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/debian.sources \
    && sed -i 's@security.debian.org@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    g++ gcc make cmake python3-dev curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
#  2. Install uv package manager
# ------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ------------------------------------------------------------------
#  3. Copy source code
#     Context = inference/  (parent of language/ and loadgen/)
# ------------------------------------------------------------------
WORKDIR /workspace

# Copy project + loadgen (local path dependency)
COPY language/glm-5.1/   ./language/glm-5.1/
COPY language/deepseek-r1/   ./language/deepseek-r1/
COPY loadgen/            ./loadgen/

# ------------------------------------------------------------------
#  4. Build & install Python dependencies (including the C++ loadgen)
# ------------------------------------------------------------------
WORKDIR /workspace/language/glm-5.1

# 使用清华 TUNA PyPI 镜像加速 uv 下载
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# Install all dependencies including the mlperf extras and the local
# mlcommons-loadgen path dependency (compiled from C++ via pybind11).
RUN uv sync --all-extras
RUN cd /workspace/language/deepseek-r1 && uv sync --all-extras

# ------------------------------------------------------------------
#  4.5 Pre-download tokenizer to local directory (avoids network at runtime)
# ------------------------------------------------------------------
ENV HF_ENDPOINT=https://hf-mirror.com
RUN cd /workspace/language/deepseek-r1 && \
    uv run python download_tokenizer.py --local-dir /workspace/tokenizer

# Prepend venv to PATH so that `python` and `uv run` find the installed
# packages without needing to activate.
ENV PATH="/workspace/language/glm-5.1/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    MLPERF_BACKEND=sglang \
    HF_ENDPOINT=https://hf-mirror.com \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZER_PATH=/workspace/tokenizer



# ------------------------------------------------------------------
#  5. Entrypoint
# ------------------------------------------------------------------
COPY language/glm-5.1/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["glm-5.1", "run_session_mlperf.py", "--help"]
