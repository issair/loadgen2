# MLPerf Inference — GLM-5.1 编译配置与安装说明

本文档总结了 `inference/` 目录下与 **GLM-5.1** 基准测试相关的三个核心组件的编译、安装与配置方法：

1. [LoadGen](#1-loadgen) — MLPerf 官方负载生成器
2. [GLM-5.1 参考实现](#2-glm-51-参考实现) — 模型端 MLPerf 客户端
3. [Mock Server](#3-mock-server) — 本地模拟 OpenAI API 服务端

---

## 1. LoadGen

**路径**: `inference/loadgen`

### 概述

LoadGen 是 MLPerf Inference 的**核心负载生成与测量工具**，用 C++ 编写并提供 Python 绑定。它按 MLPerf 定义的场景（Offline / Server / SingleStream 等）和模式（Performance / Accuracy / FindPeakPerformance）生成查询流量并记录结果。

## 2. GLM-5.1 参考实现

**路径**: `inference/language/glm-5.1`

### 概述

GLM-5.1 的 MLPerf Inference 参考实现，作为 **OpenAI 兼容 API 的客户端** 工作，对接外部推理服务端（如 vLLM、SGLang、GLM API 等）。**不管理服务端生命周期**，只负责发送请求并处理响应。

```
glm-5.1/
├── backends/          # 后端抽象（openai_backend.py）
├── mlperf/            # SUT、QSL、场景实现
├── utils/             # 工具（注册、数据、验证、tokenization）
├── docker/            # Docker 配置（目前为空）
├── mock_server/       # 模拟服务端（Rust）
├── run_mlperf.py      # MLPerf 基准测试入口
├── run_eval.py        # 评估入口
├── eval_accuracy.py   # 精度评估
├── setup.sh           # 一键安装脚本（基于 uv）
├── pyproject.toml     # 项目元数据与依赖声明
└── data-process.md    # 数据处理说明
```

### 安装

```
cd language/gml-5.1

# 安装所有的包， 包括 loadgen 模块
uv sync --all-extras --refresh --reinstall-package mlcommons-loadgen
```

### 配置

通过环境变量或修改 `utils/backend_registry.py`：

| 环境变量 | 用途 | 默认值 |
|---|---|---|
| `MLPERF_BACKEND` | 后端选择 | `openai` |
| `OPENAI_API_KEY` / `GLM_API_KEY` | API 密钥 | — |
| `OPENAI_API_BASE` | API 端点地址 | `http://localhost:8000/v1` |
| `model` | 模型名称 | `glm-5-1` |
| `max_tokens` | 最大输出 token 数 | 8192 |
| `temperature` | 采样温度 | 0.0 |
| `max_concurrent_requests` | 最大并发请求数 | 64 |

### 运行

```bash
cd language/glm-5.1

# 将数据集解压到一个目录.
unzip 算力伙伴测试数据集.zip -d datasets

# 预处理一下数据, 通常只需要一次
uv run python data_tools/split_conversation.py \
    --input "datasets/extracted_files/*.jsonl" \
    -o preprocess.json

# 运行测试前，编辑 run_test_session.sh 调整以下参数：
#   OPENAI_API_BASE  → 指向你的推理服务端地址
#   OPENAI_API_KEY   → API 密钥
#   OPENAI_MODEL     → 模型名称
#   INPUT_DATA       → 预处理后的 JSON 文件路径
#   MAX_TRIGGER      → 最大并发 trigger 数（默认为 10）
#   POISSON_LAM      → 用户间泊松间隔参数（0 表示无间隔）
#   MAX_TRACE        → 限制 VU 数量（可选）
sh run_test_session.sh
```

如需更细粒度的参数控制，可以直接调用 `run_session_mlperf.py`：

```bash
uv run python run_session_mlperf.py \
    --input preprocess.json \
    --output-dir traces_mlperf_results \
    --max-trigger 10 \
    --poisson-lam 0 \
    --poisson-seed 42 \
    --time-unit s \
    --tpm-interval 60.0 \
    --plot-interval 5.0 \
    --mlperf-conf ../../mlperf.conf
```


## 3. Mock Server

**路径**: `inference/language/glm-5.1/mock_server`

### 概述

一个 **Rust 编写的 Mock OpenAI API 服务端**，用于在不依赖真实 GPU 服务端的情况下测试 GLM-5.1 的 MLPerf 流程。支持非流式和流式（SSE）的 Chat Completions API。

### 编译

依赖：Rust 工具链（安装见 [rustup.rs](https://rustup.rs/)）

```bash
cd inference/language/glm-5.1/mock_server
cargo build --release

## 或者
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

- **Rust edition**：2021
- **编译产物**：`target/release/mock_openai`

### 依赖（`Cargo.toml`）

| 依赖 | 版本 | 用途 |
|---|---|---|
| `axum` | 0.7 | HTTP 框架 |
| `tokio` | 1（full features） | 异步运行时 |
| `serde` / `serde_json` | 1 | JSON 序列化/反序列化 |
| `tower-http` | 0.5（cors） | CORS 中间件 |
| `tracing` / `tracing-subscriber` | 0.3 / 0.3 | 结构化日志 |
| `uuid` | 1（v4） | 生成唯一请求 ID |
| `async-stream` | 0.3 | 流式响应生成 |
| `tokio-stream` | 0.1 | Tokio 流扩展 |

### 运行时配置（环境变量）

| 变量 | 用途 | 默认值 |
|---|---|---|
| `PORT` | 监听端口 | `21000` |
| `CONCURRENCY` | 最大并发数（信号量） | `1024` |
| `STATS_INTERVAL` | 统计日志间隔（秒，0=禁用） | `0` |
| `DUMP_FILE` | 请求/响应 JSON 日志路径（可选） | 无 |

### 启动

```bash
# 基本启动
./target/release/mock_openai

# 自定义配置
PORT=21000 CONCURRENCY=64 STATS_INTERVAL=10 DUMP_FILE=traffic.jsonl ./target/release/mock_openai
```

### 支持的功能

- **端点**：`POST /chat/completions` 和 `POST /v1/chat/completions`
- **流式**：支持 `stream=true` 参数，返回 SSE 事件流
- **非流式**：标准 JSON 响应
- **Token 用量**：支持 `stream_options.include_usage`
- **CORS**：完全开放（`CorsLayer::permissive()`）
- **并发控制**：基于 Tokio `Semaphore`
- **日志 dump**：可选，以 JSONL 格式记录每个请求/响应/流块
- **统计日志**：可选，周期性输出总请求数、正在处理数、可用许可数

---

## 整体架构

```
LoadGen (C++ / Python)
    │  IssueQuery()
    ▼
GLM-5.1 SUT (Python)
    │  OpenAI API (HTTP)
    ▼
┌─────────────────────────────┐
│ Mock Server (Rust / axum)   │  ← 本地测试/调试
│ Or vLLM / SGLang / GLM API  │  ← 正式运行
└─────────────────────────────┘
```

- **LoadGen** 生成查询 → **GLM-5.1 SUT** 通过 OpenAI API 转发 → **外部/模拟服务端** 处理并返回响应。
- Mock Server 模拟了完整的 Chat Completions 接口，适合在无 GPU 环境下快速验证流程正确性。



#### Docker 运行


##### GLM5.1 数据集

```
# 需要在 inference/ 根目录下执行（build context 要包含 language/ 和 loadgen/）
cd /home/ldx/mlperf/inference
docker build -f language/glm-5.1/docker/Dockerfile -t glm-5.1-mlperf .
```


```
# 下载程序镜像， 减少GLIBC 版本不一致
curl -O http://10.188.128.16:35612/glm-5.1-mlperf.tar

# 加载镜像
docker load < glm-5.1-mlperf.tar

# 下载数据集
curl -O http://10.188.128.16:35612/glm51-dataset.zip 

unzip glm51-dataset.zip

# 查看帮助
docker run --rm glm-5.1-mlperf run_session_mlperf.py --help

# 传入数据文件运行
# 也支持 --target-qps 等新参数
docker run --rm -it \
    -v /home/ldx/loadgen/extracted_files:/data \
    -e OPENAI_API_KEY=YourKey \
    -e OPENAI_API_BASE=http://10.188.128.16:21000 \
    -e OPENAI_MODEL=YourModel \
    -e MLPERF_MAX_OSL=8192 \
    glm-5.1-mlperf run_session_mlperf.py \
        --input /data \
        --target-qps 2.0 \
        --min-duration 600 \
        --min-query-count 100 
```

环境变量
```
OPENAI_API_KEY=    # API_KEY
OPENAI_API_BASE=   # 模型请求url 前缀
OPENAI_MODEL=      # 模型名称
MLPERF_MAX_OSL=    # 要求server 输出最大token数
```

一些参数说明

```
options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Conversation JSON file/dir
  --output-dir OUTPUT_DIR
                        Output directory for LoadGen logs
  --num-slots NUM_SLOTS
                        并发度的数量 (default 10)
  --poisson-lam POISSON_LAM
                        泊松间隔的时间期望值， 比如取60， 两个用户交互之间的间隔时间在60左右做泊松分布
  --poisson-seed POISSON_SEED
                        泊松分布随机种子
  --poisson-pool-size POISSON_POOL_SIZE
                        Number of pre-generated Poisson intervals per worker
                        (default 1000)
  --time-unit {s,ms}    Unit of ps_delta / executed_time (default s)
  --max-vu MAX_VU       从数据集里面取多少会话快照 (truncates sessions)
  --min-query-count MIN_QUERY_COUNT
                        要发生用户交互的次数(不是请求数)
  --go-to-end           Iterate all requests to conversation end (instead of
                        stopping at next user), sleeping Poisson intervals
                        between users
```



##### DeekSeek

有两个可以选的数据集, 
`mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl`
和
`mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl`

--mode 参数可以指定 模式

```
docker run --rm -it    \
 -v $PWD/output:/output \
 -e OPENAI_API_KEY=YourKey \
 -e OPENAI_API_BASE=http://10.188.128.16:21000 \
 -e OPENAI_MODEL=Mock-Model \
 glm-5.1-mlperf  deepseek run_mlperf.py     \
 --input-file=mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl \
 --output-dir /output
  --target-qps 14.72 \
  --min-duration 600000 \
  --min-query-count 105312 
```

or

```
docker run --rm -it    \
 -v $PWD/output:/output \
 -e OPENAI_API_KEY=YourKey \
 -e OPENAI_API_BASE=http://10.188.128.16:21000 \
 -e OPENAI_MODEL=Mock-Model \
 glm-5.1-mlperf  deepseek run_mlperf.py     \
 --input-file=mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl \
 --output-dir /output
  --target-qps 14.72 \
  --min-duration 600000 \
  --min-query-count 105312 \
  --mode server
```
