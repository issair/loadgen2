# Loadgen2

---

## 背景与动机：为什么需要 LoadGen 2.0？

### 从对话式 AI 到自主智能体

2026 年以来，随着 AI Agent（智能体）技术的全面破圈，大语言模型（LLM）正从简单的对话机器人加速演进为能够**自主规划、推理并采取行动**以达成复杂目标的**长时运行系统**。这一趋势使得大模型推理算力需求呈井喷式增长，"Token 工厂"概念跃升为行业核心焦点。

### 智能体负载的全新挑战

智能体（Agentic）的工作负载与传统人类对话交互在结构上截然不同，呈现出三大核心特征：

| 特征 | 描述 |
|---|---|
| **长周期多轮循环** | 单次任务涉及数十次推理循环、工具调用与自我反思的叠加，而非一问一答 |
| **上下文指数级膨胀 (Context Ballooning)** | 随推理深入，上下文窗口不断扩张，对 KV Cache 造成巨大压力 |
| **高频状态切换** | 在"推理阶段"（产生中间思考）与"行动阶段"（发起外部工具调用、接收结果）之间反复跳转 |

这些特征导致 **KV Cache 被频繁置换**、**轮次间隔严重抖动**，使得传统的静态压测指标完全失效。

### 传统压测工具的局限

传统的压测框架（包括 MLPerf 原版 LoadGen）擅长模拟**规整、线性的恒定流量**（如固定 QPS 的 Offline / Server 场景），但在面对智能体负载时则捉襟见肘：

- **无法模拟多轮对话的状态保持**：每次请求独立，不知道哪些请求属于同一会话
- **无法复现混沌的用户到达模式**：真实用户的行为间隔呈泊松分布，而非均匀间隔
- **无法刻画上下文膨胀效应**：传统压测看不到长对话中 context 增长对延迟和吞吐的渐进影响

### LoadGen 2.0 的核心突破

LoadGen 2.0 基于 MLPerf LoadGen 进行了深度重构，实现了从**静态并发注入**到**动态行为仿真**的跨越：

- **多轮状态保持（Stateful Turn-based）模拟**：以"会话"为最小压测单元，完整保留对话上下文链路，真实还原多轮工具调用的长周期负载
- **混合泊松分布逻辑**：通过泊松过程控制虚拟用户（VU）的到达间隔，模拟真实生产环境中交织、重叠且不可预测的计算请求
- **混沌负载能力**：能够复现 Context Ballooning 导致的渐进式性能退化，帮助开发者和运营者在上线前探明集群的性能崩溃边界与资源调度瓶颈

![用户行为与负载模型概览](overview.png)

### 面向谁？

本工具旨在为 AI 基础设施生态的四个关键角色提供统一的评测基准：

| 角色 | 核心诉求 |
|---|---|
| **建设者** | 针对长上下文频繁复用优化架构 |
| **运营者** | 预估动态波动下的并发水位 |
| **使用者** | 获取明确的 SLA 采购依据 |
| **最终用户** | 避免"首字延迟（TTFT）不可控"和"推理中途断线" |

---

## 组件概览

项目下和 **GLM-5.1** 基准测试相关的三个核心组件的编译、安装与配置方法：

1. [LoadGen2](#1-loadgen2) —  官方负载生成器
2. [GLM-5.1 模型](#2-glm-51-模型压测参考实现) — 模型端 MLPerf 客户端

---

## 1. LoadGen2

**路径**: `loadgen2`

### 概述

LoadGen2 是本项目的**核心负载生成与测量工具**，基于 MLPerf LoadGen（v1）深度重构，用 C++ 编写并提供 Python 绑定。与 v1 仅支持恒定流量不同，LoadGen2 专为**智能体长时运行负载**设计，核心能力包括：

- **会话级状态保持**：以完整多轮对话为压测单元，自动维护同一会话内的上下文链路，真实模拟 Agent 的多轮推理-工具调用循环
- **泊松到达模型**：通过泊松过程控制虚拟用户（VU）的到达间隔，复现生产环境中不可预测的并发波动，而非均匀排队的理想化流量
- **混沌负载仿真**：支持动态上下文膨胀（Context Ballooning）场景，能够探明 KV Cache 频繁置换下的性能退化曲线与集群崩溃边界
- **精准测量与日志**：记录每个请求的延迟分布、首字延迟（TTFT）、Token 间延迟（TPOT）及吞吐量汇总报告

## 2. GLM-5.1 模型压测参考实现

**路径**: `language/glm-5.1`

## 3. 其他压测模型压测参考实现

正在紧锣密鼓地进行其他模型的参考实现，尽情期待...

### 概述

LoadGen2 的模型端参考实现以 **会话（Trace）为最小压测单元**，通过**多轮状态保持**完整保留对话上下文链路，真实还原智能体长周期工具调用的负载特征。借助**泊松过程**控制虚拟用户到达间隔，模拟生产环境中交织重叠的请求分布，从而**复现 Context Ballooning 导致的渐进式性能退化**（混沌负载），帮助在上线前探明集群的性能崩溃边界。

```
glm-5.1/
├── backends/          # 后端抽象（openai_backend.py）
├── mlperf/            # SUT、QSL、场景实现
├── utils/             # 工具（注册、数据、验证、tokenization）
├── docker/            # Docker 配置
├── run_session_mlperf.py      # MLPerf 基准测试入口
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
uv sync --all-extras
```

### 配置

通过环境变量或修改 `utils/backend_registry.py`：

| 环境变量 | 用途 | 默认值 |
|---|---|---|
| `MLPERF_BACKEND` | 后端选择 | `openai` |
| `OPENAI_API_KEY` | API 密钥 | — |
| `OPENAI_API_BASE` | API 端点地址 | `http://localhost:8000/v1` |
| `model` | 模型名称 | `glm-5-1` |
| `max_tokens` | 最大输出 token 数 | 8192 |
| `temperature` | 采样温度 | 0.0 |
| `max_concurrent_requests` | 最大并发请求数 | 64 |

### 运行

```bash
cd language/glm-5.1

https://hf-mirror.com/datasets/Inferact/codex_swebenchpro_traces

# 下载数据获取里面的 codex_swebenchpro_format.json
# https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces

# 预处理一下数据集
uv run python scripts/split_codex_swebenchpro.py \
    --input codex_swebenchpro_format.json \
    --output-dir codex_swebenchpro

OPENAI_API_BASE="http://localhost:21000" \
OPENAI_API_KEY="Your-Key" \
OPENAI_MODEL="Mock-Model" \
uv run python run_session_mlperf.py \
    --input codex_swebenchpro \
    --output-dir ./output \
    --max-trigger 10 \
    --poisson-lam 0 \
    --poisson-seed 42 \
    --time-unit s 
    
```


#### Docker 构建

```bash
# Build context 为项目根路径
docker build -f language/glm-5.1/docker/Dockerfile -t glm-5.1-mlperf .
```

镜像基于 `python:3.12-slim`，构建过程会：

- 安装 C++ 编译工具链（g++/gcc/cmake）用于编译 LoadGen 的 pybind11 扩展
- 通过 `uv` 安装 Python 依赖及本地 `mlcommons-loadgen`
- 预下载 tokenizer 到本地目录（`/workspace/tokenizer`）避免运行时网络访问

#### Docker 运行

##### Loadgen2 数据集

```


# 下载数据集 

# 下载数据获取里面的 codex_swebenchpro_format.json
# https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces

# 预处理一下数据集
uv run python scripts/split_codex_swebenchpro.py \
    --input codex_swebenchpro_format.json \
    --output-dir codex_swebenchpro

# 查看帮助
docker run --rm glm-5.1-mlperf run_session_mlperf.py --help

# 传入数据文件运行
# 也支持 --target-qps 等新参数
docker run --rm -it \
    -v $PWD/codex_swebenchpro:/data \
    -e OPENAI_API_KEY=YourKey \
    -e OPENAI_API_BASE=http://10.188.128.16:21000 \
    -e OPENAI_MODEL=YourModel \
    -e MLPERF_MAX_OSL=666 \
    glm-5.1-mlperf run_session_mlperf.py \
        --input /data \
        --max-trigger 10 \
        --poisson-lam 0 \
        --poisson-seed 42 \
        --time-unit s 
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
                        Trace JSON file/dir
  --output-dir OUTPUT_DIR
                        Output directory for LoadGen logs
  --max-trigger MAX_TRIGGER
                        Maximum number of triggers (default 10)
  --poisson-lam POISSON_LAM
                        Poisson lambda for inter-turn intervals (default 1)
  --poisson-seed POISSON_SEED
                        RNG seed for Poisson generation (default 0)
  --poisson-pool-size POISSON_POOL_SIZE
                        Number of pre-generated Poisson intervals per trigger (default 1000)
  --time-unit {s,ms}    Unit of ps_delta / executed_time (default s)
  --max-trace MAX_TRACE
                        Maximum number of traces (limits traces, truncates if needed)
  --mlperf-conf MLPERF_CONF
                        Path to mlperf.conf
  --target-qps TARGET_QPS
                        Override *.Offline.target_qps (or *.Server.target_qps if --scenario Server)
  --min-duration MIN_DURATION
                        Override *.Offline.min_duration (ms)
  --min-query-count MIN_QUERY_COUNT
                        Override *.Offline.min_query_count
  --go-to-end           Iterate all requests to trace end (instead of stopping at next user), sleeping Poisson intervals between users
  --tpm-interval TPM_INTERVAL
                        Time interval in seconds for per-interval TPM calculation (default 60.0 = 1 min)
  --plot-interval PLOT_INTERVAL
                        Time interval in seconds for input-token timeline plot (default 5.0)
```
