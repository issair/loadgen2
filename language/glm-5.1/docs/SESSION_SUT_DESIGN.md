# SessionSUT 设计文档

> **目标**：实现基于虚拟用户（VU）会话的 Server 模型测试。
> LoadGen 退化为纯测量上报工具，SUT 自管调度节奏。
>
> **路径**：路径 B —— 自建调度器，10 个并发槽位，每 VU 独立 Poisson 间隔。

---

## 一、架构总览

```
                         ┌─ SessionSUT ──────────────────────────────────┐
                         │                                                │
  LoadGen (Offline)      │  ┌─ SessionScheduler ────────────────────────┐ │
                         │  │                                           │ │
  issue_queries(         │  │  sample_pool: Queue[QuerySample]          │ │
    全部 user turn)      │  │       │ (全局 FIFO)                        │ │
       │                 │  │       ▼                                   │ │
       │                 │  │  PriorityQueue[(ready_time, session_id)]  │ │
       │                 │  │       │       │              │            │ │
       │                 │  │       ▼       ▼              ▼            │ │
       │                 │  │  asyncio.Semaphore(10)                    │ │
       │                 │  │    │     │     │     │     │     │        │ │
       │                 │  │  [w0]  [w1]  [w2]  [w3]  ...  [w9]       │ │
       │                 │  │    │     │     │     │     │     │        │ │
       │                 │  │    └─────┴─────┴──┬──┴─────┴─────┘        │ │
       │                 │  │                   ▼                       │ │
       └──► 全部 sample ─┤  │        LargeModelSession                  │ │
           注入全局池     │  │        .next_execute_request()            │ │
                         │  │        .iter_requests_until_next_user()   │ │
                         │  │        .end_request()                     │ │
                         │  │                   │                       │ │
                         │  │                   ▼                       │ │
                         │  │        Backend.generate_stream()          │ │
                         │  │                   │                       │ │
                         │  │                   ▼                       │ │
                         │  │  lg.QuerySampleStart                      │ │
                         │  │  lg.FirstTokenComplete                    │ │
                         │  │  lg.QuerySamplesComplete                  │ │
                         │  └───────────────────────────────────────────┘ │
                         └────────────────────────────────────────────────┘
```

**关键原则**：

- LoadGen 只负责测量上报，不控制调度节奏。
- SUT 通过 `PriorityQueue` + `Semaphore(10)` 自管节奏。
- Sample 注入全局 FIFO 池，worker 执行 turn 时按需取用，不做 per-session 预分配。
- 不需要 `dependencies` 和 `all_requests`——`LargeModelSession.iter_requests_until_next_user()` 实时生成请求切片。
- TTFT / TPOT 由 SessionSUT 从推理流中计算。

---

## 二、组件

### 2.1 `LargeModelSession`（已有）

位置：`data_tools/split_andmapping.py`

| 字段 / 方法 | 说明 |
|---|---|
| `snap_shot: dict` | 原始对话 JSON |
| `all_conv_request: list[int]` | 每个请求切片的 `end_at` |
| `user_conv_request: list[int]` | user-ending 请求的 local index |
| `user_conv_ps_delta: list[int]` | user 轮次间的泊松间隔（`ps_delta[0] == 0`） |
| `executed_time: list[int]` | 已执行时间；0 = 未执行 |
| `next_execute_request() -> (wait_time, execute_index)` | 获取下一个待执行的 user 请求 |
| `iter_requests_until_next_user(execute_index) -> Iterator[dict]` | 遍历 user 轮次的全部请求切片 |
| `end_request(execute_index, elapsed)` | 标记一个 user 轮次完成 |

### 2.2 `SessionScheduler`（新增）

位置：`mlperf/session_scheduler.py`

核心数据结构：

```
ready_queue: asyncio.PriorityQueue[(ready_time: float, session_id: str)]
semaphore:   asyncio.Semaphore(10)
sample_pool: asyncio.Queue[lg.QuerySample]
    # 全局 FIFO 池，存放 LoadGen 下发的全部 QuerySample
    # worker 执行每个 turn 时从中取一个 —— 不按 session 预分配
```

核心方法：

```
async run()
    # 1. 将所有 session 的第一轮入队 (ready_time=now, wait_time=0)
    # 2. 启动 10 个 worker coroutine
    # 3. 等待全部完成

async worker(worker_id)
    # 循环：
    #   1. 从 ready_queue 取 (ready_time, session_id)
    #   2. sleep 到 ready_time
    #   3. 获取 semaphore
    #   4. 调用 _process_turn(session_id)
    #   5. 释放 semaphore（通过 async with）
    #   6. 如果该 session 还有下一轮，放回 ready_queue

async _process_turn(session_id)
    # 1. 从 sample_pool 取一个 QuerySample（全局 FIFO）
    # 2. session.next_execute_request() → (wait_time, exec_index)
    # 3. for req in session.iter_requests_until_next_user(exec_index):
    #        流式推理 + LoadGen 上报 (TTFT/TPOT)
    # 4. session.end_request(exec_index, elapsed)
    # 5. 检查是否还有下一轮 → 如有，入队 (now + next_wait, session_id)
```

### 2.3 `SessionSUT`（新增）

位置：`mlperf/session_sut.py`

桥接 LoadGen 和 `SessionScheduler`：

```
class SessionSUT(BaseSUT):
    def __init__(backend, sessions, num_slots=10)
        # sessions: Dict[str, LargeModelSession]
        # _total_turns = sum(len(s.user_conv_request) for s in sessions.values())

    def issue_queries(query_samples)
        # 1. 校验 len(query_samples) vs _total_turns
        # 2. 将全部 QuerySample 注入 scheduler.sample_pool
        # 3. 启动 scheduler.run()

    def flush_queries()
        # 等待 scheduler 所有 worker 完成
```

---

## 三、数据流

### 3.1 预处理（离线）

```
原始 JSON 对话文件
  │
  └─► process_dataset(input_path, poisson_lam, poisson_seed)
        │
        └─► large_model_sessions: Dict[str, LargeModelSession]
              # key = fpath（文件路径），每个文件一个会话
              # 实际使用时，key 可改为任意 session_id
```

### 3.2 运行时

```
large_model_sessions
  │
  ├─ total_user_turns = sum(len(s.user_conv_request) for s in sessions.values())
  │
  ├─► QSL(count=total_user_turns)
  │     # 不需要 dependency replay —— SUT 自己展开链
  │
  ├─► LoadGen Offline
  │     # 一次性 issue_queries(total_user_turns 个 sample)
  │
  └─► SessionSUT.issue_queries(query_samples)
        │
        ├─ 全部 sample 注入 scheduler.sample_pool（全局 FIFO）
        │
        └─► SessionScheduler.run()
              │
              ├─ bootstrap: 所有 session 的第一轮入队
              │
              └─ 10 workers 并行消费:
                    │
                    ├─ worker 取到 (ready_time, sid)
                    ├─ 从 sample_pool 取一个 QuerySample
                    ├─ session.next_execute_request()
                    ├─ for req in iter_requests_until_next_user():
                    │     backend.generate_stream() → LoadGen API
                    ├─ session.end_request(elapsed)
                    └─ 下一轮入队 或 session 完成
```

### 3.3 时序示例（1 个 VU，3 轮）

```
VU-001 (ps_delta = [0, 8s, 12s]):

  t=0       t=5s           t=13s      t=18s          t=30s      t=35s
  ├─────────┼──────────────┼──────────┼──────────────┼──────────┼──────
  │ user1   │ tool_r1      │ user2    │ tool_r2      │ user3    │ tool_r3
  │◄─chain0─►│              │◄─chain1─►│              │◄─chain2─►│
  └─ slot3 ────────────────┘─ slot7 ─────────────────┘─ slot1 ──

worker 视角:
  第1次 next_execute: (0ms, local_idx_0)      → end_request(0, 5000ms)
  第2次 next_execute: (8000ms, local_idx_1)   → end_request(1, 5000ms)
  第3次 next_execute: (12000ms, local_idx_2)  → end_request(2, 5000ms)
  第4次 next_execute: (0, -1)                  → VU-001 完成
```

---

## 四、LoadGen 承载

| 角色 | 说明 |
|---|---|
| `QSL.count` | `total_user_turns`（所有 VU 的所有 user 轮次之和） |
| `LoadSamplesToRam` | no-op（数据已在内存） |
| `UnloadSamplesFromRam` | no-op |
| `get_dependency_chain` | **不需要**（SUT 自管） |
| `TestScenario` | `Offline`（一次性拿到全部 sample） |
| `enable_dependency_replay` | `False` |
| `min_query_count` / `max_query_count` | 均设为 `total_user_turns` |

每个 user turn 在 LoadGen 视角下是一个独立的 sample。SUT 收到后自行展开依赖链、控制调度节奏、上报测量。

---

## 五、文件清单

| 文件 | 状态 | 说明 |
|---|---|---|
| `data_tools/split_andmapping.py` | ✅ 已有 | `process_dataset()` 产出 `LargeModelSession` |
| `mlperf/session_scheduler.py` | ✅ 已有 | `SessionScheduler` 类（全局 sample 池） |
| `mlperf/session_sut.py` | ✅ 已有 | `SessionSUT` 类 |
| `run_session_mlperf.py` | 🆕 新建 | 入口脚本（替代 `run_mlperf.py`） |
| `mlperf/base_sut.py` | ✅ 已有 | `BaseSUT` 基类 |
| `backends/base_backend.py` | ✅ 已有 | `BaseBackend` 接口 |

---

## 六、配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `num_slots` | 10 | 并发槽位数（即并发 VU 数上限） |
| `poisson_lam` | 1 | 泊松分布 λ（user 轮次间隔期望值） |
| `poisson_seed` | 0 | 泊松 RNG 种子 |
| `total_vu` | 2000 | 虚拟用户总数（从 `large_model_sessions` 长度自动确定） |
| `test_duration` | — | 测试持续时间（可选；跑完所有轮次即结束） |
