# 单个请求处理过程数据处理过程

file1.json
输入 `[system, user, assistant, tool, user, assistant, tool, user, assistant]`

输出：
| 请求 | 内容 | 结尾 |
|------|------|------|
| 0 | `[system, user]` | `user` ✓ |
| 1 | `[system, user, assistant, tool]` | `tool` ✓ |
| 2 | `[system, user, assistant, tool, user]` | `user` ✓ |
| 3 | `[system, user, assistant, tool, user, assistant, tool]` | `tool` ✓ |
| 4 | `[system, user, assistant, tool, user, assistant, tool, user]` | `user` ✓ |
| 5 | `[system, user, assistant, tool, user, assistant, tool, user, assistant]` | 全部(最后一条) |

**逻辑总结：**
- 每个请求都是 **从头开始累积追加**
- 每当消息序列以 `user` 或 `tool` 结尾时，就生成一个新请求
- 最后一个请求总是包含全部消息





全部正确！拆分逻辑现在生成的是：

| 请求 | 内容 | last_role | 采样？ |
|------|------|-----------|--------|
| 0 | `[system, user_q1]` | user | ✅ sample_pool |
| 1 | `[system, user_q1, assistant]` | assistant | 依赖 |
| 2 | `[system, ..., user_q2]` | user | ✅ sample_pool |
| 3 | `[system, ..., user_q2, assistant]` | assistant | 依赖 |
| 4 | `[system, ..., user_q3]` | user | ✅ sample_pool |
| 5 | `[system, ..., assistant]` | assistant | 依赖 |

dependencies: `[[0, 1], [2, 3], [4, 5]]` — 每个采样点后面跟一个 assistant-ending，停在**下一个 User 之前**。




```
[system, user_q1, assistant, tool_r1, user_q2, assistant, ..., user_q3, assistant]
        │               │                      │
        └── req0: [system, user_q1]            ← user-ending  (采样)
        └── req1: [system, user_q1, assistant] ← assistant-ending (依赖)
                    ...
                        └── req2: [system, ..., user_q2]    ← user-ending
                        └── req3: [system, ..., assistant]  ← assistant-ending
                                    ...

```





======================================================================
Test 1: Multi-turn with tool calls
Messages: ['system', 'user', 'assistant', 'tool', 'user', 'assistant', 'tool', 'assistant', 'tool','assistant', 'user', 'assistant','tool']
======================================================================

llm_requests (6):
  [0] request_idx=?, last_role=user, end_at=1, roles=['system', 'user']
  [1] request_idx=?, last_role=user, end_at=4, roles=['system', 'user', 'assistant', 'tool', 'user']
  [2] request_idx=?, last_role=tool, end_at=6, roles=['system', 'user', 'assistant', 'tool', 'user', 'assistant', 'tool']
  [3] request_idx=?, last_role=tool, end_at=8, roles=['system', 'user', 'assistant', 'tool', 'user', 'assistant', 'tool', 'assistant', 'tool']
  [4] request_idx=?, last_role=tool, end_at=10, roles=['system', 'user', 'assistant', 'tool', 'user', 'assistant', 'tool', 'assistant', 'tool','assistant', 'user']
  [5] request_idx=?, last_role=tool, end_at=10, roles=['system', 'user', 'assistant', 'tool', 'user', 'assistant', 'tool', 'assistant', 'tool','assistant', 'user', 'assistant','tool']

sample_pool: [0, 1, 4]
dependencies: [[0], [1,2, 3], [4, 5]]

======================================================================
Test 2: Multi-turn without tool calls
Messages: ['system', 'user', 'assistant', 'user', 'assistant', 'user', 'assistant']
======================================================================

llm_requests (6):
  [0] request_idx=?, last_role=user, end_at=1, roles=['system', 'user']
  [1] request_idx=?, last_role=user, end_at=3, roles=['system', 'user', 'assistant', 'user']
  [2] request_idx=?, last_role=user, end_at=5, roles=['system', 'user', 'assistant', 'user', 'assistant', 'user']
  [3] request_idx=?, last_role=assistant, end_at=6, roles=['system', 'user', 'assistant', 'user', 'assistant', 'user', 'assistant']

sample_pool: [0, 1, 2]
dependencies: [[0], [1], [2, 3]]

======================================================================
Test 3: Single assistant reply
Messages: ['system', 'user', 'assistant']
======================================================================

llm_requests (2):
  [0] request_idx=?, last_role=user, end_at=1, roles=['system', 'user']
  [1] request_idx=?, last_role=assistant, end_at=2, roles=['system', 'user', 'assistant']

sample_pool: [0]
dependencies: [[0, 1]]

```
uv sync --all-extras --refresh --reinstall-package mlcommons-loadgen
```
