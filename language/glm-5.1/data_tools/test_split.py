"""
Quick test for split_conversation_for_loadgen.

Tests: assistant-before slices + full-list, user-only sample_pool,
subsequent deps until next user-ending.
"""

import json
import sys

sys.path.insert(0, ".")
from split_conversation import split_conversation_for_loadgen


def make_messages(*roles):
    """Helper: list of roles -> list of dicts."""
    return [{"role": r, "content": r} for r in roles]


def show_requests(requests):
    """Pretty-print a list of request dicts."""
    for i, req in enumerate(requests):
        roles = [m["role"] for m in req["messages"]]
        print(
            f"  [{i}] request_idx={req.get('request_idx', '?')}, "
            f"last_role={req['last_role']}, "
            f"end_at={req['end_at']}, "
            f"roles={roles}"
        )


def compute_deps(requests):
    """Simulate process_dataset dependency logic."""
    sample_pool = []
    deps = []
    for i, req in enumerate(requests):
        req["request_idx"] = i
        if req["last_role"] != "user":
            continue
        sample_pool.append(i)
        chain = [i]
        for j in range(i + 1, len(requests)):
            if requests[j]["last_role"] == "user":
                break
            chain.append(j)
        deps.append(chain)
    return sample_pool, deps


# ---------------------------------------------------------------------------
# Test 1: Multi-turn with tool calls
# ---------------------------------------------------------------------------
conv1 = {
    "messages": make_messages(
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
        "tool",
    )
}

print("=" * 70)
print("Test 1: Multi-turn with tool calls")
print(f"Messages: {[m['role'] for m in conv1['messages']]}")
print("=" * 70)
reqs1 = split_conversation_for_loadgen(conv1, original_idx=0)
print(f"\nllm_requests ({len(reqs1)}):")
show_requests(reqs1)

sp1, dep1 = compute_deps(reqs1)
print(f"\nsample_pool: {sp1}")
print(f"dependencies: {dep1}")

# ---------------------------------------------------------------------------
# Test 2: Multi-turn without tool calls
# ---------------------------------------------------------------------------
conv2 = {
    "messages": make_messages(
        "system", "user", "assistant", "user", "assistant", "user", "assistant"
    )
}

print("\n" + "=" * 70)
print("Test 2: Multi-turn without tool calls")
print(f"Messages: {[m['role'] for m in conv2['messages']]}")
print("=" * 70)
reqs2 = split_conversation_for_loadgen(conv2, original_idx=1)
print(f"\nllm_requests ({len(reqs2)}):")
show_requests(reqs2)

sp2, dep2 = compute_deps(reqs2)
print(f"\nsample_pool: {sp2}")
print(f"dependencies: {dep2}")

# ---------------------------------------------------------------------------
# Test 3: Single user -> assistant
# ---------------------------------------------------------------------------
conv3 = {"messages": make_messages("system", "user", "assistant")}

print("\n" + "=" * 70)
print("Test 3: Single assistant reply")
print(f"Messages: {[m['role'] for m in conv3['messages']]}")
print("=" * 70)
reqs3 = split_conversation_for_loadgen(conv3, original_idx=2)
print(f"\nllm_requests ({len(reqs3)}):")
show_requests(reqs3)

sp3, dep3 = compute_deps(reqs3)
print(f"\nsample_pool: {sp3}")
print(f"dependencies: {dep3}")
