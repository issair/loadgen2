"""
Multi-turn conversation splitter for MLPerf LoadGen.

Splits multi-turn conversations into LLM requests:
  * For each ``assistant`` message at position *i*, produce ``messages[:i]``
    (the LLM input *before* that assistant).
  * Finally produce the full ``messages[:]`` slice (including the last assistant).

A ``sample_pool`` contains only requests whose ``last_role == "user"`` —
these are what LoadGen Poisson-samples from.

When a sample is drawn, the SUT replays the dependency chain —
**the sampled request plus all subsequent requests that come before
the next user-ending request** — and measures every one.

Example
-------
Messages:
  [system, user_q1, assistant, tool_r1, user_q2, assistant, tool_r2,
   assistant, tool_r3, assistant, user_q3, assistant, tool_r4]

Split:
  idx=0: [system, user_q1]                                                     user
  idx=1: [system, user_q1, assistant, tool_r1, user_q2]                        user
  idx=2: [system, user_q1, assistant, tool_r1, user_q2, assistant, tool_r2]    tool
  idx=3: [system, ..., assistant, tool_r3]                                     tool
  idx=4: [system, ..., user_q3]                                                user
  idx=5: [system, ..., tool_r4]                                                tool

Sample pool (user-ending only):  [0, 1, 4]

Dependencies (until the next user):
  0 → [0]          (next is req 1 which is user-ending → stop)
  1 → [1, 2, 3]    (reqs 2, 3 are tool-ending, stop before req 4)
  4 → [4, 5]       (req 5 is tool-ending, no more user-ending)

If LoadGen samples idx=1, the SUT replays requests 1, 2, 3 in order.

Input format (JSON array or single object):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    ...
  ]
}

Output format:
{
  "original_conversations": [...],
  "llm_requests": [
    {
      "messages": [...],
      "request_idx": 0,          # global index across all conversations
      "original_idx": 0,         # index in original_conversations
      "last_role": "user",       # role of the last message in this slice
      "end_at": 1                # index (0-based) of last message in messages
    },
    ...
  ],
  "sample_pool": [0, 1, 4],     # indices into llm_requests (last_role=="user")
  "dependencies": [
    [0],
    [1, 2, 3],
    [4, 5],
  ]
}
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------


def split_conversation_for_loadgen(
    conversation: Dict[str, Any],
    original_idx: int,
) -> List[Dict[str, Any]]:
    """
    Split a multi-turn conversation into LLM requests.

    For each ``assistant`` message at position *i* (0-based), produce a
    slice ``messages[:i]`` (the LLM input *before* that assistant).
    Finally, produce the full ``messages[:]`` slice.

    Parameters
    ----------
    conversation : dict
        A conversation dict with a ``"messages"`` list.
    original_idx : int
        Index of this conversation in the original dataset (for provenance).

    Returns
    -------
    list of dict
        Each entry has ``messages``, ``original_idx``, ``last_role``, ``end_at``.
    """
    messages = conversation.get("messages", [])
    if not messages:
        return []

    # Extra keys from the original conversation to preserve in every request
    # (e.g. "tools", "model", "stream", "temperature", "top_p")
    extra_keys = {k: v for k, v in conversation.items() if k != "messages"}

    requests: List[Dict[str, Any]] = []

    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            # Slice up to (but not including) this assistant
            if i > 0:
                chunk = messages[:i]
                requests.append(
                    {
                        "messages": chunk,
                        **extra_keys,
                        "original_idx": original_idx,
                        "last_role": chunk[-1]["role"],
                        "end_at": i - 1,
                    }
                )

    # Append the full message list
    requests.append(
        {
            "messages": messages[:],
            **extra_keys,
            "original_idx": original_idx,
            "last_role": messages[-1]["role"] if messages else "",
            "end_at": len(messages) - 1,
        }
    )

    return requests


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def load_dataset(input_path: str) -> List[Dict[str, Any]]:
    """Load a JSON file (single object or array) into a list of dicts."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON format: {type(data)}")


def discover_input_files(path: str) -> List[str]:
    """
    Given a path that may be a single file, multiple comma-separated files,
    or a directory, return a flat list of JSON file paths.
    """
    files: List[str] = []
    for part in path.split(","):
        part = part.strip()
        p = Path(part)
        if p.is_dir():
            files.extend(sorted(str(f) for f in p.glob("*.json")))
        elif p.is_file():
            files.append(str(p))
        else:
            # Try glob pattern
            import glob as _glob

            matched = _glob.glob(part)
            if matched:
                files.extend(sorted(matched))
            else:
                raise FileNotFoundError(f"No files found matching: {part}")
    return files


def estimate_memory_mb(all_requests: List[Dict[str, Any]]) -> float:
    """Rough memory estimate for the requests list in MB."""
    total_chars = 0
    for req in all_requests:
        for msg in req.get("messages", []):
            total_chars += len(str(msg.get("content", "")))
    # JSON overhead + dict overhead ~ 4x
    return total_chars * 4 / (1024 * 1024)


def process_dataset(input_path: str, output_path: str) -> int:
    """
    Process one or more datasets and write a merged output JSON.

    *input_path* may be a single JSON file, a comma-separated list of files,
    or a directory (all ``*.json`` files inside are loaded).

    Returns the number of user-ending requests in the merged sample pool.
    """
    input_files = discover_input_files(input_path)
    if not input_files:
        raise FileNotFoundError(f"No JSON files found at: {input_path}")

    print(f"Discovered {len(input_files)} input file(s)")

    all_requests: List[Dict[str, Any]] = []
    conv_request_indices: Dict[int, List[int]] = {}
    global_conv_idx = 0

    for file_idx, fpath in enumerate(input_files):
        dataset = load_dataset(fpath)
        if file_idx == 0:
            print(f"Loaded {len(dataset)} conversations from {fpath}")

        for conv in dataset:
            requests = split_conversation_for_loadgen(conv, global_conv_idx)
            start = len(all_requests)
            all_requests.extend(requests)
            end = len(all_requests)
            conv_request_indices[global_conv_idx] = list(range(start, end))
            global_conv_idx += 1

        if (file_idx + 1) % 100 == 0 or file_idx == len(input_files) - 1:
            est_mb = estimate_memory_mb(all_requests)
            print(
                f"  [{file_idx + 1}/{len(input_files)} files]  "
                f"{global_conv_idx} conversations, "
                f"{len(all_requests)} requests, "
                f"~{est_mb:.1f} MB",
                flush=True,
            )

    # Assign global request indices
    for req_idx, req in enumerate(all_requests):
        req["request_idx"] = req_idx

    # Build sample_pool (user-ending only) and dependencies.
    sample_pool: List[int] = []
    dependencies: List[List[int]] = []

    for req_idx, req in enumerate(all_requests):
        if req["last_role"] != "user":
            continue

        sample_pool.append(req_idx)

        orig_idx = req["original_idx"]
        all_conv_indices = conv_request_indices[orig_idx]
        pos = all_conv_indices.index(req_idx)

        chain = [req_idx]
        for next_idx in all_conv_indices[pos + 1 :]:
            if all_requests[next_idx]["last_role"] == "user":
                break
            chain.append(next_idx)

        dependencies.append(chain)

    # ── Store llm_requests in diskcache instead of JSON ────────────────
    import shutil

    import diskcache

    out_dir = os.path.dirname(output_path) or "."
    basename = os.path.splitext(os.path.basename(output_path))[0]
    cache_dir = os.path.join(out_dir, basename + "_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    cache = diskcache.Cache(cache_dir, size_limit=100 * (1024**3))
    for req in all_requests:
        cache[req["request_idx"]] = req
    cache["__count__"] = len(all_requests)
    cache.close()

    est_mb = estimate_memory_mb(all_requests)
    print(
        f"Done — {len(all_requests)} requests stored to diskcache at {cache_dir}, "
        f"{len(sample_pool)} in sample_pool, "
        f"~{est_mb:.1f} MB "
        f"(LoadGen samples user-ending, SUT replays dependencies)",
        flush=True,
    )

    # ── Write lightweight output JSON (no llm_requests) ────────────────
    output = {
        "llm_requests_cache": os.path.abspath(cache_dir),
        "llm_requests_count": len(all_requests),
        "sample_pool": sample_pool,
        "dependencies": dependencies,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return len(sample_pool)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split multi-turn conversations into LLM requests for MLPerf LoadGen.\n\n"
            "Outputs:\n"
            "  llm_requests    — all request slices (per-assistant + full list)\n"
            '  sample_pool     — indices where last_role=="user" '
            "(LoadGen Poisson-samples from here)\n"
            "  dependencies    — per sample_pool entry: subsequent requests "
            "until the next user-ending\n\n"
            "Usage: the SUT receives a user-ending sample from the pool,\n"
            "looks up its dependencies, replays them in order, "
            "and records metrics for each."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help=(
            "Input: a JSON file, comma-separated file list, "
            "directory of JSON files, or glob pattern"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Merged output JSON file",
    )
    args = parser.parse_args()

    process_dataset(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
