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
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from transformers import JinaEmbeddingsV3Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------


def split_conversation_for_loadgen(
    conversation: Dict[str, Any],
    original_idx: str,
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


def load_dataset(
    input_path: str, encryption_key: str | None = None
) -> List[Dict[str, Any]]:
    """Load a JSON file (single object or array) into a list of dicts.

    If ``encryption_key`` is provided or the ``ENCRYPTION_KEY`` environment
    variable is set, the file will be decrypted with AES-256-GCM before
    parsing.  Otherwise the file is read as plain JSON.
    """
    key = encryption_key or os.environ.get("ENCRYPTION_KEY")

    if key is not None:
        try:
            return _load_encrypted_dataset(input_path, key)
        except Exception:
            logger.warning(
                "AES decryption failed for %s, falling back to plain-text.",
                input_path,
            )

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported JSON format: {type(data)}")


def _derive_key(key_str: str) -> bytes:
    """Derive a 32-byte AES-256 key from a string passphrase using PBKDF2."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    # Fixed salt so the same passphrase always produces the same key.
    # In production you may want a per-file salt stored alongside the nonce.
    salt = b"mlperf-glm5.1-aes256-salt"
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=600_000,
    )
    return kdf.derive(key_str.encode("utf-8"))


def _load_encrypted_dataset(input_path: str, key_str: str) -> List[Dict[str, Any]]:
    """Decrypt a .json.aes file with AES-256-GCM and return parsed data."""
    key = _derive_key(key_str)
    aesgcm = AESGCM(key)

    with open(input_path, "rb") as f:
        nonce = f.read(12)  # 96-bit nonce / IV
        ciphertext = f.read()  # ciphertext || 16-byte tag

    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    data = json.loads(plaintext.decode("utf-8"))

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
            files.extend(sorted(str(f) for f in p.glob("*.jsonl")))
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


def process_dataset(
    input_path: str,
    output_path: str,
    poisson_lam: int = 1,
    poisson_seed: int = 0,
    time_unit: str = "s",
) -> Dict[str, "LargeModelSession"]:
    """
    Build LargeModelSession objects for every conversation in *input_path*.

    *input_path* may be a single JSON file, a comma-separated list of files,
    or a directory (all ``*.json`` files inside are loaded).

    Returns
    -------
    dict[str, LargeModelSession]
        Map from file path to the LargeModelSession built from each
        conversation.
    """
    input_files = discover_input_files(input_path)
    if not input_files:
        raise FileNotFoundError(f"No JSON files found at: {input_path}")

    print(f"Discovered {len(input_files)} input file(s)")

    import numpy as np

    large_model_sessions: Dict[str, LargeModelSession] = {}

    for file_idx, fpath in enumerate(input_files):
        dataset = load_dataset(fpath)

        for conv in dataset:
            requests = split_conversation_for_loadgen(conv, fpath)

            # ── Build LargeModelSession for this conversation ──────────
            # Identify user-ending requests among the split slices.
            user_conv_indices: List[int] = []
            for local_idx, req in enumerate(requests):
                if req["last_role"] == "user":
                    user_conv_indices.append(local_idx)

            # Zero-length intervals between consecutive user turns.
            # Poisson pacing is handled by the scheduler's worker pool instead.
            _n_user = len(user_conv_indices)
            user_ps_delta: List[int] = [0] * _n_user if _n_user > 0 else []

            large_model_sessions[fpath] = LargeModelSession(
                snap_shot=conv,
                all_conv_request=[req["end_at"] for req in requests],
                user_conv_request=user_conv_indices,
                user_conv_ps_delta=user_ps_delta,
                time_unit=time_unit,
            )

        if (file_idx + 1) % 100 == 0 or file_idx == len(input_files) - 1:
            print(
                f"  [{file_idx + 1}/{len(input_files)} files]  ",
                flush=True,
            )
    return large_model_sessions


# ---------------------------------------------------------------------------
# Large Model Session
# ---------------------------------------------------------------------------


class LargeModelSession:
    """
    大模型会话

    Manages a multi-turn conversation session, tracking user request indices,
    Poisson-distributed intervals between user turns, and scheduling the next
    request to execute.

    Attributes
    ----------
    snap_shot : dict
        原始请求会话 (the raw conversation request).
    all_conv_request : list[int]
        每个请求切片对应的 end_at 值（messages 中最后一条消息的下标）。
    user_conv_request : list[int]
        user 请求的索引 (subset of all_conv_request).
    user_conv_ps_delta : list[int]
        本轮会话 user 和 user 之间的泊松分布间隔（单位由 time_unit 决定）。
    executed_time : list[float]
        每轮结束的墙上时间（秒），0.0 表示未执行。
    time_unit : str
        ``"s"`` 或 ``"ms"`` —— ps_delta 的时间单位。
    """

    def __init__(
        self,
        snap_shot: dict,
        all_conv_request: list[int],
        user_conv_request: list[int],
        user_conv_ps_delta: list[int],
        time_unit: str = "s",
        executed_time: list[float] | None = None,
    ) -> None:
        self.snap_shot = snap_shot
        self.all_conv_request = all_conv_request
        self.user_conv_request = user_conv_request
        self.user_conv_ps_delta = user_conv_ps_delta
        self.time_unit = time_unit
        self.executed_time: list[float] = (
            executed_time
            if executed_time is not None
            else [0.0] * len(user_conv_request)
        )

    # ------------------------------------------------------------------
    #  Unit conversion helpers
    # ------------------------------------------------------------------

    def _ps_delta_to_seconds(self, value: int) -> float:
        """Convert a ``user_conv_ps_delta`` value to seconds."""
        return value / 1000.0 if self.time_unit == "ms" else float(value)

    def _seconds_to_ps_delta_unit(self, value: float) -> int:
        """Convert seconds back to the ``user_conv_ps_delta`` unit."""
        return int(value * 1000) if self.time_unit == "ms" else int(value)

    def next_execute_request(self, now_sec: float) -> tuple[int, int]:
        """
        Return (next_execute_wait_time, pos).

        Computes the next user request to execute based on Poisson intervals
        and already-executed times.  Uses *now_sec* (wall-clock time in
        seconds) to compute remaining wait.

        Returns
        -------
        tuple[int, int]
            - ``next_execute_wait_time``: how long to wait before the next
              execution (in the same time unit as ``user_conv_ps_delta``).
            - ``pos``: the **position** (index into ``user_conv_request``
              / ``executed_time``) of the turn to execute next, or -1 if
              all turns have been executed.
        """
        # If any request is already being processed (sentinel value),
        # another worker has claimed this session — skip it.
        if any(et >= 9_999_999_999_999 for et in self.executed_time):
            return 0, -1

        # Find the first unexecuted user request (executed_time == 0.0)
        logger.debug(
            "next_execute_request: now_sec=%.3f, user_conv_request=%s, executed_time=%s",
            now_sec,
            self.user_conv_request,
            self.executed_time,
        )

        for idx, et in enumerate(self.executed_time):
            if et < 0.01:
                if idx == 0:
                    wait_time = 0
                else:
                    # Convert Poisson interval to seconds.
                    ps_sec = self._ps_delta_to_seconds(self.user_conv_ps_delta[idx])
                    if ps_sec != 0:
                        # Time already elapsed since the previous turn ended.
                        elapsed_sec = now_sec - self.executed_time[idx - 1]
                        wait_sec = ps_sec - elapsed_sec
                        if wait_sec <= 0.0:
                            wait_time = 0
                        else:
                            wait_time = self._seconds_to_ps_delta_unit(wait_sec)
                    else:
                        wait_time = 0
                logger.debug(
                    "next_execute_request: found idx=%d, value=%d, wait_time=%d",
                    idx,
                    self.user_conv_request[idx],
                    wait_time,
                )
                rel_indx = idx if wait_time <= 0.01 else -1
                if rel_indx >= 0:
                    self.executed_time[rel_indx] = (
                        9999999999999  # 预占了这个情况,其他work,不能拿
                    )
                return wait_time, rel_indx

        # All requests have been executed
        # logger.warning(
        #     "next_execute_request: all executed! "
        #     "user_conv_request=%s, executed_time=%s, ps_delta=%s",
        #     self.user_conv_request,
        #     self.executed_time,
        #     self.user_conv_ps_delta,
        # )
        return 0, -1

    def end_request(
        self, pos: int, end_time_sec: float, sid: str, full_reset: bool = False
    ) -> bool:
        """
        Mark a user request as completed.

        Records the wall-clock end time for the request at position *pos*.
        After this call, :meth:`next_execute_request` will skip this request
        on subsequent invocations.

        Parameters
        ----------
        pos : int
            The position (from ``next_execute_request``) to mark as done.
        end_time_sec : float
            Wall-clock time (``time.time()``) when the turn finished.
        full_reset : bool
            If ``True``, reset *all* ``executed_time`` entries to 0.0 so
            the session can be retried from the very beginning.  Used when
            ``go_to_end=True`` in the scheduler.

        Returns
        -------
        bool
            ``True`` if all requests in this session have been executed,
            ``False`` otherwise.
        """
        if pos < 0 or pos >= len(self.executed_time):
            logger.warning(
                "end_request: pos=%d out of range (len=%d)",
                pos,
                len(self.executed_time),
            )
            return False
        logger.debug(
            "end_request: pos=%d, end_time_sec=%.3f, executed_time before=%s",
            pos,
            end_time_sec,
            self.executed_time,
        )
        last_et = self.executed_time[pos]
        self.executed_time[pos] = end_time_sec
        logger.debug(
            "end_request: %s executed_time befor=%s, after=%s",
            sid,
            last_et,
            self.executed_time,
        )

        # Determine if all requests in this session are done.
        all_done = all(et != 0.0 for et in self.executed_time)

        # If full_reset is requested, reset all turns so the session can
        # be retried from the very beginning.
        if full_reset:
            for i in range(len(self.executed_time)):
                self.executed_time[i] = end_time_sec
            logger.debug(
                "end_request: full_reset — all %d turns reset for %s",
                len(self.executed_time),
                sid,
            )
            return True
        elif all_done:
            # for i in range(len(self.executed_time)):
            # self.executed_time[i] = 0.0
            logger.debug(
                "end_request: all %d turns executed — resetting executed_time",
                len(self.executed_time),
            )

        return all_done

    async def iter_requests_until_next_user(
        self, pos: int, go_to_end: bool = False
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Yield all request slices from position *pos* up to the next user request.

        Starting from *pos* (returned by :meth:`next_execute_request`),
        yields every request slice — user, tool, and assistant messages —
        up to, but **not including**, the next user-ending request.
        If *pos* is the last user request, yields through the end
        of the conversation.

        When *go_to_end* is ``True``, the iterator does **not** stop at the
        next user boundary but continues all the way to the end of the
        conversation.  Between consecutive user turns (the Poisson-interval
        boundaries), the generator sleeps for the corresponding
        Poisson-distributed interval stored in ``user_conv_ps_delta``.

        Each yielded dict has the same shape as the output of
        :func:`split_conversation_for_loadgen`:
        ``messages``, ``last_role``, ``end_at``, plus extra keys from the
        original conversation.

        Parameters
        ----------
        pos : int
            Position into ``user_conv_request`` (from ``next_execute_request``).
        go_to_end : bool
            If ``True``, iterate through the entire remaining conversation
            instead of stopping at the next user boundary, sleeping for
            Poisson intervals at each user-turn boundary.

        Yields
        ------
        dict
            Reconstructed request slice.
        """
        if pos < 0 or pos >= len(self.user_conv_request):
            return

        messages = self.snap_shot.get("messages", [])
        if not messages:
            return

        extra_keys = {k: v for k, v in self.snap_shot.items() if k != "messages"}

        # Determine the local-index range to iterate.
        start_local = self.user_conv_request[pos]
        if go_to_end:
            stop_local = len(self.all_conv_request)
        elif pos + 1 < len(self.user_conv_request):
            stop_local = self.user_conv_request[pos + 1]
        else:
            stop_local = len(self.all_conv_request)

        # When go_to_end, track the next user boundary for Poisson sleeps.
        next_user_pos = pos + 1

        for local_idx in range(start_local, stop_local):
            # In go_to_end mode, sleep at each user-turn boundary.
            if (
                go_to_end
                and next_user_pos < len(self.user_conv_request)
                and local_idx == self.user_conv_request[next_user_pos]
            ):
                ps_sec = self._ps_delta_to_seconds(
                    self.user_conv_ps_delta[next_user_pos]
                )
                logger.debug(
                    "iter_requests_until_next_user: sleeping %.3f s "
                    "at user boundary pos=%d (ps_delta=%d)",
                    ps_sec,
                    next_user_pos,
                    self.user_conv_ps_delta[next_user_pos],
                )
                await asyncio.sleep(ps_sec)
                next_user_pos += 1

            end_at = self.all_conv_request[local_idx]
            chunk = messages[: end_at + 1]
            yield {
                "messages": chunk,
                **extra_keys,
                "last_role": chunk[-1]["role"] if chunk else "",
                "end_at": end_at,
            }


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
    parser.add_argument(
        "--poisson-lam",
        type=int,
        default=1,
        help="Poisson λ for inter-turn user-request intervals (default: 1)",
    )
    parser.add_argument(
        "--poisson-seed",
        type=int,
        default=0,
        help="Random seed for Poisson RNG (default: 0)",
    )
    args = parser.parse_args()

    process_dataset(
        input_path=args.input,
        output_path=args.output,
        poisson_lam=args.poisson_lam,
        poisson_seed=args.poisson_seed,
    )


if __name__ == "__main__":
    main()
