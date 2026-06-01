"""Per-request metrics collector and aggregator for SessionScheduler.

Records start / end events for every turn processed by the scheduler and
produces the same output format as :class:`ServerSUT` (see
:mod:`.server_sut`).

Usage inside :class:`SessionScheduler`::

    self._stats = SessionStats()

    # In _process_turn:
    rid = self._stats.record_start(worker_id, sid, pos, start_time)
    ...
    self._stats.record_end(rid, result, tokenizer=...)

    # After run() completes:
    results = self._stats.get_results()
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backends.base_backend import StreamingChunk

from .stream_process_pool import StreamResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Per-request record
# ---------------------------------------------------------------------------


@dataclass
class RequestRecord:
    """All metrics captured for a single request.

    Attributes
    ----------
    request_id : str
        UUID v4 — unique per request, used to match start ↔ end.
    worker_id : int
        Worker slot that processed this request.
    sid : str
        Session id (virtual user).
    pos : int
        Turn position within the session (index into ``user_conv_request``).
    start_time : float
        ``time.perf_counter()`` when the request started.
    end_time : float
        ``time.perf_counter()`` when the request ended.
    first_token_time : float or None
        ``time.perf_counter()`` of the first token, or ``None``.
    accumulated_text : str
        Full concatenated token text.
    tokens : list[int]
        Tokenised output (populated by the caller if a tokenizer is available).
    prompt_tokens : int
        From ``usage.prompt_tokens``.
    cached_tokens : int
        From ``usage.prompt_tokens_details.cached_tokens``.
    reasoning_tokens : int
        From ``usage.completion_tokens_details.reasoning_tokens``.
    completion_tokens : int
        From ``usage.completion_tokens``.
    """

    request_id: str = ""
    worker_id: int = 0
    sid: str = ""
    pos: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    first_token_time: Optional[float] = None
    accumulated_text: str = ""
    tokens: List[int] = field(default_factory=list)
    prompt_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    completion_tokens: int = 0
    end_at: int = -1
    role: str = "unkown"


# ---------------------------------------------------------------------------
#  Stats collector
# ---------------------------------------------------------------------------


class SessionStats:
    """Collect per-request metrics and produce aggregate results.

    Uses UUIDs to match ``record_start`` ↔ ``record_end`` so that
    concurrent workers never collide.
    """

    def __init__(self) -> None:
        self._records: List[RequestRecord] = []
        self._by_id: Dict[str, RequestRecord] = {}

    # -- recording -----------------------------------------------------------

    def record_start(
        self,
        worker_id: int,
        sid: str,
        pos: int,
        end_at: int,
        start_time: float,
        role: str,
    ) -> str:
        """Create a new record when a request begins.

        Returns the *request_id* (UUID hex string) that must be passed to
        :meth:`record_end`.
        """
        rid = uuid.uuid4().hex
        rec = RequestRecord(
            request_id=rid,
            worker_id=worker_id,
            sid=sid,
            pos=pos,
            end_at=end_at,
            start_time=start_time,
            role=role,
        )
        self._records.append(rec)
        self._by_id[rid] = rec
        logger.debug(
            "record_start: id=%s worker=%d sid=%s pos=%d time=%.3f",
            rid,
            worker_id,
            sid,
            pos,
            start_time,
        )
        return rid

    def record_end(
        self,
        request_id: str,
        result: StreamResult,
        tokenizer=None,
    ) -> Optional[RequestRecord]:
        """Finalise the record identified by *request_id*.

        Parameters
        ----------
        request_id : str
            The id returned by :meth:`record_start`.
        result : StreamResult
            The streaming result from the process pool.
        tokenizer : optional
            A tokenizer callable ``encode(text) -> list[int]`` for
            populating ``tokens``.
        """
        rec = self._by_id.get(request_id)
        if rec is None:
            logger.warning("record_end: unknown request_id=%s", request_id)
            return None

        rec.start_time = result.start_time
        rec.end_time = result.end_time
        rec.first_token_time = result.first_token_time
        rec.accumulated_text = result.accumulated_text

        # -- usage from the final chunk -----------------------------------
        _apply_usage(rec, result.last_chunk)

        # -- tokenisation -------------------------------------------------
        if tokenizer is not None and rec.accumulated_text:
            try:
                rec.tokens = tokenizer.encode(rec.accumulated_text)
            except Exception:
                rec.tokens = [1]
        elif not rec.accumulated_text:
            rec.tokens = [1]
        return rec

    # -- query ---------------------------------------------------------------

    def get_results(self) -> List[Dict[str, Any]]:
        """Return results in the same format as :meth:`ServerSUT.get_results`.

        Each dict contains
        ``dep_idx``, ``is_target``, ``workid``, ``sid``,
        ``model_output``,
        ``tok_model_output``, ``tok_model_output_len``,
        ``start_time``, ``end_time``,
        ``first_token_time``, ``completion_time``,
        ``prompt_tokens``, ``cached_tokens``, ``reasoning_tokens``,
        ``completion_tokens``.
        """
        ordered: List[Dict[str, Any]] = []
        for i, rec in enumerate(self._records):
            ordered.append(
                {
                    "dep_idx": i,
                    "is_target": True,
                    "workid": rec.worker_id,
                    "sid": rec.sid,
                    "pos": rec.pos,
                    "role": rec.role,
                    "end_at": rec.end_at,
                    "tok_model_output_len": len(rec.tokens),
                    "start_time": rec.start_time,
                    "end_time": rec.end_time if rec.end_time > 0 else None,
                    "first_token_time": _ttft(rec),
                    "completion_time": (
                        rec.end_time - rec.start_time if rec.end_time > 0 else None
                    ),
                    "prompt_tokens": rec.prompt_tokens,
                    "cached_tokens": rec.cached_tokens,
                    "reasoning_tokens": rec.reasoning_tokens,
                    "completion_tokens": rec.completion_tokens,
                }
            )
        logger.info("SessionStats: %d results collected", len(ordered))
        return ordered

    @property
    def count(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _ttft(rec: RequestRecord) -> Optional[float]:
    """Time-to-first-token in seconds (relative to start)."""
    if rec.first_token_time is not None and rec.start_time > 0:
        return rec.first_token_time - rec.start_time
    return None


def _apply_usage(rec: RequestRecord, chunk: Optional[StreamingChunk]) -> None:
    """Extract usage fields from the final chunk (if any)."""
    if chunk is None or chunk.usage is None:
        return
    usage = chunk.usage
    rec.prompt_tokens = usage.get("prompt_tokens", 0)
    rec.completion_tokens = usage.get("completion_tokens", 0)
    rec.cached_tokens = (usage.get("prompt_tokens_details", {}) or {}).get(
        "cached_tokens", 0
    )
    rec.reasoning_tokens = usage.get("completion_tokens_details", {}).get(
        "reasoning_tokens", 0
    )
