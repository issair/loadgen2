"""Multi-process streaming request executor.

Replaces the in-process ``_stream_single_request`` with a pool of worker
**processes**.  Each child process creates its own backend instance **once**
at process start (via ``ProcessPoolExecutor(initializer=...)``) and reuses
it for all requests dispatched to that process.

Usage inside :class:`SessionScheduler`::

    self._stream_pool = StreamProcessPool(
        num_workers=self.num_slots,
        backend_name=backend_name,
    )
    self._stream_pool.start()

    # In _stream_single_request:
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        self._stream_pool.executor,
        _stream_worker,
        prompt_str,
        self._stream_pool.backend_name,
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

from backends.base_backend import StreamingChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Result type returned by every worker process
# ---------------------------------------------------------------------------


@dataclass
class StreamResult:
    """Summary of a single streaming inference call.

    Attributes
    ----------
    start_time : float
        ``time.perf_counter()`` value captured just before streaming begins.
    first_token_time : float or None
        ``time.perf_counter()`` when the first non-empty token arrived.
        ``None`` if the stream produced no tokens.
    first_chunk : StreamingChunk or None
        The first chunk that contained a non-empty token / token_ids.
    last_chunk : StreamingChunk or None
        The final chunk where ``is_finished == True``.  Carries ``usage``
        when the backend provides it.
    end_time : float
        ``time.perf_counter()`` value captured after the stream finished.
    accumulated_text : str
        Full concatenated text of all tokens in the stream.
    """

    start_time: float
    first_token_time: Optional[float] = None
    first_chunk: Optional[StreamingChunk] = None
    last_chunk: Optional[StreamingChunk] = None
    end_time: float = 0.0
    accumulated_text: str = ""


# ---------------------------------------------------------------------------
#  Per-process state — set once by _pool_initializer
# ---------------------------------------------------------------------------

_worker_backend = None
_worker_loop = None


def _pool_initializer(backend_name: str) -> None:
    """Called **once** by ProcessPoolExecutor when each worker process starts.

    Creates and initialises a backend instance and a persistent event loop
    that live for the lifetime of the worker process.  All subsequent
    :func:`_stream_worker` calls in the same process reuse them.
    """
    global _worker_backend, _worker_loop

    _worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_loop)

    from utils.backend_registry import get_backend_instance

    _worker_backend = get_backend_instance(backend_name)
    _worker_backend.initialize()
    logger.info("Trigger process backend initialised — backend=%s", backend_name)


# ---------------------------------------------------------------------------
#  Worker function (module-level → picklable by ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _stream_worker(prompt_str: str, backend_name: str) -> Optional[StreamResult]:
    """Run a single streaming request inside a child process.

    Uses the persistent event loop created by :func:`_pool_initializer`,
    allowing the backend's :class:`httpx.AsyncClient` connection pool to
    be reused across requests within the same worker process.

    Parameters
    ----------
    prompt_str : str
        JSON-encoded request payload.
    backend_name : str
        Backend registry key (unused; present for signature compatibility).

    Returns
    -------
    StreamResult
        Timing, chunk, and text data for the parent process.
    """
    asyncio.set_event_loop(_worker_loop)
    return _worker_loop.run_until_complete(
        _async_stream_one(_worker_backend, prompt_str)
    )


async def _async_stream_one(backend, prompt_str: str) -> Optional[StreamResult]:
    """Async body executed inside the child process's temporary event loop.

    Returns :class:`StreamResult` on success, or ``None`` if an error
    occurred during streaming.
    """
    try:
        start_time = time.perf_counter()

        stream_gens = await backend.generate_stream(text_prompts=[prompt_str])
        stream_gen = stream_gens[0]

        first_chunk: Optional[StreamingChunk] = None
        first_token_time: Optional[float] = None
        last_chunk: Optional[StreamingChunk] = None
        accumulated_text_parts: List[str] = []

        async for chunk in stream_gen:
            if chunk.token:
                accumulated_text_parts.append(chunk.token)
            if first_chunk is None and (chunk.token or chunk.token_ids):
                first_chunk = chunk
                first_token_time = time.perf_counter()
            if chunk.is_finished:
                last_chunk = chunk
            if chunk.start_time:
                start_time = chunk.start_time

        end_time = time.perf_counter()
        return StreamResult(
            start_time=start_time,
            first_token_time=first_token_time,
            first_chunk=first_chunk,
            last_chunk=last_chunk,
            end_time=end_time,
            accumulated_text="".join(accumulated_text_parts),
        )
    except Exception as e:
        logger.exception(f"_async_stream_one | streaming failed: {e}")
        return None


# ---------------------------------------------------------------------------
#  Process-pool manager
# ---------------------------------------------------------------------------


class StreamProcessPool:
    """Pool of worker processes for running streaming inference requests.

    Each child process runs :func:`_pool_initializer` exactly once at
    start-up, creating a backend instance that is reused for every
    subsequent request dispatched to that process.

    Parameters
    ----------
    num_workers : int
        Maximum number of concurrent worker processes.
    backend_name : str
        Backend registry key passed to each worker's initializer.
    """

    def __init__(self, num_workers: int, backend_name: str) -> None:
        self.num_workers = num_workers
        self.backend_name = backend_name
        self._executor: Optional[ProcessPoolExecutor] = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Launch the process pool with per-worker backend initialisation."""
        if self._executor is not None:
            return
        self._executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_pool_initializer,
            initargs=(self.backend_name,),
        )
        logger.info(
            "StreamProcessPool started — %d triggers, backend=%s",
            self.num_workers,
            self.backend_name,
        )

    def shutdown(self) -> None:
        """Shut down the pool, waiting for all pending work to finish."""
        if self._executor is None:
            return
        logger.info("StreamProcessPool shutting down …")
        self._executor.shutdown(wait=True)
        self._executor = None
        logger.info("StreamProcessPool shut down")

    @property
    def executor(self) -> ProcessPoolExecutor:
        """The underlying :class:`~concurrent.futures.ProcessPoolExecutor`."""
        if self._executor is None:
            raise RuntimeError("StreamProcessPool not started — call start() first")
        return self._executor

    # -- submission ----------------------------------------------------------

    def submit(self, prompt_str: str) -> Future:
        """Submit one prompt and return a :class:`~concurrent.futures.Future`.

        The future resolves to a :class:`StreamResult`.
        """
        return self.executor.submit(
            _stream_worker,
            prompt_str,
            self.backend_name,
        )

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> StreamProcessPool:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.shutdown()
