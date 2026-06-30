"""SessionScheduler — per-VU session scheduling with N concurrent triggers.

Receives ``issue_queries`` samples from LoadGen (Offline, all at once)
and dispatches them through a fixed number of worker triggers.  Each trigger
processes one virtual user's turn (the full dependency chain) at a time.
Poisson-distributed inter-turn intervals (``user_conv_ps_delta``) control
when a VU's next turn becomes ready.

Samples are held in a single global FIFO pool.  When a worker picks up
a turn it draws the next sample from the pool — no per-session mapping
is needed.  TTFT / TPOT are computed by SessionSUT from the stream.
"""

import asyncio
import io
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import mlperf_loadgen as lg
import numpy as np
import orjson

from backends.base_backend import BaseBackend

from .session_stats import SessionStats
from .stream_process_pool import StreamProcessPool, StreamResult, _stream_worker

logger = logging.getLogger(__name__)

# Sentinel pushed into ready_queue when all turns are done.
_SENTINEL = object()


def warnup(s: str, p: str):
    pass


class SessionScheduler:
    """Schedule virtual-user conversation turns across a fixed triggers pool.

    Parameters
    ----------
    sessions : dict[str, LargeModelSession]
        All virtual-user sessions keyed by session id.
    backend : BaseBackend
        Initialised inference backend.
    num_triggers : int
        Maximum number of concurrently executing turns (default 10).
    time_unit : str
        Unit of ``user_conv_ps_delta`` / ``executed_time`` values.
        ``"s"`` (seconds) or ``"ms"`` (milliseconds).  Default ``"s"``.
    backend_name : str
        Backend registry key used by child processes (default ``"openai"``).
    """

    def __init__(
        self,
        sessions: Dict[str, Any],  # Dict[str, LargeModelSession]
        backend: BaseBackend,
        num_triggers: int = 10,
        time_unit: str = "s",
        backend_name: str = "openai",
        go_to_end: bool = False,
        poisson_seed: int = 0,
        poisson_lam: int = 10,
        poisson_pool_size: int = 1000,
        round_from: int = 0,
        round_to: int = 1,
    ) -> None:
        self.sessions = sessions
        self.backend = backend
        self.num_triggers = num_triggers
        self.time_unit = time_unit
        self.backend_name = backend_name
        self.go_to_end = go_to_end
        self._poisson_seed = poisson_seed
        self._poisson_lam = poisson_lam
        self._poisson_pool_size = poisson_pool_size
        self._round_from = round_from
        self._round_to = round_to
        self._early_term_k = float(os.getenv("EARLY_TERM_K", "1.0"))  # 0.0-1.0

        # Global RNG — each worker generates its own pool from it.
        self._rng = np.random.default_rng(seed=poisson_seed)

        # Multi-process stream pool — one backend per child process.
        self._stream_pool = StreamProcessPool(
            num_workers=num_triggers, backend_name=backend_name
        )

        # Dedicated thread pool for orjson serialization.
        self._json_executor = ThreadPoolExecutor(max_workers=4)

        # Per-request metrics collector.
        self._stats = SessionStats()

        # Concurrency gate.
        self._semaphore = asyncio.Semaphore(num_triggers)

        # Global FIFO pool of LoadGen QuerySample objects.
        # Workers draw one sample per turn — no per-session pre-allocation.
        self._sample_pool: asyncio.Queue[lg.QuerySample] = asyncio.Queue()

        # Total progress tracking.
        rounds = round_to - round_from
        self._total_turns = sum(len(s.user_conv_request) for s in sessions.values()) * rounds
        self._completed_turns = 0
        self._session_turns: Dict[str, int] = {sid: 0 for sid in sessions}
        self._turns_lock = asyncio.Lock()

        # Session completion tracking (for early termination).
        self._completed_sessions: int = 0
        self._total_sessions: int = len(sessions)

        # Per-session round tracking.
        self._session_rounds: Dict[str, int] = {sid: round_from for sid in sessions}

        # Set when all turns have completed.
        self._done_event = asyncio.Event()

    # ------------------------------------------------------------------
    #  Public API (called from SessionSUT)
    # ------------------------------------------------------------------

    def register_samples(self, query_samples: List[lg.QuerySample]) -> None:
        """Feed LoadGen *query_samples* into the global FIFO pool.

        Must be called once, from ``SessionSUT.issue_queries``, before
        :meth:`run` is scheduled.

        Parameters
        ----------
        query_samples : list[lg.QuerySample]
            The samples delivered by LoadGen.
        """
        for qs in query_samples:
            self._sample_pool.put_nowait(qs)
        logger.debug(f"Registered {len(query_samples)} samples into global pool")

    async def run(self) -> None:
        """Bootstrap ready-queue and launch worker coroutines.

        Blocks until every user turn of every session has been processed.
        """
        await self._bootstrap()
        self._stream_pool.start()

        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(
                self._stream_pool.executor,
                warnup,
                "warnup",
                self._stream_pool.backend_name,
            )
        except Exception as e:
            logger.exception(f"execept,{e}")

        await asyncio.sleep(3)
        try:
            workers = [
                asyncio.create_task(self._worker(i)) for i in range(self.num_triggers)
            ]

            # Periodically dump task stacks while waiting for workers.
            async def _stack_monitor():
                while True:
                    await asyncio.sleep(30)
                    logger.warning("--- Task stacks (every 30s) ---")
                    try:
                        for t in asyncio.all_tasks():
                            if t is not asyncio.current_task() and not t.done():
                                with io.StringIO() as buf:
                                    t.print_stack(file=buf)
                                    frames = buf.getvalue()
                                # Also traverse the await chain to see deeper.
                                coro = t.get_coro()
                                depth = 0
                                while (
                                    coro is not None
                                    and hasattr(coro, "cr_await")
                                    and coro.cr_await is not None
                                ):
                                    await_coro = coro.cr_await
                                    if (
                                        hasattr(await_coro, "cr_frame")
                                        and await_coro.cr_frame is not None
                                    ):
                                        fname = await_coro.cr_frame.f_code.co_filename
                                        lineno = await_coro.cr_frame.f_lineno
                                        with open(fname) as fp:
                                            line = fp.readlines()[lineno - 1].strip()
                                        depth += 1
                                        frames += f"  [await depth {depth}] {fname}:{lineno}  {line}\n"
                                    else:
                                        depth += 1
                                        frames += f"  [await depth {depth}] {type(await_coro).__name__}\n"
                                    coro = await_coro
                                logger.warning("Task %s:\n%s", t.get_name(), frames)
                    except Exception as e:
                        logger.error(f"{e}")
                    logger.warning("--- End task stacks ---")

            if not os.getenv("PRINT_STACK", None):

                async def noop():
                    pass

                _stack_monitor = noop

            monitor = asyncio.create_task(_stack_monitor())
            try:
                await asyncio.gather(*workers)
            finally:
                monitor.cancel()
                try:
                    await monitor
                except asyncio.CancelledError:
                    pass

            await self._drain_remaining_samples()
        finally:
            self._stream_pool.shutdown()
            self._json_executor.shutdown(wait=True)
        logger.info("Scheduler finished — all turns processed")

    # ------------------------------------------------------------------
    #  Bootstrap
    # ------------------------------------------------------------------

    async def _bootstrap(self) -> None:
        """Seed the ready-queue with the first turn of every trace."""
        logger.info(
            f"Bootstrapped {len(self.sessions)} traces, "
            f"{self._total_turns} total turns, "
            f"{self.num_triggers} triggers"
        )

    # ------------------------------------------------------------------
    #  Poisson delay generator
    # ------------------------------------------------------------------

    async def _poisson_delays(self):
        """Async generator yielding Poisson-distributed delays, cycling forever.

        Each invocation (each worker) generates its own pool from the global RNG,
        so workers get independent yet deterministic sequences.
        """
        pool = self._rng.poisson(
            lam=self._poisson_lam, size=self._poisson_pool_size
        ).tolist()
        idx = 0
        while True:
            yield pool[idx]
            idx += 1
            if idx >= self._poisson_pool_size:
                idx = 0

    # ------------------------------------------------------------------
    #  Worker loop
    # ------------------------------------------------------------------

    async def _worker(self, worker_id: int) -> None:
        logger.debug(f"Trigger-{worker_id:02d} started")

        poisson_iter = self._poisson_delays().__aiter__()

        while not self._done_event.is_set():
            sid: str = self._rng.choice(list(self.sessions.keys()))
            async with self._semaphore:
                if self._done_event.is_set():
                    break
                processed = await self._process_turn(worker_id, sid, poisson_iter)

            # Only consume a Poisson interval when a turn was actually processed.
            if processed:
                delay = await anext(poisson_iter)
                await asyncio.sleep(self._to_seconds(delay))

        logger.debug(f"Trigger-{worker_id:02d} exiting — done_event set")

    # ------------------------------------------------------------------
    #  Turn processing
    # ------------------------------------------------------------------

    async def _process_turn(
        self,
        worker_id: int,
        sid: str,
        poisson_iter,
    ) -> bool:
        """Process one turn if any session is ready.

        Parameters
        ----------
        poisson_iter : AsyncIterator[int]
            Async iterator yielding Poisson-distributed delays, shared
            with the worker loop (cycles forever internally).

        Returns
        -------
        bool
            ``True`` if a turn was actually processed, ``False`` if no
            session had a ready request.
        """
        session = self.sessions[sid]

        wait_time, pos = session.next_execute_request(time.time())
        if pos == -1:
            # The randomly-chosen session isn't ready.  Try the
            # remaining sessions in *random* order so that concurrent
            # workers spread across different ready sessions instead
            # of converging on the same insertion-order prefix.
            keys = list(self.sessions.keys())
            random.shuffle(keys)
            for next_sid in keys:
                if next_sid == sid:
                    continue
                session = self.sessions[next_sid]
                wait_time, pos = session.next_execute_request(time.time())
                if pos != -1:
                    sid = next_sid
                    break
            else:
                # No session has a ready request — back off briefly.
                await asyncio.sleep(0.01)
                return False

        # Determine the current round for this session.
        round_num = self._session_rounds[sid]

        logger.info(f"Trigger-{worker_id:02d} | {sid} | turn pos={pos} round={round_num}")

        chain_start = time.perf_counter()

        request_in_chain = 0
        try:
            async for req_dict in session.iter_requests_until_next_user(pos, go_to_end=self.go_to_end):
                end_at = req_dict.get("end_at", -1)
                role = req_dict.get("last_role", "unkown")

                # If this is not the first request in the chain and the
                # role is user, sleep for a Poisson-distributed interval
                # (same pool as the worker-level delay between turns).
                if request_in_chain > 0 and role == "user":
                    delay = await anext(poisson_iter)
                    await asyncio.sleep(self._to_seconds(delay))

                if self._done_event.is_set():
                    break

                # Offload orjson serialization to the dedicated thread pool.
                loop = asyncio.get_running_loop()
                prompt_bytes: bytes = await loop.run_in_executor(
                    self._json_executor,
                    orjson.dumps,
                    req_dict,
                )
                prompt_str = prompt_bytes.decode("utf-8")
                # ── record start (returns UUID) ───────────────────────
                rid = self._stats.record_start(
                    worker_id=worker_id,
                    sid=sid,
                    pos=pos,
                    end_at=end_at,
                    role=role,
                    start_time=time.perf_counter(),
                )

                result = await self._stream_single_request(sid, prompt_str, round_num)

                if result is None:
                    raise RuntimeError(
                        f"Trigger-{worker_id:02d} | {sid} | "
                        f"_stream_single_request returned None at pos={pos}"
                    )

                if result.last_chunk:
                    last_chunk = result.last_chunk
                    if last_chunk.usage:
                        usage = last_chunk.usage.get("")

                # ── record end (by UUID) ──────────────────────────────
                self._stats.record_end(
                    request_id=rid,
                    result=result,
                    tokenizer=self._get_tokenizer(),
                )

                request_in_chain = request_in_chain + 1
        except Exception as e:
            logger.exception(f"Trigger-{worker_id:02d} | {sid} | FAIL pos={pos}, {e}")
        finally:
            # Always mark the turn as ended, regardless of success/error.
            # When there are more rounds, reset the session.
            should_reset = (round_num + 1 < self._round_to)
            session_all_done = session.end_request(
                pos, time.time(), sid, full_reset=should_reset
            )
            if session_all_done and should_reset:
                self._session_rounds[sid] = round_num + 1

        chain_elapsed_s = time.perf_counter() - chain_start

        async with self._turns_lock:
            self._completed_turns += 1
            self._session_turns[sid] += 1

            # Count completed sessions — only when permanently done
            # (no more rounds to reset for).
            if session_all_done:
                if round_num + 1 >= self._round_to:
                    self._completed_sessions += 1

            done = self._completed_turns
            session_done = self._session_turns[sid]
            sessions_done = self._completed_sessions

            # Early termination: if the number of remaining sessions that can
            # still run is fewer than the number of triggers, signal done
            # early — the remaining turns belong to sessions already inflight
            # and will be drained naturally.
            remaining_sessions = self._total_sessions - sessions_done
            if remaining_sessions < self.num_triggers * self._early_term_k and not self._done_event.is_set():
                logger.warning(
                    f"Early done: remaining sessions ({remaining_sessions}) "
                    f"< triggers ({self.num_triggers}), "
                    f"turns done ({done}/{self._total_turns}), "
                    f"sessions done ({sessions_done}/{self._total_sessions})"
                )
                self._done_event.set()

        logger.info(
            f"Trigger-{worker_id:02d} | {sid} | {request_in_chain} request(s) done "
            f"({done + 1}/{self._total_turns}) session_runs={session_done}"
            f"  elapsed={chain_elapsed_s:.1f}s"
        )

        # If all turns are done, push sentinel so workers drain.
        if done >= self._total_turns:
            self._done_event.set()

        return True

    # ------------------------------------------------------------------
    #  Streaming helpers
    # ------------------------------------------------------------------

    async def _stream_single_request(
        self, sid: str, prompt_str: str, round_num: int
    ) -> Optional[StreamResult]:
        """Stream a single request through a **child process**.

        Returns the :class:`~.stream_process_pool.StreamResult` or ``None``
        if the worker encountered an error.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._stream_pool.executor,
            _stream_worker,
            prompt_str,
            self._stream_pool.backend_name,
            round_num,
            sid,
        )

        if result is not None:
            if result.first_chunk is not None:
                await self._send_first_token()

            if result.last_chunk is not None:
                await self._send_final_response()

        return result

    async def _send_first_token(self) -> None:
        pass

    async def _send_final_response(self) -> None:
        pass

    # ------------------------------------------------------------------
    #  Tokenizer helper
    # ------------------------------------------------------------------

    def _get_tokenizer(self):
        """Return a tokenizer ``encode(text) -> list[int]`` if available."""
        if hasattr(self.backend, "tokenizer") and self.backend.tokenizer:
            return self.backend.tokenizer
        return None

    # ------------------------------------------------------------------
    #  Drain & shutdown
    # ------------------------------------------------------------------

    async def _drain_remaining_samples(self) -> None:
        """Drain every remaining sample from the pool.

        Sends ``FirstTokenComplete`` + ``QuerySamplesComplete`` for
        each sample so LoadGen can finish cleanly.
        """
        remaining: List[lg.QuerySample] = []
        while True:
            try:
                sample = self._sample_pool.get_nowait()
                remaining.append(sample)
            except asyncio.QueueEmpty:
                break

        if not remaining:
            return

        logger.debug(f"Draining {len(remaining)} remaining samples from pool")
        for sample in remaining:
            token_array = np.array([1], dtype=np.int32)
            buf_ptr = token_array.ctypes.data
            buf_nbytes = token_array.nbytes

            first_resp = lg.QuerySampleResponse(sample.id, buf_ptr, buf_nbytes, 1)
            lg.FirstTokenComplete([first_resp])

            final_resp = lg.QuerySampleResponse(sample.id, buf_ptr, buf_nbytes, 2)
            lg.QuerySamplesComplete([final_resp])

        logger.debug(f"Drained {len(remaining)} samples")

    async def _send_empty_response(self, sample: lg.QuerySample) -> None:
        token_array = np.array([1], dtype=np.int32)
        response = lg.QuerySampleResponse(
            sample.id, token_array.ctypes.data, token_array.nbytes, 2
        )
        lg.QuerySamplesComplete([response])

    # ------------------------------------------------------------------
    #  Utility
    # ------------------------------------------------------------------

    def _to_seconds(self, value: int) -> float:
        """Convert a time value from *time_unit* to seconds."""
        return value / 1000.0 if self.time_unit == "ms" else float(value)

    @property
    def completed(self) -> int:
        return self._completed_turns

    @property
    def total(self) -> int:
        return self._total_turns

    # -- metrics -----------------------------------------------------------

    def get_stats_results(self) -> List[Dict[str, Any]]:
        """Return collected metrics in the same format as ``ServerSUT.get_results``."""
        return self._stats.get_results()

    @property
    def stats(self) -> SessionStats:
        return self._stats
