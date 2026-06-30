"""SessionSUT — Path-B SUT: self-managed scheduling, LoadGen as measurement harness.

LoadGen runs in Offline mode, issuing every user turn as a single sample
all at once.  Samples are fed into a global FIFO pool inside
:class:`SessionScheduler` — no per-session pre-allocation is needed.
TTFT / TPOT are computed by the SUT from the inference stream.
"""

import asyncio
import logging
import threading
from typing import Any, Dict, List

import mlperf_loadgen as lg

from backends.base_backend import BaseBackend

from .base_sut import BaseSUT
from .session_scheduler import SessionScheduler

logger = logging.getLogger(__name__)


class SessionSUT(BaseSUT):
    """Path-B System Under Test.

    Parameters
    ----------
    backend : BaseBackend
        Initialised inference backend.
    sessions : dict[str, LargeModelSession]
        Virtual-user sessions produced by :func:`process_dataset`.
    num_triggers : int
        Maximum concurrent turns (default 10).
    time_unit : str
        ``"s"`` or ``"ms"`` — must match the Poisson-delta unit in *sessions*.
    backend_name : str
        Backend registry key for the process-pool workers (default ``"openai"``).
    """

    def __init__(
        self,
        backend: BaseBackend,
        sessions: Dict[str, Any],
        num_triggers: int = 10,
        time_unit: str = "s",
        backend_name: str = "openai",
        go_to_end: bool = False,
        poisson_seed: int = 0,
        poisson_lam: int = 10,
        poisson_pool_size: int = 1000,
        round_from: int = 0,
        round_to: int = 1,
        name: str = "SessionSUT",
    ) -> None:
        super().__init__(name)
        self.backend = backend
        self.sessions = sessions
        self.num_triggers = num_triggers
        self.time_unit = time_unit
        self.backend_name = backend_name
        self.go_to_end = go_to_end

        self._total_turns = sum(len(s.user_conv_request) for s in sessions.values())
        logger.info(
            f"SessionSUT: {len(sessions)} trace, "
            f"{self._total_turns} total turns, "
            f"{num_triggers} triggers"
        )

        self._scheduler = SessionScheduler(
            sessions=sessions,
            backend=backend,
            num_triggers=num_triggers,
            time_unit=time_unit,
            backend_name=backend_name,
            go_to_end=go_to_end,
            poisson_seed=poisson_seed,
            poisson_lam=poisson_lam,
            poisson_pool_size=poisson_pool_size,
            round_from=round_from,
            round_to=round_to,
        )

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._scheduler_future: Any = None  # asyncio.Future

    # ------------------------------------------------------------------
    #  LoadGen callbacks
    # ------------------------------------------------------------------

    def issue_queries(self, query_samples: List[lg.QuerySample]) -> None:
        """Receive all user-turn samples from LoadGen (Offline --- once).

        Registers samples with the scheduler and launches the worker pool.
        """
        logger.debug(f"issue_queries: {len(query_samples)} samples")

        if len(query_samples) != self._total_turns:
            logger.debug(
                f"Sample count mismatch: expected {self._total_turns}, "
                f"got {len(query_samples)}"
            )

        # Feed all samples into the scheduler's global pool.
        self._scheduler.register_samples(query_samples)

        # Launch the scheduler in the background event loop.
        if self._loop is None:
            raise RuntimeError("Event loop not started --- call start() first")

        self._scheduler_future = asyncio.run_coroutine_threadsafe(
            self._scheduler.run(), self._loop
        )

    def flush_queries(self) -> None:
        """Block until the scheduler finishes every turn."""
        logger.info("flush_queries: waiting for scheduler to drain ...")
        if self._scheduler_future is None:
            logger.warning("flush_queries called before issue_queries")
            return

        timeout_s = 7200  # 2 h
        waited = 0.0
        poll_interval = 2.0
        while not self._scheduler_future.done():
            if waited >= timeout_s:
                logger.error(
                    "flush_queries timed out after %.0fs --- %d/%d turns done",
                    waited,
                    self._scheduler.completed,
                    self._scheduler.total,
                )
                return
            try:
                self._scheduler_future.result(timeout=poll_interval)
            except TimeoutError:
                pass
            waited += poll_interval
        logger.info(
            "flush_queries complete --- %d/%d turns",
            self._scheduler.completed,
            self._scheduler.total,
        )

    # ------------------------------------------------------------------
    #  Event loop
    # ------------------------------------------------------------------

    def start(self) -> lg.ConstructSUT:
        """Start async event loop (background thread), construct LoadGen SUT."""
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()
        logger.info("SessionSUT event loop started")
        return super().start()

    def stop(self) -> None:
        """Tear down event loop and LoadGen SUT."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)
        if self._loop is not None:
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None
        super().stop()

    def _run_event_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    # ------------------------------------------------------------------
    #  Results
    # ------------------------------------------------------------------

    def get_results(self) -> List[Dict[str, Any]]:
        """Return collected metrics (same format as ``ServerSUT.get_results``)."""
        return self._scheduler.get_stats_results()
