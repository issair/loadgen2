#!/usr/bin/env python3
"""Server SUT implementation for MLPerf inference benchmarks.

Supports multi-turn conversation dependency replay via LoadGen's
enable_dependency_replay: LoadGen samples user-ending requests,
expands each into a multi-sample query (the full dependency chain),
and each sample gets independent latency measurement.

The SUT processes samples sequentially and calls QuerySampleStart
before each, then QuerySamplesComplete after each.
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Set

import mlperf_loadgen as lg
import numpy as np

from backends import BaseBackend
from backends.base_backend import StreamingChunk

from .base_sut import BaseSUT

logger = logging.getLogger(__name__)


@dataclass
class DependencyResult:
    """Result of a single dependency in the chain."""

    dep_idx: int
    is_target: bool
    first_token_time: Optional[float] = None
    completion_time: Optional[float] = None
    accumulated_text: str = ""
    tokens: List[int] = field(default_factory=list)
    prompt_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    completion_tokens: int = 0


class ServerSUT(BaseSUT):
    """Server scenario SUT with multi-turn dependency replay."""

    def __init__(
        self,
        backend: BaseBackend,
        llm_requests: Any,
        sample_pool: List[int],
        dependencies: List[List[int]],
        name: str = "ServerSUT",
    ):
        super().__init__(name)
        self.backend = backend
        self.llm_requests = llm_requests
        self.sample_pool = sample_pool
        self.dependencies = dependencies

        self.backend_name = getattr(
            backend, "backend_name", type(backend).__name__.lower()
        )

        self.loop = None
        self.loop_thread = None
        self.active_tasks: Set[asyncio.Task] = set()
        self.active_tasks_lock = asyncio.Lock()
        self.chain_results: Dict[int, List[DependencyResult]] = {}
        self.chain_results_lock = asyncio.Lock()
        self.index_to_id: Dict[int, int] = {}

    # ------------------------------------------------------------------
    #  LoadGen callbacks
    # ------------------------------------------------------------------

    def issue_queries(self, query_samples: List[lg.QuerySample]) -> None:
        for sample in query_samples:
            self.index_to_id[sample.index] = sample.id

        logger.info(
            f"issue_queries: {len(query_samples)} samples, {len(self.active_tasks)} active_tasks"
        )
        if self.loop:
            coro = self._process_query_samples(query_samples)
            asyncio.run_coroutine_threadsafe(self._track_chain_task(coro), self.loop)

    def flush_queries(self) -> None:
        logger.info("Flushing queries...")
        if self.loop:
            future = asyncio.run_coroutine_threadsafe(
                self._wait_for_chains(), self.loop
            )
            waited = 0
            while True:
                try:
                    future.result(timeout=5)
                    break
                except Exception:
                    waited += 5
                    logger.warning(
                        f"Flush pending: waited {waited}s, "
                        f"{len(self.active_tasks)} active tasks"
                    )

    # ------------------------------------------------------------------
    #  Sequential sample processing
    # ------------------------------------------------------------------

    async def _process_query_samples(self, query_samples: List[lg.QuerySample]) -> None:
        try:
            chain_results: List[DependencyResult] = []
            for i, sample in enumerate(query_samples):
                dep_idx = sample.index
                dep_result = DependencyResult(
                    dep_idx=dep_idx,
                    is_target=(i == len(query_samples) - 1),
                )

                request = self.llm_requests[dep_idx]
                prompt_str = json.dumps(request, ensure_ascii=False)

                # --- per-sample independent start time ---
                start_marker = lg.QuerySampleResponse(sample.id, 0, 0, 0)
                lg.QuerySampleStart([start_marker])

                try:
                    stream_gens = await self.backend.generate_stream(
                        text_prompts=[prompt_str]
                    )
                    await self._process_single_sample(
                        sample, dep_result, stream_gens[0]
                    )
                except Exception:
                    logger.exception(f"[SUT] FAIL sample dep_idx={dep_idx}")
                    await self._send_empty_response_for(sample)

                chain_results.append(dep_result)

            first_id = query_samples[0].id if query_samples else 0
            async with self.chain_results_lock:
                self.chain_results[first_id] = chain_results
        except Exception as e:
            logger.exception(f"[SUT] _process_query_samples: {e}")
            raise

    async def _process_single_sample(
        self,
        sample: lg.QuerySample,
        dep_result: DependencyResult,
        stream_gen: AsyncIterator[StreamingChunk],
    ) -> None:
        start_time = time.perf_counter()

        async def _consume():
            first_token_sent = False
            async for chunk in stream_gen:
                if chunk.token:
                    dep_result.accumulated_text += chunk.token
                if not first_token_sent and chunk.token:
                    dep_result.first_token_time = time.perf_counter() - start_time
                    first_token_sent = True
                    await self._send_first_token_response_for(sample)
                if chunk.is_finished:
                    dep_result.completion_time = time.perf_counter() - start_time
                    self._extract_usage(dep_result, chunk)
                    await self._send_final_response_for(sample, dep_result)
                    return

        await asyncio.wait_for(_consume(), timeout=3000.0)

    async def _send_first_token_response_for(self, sample: lg.QuerySample) -> None:
        token_array = np.array([1], dtype=np.int32)
        response = lg.QuerySampleResponse(
            sample.id, token_array.ctypes.data, token_array.nbytes, 1
        )
        lg.FirstTokenComplete([response])

    async def _send_final_response_for(
        self, sample: lg.QuerySample, dep_result: DependencyResult
    ) -> None:
        tokens = dep_result.tokens or [1, 1]
        if len(tokens) < 2:
            tokens = [1, 1]
        n_completion = dep_result.completion_tokens or len(tokens)
        if n_completion < 2:
            n_completion = len(tokens)
        token_array = np.array(tokens, dtype=np.int32)
        response = lg.QuerySampleResponse(
            sample.id,
            token_array.ctypes.data if token_array.size > 0 else 0,
            token_array.nbytes,
            n_completion,
        )
        lg.QuerySamplesComplete([response])

    async def _send_empty_response_for(self, sample: lg.QuerySample) -> None:
        token_array = np.array([1], dtype=np.int32)
        response = lg.QuerySampleResponse(
            sample.id, token_array.ctypes.data, token_array.nbytes, 2
        )
        lg.QuerySamplesComplete([response])

    # ------------------------------------------------------------------
    #  Utility
    # ------------------------------------------------------------------

    def _extract_usage(
        self, dep_result: DependencyResult, chunk: StreamingChunk
    ) -> None:
        if chunk.usage:
            usage = chunk.usage
            dep_result.prompt_tokens = usage.get("prompt_tokens", 0)
            dep_result.completion_tokens = usage.get("completion_tokens", 0)
            dep_result.cached_tokens = usage.get("prompt_tokens_details", {}).get(
                "cached_tokens", 0
            )
            dep_result.reasoning_tokens = usage.get(
                "completion_tokens_details", {}
            ).get("reasoning_tokens", 0)

        text = dep_result.accumulated_text
        if text:
            try:
                if hasattr(self.backend, "tokenizer") and self.backend.tokenizer:
                    dep_result.tokens = self.backend.tokenizer.encode(text)
                else:
                    dep_result.tokens = [1]
            except Exception:
                dep_result.tokens = [1]
        else:
            dep_result.tokens = [1]

    # ------------------------------------------------------------------
    #  Event loop management
    # ------------------------------------------------------------------

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self) -> lg.ConstructSUT:
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()
        return super().start()

    def stop(self) -> None:
        if self.loop:
            try:
                stop_future = asyncio.run_coroutine_threadsafe(
                    self._cancel_all_tasks(), self.loop
                )
                stop_future.result(timeout=5)
            except Exception:
                pass
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=5)
        if self.loop:
            try:
                self.loop.close()
            except Exception:
                pass
            self.loop = None
        super().stop()

    async def _track_chain_task(self, coro) -> None:
        task = asyncio.create_task(coro)
        async with self.active_tasks_lock:
            self.active_tasks.add(task)
        try:
            await task
        finally:
            async with self.active_tasks_lock:
                self.active_tasks.discard(task)
                logger.info(
                    f"Task done, {len(self.active_tasks)} active_tasks remaining"
                )

    async def _cancel_all_tasks(self):
        async with self.active_tasks_lock:
            tasks = list(self.active_tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self.active_tasks.clear()

    async def _wait_for_chains(self):
        start = time.time()
        timeout = 60000
        while True:
            async with self.active_tasks_lock:
                if not self.active_tasks:
                    break
            if time.time() - start > timeout:
                logger.warning("Timeout waiting for dependency chains")
                break
            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    #  Result collection
    # ------------------------------------------------------------------

    def get_results(self) -> List[Dict[str, Any]]:
        ordered_results = []
        for query_key in sorted(self.chain_results.keys()):
            chain = self.chain_results[query_key]
            for dep in chain:
                ordered_results.append(
                    {
                        "dep_idx": dep.dep_idx,
                        "is_target": dep.is_target,
                        "model_output": dep.accumulated_text,
                        "tok_model_output": dep.tokens,
                        "tok_model_output_len": len(dep.tokens),
                        "first_token_time": dep.first_token_time,
                        "completion_time": dep.completion_time,
                        "prompt_tokens": dep.prompt_tokens,
                        "cached_tokens": dep.cached_tokens,
                        "reasoning_tokens": dep.reasoning_tokens,
                        "completion_tokens": dep.completion_tokens,
                    }
                )
        logger.info(
            f"Retrieved {len(ordered_results)} dependency results "
            f"({len(self.chain_results)} chains)"
        )
        return ordered_results
