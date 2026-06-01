#!/usr/bin/env python3
"""Offline SUT implementation for MLPerf inference benchmarks.

Supports multi-turn conversation dependency replay:
LoadGen samples only user-ending requests (via sample_pool).
When a sample is received, the SUT replays the subsequent
assistant-ending requests in the same conversation (dependencies)
in order, stopping before the next user speaks.  Every dependency
is measured.
"""

import asyncio
import json
import logging
import threading
from typing import Any, Dict, List

import mlperf_loadgen as lg
import numpy as np

from backends import BaseBackend

from .base_sut import BaseSUT

logger = logging.getLogger(__name__)


class OfflineSUT(BaseSUT):
    """Offline scenario SUT with dependency replay.

    Collects all queries into a batch, then replays each query's
    dependency chain through the backend.
    """

    def __init__(
        self,
        backend: BaseBackend,
        llm_requests: Any,  # List[Dict] or DiskCacheRequestStore
        sample_pool: List[int],
        dependencies: List[List[int]],
        name: str = "OfflineSUT",
    ):
        super().__init__(name)
        self.backend = backend
        self.llm_requests = llm_requests
        self.sample_pool = sample_pool
        self.dependencies = dependencies

        self.loop = None
        self.loop_thread = None
        self.query_samples = []
        self.queries_lock = threading.Lock()
        self.batch_future = None

        # chain_results[query_id] = [DependencyResult, ...]
        self.chain_results: Dict[int, List[Dict[str, Any]]] = {}
        self.chain_results_lock = threading.Lock()
        self.index_to_id: Dict[int, int] = {}

    # ------------------------------------------------------------------
    #  LoadGen callbacks
    # ------------------------------------------------------------------

    def issue_queries(self, query_samples: List[lg.QuerySample]) -> None:
        """Issue queries — collect all queries for batch processing."""
        logger.info(f"Issuing {len(query_samples)} queries (dependency replay mode)")

        with self.queries_lock:
            self.query_samples = query_samples
            for sample in query_samples:
                self.index_to_id[sample.index] = sample.id

        future = asyncio.run_coroutine_threadsafe(
            self._process_all_chains_async(), self.loop
        )
        self.batch_future = future

    def flush_queries(self) -> None:
        """Wait for all dependency chains to complete."""
        logger.info("Flushing queries...")
        if self.batch_future is not None:
            try:
                self.batch_future.result(timeout=1200)
                logger.info("Batch processing completed")
            except Exception as e:
                logger.error(f"Error waiting for batch completion: {e}")
            finally:
                self.batch_future = None

    # ------------------------------------------------------------------
    #  Batch chain processing
    # ------------------------------------------------------------------

    async def _process_all_chains_async(self) -> None:
        """Process all queries' dependency chains in the batch."""
        try:
            with self.queries_lock:
                query_samples = self.query_samples.copy()

            if not query_samples:
                logger.warning("No queries to process")
                return

            logger.info(f"Processing {len(query_samples)} queries' dependency chains")

            for sample in query_samples:
                chain_results = await self._replay_single_chain(sample)
                with self.chain_results_lock:
                    self.chain_results[sample.id] = chain_results

        except Exception as e:
            logger.error(f"Error during batch chain processing: {e}")
            import traceback

            traceback.print_exc()
            raise

    async def _replay_single_chain(
        self, sample: lg.QuerySample
    ) -> List[Dict[str, Any]]:
        """Replay a single query's dependency chain in order."""
        sp_idx = sample.index
        dep_indices = self.dependencies[sp_idx]
        chain_results: List[Dict[str, Any]] = []

        for i, dep_idx in enumerate(dep_indices):
            is_target = i == len(dep_indices) - 1

            # Build the full payload: messages + extra keys (tools, etc.)
            request = self.llm_requests[dep_idx]
            prompt_str = json.dumps(request, ensure_ascii=False)

            # Send to backend (non-streaming for offline)
            results = self.backend.generate(text_prompts=[prompt_str])
            result = results[0] if results else {}

            dep_result = {
                "dep_idx": dep_idx,
                "is_target": is_target,
                "model_output": result.get("text", ""),
                "tok_model_output": result.get("tokens", []),
                "tok_model_output_len": len(result.get("tokens", [])),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "cached_tokens": result.get("cached_tokens", 0),
                "reasoning_tokens": result.get("reasoning_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
            }
            chain_results.append(dep_result)

        # Signal completion for the target dependency (the sample itself)
        target_result = chain_results[-1]
        await self._send_result_to_loadgen(sample, target_result)

        return chain_results

    # ------------------------------------------------------------------
    #  LoadGen response
    # ------------------------------------------------------------------

    async def _send_result_to_loadgen(
        self, sample: lg.QuerySample, dep_result: Dict[str, Any]
    ) -> None:
        """Send the target dependency's result to LoadGen."""
        tokens = dep_result.get("tok_model_output", [1])
        if not tokens:
            tokens = [1]

        token_array = np.array(tokens, dtype=np.int32)
        n_tokens = dep_result.get("completion_tokens", len(tokens))

        response = lg.QuerySampleResponse(
            sample.id,
            token_array.ctypes.data if token_array.size > 0 else 0,
            token_array.nbytes,
            n_tokens,
        )
        lg.QuerySamplesComplete([response])
        logger.debug(
            f"Sent {n_tokens} tokens for query {sample.id} "
            f"(target dep_idx={dep_result.get('dep_idx')})"
        )

    # ------------------------------------------------------------------
    #  Event loop management
    # ------------------------------------------------------------------

    def _run_event_loop(self):
        """Run the async event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self) -> lg.ConstructSUT:
        """Start the SUT and async event loop."""
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()
        return super().start()

    def stop(self) -> None:
        """Stop the SUT and clean up."""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join()
        if self.loop:
            self.loop.close()
            self.loop = None
        super().stop()

    # ------------------------------------------------------------------
    #  Result collection
    # ------------------------------------------------------------------

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all dependency results in order of sample_pool indices.

        Returns one entry per dependency, flattened across all queries.
        """
        ordered_results = []
        queried_sp_indices = sorted(self.index_to_id.keys())

        for sp_idx in queried_sp_indices:
            query_id = self.index_to_id[sp_idx]
            chain = self.chain_results.get(query_id, [])
            for dep in chain:
                ordered_results.append(dep)
            if not chain:
                raise RuntimeError(
                    f"No dependency chain result for sample_pool index {sp_idx}, "
                    f"query_id {query_id}"
                )

        logger.info(
            f"Retrieved {len(ordered_results)} dependency results "
            f"({len(queried_sp_indices)} queries)"
        )
        return ordered_results
