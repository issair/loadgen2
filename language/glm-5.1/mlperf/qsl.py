"""QuerySampleLibrary implementations for MLPerf LoadGen."""

import logging
from typing import Any, List

import mlperf_loadgen as lg


class QuerySampleLibrary:
    """MLPerf QuerySampleLibrary implementation.

    Exposes all llm_requests.  When enable_dependency_replay is on,
    LoadGen calls get_dependency_chain to expand each sampled index
    into a multi-sample query where each sample gets independent
    latency measurement.
    """

    def __init__(
        self,
        llm_requests: Any,  # List[Dict] or DiskCacheRequestStore
        dependencies: List[List[int]],
        sample_pool: List[int],
        name: str = "QSL",
    ):
        self.llm_requests = llm_requests
        self.dependencies = dependencies
        self.sample_pool = sample_pool
        self.count = len(sample_pool)
        self.perf_count = self.count
        self.name = name
        self.logger = logging.getLogger(__name__)

        self.qsl = lg.ConstructQSL(
            self.count,
            self.perf_count,
            lambda x: None,  # LoadSamplesToRam
            lambda x: None,  # UnloadSamplesFromRam
            self._get_dependency_chain,
        )
        self.logger.info(
            f"Created {self.name} with {self.count} samples"
            f" (+ dependency chain support)"
        )

    def _get_dependency_chain(self, sample_pool_index: int) -> List[int]:
        """Return the full dependency chain for a sample_pool entry."""
        rs = self.dependencies[sample_pool_index]
        self.logger.info(f"dependency_chain: {sample_pool_index} -> {rs}")
        return rs

    def __del__(self):
        if self.qsl is not None:
            lg.DestroyQSL(self.qsl)
            self.logger.info(f"{self.name} destroyed")
