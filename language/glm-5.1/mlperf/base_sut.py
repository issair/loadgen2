#!/usr/bin/env python3
"""Base System Under Test (SUT) class for MLPerf inference benchmarks."""

import abc
import logging
from typing import List

import mlperf_loadgen as lg

logger = logging.getLogger(__name__)


class BaseSUT(abc.ABC):
    """Base class for MLPerf inference System Under Test (SUT)."""

    def __init__(self, name: str = "BaseSUT"):
        self.name = name
        self.sut = None
        logger.info(f"Initializing {self.name}")

    @abc.abstractmethod
    def issue_queries(self, query_samples: List[lg.QuerySample]) -> None:
        """Issue queries to the SUT."""
        raise NotImplementedError("Subclasses must implement issue_queries")

    @abc.abstractmethod
    def flush_queries(self) -> None:
        """Flush all pending queries."""
        raise NotImplementedError("Subclasses must implement flush_queries")

    def start(self) -> lg.ConstructSUT:
        """Start the SUT and return the LoadGen SUT handle."""
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        logger.info(f"{self.name} started")
        return self.sut

    def stop(self) -> None:
        """Stop the SUT and clean up resources."""
        if self.sut:
            lg.DestroySUT(self.sut)
            self.sut = None
            logger.info(f"{self.name} stopped")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
