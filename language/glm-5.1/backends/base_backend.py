import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional


@dataclass(slots=True)
class StreamingChunk:
    """Standardized streaming response chunk."""

    token: str
    token_ids: List[int]
    is_finished: bool
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    start_time: Optional[float] = None


class BaseBackend(ABC):
    """Abstract base class for all inference backends."""

    def __init__(self):
        """Initialize base backend attributes."""
        self.is_initialized = False

    @property
    def backend_name(self) -> str:
        """Get backend name from class name or registry."""
        class_name = self.__class__.__name__
        if class_name.endswith("Backend"):
            name = class_name[:-7].lower()
        else:
            name = class_name.lower()

        from utils.backend_registry import BACKEND_REGISTRY

        if name not in BACKEND_REGISTRY:
            return class_name.lower()
        return name

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend (load tokenizer, setup clients, etc.)."""
        pass

    @abstractmethod
    def generate(
        self,
        tokenized_prompts: Optional[List[List[int]]] = None,
        text_prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for a list of prompts synchronously.

        Args:
            tokenized_prompts: List of pre-tokenized prompts (token IDs)
            text_prompts: List of text prompts

        Returns:
            List of dictionaries with standardized output format
        """
        pass

    @abstractmethod
    def generate_async(
        self,
        tokenized_prompts: Optional[List[List[int]]] = None,
        text_prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[asyncio.Future]:
        """
        Generate responses for a list of prompts asynchronously.

        Returns:
            List of futures that will resolve to dictionaries with standardized output format
        """
        pass

    async def generate_stream(
        self,
        tokenized_prompts: Optional[List[List[int]]] = None,
        text_prompts: Optional[List[str]] = None,
        sid: Optional[str] = None,
        **kwargs,
    ) -> List[AsyncIterator[StreamingChunk]]:
        """
        Generate responses for a list of prompts with streaming.

        Raises:
            NotImplementedError: If backend doesn't support streaming
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming generation"
        )

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources and shut down the backend."""
        pass

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
