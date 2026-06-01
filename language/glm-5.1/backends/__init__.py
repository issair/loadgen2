"""
Modular backend system for MLPerf GLM-5.1 reference implementation.

Supports OpenAI-compatible API backend for GLM model inference.
"""

from .base_backend import BaseBackend

__all__ = [
    "BaseBackend",
]
