"""
Utility functions for GLM-5.1 backends.
"""

import os
import random
from pathlib import Path

import numpy as np


def get_cache_directory() -> Path:
    """Get the cache directory.

    Returns:
        Path: The cache directory path
    """
    user = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
    cache_dir = Path(f"/raid/data/{user}/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def set_all_seeds(seed: int = 42) -> None:
    """Set seeds for all random number generators for reproducibility.

    Args:
        seed: The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
