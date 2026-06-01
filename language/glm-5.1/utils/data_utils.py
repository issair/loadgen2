"""
Data utilities for loading datasets and saving results.
"""

import os
import pickle
import time
from typing import Any, Dict, Optional

import diskcache
import pandas as pd


def generate_timestamped_filename(output_file: str, add_timestamp: bool = True) -> str:
    """Generate filename with optional timestamp."""
    if not add_timestamp:
        return output_file
    timestamp_suffix = time.strftime("%Y%m%d_%H%M%S")
    base_name, ext = os.path.splitext(output_file)
    return f"{base_name}_{timestamp_suffix}{ext}"


def load_dataset(
    file_path: str, num_samples: Optional[int] = None, skip_samples: int = 0
) -> pd.DataFrame:
    """Load dataset from pickle file."""
    if not os.path.exists(file_path):
        from utils.validation import ValidationError

        raise ValidationError(f"Input file not found: {file_path}")

    print(f"Loading dataset from {file_path}...")
    try:
        with open(file_path, "rb") as f:
            df = pd.read_pickle(f)
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {str(e)}")

    print(f"Loaded {len(df)} samples")

    if skip_samples > 0:
        if skip_samples >= len(df):
            raise ValueError(
                f"skip_samples ({skip_samples}) must be less than total samples ({len(df)})"
            )
        original_length = len(df)
        df = df.iloc[skip_samples:].reset_index(drop=True)
        print(f"Skipped first {skip_samples} samples (from {original_length} total)")

    if num_samples is not None:
        original_length = len(df)
        df = df.head(num_samples)
        print(
            f"Limited to {len(df)} samples (from {original_length} total after skipping)"
        )

    return df


def save_results(df: pd.DataFrame, output_file: str, add_timestamp: bool = True) -> str:
    """Save results DataFrame to pickle file."""
    if add_timestamp:
        timestamp_suffix = time.strftime("%Y%m%d_%H%M%S")
        base_name, ext = os.path.splitext(output_file)
        output_file = f"{base_name}_{timestamp_suffix}{ext}"

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving results to {output_file}...")
    df_to_save = df.reset_index(drop=True)

    try:
        with open(output_file, "wb") as f:
            pickle.dump(df_to_save, f)
        print(f"Save completed: {len(df_to_save)} samples saved to {output_file}")
    except Exception as e:
        raise ValueError(f"Failed to save results: {str(e)}")

    return output_file


def validate_dataset(df: pd.DataFrame, backend_name: Optional[str] = None) -> None:
    """Validate that the dataset has required columns."""
    from utils.validation import validate_dataset_extended

    validate_dataset_extended(df, backend_name)


# ---------------------------------------------------------------------------
# DiskCache-backed request store (drop-in replacement for List[Dict])
# ---------------------------------------------------------------------------


class DiskCacheRequestStore:
    """A list-like store backed by diskcache.Cache.

    Each request is stored individually keyed by its integer index.
    Supports ``__getitem__``, ``__len__``, and ``__iter__`` so it can
    be used wherever ``llm_requests: List[Dict]`` was expected.
    """

    def __init__(self, cache_path: str):
        self._cache = diskcache.Cache(cache_path, size_limit=100 * (1024**3))
        # Cache stores request_idx -> request.  We also store a special
        # key "__count__" with the total number of requests.
        self._count = self._cache.get("__count__", 0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self._count:
            raise IndexError(f"Request index {idx} out of range [0, {self._count})")
        return self._cache[idx]

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        for i in range(self._count):
            yield self._cache[i]

    def close(self) -> None:
        """Close the underlying diskcache."""
        self._cache.close()

    def __del__(self):
        try:
            self._cache.close()
        except Exception:
            pass
