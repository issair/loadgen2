#!/usr/bin/env python3
"""Download DeepSeek-R1 tokenizer only (no model weights).

Usage:
    python download_tokenizer.py
    python download_tokenizer.py --cache-dir /path/to/cache
    HF_ENDPOINT=https://hf-mirror.com python download_tokenizer.py
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

# Set HF mirror for China if not already configured
_HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = _HF_MIRROR

MODEL_ID = "deepseek-ai/DeepSeek-R1"
REVISION = "56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad"

# Files needed for tokenization only (tokenizer config, vocab, etc.)
TOKENIZER_PATTERNS = [
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",  # useful but optional
]


def main():
    parser = argparse.ArgumentParser(description="Download DeepSeek-R1 tokenizer")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory (default: ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local directory to download tokenizer files directly (flat, non-cached). "
        "Use this to get a path suitable for TOKENIZER_PATH.",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or str(Path.cwd() / ".cache" / "huggingface")
    local_dir = args.local_dir

    if local_dir:
        print(f"Downloading tokenizer for {MODEL_ID} (revision: {REVISION})")
        print(f"Local directory: {local_dir}")
        print()

        snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            local_dir=local_dir,
            allow_patterns=TOKENIZER_PATTERNS,
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt"],
        )
        local_path = local_dir
    else:
        print(f"Downloading tokenizer for {MODEL_ID} (revision: {REVISION})")
        print(f"Cache directory: {cache_dir}")
        print(f"Patterns: {TOKENIZER_PATTERNS}")
        print()

        local_path = snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            cache_dir=cache_dir,
            allow_patterns=TOKENIZER_PATTERNS,
            # This ensures we only download matching files, not model weights
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt"],
        )

    print(f"\nTokenizer downloaded to: {local_path}")

    # Verify key files exist
    tokenizer_json = Path(local_path) / "tokenizer.json"
    tokenizer_config = Path(local_path) / "tokenizer_config.json"

    for f in [tokenizer_json, tokenizer_config]:
        if f.exists():
            print(f"  ✓ {f.name} ({f.stat().st_size:,} bytes)")
        else:
            print(f"  ✗ {f.name} NOT FOUND")

    print("\nDone. Tokenizer is ready for use.")


if __name__ == "__main__":
    main()
