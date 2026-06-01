#!/usr/bin/env python3
"""Generate a minimal test dataset for GLM-5.1 evaluation.

Usage:
    python gen_test_dataset.py <directory>

Reads all .jsonl files from the given directory, extracts prompt text,
and creates test_input.pkl with a "text_input" column.
"""

import json
import os
import pickle
import sys

import pandas as pd


def find_jsonl_files(directory: str) -> list[str]:
    """Find all .jsonl files in the given directory (non-recursive)."""
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory", file=sys.stderr)
        sys.exit(1)

    jsonl_files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".jsonl")
    )

    if not jsonl_files:
        print(f"Error: No .jsonl files found in '{directory}'", file=sys.stderr)
        sys.exit(1)

    return jsonl_files


def extract_prompts_from_jsonl(filepath: str) -> list[str]:
    """Extract prompt text from a JSONL file.

    Each line is a JSON object. The function tries to find the prompt
    from common keys in order: 'prompt', 'text_input', 'text', 'input', 'content'.
    Falls back to the first string value found.
    """
    prompts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"Warning: Skipping invalid JSON on line {line_num} "
                    f"in {filepath}: {e}",
                    file=sys.stderr,
                )
                continue
            prompts.append(line)
    return prompts


def main():
    if len(sys.argv) != 2:
        print("Usage: python gen_test_dataset.py <directory>", file=sys.stderr)
        print(
            "Scans all .jsonl files in the directory and builds test_input.pkl",
            file=sys.stderr,
        )
        sys.exit(1)

    directory = sys.argv[1]

    jsonl_files = find_jsonl_files(directory)

    df = pd.DataFrame(columns=["text_input"])
    batch = []
    for filepath in jsonl_files:
        prompts = extract_prompts_from_jsonl(filepath)
        batch.extend(prompts)
        while len(batch) >= 50:
            df = pd.concat(
                [df, pd.DataFrame({"text_input": batch[:50]})], ignore_index=True
            )
            batch = batch[50:]
        print(f"Loaded {len(prompts)} prompts from {os.path.basename(filepath)}")
    if batch:
        df = pd.concat([df, pd.DataFrame({"text_input": batch})], ignore_index=True)

    if df.empty:
        print("Error: No prompts extracted from any .jsonl files.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal prompts extracted: {len(df)}")
    output_path = "test_input.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(df, f)

    print(f"Created {output_path} with {len(df)} samples")
    print(df.head(10))
    if len(df) > 10:
        print(f"  ... ({len(df) - 10} more rows)")


if __name__ == "__main__":
    main()
