#!/usr/bin/env python3
"""Evaluation runner for GLM-5.1 with OpenAI backend."""

import argparse
import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from backends import BaseBackend
from utils import (
    StandardTokenizer,
    create_base_argument_parser,
    generate_timestamped_filename,
    get_backend_instance,
    handle_runner_error,
    load_dataset,
    print_runner_header,
    process_inference_results,
    save_results,
    setup_output_paths,
    supports_async,
    uses_chat_template,
    uses_text_input,
    validate_dataset_extended,
    validate_runner_args,
    validate_runner_for_backend,
)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = create_base_argument_parser(
        "GLM-5.1 evaluation system with OpenAI-compatible backend"
    )
    parser.add_argument(
        "--async",
        action="store_true",
        help="Use async generation instead of synchronous",
    )
    return parser


async def run_async_inference(
    backend: BaseBackend,
    tokenized_prompts: List[List[int]],
    text_prompts: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run async inference."""
    try:
        if uses_text_input():
            futures = backend.generate_async(text_prompts=text_prompts)
        else:
            futures = backend.generate_async(tokenized_prompts=tokenized_prompts)

        results = [None] * len(futures)
        indexed_futures = [(i, future) for i, future in enumerate(futures)]
        completed_indices = set()
        pending = {future for _, future in indexed_futures}

        with async_tqdm(
            total=len(futures), desc="Async inference", unit="prompt"
        ) as pbar:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )

                for completed_future in done:
                    original_idx = None
                    for idx, future in indexed_futures:
                        if future is completed_future:
                            original_idx = idx
                            break
                    if original_idx is None or original_idx in completed_indices:
                        continue

                    try:
                        result = await completed_future
                        results[original_idx] = result
                        completed_indices.add(original_idx)
                    except Exception as e:
                        print(f"\nError processing prompt {original_idx}: {e}")
                        raise RuntimeError(
                            f"Backend failed for prompt {original_idx}: {e}"
                        )
                    pbar.update(1)

        if len(completed_indices) != len(futures):
            raise RuntimeError(
                f"Missing results: {len(completed_indices)} != {len(futures)}"
            )
        for i, r in enumerate(results):
            if r is None:
                raise RuntimeError(f"Missing result for prompt {i}")

        print(f"\nCompleted all {len(completed_indices)} prompts successfully")
        return results

    except Exception as e:
        print(f"Error during async inference: {e}")
        import traceback

        traceback.print_exc()
        raise


def run_sync_inference(
    backend: BaseBackend,
    tokenized_prompts: List[List[int]],
    text_prompts: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Run sync inference."""
    try:
        if uses_text_input():
            results = backend.generate(text_prompts=text_prompts)
        else:
            results = backend.generate(tokenized_prompts=tokenized_prompts)
        return results
    except Exception as e:
        print(f"Error during sync inference: {e}")
        raise


def main():
    """Main evaluation function."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        validate_runner_args(args, "eval")
        backend_name = validate_runner_for_backend("eval")

        output_dir, output_file = setup_output_paths(args)
        if args.output_file is None:
            args.output_file = output_file

        actual_output_file = generate_timestamped_filename(
            args.output_file, add_timestamp=True
        )

        use_async = getattr(args, "async", False)
        if use_async and not supports_async():
            raise RuntimeError(f"Backend {backend_name} does not support async")

        print_runner_header("GLM-5.1 Evaluation System", backend_name, args)
        print(f"Mode: {'Async' if use_async else 'Sync'}")
        print("=" * 80)

        df = load_dataset(args.input_file, args.num_samples, args.skip_samples)
        validate_dataset_extended(df)
        prompts = df["text_input"].tolist()

        tokenizer = StandardTokenizer()
        use_chat_template = uses_chat_template()

        if uses_text_input():
            print(f"Backend {backend_name} uses text prompts directly")
            tokenized_prompts = None
            processed_strings = prompts
        else:
            print("Tokenizing prompts...")
            print(f"Using chat template: {use_chat_template}")
            tokenized_prompts, processed_strings = tokenizer.tokenize_prompts(
                prompts, use_chat_template
            )
            print(f"Tokenized {len(tokenized_prompts)} prompts")
            print(f"Tokenizer Max length: {tokenizer.max_length}")

        print(f"\nInitializing {backend_name.upper()} backend...")
        backend = get_backend_instance(backend_name)

        with backend:
            df_output = df.copy()

            if use_async:
                print("Running async inference...")
                raw_results = asyncio.run(
                    run_async_inference(
                        backend, tokenized_prompts, text_prompts=prompts
                    )
                )
            else:
                print("Running sync inference...")
                raw_results = run_sync_inference(
                    backend, tokenized_prompts, text_prompts=prompts
                )

            valid_indices = []
            valid_results = []
            for i, res in enumerate(raw_results):
                if "error" not in res:
                    valid_indices.append(i)
                    valid_results.append(res)
                else:
                    print(
                        f"Skipping prompt {i} due to error: {res.get('error', 'Unknown error')}"
                    )

            if len(valid_results) < len(raw_results):
                print(
                    f"Filtered out {len(raw_results) - len(valid_results)} failed prompts"
                )
                raw_results = valid_results
                df_output = df_output.iloc[valid_indices].reset_index(drop=True)

            print("Processing results...")
            standardized_results = process_inference_results(raw_results, tokenizer)

            df_output["model_output"] = [
                r["model_output"] for r in standardized_results
            ]
            df_output["tok_model_output"] = [
                r["tok_model_output"] for r in standardized_results
            ]
            df_output["tok_model_output_len"] = [
                r["tok_model_output_len"] for r in standardized_results
            ]
            df_output["model_backend"] = [
                r["model_backend"] for r in standardized_results
            ]

            output_file = save_results(df_output, args.output_file, add_timestamp=True)
            print("\nEvaluation completed successfully!")
            print(f"Results saved to: {output_file}")
            print(f"Output columns: {list(df_output.columns)}")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        handle_runner_error(e, "run_eval.py")


if __name__ == "__main__":
    main()
