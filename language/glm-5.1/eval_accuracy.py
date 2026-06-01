#!/usr/bin/env python3
"""
Standalone evaluation script for MLPerf GLM-5.1 dataset.

Expected input format (pickle file with DataFrame):
- model_output: The model's response text
- tok_model_output_len: The length of the model's response tokens
- ground_truth: The expected answer
- dataset: Dataset name
- question: The question text

Output adds two columns:
- extracted_answer: Parsed answer from model output
- prompt_accuracy: 100.0 if correct, 0.0 if incorrect
"""

import argparse
import json
import logging
import multiprocessing
import os
import pickle
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# MLPerf Log Accuracy Processing
# =============================================================================


def process_mlperf_log_accuracy(
    mlperf_log_file: Union[str, Path],
    dataset_file: Union[str, Path],
    checkpoint_path: str,
    dtype: str = "int32",
    output_dir: Optional[Union[str, Path]] = None,
    base_filename: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """Process MLPerf log accuracy file and evaluate results.

    Args:
        mlperf_log_file: Path to mlperf_log_accuracy.json
        dataset_file: Path to ground truth dataset pickle file
        checkpoint_path: Path to tokenizer checkpoint
        dtype: Data type for numpy conversion ("int32", "int64", "float")
        output_dir: Directory to save evaluated results
        base_filename: Base filename for output file

    Returns:
        Tuple of (evaluated_dataframe, saved_file_path)
    """
    mlperf_log_file = Path(mlperf_log_file)
    dataset_file = Path(dataset_file)

    if not mlperf_log_file.exists():
        raise FileNotFoundError(f"MLPerf log file not found: {mlperf_log_file}")
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    logger.info(f"Processing MLPerf log: {mlperf_log_file}")
    logger.info(f"Using dataset: {dataset_file}")
    logger.info(f"Using checkpoint: {checkpoint_path}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {checkpoint_path}: {e}")

    # Load ground truth dataset
    try:
        with open(dataset_file, "rb") as f:
            dataset_df = pickle.load(f)
        if isinstance(dataset_df, pd.DataFrame):
            logger.info(f"Loaded ground truth dataset: {len(dataset_df)} samples")
        else:
            raise TypeError(f"Expected DataFrame, got {type(dataset_df)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    # Load MLPerf accuracy log
    try:
        with open(mlperf_log_file, "r") as f:
            accuracy_data = json.load(f)
        logger.info(f"Loaded MLPerf accuracy log: {len(accuracy_data)} entries")
    except Exception as e:
        raise RuntimeError(f"Failed to load MLPerf log: {e}")

    # Decode tokens from MLPerf log
    decoded_responses = []
    for entry in tqdm(accuracy_data, desc="Decoding MLPerf log tokens"):
        qsl_idx = entry["qsl_idx"]
        data = entry["data"]

        # Decode the token data based on dtype
        try:
            if dtype == "int32":
                token_array = np.frombuffer(bytes(data), dtype=np.int32)
            elif dtype == "int64":
                token_array = np.frombuffer(bytes(data), dtype=np.int64)
            elif dtype == "float":
                token_array = np.frombuffer(bytes(data), dtype=np.float32)
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

            # Decode tokens to text
            response_text = tokenizer.decode(token_array, skip_special_tokens=True)

            decoded_responses.append(
                {
                    "qsl_idx": qsl_idx,
                    "response_tokens": token_array.tolist(),
                    "response_text": response_text,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to decode entry {qsl_idx}: {e}")

    logger.info(f"Decoded {len(decoded_responses)} responses")

    # Merge with ground truth
    merged_data = []
    for resp in decoded_responses:
        qsl_idx = resp["qsl_idx"]
        if qsl_idx < len(dataset_df):
            row = dataset_df.iloc[qsl_idx].to_dict()
            row["model_output"] = resp["response_text"]
            row["tok_model_output"] = resp["response_tokens"]
            row["tok_model_output_len"] = len(resp["response_tokens"])
            merged_data.append(row)

    result_df = pd.DataFrame(merged_data)
    logger.info(f"Merged {len(result_df)} samples with ground truth")

    # Run evaluation
    result_df = evaluate_dataframe(result_df)

    # Compute overall accuracy
    if "prompt_accuracy" in result_df.columns:
        mean_accuracy = result_df["prompt_accuracy"].mean()
        mean_tok_len = result_df["tok_model_output_len"].mean()
        logger.info(
            f"Evaluation Results: mean-accuracy={mean_accuracy:.4f}, "
            f"mean-output-tok-len={mean_tok_len:.4f}"
        )

    # Save results
    if output_dir is None:
        output_dir = mlperf_log_file.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if base_filename is None:
        base_filename = "mlperf_accuracy_evaluated.pkl"

    output_file = output_dir / base_filename
    with open(output_file, "wb") as f:
        pickle.dump(result_df, f)
    logger.info(f"Saved evaluated results to: {output_file}")

    return result_df, str(output_file)


# =============================================================================
# Evaluation Logic
# =============================================================================


def extract_answer(text: str) -> str:
    """Extract answer from model output."""
    if not text:
        return ""

    # Try to find answer in boxed format
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Try to find answer after "Answer:" or "answer:"
    answer_match = re.search(
        r"(?:Answer|answer|The answer is)[:\s]+([A-Da-d]|[0-9]+(?:\.[0-9]+)?)",
        text,
    )
    if answer_match:
        return answer_match.group(1).strip()

    return text.strip()


def evaluate_single(row: pd.Series) -> Dict[str, Any]:
    """Evaluate a single row."""
    model_output = str(row.get("model_output", ""))
    ground_truth = str(row.get("ground_truth", ""))

    extracted = extract_answer(model_output)
    is_correct = (
        100.0 if extracted.strip().lower() == ground_truth.strip().lower() else 0.0
    )

    return {
        "extracted_answer": extracted,
        "prompt_accuracy": is_correct,
    }


def evaluate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all rows in the dataframe."""
    if "model_output" not in df.columns:
        logger.warning("No model_output column found, skipping evaluation")
        return df

    logger.info(f"Evaluating {len(df)} samples...")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        result = evaluate_single(row)
        results.append(result)

    result_df = pd.DataFrame(results)
    df["extracted_answer"] = result_df["extracted_answer"]
    df["prompt_accuracy"] = result_df["prompt_accuracy"]

    correct = df["prompt_accuracy"].sum()
    logger.info(f"Accuracy: {correct}/{len(df)} = {correct / len(df) * 100:.2f}%")

    return df


def process_dataframe(
    input_file: str, output_file: str, checkpoint_path: Optional[str] = None
) -> pd.DataFrame:
    """Load, evaluate, and save results."""
    logger.info(f"Loading data from {input_file}")
    with open(input_file, "rb") as f:
        df = pickle.load(f)

    logger.info(f"Loaded {len(df)} samples")
    df = evaluate_dataframe(df)

    with open(output_file, "wb") as f:
        pickle.dump(df, f)
    logger.info(f"Saved evaluated results to {output_file}")

    return df


def print_evaluation_results(df: pd.DataFrame) -> None:
    """Print evaluation results summary."""
    if "prompt_accuracy" in df.columns:
        mean_accuracy = df["prompt_accuracy"].mean()
        mean_tok_len = (
            df["tok_model_output_len"].mean()
            if "tok_model_output_len" in df.columns
            else 0
        )
        logger.info(
            f"Evaluation Results: "
            f"mean-accuracy={mean_accuracy:.4f}, "
            f"mean-output-tok-len={mean_tok_len:.4f}"
        )


def process_and_save_dataframe(input_file: str, output_file: str) -> str:
    """Process dataframe and save results, return output file path."""
    df = process_dataframe(input_file, output_file)
    print_evaluation_results(df)
    return output_file


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate MLPerf GLM-5.1 inference results"
    )
    parser.add_argument(
        "--input-file", type=str, required=True, help="Input pickle file with results"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output pickle file for evaluated results",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Tokenizer checkpoint path"
    )
    parser.add_argument(
        "--mlperf-log", type=str, default=None, help="MLPerf log accuracy json file"
    )
    parser.add_argument(
        "--dataset-file", type=str, default=None, help="Ground truth dataset file"
    )
    args = parser.parse_args()

    if args.mlperf_log and args.dataset_file:
        checkpoint = args.checkpoint or "THUDM/glm-5-1"
        output_dir = Path(args.output_file).parent if args.output_file else None
        result_df, output_file = process_mlperf_log_accuracy(
            mlperf_log_file=args.mlperf_log,
            dataset_file=args.dataset_file,
            checkpoint_path=checkpoint,
            output_dir=output_dir,
        )
        print_evaluation_results(result_df)
        print(f"Results saved to: {output_file}")
    else:
        output_file = args.output_file or args.input_file.replace(
            ".pkl", "_evaluated.pkl"
        )
        process_and_save_dataframe(args.input_file, output_file)


if __name__ == "__main__":
    main()
