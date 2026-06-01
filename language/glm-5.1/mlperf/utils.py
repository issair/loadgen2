"""MLPerf-specific utilities for tokenization and dataset handling."""

from typing import Any, Dict, List, Optional

import pandas as pd

from utils import load_dataset, validate_dataset
from utils.backend_registry import uses_chat_template, uses_text_input
from utils.tokenization import StandardTokenizer


def prepare_mlperf_dataset(
    input_file: str,
    backend_name: Optional[str] = None,
    tokenizer: StandardTokenizer = None,
    num_samples: Optional[int] = None,
    skip_samples: int = 0,
    use_chat_template: Optional[bool] = None,
) -> Dict[str, Any]:
    """Prepare dataset for MLPerf inference."""
    if backend_name is None:
        from utils.backend_registry import detect_backend

        backend_name = detect_backend()

    df = load_dataset(input_file, num_samples, skip_samples)
    validate_dataset(df)

    prompts = df["text_input"].tolist()
    print(f"[MLPerf] Loaded {len(prompts)} prompts from dataset")

    uses_text_prompts = uses_text_input()

    if use_chat_template is None:
        use_chat_template = uses_chat_template()
        print(f"[MLPerf] Using chat template from registry: {use_chat_template}")

    if uses_text_prompts:
        print(f"[MLPerf] Backend {backend_name} uses text prompts directly")
        return {
            "dataframe": df,
            "prompts": prompts,
            "tokenized_prompts": prompts,
            "processed_strings": prompts,
            "uses_text_prompts": True,
        }
    else:
        print(f"[MLPerf] Tokenizing prompts for {backend_name} backend...")
        tokenized_prompts, processed_strings = tokenizer.tokenize_prompts(
            prompts, use_chat_template
        )
        print(f"[MLPerf] Tokenized {len(tokenized_prompts)} prompts")
        return {
            "dataframe": df,
            "prompts": prompts,
            "tokenized_prompts": tokenized_prompts,
            "processed_strings": processed_strings,
            "uses_text_prompts": False,
        }


def process_mlperf_results(
    sut_results: List[Dict[str, Any]],
    tokenizer: Optional[StandardTokenizer] = None,
    backend_name: Optional[str] = None,
    uses_text_prompts: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Process MLPerf SUT results into standardized format."""
    from utils.tokenization import process_inference_results

    if backend_name is None:
        from utils.backend_registry import detect_backend

        backend_name = detect_backend()

    if uses_text_prompts is None:
        uses_text_prompts = uses_text_input()

    return process_inference_results(
        sut_results, tokenizer, uses_text_prompts=uses_text_prompts
    )


def create_mlperf_output_dataframe(
    input_df: pd.DataFrame,
    results: List[Dict[str, Any]],
    backend_name: Optional[str] = None,
) -> pd.DataFrame:
    """Create output dataframe with MLPerf results."""
    if backend_name is None:
        from utils.backend_registry import detect_backend

        backend_name = detect_backend()

    df_output = input_df.copy()
    df_output["model_output"] = [r["model_output"] for r in results]
    df_output["tok_model_output"] = [r["tok_model_output"] for r in results]
    df_output["tok_model_output_len"] = [r["tok_model_output_len"] for r in results]
    df_output["model_backend"] = backend_name
    df_output["prompt_tokens"] = [r.get("prompt_tokens", 0) for r in results]
    df_output["cached_tokens"] = [r.get("cached_tokens", 0) for r in results]
    df_output["reasoning_tokens"] = [r.get("reasoning_tokens", 0) for r in results]
    df_output["completion_tokens"] = [r.get("completion_tokens", 0) for r in results]
    df_output["cached_ratio"] = [
        (r.get("cached_tokens", 0) / r.get("prompt_tokens", 1) * 100)
        if r.get("prompt_tokens", 0) > 0
        else 0.0
        for r in results
    ]
    df_output["reasoning_ratio"] = [
        (r.get("reasoning_tokens", 0) / r.get("completion_tokens", 1) * 100)
        if r.get("completion_tokens", 0) > 0
        else 0.0
        for r in results
    ]
    return df_output
