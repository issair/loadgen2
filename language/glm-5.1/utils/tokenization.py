"""
Shared tokenization utilities for all runners.
"""

from typing import List, Optional, Tuple

from transformers import AutoTokenizer

from utils.backend_registry import (
    detect_backend,
    get_backend_config,
    get_supported_backends,
    uses_chat_template,
)


class StandardTokenizer:
    """Standard tokenizer for GLM-5.1 models."""

    DEFAULT_MODEL = "THUDM/glm-5-1"
    DEFAULT_MAX_LENGTH = 16 * 1024

    def __init__(self, model_name: str = None, max_length: int = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_length = max_length or self.DEFAULT_MAX_LENGTH
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            print(f"Loading tokenizer: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
        return self._tokenizer

    def tokenize_prompts(
        self,
        prompts: List[str],
        use_chat_template: Optional[bool] = None,
        backend_name: Optional[str] = None,
    ) -> Tuple[List[List[int]], List[str]]:
        """Tokenize prompts with backend-specific handling.

        Returns:
            Tuple of (tokenized_prompts, processed_strings)
        """
        if backend_name is None:
            backend_name = detect_backend()

        if use_chat_template is None:
            use_chat_template = uses_chat_template(backend_name)
            print(
                f"[{backend_name}] Using chat template from registry: {use_chat_template}"
            )

        tokenized = []
        processed_strings = []

        for prompt in prompts:
            if use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
                tokens = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    max_length=self.max_length,
                    truncation=True,
                )
                processed_string = self.tokenizer.decode(
                    tokens, skip_special_tokens=False
                )
            else:
                tokens = self.tokenizer.encode(
                    prompt, truncation=True, max_length=self.max_length
                )
                processed_string = prompt

            tokenized.append(tokens)
            processed_strings.append(processed_string)

        return tokenized, processed_strings

    def decode_tokens(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode tokens to text."""
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self, token_lists: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """Batch decode multiple token lists."""
        return self.tokenizer.batch_decode(
            token_lists, skip_special_tokens=skip_special_tokens
        )


def process_inference_results(
    raw_results: List[dict],
    tokenizer: Optional[StandardTokenizer] = None,
    backend_name: Optional[str] = None,
    uses_text_prompts: bool = False,
) -> List[dict]:
    """Process raw inference results into standardized format."""
    if backend_name is None:
        backend_name = detect_backend()
    if backend_name not in get_supported_backends():
        raise ValueError(f"Backend {backend_name} is not supported")

    standardized_results = []
    for raw_result in raw_results:
        if uses_text_prompts and "text" in raw_result:
            text = raw_result["text"]
            tokens = raw_result.get("tokens", [])
        else:
            tokens = raw_result.get("tokens", [])
            text = ""
            if tokenizer and tokens:
                try:
                    text = tokenizer.decode_tokens(tokens)
                except BaseException:
                    pass

        standardized = {
            "model_output": text,
            "tok_model_output": tokens,
            "tok_model_output_len": len(tokens),
            "model_backend": backend_name,
            "prompt_tokens": raw_result.get("prompt_tokens", 0),
            "cached_tokens": raw_result.get("cached_tokens", 0),
            "reasoning_tokens": raw_result.get("reasoning_tokens", 0),
            "completion_tokens": raw_result.get("completion_tokens", 0),
        }
        standardized_results.append(standardized)

    return standardized_results
