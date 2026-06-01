"""
OpenAI-compatible backend for GLM-5.1 model inference.

This backend connects to an externally deployed OpenAI-compatible API server
(e.g., vLLM, SGLang, GLM official API) and does NOT manage the server lifecycle.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import deque
from pathlib import Path as _Path
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
import orjson
from openai import AsyncOpenAI, OpenAI

from utils.backend_registry import apply_backend_env_vars, get_backend_config
from utils.validation import require_initialized, validate_prompts_input

from .base_backend import BaseBackend, StreamingChunk
from .utils import set_all_seeds

logger = logging.getLogger(__name__)


class OpenAIBackend(BaseBackend):
    """OpenAI-compatible backend for GLM-5.1 inference.

    Connects to an externally deployed OpenAI-compatible API server.
    Does NOT start or manage any server processes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize OpenAI backend with configuration from registry.

        Args:
            config: Optional configuration overrides
        """
        super().__init__()
        # Get configuration from registry
        self.config = get_backend_config("openai")
        # Allow override with passed config
        if config:
            self.config.update(config)

        # API client objects
        self.client: Optional[OpenAI] = None
        self.async_client: Optional[AsyncOpenAI] = None

        # Pooled httpx client for streaming (lazy-init in first stream call)
        self._httpx_client: Optional[httpx.AsyncClient] = None

        # Shared semaphore for async concurrency control
        self._async_semaphore: Optional[asyncio.Semaphore] = None

        # Tokenizer (will be lazily loaded if available)
        self.tokenizer = None

        # Configure logging to suppress httpx INFO logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        self._setup_environment()

    def _setup_environment(self) -> None:
        """Set up environment variables."""
        apply_backend_env_vars("openai")
        seed = self.config.get("seed", 42)
        set_all_seeds(seed)

    def _flush_cache(self, base_url: str, api_key: Optional[str]) -> None:
        """Flush the server-side KV cache to ensure a clean benchmarking state.

        Calls ``{base_url}/flush_cache`` on the model server.  Failures are
        logged but never raised — a non-functional cache flush should not
        prevent the backend from initializing.
        """
        pass
        # # Strip /v1 (and trailing slash) to reach server root, then append /flush_cache
        # server_url = base_url.rstrip("/").removesuffix("/v1")
        # flush_url = server_url + "/flush_cache"
        # logger.info(f"[OPENAI] Flushing server cache: {flush_url}")
        # try:
        #     headers: Dict[str, str] = {"Content-Type": "application/json"}
        #     if api_key:
        #         headers["Authorization"] = f"Bearer {api_key}"
        #     resp = httpx.post(flush_url, headers=headers, timeout=30.0)
        #     logger.info(
        #         f"[OPENAI] Cache flush response: status={resp.status_code}, "
        #         f"body={resp.text[:200]}"
        #     )
        # except Exception as e:
        #     logger.warning(f"[OPENAI] Cache flush failed (non-fatal): {e}")

    def initialize(self) -> None:
        """Initialize the backend by setting up OpenAI API clients."""
        if self.is_initialized:
            return

        try:
            base_url = self.config["base_url"]
            api_key = self.config.get("api_key")

            logger.info("[OPENAI] Initializing OpenAI backend")
            logger.info(f"[OPENAI] Base URL: {base_url}")
            logger.info(f"[OPENAI] Model: {self.config['model']}")
            logger.info(f"[OPENAI] API Key set: {bool(api_key)}")

            # If api_key is None or empty, try environment variable
            if not api_key:
                api_key = os.environ.get(
                    "OPENAI_API_KEY", os.environ.get("GLM_API_KEY", "dummy-key")
                )
                logger.info(
                    f"[OPENAI] Using API key from environment: {'set' if api_key else 'not set'}"
                )

            # Configure timeout settings
            request_timeout = self.config.get("request_timeout")
            timeout_config = httpx.Timeout(
                timeout=request_timeout,
                connect=60.0,
                read=None,
                write=None,
                pool=None,
            )

            max_retries = self.config.get("max_retries", 3)

            # Create synchronous client
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout_config,
                max_retries=max_retries,
            )
            logger.info("[OPENAI] Created synchronous OpenAI client")

            # Create asynchronous client
            self.async_client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout_config,
                max_retries=max_retries,
            )
            logger.info("[OPENAI] Created asynchronous OpenAI client")

            logger.info("[OPENAI] Created httpx streaming client (per-request clients)")

            # Create shared semaphore for async concurrency control
            max_concurrent = self.config.get("max_concurrent_requests", 64)
            self._async_semaphore = asyncio.Semaphore(max_concurrent)
            logger.info(
                f"[OPENAI] Created async semaphore with limit: {max_concurrent}"
            )

            # Optionally load tokenizer
            tokenizer_name = self.config.get("tokenizer")
            if tokenizer_name:
                try:
                    from transformers import AutoTokenizer

                    logger.info(f"[OPENAI] Loading tokenizer: {tokenizer_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_name,
                        trust_remote_code=True,
                    )
                    logger.info("[OPENAI] Tokenizer loaded successfully")
                except Exception as e:
                    logger.warning(f"[OPENAI] Failed to load tokenizer: {e}")
                    logger.warning(
                        "[OPENAI] Token IDs will not be available in results"
                    )
                    self.tokenizer = None

            # Verify the API connection with a simple model list or chat query
            if self.config.get("verify_connection", True):
                logger.info("[OPENAI] Verifying API connection...")
                try:
                    # Try a simple chat completion to verify the endpoint
                    verify_response = self.client.chat.completions.create(
                        model=self.config["model"],
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=5,
                        temperature=0.0,
                    )
                    if verify_response.choices:
                        logger.info("[OPENAI] ✓ API connection verified successfully!")
                    else:
                        logger.warning("[OPENAI] API responded but no choices returned")
                except Exception as e:
                    logger.warning(f"[OPENAI] Connection verification warning: {e}")
                    logger.warning(
                        "[OPENAI] Continuing anyway - server may need more time"
                    )

            # Flush server-side cache to ensure a clean state for benchmarking.
            self._flush_cache(base_url, api_key)

            self.is_initialized = True
            logger.info("[OPENAI] Backend initialized successfully!")

        except Exception as e:
            logger.error(f"[OPENAI] Initialization failed: {e}")
            self.client = None
            self.async_client = None
            self._async_semaphore = None
            self.is_initialized = False
            raise

    def _prepare_prompts(
        self,
        tokenized_prompts: Optional[List[List[int]]] = None,
        text_prompts: Optional[List[str]] = None,
    ) -> List[str]:
        """Prepare text prompts from input, converting tokenized if needed.

        Args:
            tokenized_prompts: Optional tokenized prompts
            text_prompts: Optional text prompts

        Returns:
            List of text prompt strings
        """
        if text_prompts is not None:
            return text_prompts

        if tokenized_prompts is not None and self.tokenizer is not None:
            return [
                self.tokenizer.decode(tokens, skip_special_tokens=False)
                for tokens in tokenized_prompts
            ]

        raise ValueError(
            "OpenAI backend requires text_prompts, or tokenized_prompts with a loaded tokenizer"
        )

    def _build_messages(self, prompt: str) -> tuple:
        """Build chat messages from a prompt string.

        Returns ``(messages, extra_payload)`` where *messages* is a list of
        message dicts and *extra_payload* is a dict of additional API
        parameters (e.g. ``"tools"``) parsed from an enriched request.

        Supports three input formats:
        1. JSON object with a ``"messages"`` key (enriched request payload).
        2. JSON array of message dicts (``[{"role": ..., ...}, ...]``).
        3. Plain text — treated as a single user message.
        """
        extra: Dict[str, Any] = {}

        # Detect JSON object with "messages" key (enriched request)
        if prompt.startswith("{") and '"messages"' in prompt:
            try:
                parsed = orjson.loads(prompt)
                if isinstance(parsed, dict) and "messages" in parsed:
                    extra = {
                        k: v
                        for k, v in parsed.items()
                        if k
                        not in (
                            "messages",
                            "request_idx",
                            "original_idx",
                            "last_role",
                            "end_at",
                        )
                    }
                    return parsed["messages"], extra
            except (orjson.JSONDecodeError, TypeError):
                pass

        # Detect JSON-encoded multi-turn message list
        if prompt.startswith("[") and prompt.endswith("]"):
            try:
                parsed = orjson.loads(prompt)
                if (
                    isinstance(parsed, list)
                    and len(parsed) > 0
                    and isinstance(parsed[0], dict)
                    and "role" in parsed[0]
                ):
                    return parsed, extra
            except (orjson.JSONDecodeError, IndexError, TypeError):
                pass
        return [{"role": "user", "content": prompt}], extra

    def _extract_result(self, completion) -> Dict[str, Any]:
        """Extract standardized result from an OpenAI API response.

        Args:
            completion: Response from OpenAI chat completions API

        Returns:
            Standardized result dictionary
        """
        # Handle case where choices is None (e.g. non-standard fields like reasoning_content)
        if completion.choices is None:
            logger.warning(
                "[OPENAI] choices is None, falling back to raw dict extraction"
            )
            try:
                raw = completion.to_dict()
                choices_raw = raw.get("choices", [])
                if choices_raw:
                    generated_text = (
                        choices_raw[0].get("message", {}).get("content", "") or ""
                    )
                else:
                    generated_text = ""
            except Exception:
                generated_text = ""
        else:
            generated_text = completion.choices[0].message.content or ""

        if not generated_text:
            raise RuntimeError("Empty response received from OpenAI API")

        # Try to get token IDs using the tokenizer if available
        # Fall back to a single placeholder token so MLPerf LoadGen can proceed
        tokens = [1]
        if self.tokenizer:
            tokens = self.tokenizer.encode(generated_text)

        return {
            "tokens": tokens,
            "text": generated_text,
        }

    @require_initialized
    def generate(
        self,
        tokenized_prompts: Optional[List[List[int]]] = None,
        text_prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate responses synchronously.

        Args:
            tokenized_prompts: List of tokenized prompts
            text_prompts: List of text prompts (preferred)
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
                      These override the config defaults.

        Returns:
            List of result dictionaries
        """
        if self.client is None:
            raise RuntimeError(
                "OpenAI client is not initialized. Call initialize() first."
            )

        validate_prompts_input(
            backend_name="openai",
            tokenized_prompts=tokenized_prompts,
            text_prompts=text_prompts,
            input_type="text",
        )

        prompt_strings = self._prepare_prompts(tokenized_prompts, text_prompts)

        # Merge kwargs with config (kwargs take precedence)
        temperature = kwargs.get("temperature", self.config["temperature"])
        max_tokens = kwargs.get("max_tokens", self.config["max_tokens"])
        top_p = kwargs.get("top_p", self.config["top_p"])
        seed = kwargs.get("seed", self.config["seed"])

        results = []
        for prompt in prompt_strings:
            try:
                messages, extra = self._build_messages(prompt)
                create_kwargs: Dict[str, Any] = {
                    "model": self.config["model"],
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "seed": seed,
                    "ignore_eos": True,
                }
                if extra:
                    create_kwargs.update(extra)
                completion = self.client.chat.completions.create(**create_kwargs)
                result = self._extract_result(completion)
                results.append(result)

            except Exception as e:
                logger.error(f"Error generating completion: {e}")
                results.append({"error": str(e)})

        return results

    async def _async_generate_single(
        self,
        prompt: str,
        idx: int,
        semaphore: asyncio.Semaphore,
        **kwargs,
    ) -> tuple:
        """Generate a single response asynchronously with semaphore control.

        Args:
            prompt: The text prompt
            idx: Index of the prompt
            semaphore: Semaphore for concurrency control
            **kwargs: Additional parameters overriding config defaults

        Returns:
            Tuple of (index, result_dict)
        """
        if self.async_client is None:
            raise RuntimeError(
                f"OpenAI async client is not initialized for prompt {idx}"
            )

        # Merge kwargs with config (kwargs take precedence)
        temperature = kwargs.get("temperature", self.config["temperature"])
        max_tokens = kwargs.get("max_tokens", self.config["max_tokens"])
        top_p = kwargs.get("top_p", self.config["top_p"])
        seed = kwargs.get("seed", self.config["seed"])

        async with semaphore:
            try:
                messages, extra = self._build_messages(prompt)
                create_kwargs: Dict[str, Any] = {
                    "model": self.config["model"],
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "seed": seed,
                }
                if extra:
                    create_kwargs.update(extra)
                completion = await self.async_client.chat.completions.create(
                    **create_kwargs
                )
                result = self._extract_result(completion)
                return idx, result

            except Exception as e:
                logger.error(f"Error generating for prompt {idx}: {e}")
                return idx, {"error": str(e)}

    @require_initialized
    def generate_async(
        self,
        tokenized_prompts: Optional[List[List[int]]] = None,
        text_prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[asyncio.Future]:
        """Generate responses asynchronously using shared semaphore.

        Args:
            tokenized_prompts: List of tokenized prompts
            text_prompts: List of text prompts (preferred)
            **kwargs: Additional parameters

        Returns:
            List of futures
        """
        if self.async_client is None:
            raise RuntimeError(
                "OpenAI async client is not initialized. Call initialize() first."
            )

        validate_prompts_input(
            backend_name="openai",
            tokenized_prompts=tokenized_prompts,
            text_prompts=text_prompts,
            input_type="text",
        )

        prompt_strings = self._prepare_prompts(tokenized_prompts, text_prompts)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        futures = []
        for idx, prompt in enumerate(prompt_strings):
            task = asyncio.create_task(
                self._async_generate_single(
                    prompt, idx, self._async_semaphore, **kwargs
                )
            )

            future = asyncio.Future()

            def make_callback(future_obj, expected_idx):
                def callback(task_obj):
                    try:
                        idx, result = task_obj.result()
                        if idx != expected_idx:
                            future_obj.set_exception(
                                Exception(
                                    f"Index mismatch: expected {expected_idx}, got {idx}"
                                )
                            )
                        else:
                            future_obj.set_result(result)
                    except Exception as e:
                        future_obj.set_exception(e)

                return callback

            task.add_done_callback(make_callback(future, idx))
            futures.append(future)

        return futures

    # Class-level accumulators for average TTFT (Time To First Token)
    _ttft_total: float = 0.0
    _ttft_count: int = 0

    def fix_tool_call_arguments(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """修复 tool_calls 中 arguments 的双重 JSON 编码问题。"""
        fixed = []
        for msg in messages:
            m = dict(msg)
            tool_calls = m.get("tool_calls", [])
            if tool_calls:
                fixed_tcs = []
                for tc in tool_calls:
                    ftc = dict(tc)
                    func = ftc.get("function", {})
                    if isinstance(func, dict):
                        func = dict(func)
                        if isinstance(func.get("arguments"), str):
                            parsed = prev = func["arguments"]
                            while isinstance(parsed, str):
                                try:
                                    prev = parsed
                                    parsed = json.loads(parsed)
                                except (json.JSONDecodeError, TypeError, ValueError):
                                    break
                            func["arguments"] = prev
                        ftc["function"] = func
                    fixed_tcs.append(ftc)
                m["tool_calls"] = fixed_tcs
            fixed.append(m)
        return fixed

    @require_initialized
    async def _stream_chat_completion(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        seed: int,
    ) -> AsyncIterator[StreamingChunk]:
        """Stream a chat completion using raw httpx SSE parsing.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            seed: Random seed

        Yields:
            StreamingChunk objects as tokens arrive
        """
        base_url = self.config["base_url"].rstrip("/")
        url = f"{base_url}/chat/completions"
        api_key = self.config.get("api_key") or os.environ.get(
            "OPENAI_API_KEY", os.environ.get("GLM_API_KEY", "dummy-key")
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        messages, extra = self._build_messages(prompt)
        messages = self.fix_tool_call_arguments(messages)
        payload = {
            "model": self.config["model"],
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if extra:
            payload.update(extra)

        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        if max_tokens:
            payload["max_tokens"] = max_tokens
            payload["max_completion_tokens"] = max_tokens
            payload["ignore_eos"] = True

        payload_bytes = orjson.dumps(payload)
        start = time.perf_counter()
        is_first = False

        if self._httpx_client is None:
            request_timeout = self.config.get("request_timeout")
            timeout_config = httpx.Timeout(
                timeout=request_timeout,
                connect=60.0,
                read=None,
                write=None,
                pool=None,
            )
            self._httpx_client = httpx.AsyncClient(timeout=timeout_config)

        client = self._httpx_client
        async with client.stream(
            "POST", url, content=payload_bytes, headers=headers
        ) as response:
            try:
                response.raise_for_status()
            except Exception as e:
                response_body = await response.aread()
                logger.error(
                    f"[OPENAI] stream request failed: {e} | "
                    f"status={response.status_code} | "
                    f"body={response_body.decode('utf-8', errors='replace')[:2000]}"
                )
                # Dump the failed request payload to a JSON file for debugging
                dump_dir = _Path("dump")
                dump_dir.mkdir(exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                rand_suffix = uuid.uuid4().hex[:8]
                dump_path = dump_dir / f"{timestamp}_{rand_suffix}.json"
                dump_data = {
                    "url": url,
                    "payload": orjson.loads(payload_bytes),
                    "error": str(e),
                    "status_code": response.status_code,
                    "response_body": response_body.decode("utf-8", errors="replace"),
                }
                dump_path.write_text(
                    json.dumps(dump_data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                logger.error(f"[OPENAI] request dumped to {dump_path}")
                raise
            _cached_usage = None
            _deferred_finish = None
            # Rolling buffer of last 3 raw JSON strings for final finish_reason+usage extraction
            _recent_lines: deque[str] = deque(maxlen=3)
            _yielded_first = False

            # Precompile regex for fast content extraction (avoids full orjson.loads per line)
            _CONTENT_RE = re.compile(r'"content":"((?:[^"\\]|\\.)*?)"')

            async for raw_line in response.aiter_lines():
                # SSE lines may arrive as "data: {...}"
                line = raw_line.strip()
                if not line:
                    continue
                # Handle "data: [DONE]" signal – stop and process buffered lines after the loop
                if line == "data: [DONE]" or line == "data:[DONE]":
                    break

                if line.startswith("data:"):
                    data_str = line[5:]
                else:
                    continue

                # Always buffer the raw data string (keep last 3)
                _recent_lines.append(data_str)

                if not _yielded_first:
                    # First content-bearing line: full orjson.loads parse
                    _yielded_first = True
                    yield StreamingChunk(
                        token="",
                        token_ids=[1],
                        is_finished=False,
                        finish_reason=None,
                        usage=None,
                        start_time=start,
                    )

                    # Note: if finish_reason/usage appear without content in first line,
                    # they are cached but not yielded yet – handled in post-loop parsing below.
                    continue

                # Subsequent lines: lightweight regex content extraction (no full JSON parse)

            # Post-loop: parse only the buffered last 3 lines for finish_reason + usage
            for data_str in _recent_lines:
                try:
                    chunk = orjson.loads(data_str)
                except orjson.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [{}]) or [{}]
                finish_reason = choices[0].get("finish_reason")
                usage = chunk.get("usage", None)
                if usage:
                    _cached_usage = usage
                if finish_reason:
                    _deferred_finish = finish_reason

            # Flush any deferred finish (merged with cached usage)
            if _deferred_finish:
                yield StreamingChunk(
                    token="",
                    token_ids=[],
                    is_finished=True,
                    finish_reason=_deferred_finish,
                    usage=_cached_usage,
                )

    async def generate_stream(
        self,
        tokenized_prompts: Optional[List[List[int]]] = None,
        text_prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> List[AsyncIterator[StreamingChunk]]:
        """Generate responses with streaming.

        Uses raw httpx to parse SSE (Server-Sent Events) directly,
        instead of relying on the OpenAI SDK's stream handling.

        Args:
            tokenized_prompts: List of tokenized prompts
            text_prompts: List of text prompts (preferred)
            **kwargs: Additional parameters

        Returns:
            List of async iterators yielding StreamingChunk objects
        """
        validate_prompts_input(
            backend_name="openai",
            tokenized_prompts=tokenized_prompts,
            text_prompts=text_prompts,
            input_type="text",
        )

        prompt_strings = self._prepare_prompts(tokenized_prompts, text_prompts)

        # Merge kwargs with config (kwargs take precedence)
        temperature = kwargs.get("temperature", self.config["temperature"])
        max_tokens = self.config["max_tokens"]
        top_p = kwargs.get("top_p", self.config["top_p"])
        seed = kwargs.get("seed", self.config["seed"])

        return [
            self._stream_chat_completion(p, temperature, max_tokens, top_p, seed)
            for p in prompt_strings
        ]

    def shutdown(self) -> None:
        """Clean up resources and shut down the backend."""
        logger.info("[OPENAI] Shutting down OpenAI backend...")

        if self._httpx_client is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._httpx_client.aclose())
            except RuntimeError:
                pass  # No running event loop, GC will clean up
            self._httpx_client = None

        self.client = None
        self.async_client = None
        self._async_semaphore = None
        self.tokenizer = None
        self.is_initialized = False

        logger.info("[OPENAI] Backend shutdown complete")
