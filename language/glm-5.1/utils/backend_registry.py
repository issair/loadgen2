"""Central registry for all backend configurations and metadata for GLM-5.1."""

import importlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configuration constants for GLM-5.1
MAX_ISL = 8192  # max input sequence length for GLM
MAX_OSL = int(os.environ.get("MLPERF_MAX_OSL", "8192"))  # max output sequence length
MAX_TEMPLATE_TOKS = 10
MODEL_REVISION = "main"

# Backend Registry with all metadata
BACKEND_REGISTRY = {
    "openai": {
        "class_path": "backends.openai_backend.OpenAIBackend",
        "input_type": "text",
        "uses_chat_template": False,
        "supports_async": True,
        "supports_streaming": True,
        "compatible_runners": ["eval", "mlperf"],
        "required_torchrun": False,
        "config": {
            # API connection settings
            "base_url": "http://localhost:8000/v1",
            "api_key": None,
            "model": "glm-5-1",
            "tokenizer": None,  # "zai-org/GLM-5.1",
            # Generation parameters
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": MAX_OSL,
            "seed": 42,
            # Performance settings
            "max_concurrent_requests": 64,
            "max_retries": 3,
            "request_timeout": None,
            # Connection verification
            "verify_connection": False,
        },
        "env_vars": {
            "OPENAI_API_KEY": "",  # Can be set via environment
        },
    },
}


def detect_backend() -> str:
    """
    Detect backend from MLPERF_BACKEND environment variable.

    Returns:
        Backend name from MLPERF_BACKEND

    Raises:
        RuntimeError: If MLPERF_BACKEND is not set or invalid
    """
    if "MLPERF_BACKEND" not in os.environ:
        supported = get_supported_backends()
        raise RuntimeError(
            "MLPERF_BACKEND environment variable is required but not set.\n\n"
            "Please set it to one of the supported backends:\n"
            + "\n".join(f"  export MLPERF_BACKEND={backend}" for backend in supported)
        )

    backend = os.environ["MLPERF_BACKEND"]
    supported = get_supported_backends()
    if backend not in supported:
        raise RuntimeError(
            f"Invalid MLPERF_BACKEND value: {backend}\n\n"
            f"Supported backends: {', '.join(supported)}\n\n"
            "Please set it to one of:\n"
            + "\n".join(f"  export MLPERF_BACKEND={b}" for b in supported)
        )
    return backend


def validate_backend(backend: str) -> None:
    """Validate that the backend name is supported."""
    supported_backends = get_supported_backends()
    if backend not in supported_backends:
        raise ValueError(
            f"Unknown backend '{backend}'. Supported backends: {', '.join(supported_backends)}"
        )


def validate_runner_for_backend(runner_type: str) -> str:
    """Validate that the runner is compatible with the detected backend.

    Args:
        runner_type: One of 'eval', 'mlperf'

    Returns:
        Backend name from MLPERF_BACKEND env var

    Raises:
        RuntimeError: If MLPERF_BACKEND not set or runner incompatible with backend
    """
    backend = detect_backend()
    if not is_backend_compatible_with_runner(backend, runner_type):
        supported = get_supported_backends()
        compatible = [
            b for b in supported if is_backend_compatible_with_runner(b, runner_type)
        ]
        raise RuntimeError(
            f"Runner '{runner_type}' does not support {backend} backend.\n"
            f"Compatible backends: {', '.join(compatible)}"
        )
    return backend


def supports_streaming(backend_name: Optional[str] = None) -> bool:
    """Check if backend supports streaming generation."""
    if backend_name is None:
        backend_name = detect_backend()
    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {backend_name}")
    return BACKEND_REGISTRY[backend_name].get("supports_streaming", False)


def supports_async(backend_name: Optional[str] = None) -> bool:
    """Check if backend supports async generation."""
    if backend_name is None:
        backend_name = detect_backend()
    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {backend_name}")
    return BACKEND_REGISTRY[backend_name].get("supports_async", False)


def requires_torchrun(backend_name: Optional[str] = None) -> bool:
    """Check if backend requires torchrun for execution."""
    if backend_name is None:
        backend_name = detect_backend()
    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {backend_name}")
    return BACKEND_REGISTRY[backend_name].get("required_torchrun", False)


def get_backend_config(backend_name: Optional[str] = None) -> Dict[str, Any]:
    """Get configuration for a specific backend.

    Returns a copy of the backend config, with overrides from environment variables:
      - OPENAI_API_BASE   overrides base_url
      - OPENAI_MODEL       overrides model
      - OPENAI_API_KEY     overrides api_key
    """
    if backend_name is None:
        backend_name = detect_backend()
    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available backends: {list(BACKEND_REGISTRY.keys())}"
        )
    config = BACKEND_REGISTRY[backend_name]["config"].copy()

    # Apply environment variable overrides
    if os.environ.get("OPENAI_API_BASE"):
        config["base_url"] = os.environ["OPENAI_API_BASE"]
    if os.environ.get("OPENAI_MODEL"):
        config["model"] = os.environ["OPENAI_MODEL"]
    if os.environ.get("OPENAI_API_KEY"):
        config["api_key"] = os.environ["OPENAI_API_KEY"]

    return config


def get_backend_class_path(backend_name: Optional[str] = None) -> str:
    """Get the class path for a backend."""
    if backend_name is None:
        backend_name = detect_backend()
    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend_name}. Available backends: {list(BACKEND_REGISTRY.keys())}"
        )
    return BACKEND_REGISTRY[backend_name]["class_path"]


def uses_text_input(backend_name: Optional[str] = None) -> bool:
    """Check if a backend uses text input (vs tokenized)."""
    if backend_name is None:
        backend_name = detect_backend()
    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {backend_name}")
    return BACKEND_REGISTRY[backend_name]["input_type"] == "text"


def uses_chat_template(backend_name: Optional[str] = None) -> bool:
    """Check if a backend uses chat templates."""
    if backend_name is None:
        backend_name = detect_backend()
    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {backend_name}")
    return BACKEND_REGISTRY[backend_name]["uses_chat_template"]


def get_supported_backends() -> List[str]:
    """Get list of all supported backends."""
    return list(BACKEND_REGISTRY.keys())


def get_backend_instance(backend_name: Optional[str] = None):
    """Create and return a backend instance.

    Args:
        backend_name: Optional backend name override.
                      If None, uses MLPERF_BACKEND env var.

    Returns:
        Backend instance
    """
    if backend_name is None:
        backend_name = detect_backend()
    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend: {backend_name}. "
            f"Available backends: {list(BACKEND_REGISTRY.keys())}"
        )
    class_path = get_backend_class_path(backend_name)
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    backend_class = getattr(module, class_name)
    return backend_class()


def is_backend_compatible_with_runner(
    backend_name: Optional[str] = None,
    runner_type: str = None,
) -> bool:
    """Check if a backend is compatible with a specific runner type."""
    if runner_type is None:
        raise ValueError("runner_type is required")
    if backend_name is None:
        backend_name = detect_backend()
    if backend_name not in BACKEND_REGISTRY:
        return False
    return runner_type in BACKEND_REGISTRY[backend_name]["compatible_runners"]


def get_backend_env_vars(backend_name: Optional[str] = None) -> Dict[str, str]:
    """Get environment variables for a backend."""
    if backend_name is None:
        backend_name = detect_backend()
    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {backend_name}")
    return BACKEND_REGISTRY[backend_name]["env_vars"].copy()


def apply_backend_env_vars(backend_name: Optional[str] = None) -> None:
    """Apply environment variables for a backend."""
    if backend_name is None:
        backend_name = detect_backend()
    env_vars = get_backend_env_vars(backend_name)
    for key, value in env_vars.items():
        if value:  # Only set non-empty values
            os.environ[key] = value
