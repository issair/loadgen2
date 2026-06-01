"""
Utility modules for GLM-5.1 MLPerf reference implementation.
"""

from .backend_registry import (
    apply_backend_env_vars,
    detect_backend,
    get_backend_class_path,
    get_backend_config,
    get_backend_env_vars,
    get_backend_instance,
    get_supported_backends,
    is_backend_compatible_with_runner,
    requires_torchrun,
    supports_async,
    supports_streaming,
    uses_chat_template,
    uses_text_input,
    validate_backend,
    validate_runner_for_backend,
)
from .data_utils import (
    DiskCacheRequestStore,
    generate_timestamped_filename,
    load_dataset,
    save_results,
    validate_dataset,
)
from .error_handling import handle_backend_error, handle_runner_error
from .runner_utils import (
    create_base_argument_parser,
    print_runner_header,
    setup_output_paths,
)
from .tokenization import StandardTokenizer, process_inference_results
from .validation import (
    BackendError,
    BackendNotInitializedError,
    ValidationError,
    require_initialized,
    validate_dataset_extended,
    validate_prompts_input,
    validate_runner_args,
)

__all__ = [
    # Backend registry
    "apply_backend_env_vars",
    "detect_backend",
    "get_backend_class_path",
    "get_backend_config",
    "get_backend_env_vars",
    "get_backend_instance",
    "get_supported_backends",
    "is_backend_compatible_with_runner",
    "requires_torchrun",
    "supports_async",
    "supports_streaming",
    "uses_chat_template",
    "uses_text_input",
    "validate_backend",
    "validate_runner_for_backend",
    # Data utilities
    "DiskCacheRequestStore",
    "generate_timestamped_filename",
    "load_dataset",
    "save_results",
    "validate_dataset",
    # Validation
    "BackendError",
    "BackendNotInitializedError",
    "ValidationError",
    "require_initialized",
    "validate_dataset_extended",
    "validate_prompts_input",
    "validate_runner_args",
    # Runner utilities
    "create_base_argument_parser",
    "print_runner_header",
    "setup_output_paths",
    # Tokenization
    "StandardTokenizer",
    "process_inference_results",
    # Error handling
    "handle_backend_error",
    "handle_runner_error",
]
