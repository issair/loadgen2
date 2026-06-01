"""
Error handling utilities.
"""

import logging
import sys
import traceback


def handle_backend_error(error: Exception, context: str = "") -> None:
    """Handle backend errors gracefully."""
    logger = logging.getLogger(__name__)
    logger.error(f"Backend error{f' in {context}' if context else ''}: {error}")
    logger.error(traceback.format_exc())
    raise error


def handle_runner_error(error: Exception, runner_name: str = "") -> None:
    """Handle runner errors gracefully."""
    logger = logging.getLogger(__name__)
    logger.error(f"Runner error{f' in {runner_name}' if runner_name else ''}: {error}")
    logger.error(traceback.format_exc())
    sys.exit(1)
