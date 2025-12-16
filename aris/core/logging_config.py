"""
Deterministic logging configuration for ARIS.

Provides consistent, reproducible logging behavior with strict formatting.
"""

import logging
import sys
from typing import Final

# Constants for deterministic logging
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL: Final[int] = logging.INFO


def setup_logging(level: int = DEFAULT_LEVEL) -> logging.Logger:
    """
    Configure deterministic logging for the ARIS system.

    Args:
        level: The logging level to use (default: INFO).

    Returns:
        The root logger instance.
    """
    # Get root logger
    root_logger = logging.getLogger("aris")
    # Clear any existing handlers to ensure deterministic behavior
    root_logger.handlers.clear()
    # Set level
    root_logger.setLevel(level)
    # Create console handler with deterministic formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    # Create and set formatter
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(formatter)
    # Add handler to logger
    root_logger.addHandler(console_handler)
    # Prevent propagation to root logger to avoid duplicate logs
    root_logger.propagate = False
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: The name of the module requesting the logger.

    Returns:
        A logger instance with the specified name under the 'aris' namespace.
    """
    return logging.getLogger(f"aris.{name}")
