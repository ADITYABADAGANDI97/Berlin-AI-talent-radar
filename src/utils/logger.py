"""
Structured timestamped logging utility for Berlin AI Talent Radar.

Provides a factory function to create consistently configured loggers
across all modules. Every logger outputs ISO-8601 timestamps, log level,
module name, and message — making logs grep-friendly and production-ready.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline stage started")
"""

import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """
    Create or retrieve a named logger with structured formatting.

    Configures the logger with a StreamHandler to stdout if no handlers
    are already attached (idempotent — safe to call multiple times with
    the same name).

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level (default: ``logging.INFO``).
        fmt: Optional custom format string. Defaults to the project-wide
             structured format if not provided.

    Returns:
        Configured ``logging.Logger`` instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Collector started", extra={"source": "jsearch"})
        2025-08-01T12:00:00 | INFO     | src.collectors.jsearch | Collector started
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    format_str = fmt or (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    formatter = logging.Formatter(
        fmt=format_str,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False

    return logger