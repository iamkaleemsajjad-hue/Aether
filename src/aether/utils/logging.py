"""
Structured logging utilities.

Aether uses structlog for structured, context-rich logs. This module provides a
convenient `get_logger()` helper and configuration utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    log_file: str | Path | None = None,
    **kwargs: Any,
) -> None:
    """Configure Aether's structured logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        json_format: Whether to format logs as JSON.
        log_file: Optional file to write logs to.
        kwargs: Additional context to bind to every log message.
    """
    try:
        import structlog
    except ImportError:
        # Fallback to standard logging if structlog is not installed
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=str(log_file) if log_file else None,
        )
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer() if json_format else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    if log_file:
        structlog.configure(
            logger_factory=structlog.PrintLoggerFactory(file=Path(log_file).open("a", encoding="utf-8"))
        )


def get_logger(name: str, **kwargs: Any) -> Any:
    """Get a structured logger for a module.

    Args:
        name: Logger name (usually __name__).
        kwargs: Initial context to bind to the logger.

    Returns:
        A structlog logger or standard logging logger fallback.
    """
    try:
        import structlog
        logger = structlog.get_logger(name)
        if kwargs:
            logger = logger.bind(**kwargs)
        return logger
    except ImportError:
        return logging.getLogger(name)
