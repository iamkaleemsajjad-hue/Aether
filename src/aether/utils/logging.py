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

    level_number = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level_number)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(getattr(handler, "baseFilename", "")).resolve() == log_path.resolve()
            for handler in root_logger.handlers
        ):
            root_logger.addHandler(logging.FileHandler(log_path, encoding="utf-8"))

    # ``filter_by_level`` is a stdlib processor and requires a stdlib logger.
    # PrintLoggerFactory returns a PrintLogger without ``disabled``, which
    # previously crashed every CLI path using this processor.
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
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
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
