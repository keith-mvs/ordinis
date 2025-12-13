"""
Centralized logging configuration for Ordinis trading system.

Provides structured and simple logging formats with file rotation.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ordinis.runtime.config import LoggingConfig


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data, default=str)


class SimpleFormatter(logging.Formatter):
    """Simple human-readable log formatter."""

    def __init__(self) -> None:
        """Initialize with standard format."""
        super().__init__(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def configure_logging(
    config: LoggingConfig | None = None,
    level: str | None = None,
    format_type: str | None = None,
    log_file: str | None = None,
) -> None:
    """
    Configure application logging.

    Args:
        config: Logging configuration from Settings. If provided, other
                parameters are used as overrides.
        level: Log level override (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type override ("structured" or "simple")
        log_file: Log file path override
    """
    # Get values from config or use parameters/defaults
    effective_level = level or (config.level if config else "INFO")
    effective_format = format_type or (config.format if config else "structured")
    effective_file = log_file or (config.file if config else None)
    max_size_mb = config.max_size_mb if config else 100
    backup_count = config.backup_count if config else 5

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, effective_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Choose formatter
    if effective_format == "structured":
        formatter: logging.Formatter = StructuredFormatter()
    else:
        formatter = SimpleFormatter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if configured)
    if effective_file:
        file_path = Path(effective_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set third-party loggers to WARNING to reduce noise
    for logger_name in ["urllib3", "httpx", "httpcore", "asyncio"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
