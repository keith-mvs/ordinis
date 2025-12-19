from collections.abc import MutableMapping
import contextvars
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any
import uuid

# Context variable for trace_id
_trace_id_ctx = contextvars.ContextVar("trace_id", default=None)


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSONL for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        # Get trace_id from context var, or from record if explicitly passed
        trace_id = getattr(record, "trace_id", _trace_id_ctx.get())

        log_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": trace_id,
        }

        # Merge extra fields
        # We look for 'data' in the record, which is populated by the Adapter
        if hasattr(record, "data") and isinstance(record.data, dict):
            log_record.update(record.data)

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Adapter that injects trace_id and handles structured data."""

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        # Inject trace_id from context if not provided
        if "trace_id" not in kwargs["extra"]:
            kwargs["extra"]["trace_id"] = _trace_id_ctx.get()

        # If 'data' is passed in kwargs, move it to extra so Formatter sees it
        if "data" in kwargs:
            kwargs["extra"]["data"] = kwargs.pop("data")

        return msg, kwargs


def setup_logging(log_dir: str = "artifacts/logs", level: str = "INFO") -> None:
    """Configure the root logger for Ordinis."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # JSONL Handler
    json_handler = logging.FileHandler(log_path / "ordinis.jsonl")
    json_handler.setFormatter(JSONFormatter())

    # Console Handler (Human readable)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    # Root Logger Configuration
    root = logging.getLogger("ordinis")
    root.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    if root.hasHandlers():
        root.handlers.clear()

    root.addHandler(json_handler)
    root.addHandler(console_handler)


def get_logger(name: str) -> StructuredLoggerAdapter:
    """Get a structured logger adapter."""
    logger = logging.getLogger(name)
    return StructuredLoggerAdapter(logger, {})


class TraceContext:
    """Context manager for setting the trace_id."""

    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.token = None

    def __enter__(self) -> str:
        self.token = _trace_id_ctx.set(self.trace_id)
        return self.trace_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            _trace_id_ctx.reset(self.token)
