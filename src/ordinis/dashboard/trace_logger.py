"""
Trace Logger for Ordinis Dashboard.

Captures structured events for retroactive analysis and observability.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import time
from typing import Any
import uuid

_logger = logging.getLogger(__name__)


class TraceType(str, Enum):
    """Types of trace events."""

    SYSTEM = "system"
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    RISK_CHECK = "risk_check"
    ORDER = "order"
    FILL = "fill"

    # AI Specific
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    RAG_RETRIEVAL = "rag_retrieval"
    CORTEX_THOUGHT = "cortex_thought"
    CODE_GEN = "code_gen"


@dataclass
class TraceEvent:
    """Structured trace event."""

    trace_id: str
    timestamp: float
    type: TraceType
    component: str
    content: dict[str, Any]
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)


class TraceLogger:
    """
    Centralized logger for dashboard traces.

    Writes structured JSONL logs to artifacts/traces/ for ingestion by the dashboard.
    """

    def __init__(self, log_dir: str = "artifacts/traces"):
        """Initialize trace logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Current log file (rotate daily)
        date_str = time.strftime("%Y%m%d")
        self.log_file = self.log_dir / f"trace_{date_str}.jsonl"

        _logger.info(f"Trace logger initialized: {self.log_file}")

    def log(
        self,
        trace_type: TraceType,
        component: str,
        content: dict[str, Any],
        trace_id: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Log a trace event.

        Args:
            trace_type: Type of event
            component: Component name (e.g., "Cortex", "Helix")
            content: Main event data
            trace_id: Correlation ID (generated if None)
            parent_id: Parent event ID
            metadata: Additional context

        Returns:
            trace_id used for the event
        """
        if trace_id is None:
            trace_id = uuid.uuid4().hex

        event = TraceEvent(
            trace_id=trace_id,
            timestamp=time.time(),
            type=trace_type,
            component=component,
            content=content,
            parent_id=parent_id,
            metadata=metadata or {},
        )

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")
        except Exception as e:
            _logger.error(f"Failed to write trace: {e}")

        return trace_id

    def start_trace(self) -> str:
        """Start a new trace chain and return the ID."""
        return uuid.uuid4().hex


# Global instance
_trace_logger = None


def get_trace_logger() -> TraceLogger:
    """Get global trace logger instance."""
    global _trace_logger
    if _trace_logger is None:
        _trace_logger = TraceLogger()
    return _trace_logger
