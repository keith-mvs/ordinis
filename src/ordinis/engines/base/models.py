"""Common data models for all Ordinis engines.

This module defines shared data structures used across all engines
for consistent status reporting, error handling, and audit trails.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EngineState(Enum):
    """Lifecycle states for engines."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class HealthLevel(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class HealthStatus:
    """Engine health status report.

    Attributes:
        level: Overall health level.
        message: Human-readable status message.
        details: Additional diagnostic information.
        last_check: Timestamp of health check.
        latency_ms: Response latency in milliseconds.
    """

    level: HealthLevel
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.utcnow)
    latency_ms: float | None = None

    @property
    def is_healthy(self) -> bool:
        """Check if engine is operational."""
        return self.level in (HealthLevel.HEALTHY, HealthLevel.DEGRADED)


@dataclass(frozen=True)
class EngineMetrics:
    """Runtime metrics for an engine.

    Attributes:
        requests_total: Total requests processed.
        requests_failed: Failed request count.
        latency_p50_ms: Median latency.
        latency_p95_ms: 95th percentile latency.
        latency_p99_ms: 99th percentile latency.
        uptime_seconds: Time since initialization.
    """

    requests_total: int = 0
    requests_failed: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    uptime_seconds: float = 0.0

    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.requests_total == 0:
            return 0.0
        return (self.requests_failed / self.requests_total) * 100


@dataclass(frozen=True)
class AuditRecord:
    """Standardized audit record for governance tracking.

    Every engine operation should emit an audit record for
    compliance and debugging purposes.

    Attributes:
        engine: Engine name.
        action: Operation performed.
        timestamp: When the action occurred.
        trace_id: Distributed trace identifier.
        inputs: Sanitized input summary.
        outputs: Sanitized output summary.
        model_used: AI model used (if applicable).
        policy_version: Governance policy version applied.
        decision: Allow/deny/warn decision.
        latency_ms: Operation latency.
        metadata: Additional context.
    """

    engine: str
    action: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    trace_id: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    model_used: str | None = None
    policy_version: str | None = None
    decision: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineError:
    """Structured error information.

    Attributes:
        code: Error code for programmatic handling.
        message: Human-readable error message.
        engine: Engine that raised the error.
        recoverable: Whether the error is recoverable.
        details: Additional error context.
        timestamp: When the error occurred.
    """

    code: str
    message: str
    engine: str
    recoverable: bool = True
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
