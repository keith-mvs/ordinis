"""
Orchestration Engine Data Models.

Defines data structures for trading cycles, pipeline results,
and performance metrics.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
import uuid


class CycleStatus(Enum):
    """Status of a trading cycle."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStage(Enum):
    """Stages in the trading pipeline."""

    DATA_FETCH = "data_fetch"
    ANOMALY_DETECTION = "anomaly_detection"
    FEATURE_ENGINEERING = "feature_engineering"
    SIGNAL_GENERATION = "signal_generation"
    RISK_EVALUATION = "risk_evaluation"
    ORDER_EXECUTION = "order_execution"
    ANALYTICS_RECORDING = "analytics_recording"


@dataclass
class StageResult:
    """Result from a pipeline stage."""

    stage: PipelineStage
    success: bool
    duration_ms: float
    output_count: int = 0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CycleResult:
    """Complete result of a trading cycle."""

    cycle_id: str = field(default_factory=lambda: f"CYC-{uuid.uuid4().hex[:12].upper()}")
    status: CycleStatus = CycleStatus.PENDING
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    total_duration_ms: float = 0.0

    # Stage results
    stages: list[StageResult] = field(default_factory=list)

    # Metrics
    signals_generated: int = 0
    signals_approved: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0

    # Latency breakdown
    data_latency_ms: float = 0.0
    signal_latency_ms: float = 0.0
    risk_latency_ms: float = 0.0
    execution_latency_ms: float = 0.0
    analytics_latency_ms: float = 0.0

    # Errors
    errors: list[str] = field(default_factory=list)

    def add_stage(self, result: StageResult) -> None:
        """Add a stage result."""
        self.stages.append(result)
        # Update latency breakdown
        if result.stage == PipelineStage.DATA_FETCH:
            self.data_latency_ms = result.duration_ms
        elif result.stage == PipelineStage.SIGNAL_GENERATION:
            self.signal_latency_ms = result.duration_ms
        elif result.stage == PipelineStage.RISK_EVALUATION:
            self.risk_latency_ms = result.duration_ms
        elif result.stage == PipelineStage.ORDER_EXECUTION:
            self.execution_latency_ms = result.duration_ms
        elif result.stage == PipelineStage.ANALYTICS_RECORDING:
            self.analytics_latency_ms = result.duration_ms

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_ms": self.total_duration_ms,
            "signals_generated": self.signals_generated,
            "signals_approved": self.signals_approved,
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "latency": {
                "data_ms": self.data_latency_ms,
                "signal_ms": self.signal_latency_ms,
                "risk_ms": self.risk_latency_ms,
                "execution_ms": self.execution_latency_ms,
                "analytics_ms": self.analytics_latency_ms,
            },
            "errors": self.errors,
        }


@dataclass
class PipelineMetrics:
    """Aggregated metrics for pipeline performance."""

    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0

    # Signal metrics
    total_signals: int = 0
    approved_signals: int = 0
    rejected_signals: int = 0

    # Order metrics
    total_orders: int = 0
    filled_orders: int = 0
    rejected_orders: int = 0

    # Latency metrics (rolling averages)
    avg_cycle_duration_ms: float = 0.0
    avg_signal_latency_ms: float = 0.0
    avg_risk_latency_ms: float = 0.0
    avg_execution_latency_ms: float = 0.0

    # Latency percentiles
    p50_cycle_duration_ms: float = 0.0
    p95_cycle_duration_ms: float = 0.0
    p99_cycle_duration_ms: float = 0.0

    def update_from_cycle(self, result: CycleResult) -> None:
        """Update metrics from a cycle result."""
        self.total_cycles += 1
        if result.status == CycleStatus.COMPLETED:
            self.successful_cycles += 1
        else:
            self.failed_cycles += 1

        self.total_signals += result.signals_generated
        self.approved_signals += result.signals_approved
        self.rejected_signals += result.signals_generated - result.signals_approved

        self.total_orders += result.orders_submitted
        self.filled_orders += result.orders_filled
        self.rejected_orders += result.orders_rejected

        # Update rolling average (simple exponential)
        alpha = 0.1
        self.avg_cycle_duration_ms = (
            alpha * result.total_duration_ms + (1 - alpha) * self.avg_cycle_duration_ms
        )
        self.avg_signal_latency_ms = (
            alpha * result.signal_latency_ms + (1 - alpha) * self.avg_signal_latency_ms
        )
        self.avg_risk_latency_ms = (
            alpha * result.risk_latency_ms + (1 - alpha) * self.avg_risk_latency_ms
        )
        self.avg_execution_latency_ms = (
            alpha * result.execution_latency_ms + (1 - alpha) * self.avg_execution_latency_ms
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cycles": {
                "total": self.total_cycles,
                "successful": self.successful_cycles,
                "failed": self.failed_cycles,
            },
            "signals": {
                "total": self.total_signals,
                "approved": self.approved_signals,
                "rejected": self.rejected_signals,
            },
            "orders": {
                "total": self.total_orders,
                "filled": self.filled_orders,
                "rejected": self.rejected_orders,
            },
            "latency_avg_ms": {
                "cycle": self.avg_cycle_duration_ms,
                "signal": self.avg_signal_latency_ms,
                "risk": self.avg_risk_latency_ms,
                "execution": self.avg_execution_latency_ms,
            },
        }
