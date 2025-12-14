"""
Learning Engine Data Models.

Defines data structures for learning events, training jobs, model versions,
and evaluation results used throughout the continual learning pipeline.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
import uuid


class EventType(Enum):
    """Types of learning events captured from trading operations."""

    # Signal events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_ACCURACY = "signal_accuracy"

    # Execution events
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"

    # Portfolio events
    REBALANCE_EXECUTED = "rebalance_executed"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"

    # Risk events
    RISK_BREACH = "risk_breach"
    DRAWDOWN_EVENT = "drawdown_event"

    # Performance events
    PNL_SNAPSHOT = "pnl_snapshot"
    METRIC_RECORDED = "metric_recorded"

    # Model events
    MODEL_PREDICTION = "model_prediction"
    DRIFT_DETECTED = "drift_detected"


class TrainingStatus(Enum):
    """Status of a training job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelStage(Enum):
    """Model lifecycle stage."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class RolloutStrategy(Enum):
    """Model deployment strategy."""

    IMMEDIATE = "immediate"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    SHADOW = "shadow"


@dataclass
class LearningEvent:
    """Event captured for learning purposes."""

    event_id: str = field(default_factory=lambda: f"EVT-{uuid.uuid4().hex[:12].upper()}")
    event_type: EventType = EventType.METRIC_RECORDED
    source_engine: str = ""
    symbol: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    outcome: float | None = None  # For supervised learning (actual result)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source_engine": self.source_engine,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "labels": self.labels,
            "outcome": self.outcome,
        }


@dataclass
class TrainingJob:
    """Training job configuration and status."""

    job_id: str = field(default_factory=lambda: f"JOB-{uuid.uuid4().hex[:12].upper()}")
    model_name: str = ""
    model_type: str = ""  # signal, risk, llm, etc.
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None
    artifacts_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "config": self.config,
            "metrics": self.metrics,
            "error_message": self.error_message,
            "artifacts_path": self.artifacts_path,
        }


@dataclass
class ModelVersion:
    """Versioned model with metadata and evaluation results."""

    version_id: str = field(default_factory=lambda: f"VER-{uuid.uuid4().hex[:12].upper()}")
    model_name: str = ""
    version: str = "1.0.0"
    stage: ModelStage = ModelStage.DEVELOPMENT
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    training_job_id: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    artifacts_path: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "version": self.version,
            "stage": self.stage.value,
            "created_at": self.created_at.isoformat(),
            "training_job_id": self.training_job_id,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "artifacts_path": self.artifacts_path,
            "description": self.description,
        }


@dataclass
class EvaluationResult:
    """Result of model evaluation against benchmarks."""

    eval_id: str = field(default_factory=lambda: f"EVAL-{uuid.uuid4().hex[:12].upper()}")
    model_version_id: str = ""
    benchmark_name: str = ""
    passed: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "eval_id": self.eval_id,
            "model_version_id": self.model_version_id,
            "benchmark_name": self.benchmark_name,
            "passed": self.passed,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "details": self.details,
        }


@dataclass
class DriftAlert:
    """Alert for detected model drift."""

    alert_id: str = field(default_factory=lambda: f"DRIFT-{uuid.uuid4().hex[:12].upper()}")
    model_name: str = ""
    drift_type: str = ""  # data_drift, concept_drift, performance_drift
    severity: str = "warning"  # info, warning, critical
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metric_name: str = ""
    current_value: float = 0.0
    baseline_value: float = 0.0
    threshold: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "model_name": self.model_name,
            "drift_type": self.drift_type,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "threshold": self.threshold,
            "details": self.details,
        }
