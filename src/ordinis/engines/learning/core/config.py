"""
Learning Engine Configuration.

Defines configuration parameters for the continual learning engine
that captures events, trains models, and manages model lifecycle.
"""

from dataclasses import dataclass
from pathlib import Path

from ordinis.engines.base import BaseEngineConfig

from .models import RolloutStrategy


@dataclass
class LearningEngineConfig(BaseEngineConfig):
    """
    Configuration for the LearningEngine.

    Attributes:
        engine_id: Unique identifier for the engine.
        engine_name: Display name for the engine.
        data_dir: Directory for storing learning data and artifacts.
        max_events_memory: Maximum events to keep in memory before flush.
        flush_interval_seconds: Interval between data flushes to storage.
        enable_governance: Whether to enable governance hooks.
        require_eval_pass: Require evaluation pass before promotion.
    """

    engine_id: str = "learning"
    engine_name: str = "Learning Engine"

    # Storage settings
    data_dir: Path | None = None
    max_events_memory: int = 10000
    flush_interval_seconds: float = 60.0

    # Event collection settings
    collect_signals: bool = True
    collect_executions: bool = True
    collect_portfolio: bool = True
    collect_risk: bool = True
    collect_predictions: bool = True

    # Training settings
    min_samples_for_training: int = 1000
    training_batch_size: int = 256
    max_concurrent_jobs: int = 2

    # Evaluation settings
    require_eval_pass: bool = True
    eval_benchmark_threshold: float = 0.95  # Must beat baseline by 95%
    cross_validation_folds: int = 5

    # Drift detection settings
    enable_drift_detection: bool = True
    drift_check_interval_seconds: float = 3600.0  # 1 hour
    drift_threshold_pct: float = 0.10  # 10% change triggers alert

    # Rollout settings
    default_rollout_strategy: RolloutStrategy = RolloutStrategy.CANARY
    canary_percentage: float = 0.10  # 10% of traffic
    shadow_mode_duration_seconds: float = 86400.0  # 24 hours

    # Retention settings
    event_retention_days: int = 90
    model_version_retention: int = 10  # Keep last N versions

    # Governance
    enable_governance: bool = True

    def validate(self) -> list[str]:
        """Validate configuration parameters."""
        errors = super().validate()

        if self.max_events_memory < 100:
            errors.append("max_events_memory should be at least 100")
        if self.flush_interval_seconds < 1:
            errors.append("flush_interval_seconds must be at least 1")
        if self.min_samples_for_training < 10:
            errors.append("min_samples_for_training should be at least 10")
        if self.training_batch_size < 1:
            errors.append("training_batch_size must be positive")
        if not 0 < self.eval_benchmark_threshold <= 2:
            errors.append("eval_benchmark_threshold must be between 0 and 2")
        if not 0 < self.canary_percentage <= 1:
            errors.append("canary_percentage must be between 0 and 1")
        if self.drift_threshold_pct < 0:
            errors.append("drift_threshold_pct must be non-negative")
        if self.event_retention_days < 1:
            errors.append("event_retention_days must be at least 1")

        return errors
