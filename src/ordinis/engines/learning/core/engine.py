"""
Learning Engine - Continual Model Improvement.

Captures trading events, trains models, evaluates against benchmarks,
and manages model lifecycle with drift detection and controlled rollout.

Key Workflows:
1. Data Capture: Stream outcomes from Execution/Portfolio/Analytics
2. Training: Offline jobs for signal/risk models
3. Evaluation: Benchmark passes before promotion
4. Rollout: Feature flags, canary, blue-green deployment
5. Monitoring: Drift detection, performance decay alerts
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
import logging
from typing import TYPE_CHECKING, Any

from ordinis.engines.base import (
    AuditRecord,
    BaseEngine,
    HealthLevel,
    HealthStatus,
)

from .config import LearningEngineConfig
from .models import (
    DriftAlert,
    EvaluationResult,
    EventType,
    LearningEvent,
    ModelStage,
    ModelVersion,
    RolloutStrategy,
    TrainingJob,
    TrainingStatus,
)

if TYPE_CHECKING:
    from ordinis.engines.base import GovernanceHook

_logger = logging.getLogger(__name__)


class LearningEngine(BaseEngine[LearningEngineConfig]):
    """
    Continual learning engine for model improvement.

    Captures trading events, manages training pipelines, evaluates models
    against benchmarks, and handles controlled rollout with monitoring.

    Example:
        >>> config = LearningEngineConfig()
        >>> engine = LearningEngine(config)
        >>> await engine.initialize()
        >>> engine.record_event(LearningEvent(
        ...     event_type=EventType.SIGNAL_GENERATED,
        ...     source_engine="signalcore",
        ...     payload={"signal_id": "...", "probability": 0.75}
        ... ))
    """

    def __init__(
        self,
        config: LearningEngineConfig | None = None,
        governance_hook: GovernanceHook | None = None,
    ) -> None:
        """Initialize the LearningEngine."""
        config = config or LearningEngineConfig()
        super().__init__(config, governance_hook)

        # Event storage
        self._events: list[LearningEvent] = []
        self._events_by_type: dict[EventType, list[LearningEvent]] = defaultdict(list)

        # Training state
        self._training_jobs: dict[str, TrainingJob] = {}
        self._active_jobs: list[str] = []

        # Model registry
        self._model_versions: dict[str, list[ModelVersion]] = defaultdict(list)
        self._production_models: dict[str, ModelVersion] = {}

        # Evaluation results
        self._evaluations: list[EvaluationResult] = []

        # Drift detection
        self._drift_alerts: list[DriftAlert] = []
        self._baseline_metrics: dict[str, dict[str, float]] = {}

    async def _do_initialize(self) -> None:
        """Initialize engine resources."""
        if self.config.data_dir:
            self.config.data_dir.mkdir(parents=True, exist_ok=True)
        _logger.info("LearningEngine initialized")

    async def _do_shutdown(self) -> None:
        """Shutdown and flush pending data."""
        # Flush remaining events
        await self._flush_events()
        _logger.info("LearningEngine shutdown complete")

    async def _do_health_check(self) -> HealthStatus:
        """Check engine health."""
        active_job_count = len(self._active_jobs)
        event_count = len(self._events)
        recent_drift_alerts = [
            a
            for a in self._drift_alerts
            if (datetime.now(UTC) - a.timestamp).total_seconds() < 3600
        ]

        if len(recent_drift_alerts) > 5:
            return HealthStatus(
                level=HealthLevel.DEGRADED,
                message="Multiple drift alerts detected",
                details={
                    "drift_alert_count": len(recent_drift_alerts),
                    "event_buffer_size": event_count,
                },
            )

        return HealthStatus(
            level=HealthLevel.HEALTHY,
            message="LearningEngine operational",
            details={
                "events_buffered": event_count,
                "active_training_jobs": active_job_count,
                "production_models": len(self._production_models),
                "total_evaluations": len(self._evaluations),
            },
        )

    # -------------------------------------------------------------------------
    # Event Collection
    # -------------------------------------------------------------------------

    def record_event(self, event: LearningEvent) -> str:
        """
        Record a learning event from trading operations.

        Args:
            event: Learning event to record.

        Returns:
            Event ID.
        """
        # Validate event type collection settings
        if not self._should_collect(event.event_type):
            _logger.debug("Skipping event type %s (collection disabled)", event.event_type)
            return event.event_id

        self._events.append(event)
        self._events_by_type[event.event_type].append(event)

        # Check if flush needed
        if len(self._events) >= self.config.max_events_memory:
            _logger.info("Event buffer full, triggering flush")
            # Note: In production, this would be async
            # For now, just truncate oldest events
            self._events = self._events[-self.config.max_events_memory // 2 :]

        return event.event_id

    def _should_collect(self, event_type: EventType) -> bool:
        """Check if event type should be collected based on config."""
        signal_types = {EventType.SIGNAL_GENERATED, EventType.SIGNAL_ACCURACY}
        execution_types = {
            EventType.ORDER_SUBMITTED,
            EventType.ORDER_FILLED,
            EventType.ORDER_REJECTED,
        }
        portfolio_types = {
            EventType.REBALANCE_EXECUTED,
            EventType.POSITION_OPENED,
            EventType.POSITION_CLOSED,
        }
        risk_types = {EventType.RISK_BREACH, EventType.DRAWDOWN_EVENT}
        prediction_types = {EventType.MODEL_PREDICTION, EventType.DRIFT_DETECTED}

        if event_type in signal_types:
            return self.config.collect_signals
        if event_type in execution_types:
            return self.config.collect_executions
        if event_type in portfolio_types:
            return self.config.collect_portfolio
        if event_type in risk_types:
            return self.config.collect_risk
        if event_type in prediction_types:
            return self.config.collect_predictions

        return True  # Collect unknown types

    async def _flush_events(self) -> None:
        """Flush events to persistent storage."""
        if not self._events:
            return

        if self.config.data_dir:
            # In production, write to parquet/database
            _logger.info("Flushing %d events to storage", len(self._events))

        # Audit the flush
        if self._governance_hook:
            await self._governance_hook.audit(
                AuditRecord(
                    engine_id=self.config.engine_id,
                    operation="flush_events",
                    context={"event_count": len(self._events)},
                    result={"status": "success"},
                    duration_ms=0,
                )
            )

    def get_events(
        self,
        event_type: EventType | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 1000,
    ) -> list[LearningEvent]:
        """
        Query recorded events.

        Args:
            event_type: Filter by event type.
            start: Filter by start timestamp.
            end: Filter by end timestamp.
            limit: Maximum events to return.

        Returns:
            List of matching events.
        """
        if event_type:
            events = self._events_by_type.get(event_type, [])
        else:
            events = self._events

        # Apply time filters
        if start:
            events = [e for e in events if e.timestamp >= start]
        if end:
            events = [e for e in events if e.timestamp <= end]

        return events[-limit:]

    # -------------------------------------------------------------------------
    # Training Management
    # -------------------------------------------------------------------------

    async def submit_training_job(
        self,
        model_name: str,
        model_type: str,
        config: dict[str, Any] | None = None,
    ) -> TrainingJob:
        """
        Submit a new training job.

        Args:
            model_name: Name of the model to train.
            model_type: Type of model (signal, risk, llm).
            config: Training configuration.

        Returns:
            Created training job.
        """
        if len(self._active_jobs) >= self.config.max_concurrent_jobs:
            _logger.warning("Max concurrent jobs reached (%d)", self.config.max_concurrent_jobs)

        job = TrainingJob(
            model_name=model_name,
            model_type=model_type,
            config=config or {},
        )

        self._training_jobs[job.job_id] = job
        _logger.info("Training job submitted: %s for model %s", job.job_id, model_name)

        # Audit the submission
        if self._governance_hook:
            await self._governance_hook.audit(
                AuditRecord(
                    engine_id=self.config.engine_id,
                    operation="submit_training_job",
                    context={
                        "model_name": model_name,
                        "model_type": model_type,
                    },
                    result={"job_id": job.job_id},
                    duration_ms=0,
                )
            )

        return job

    def get_training_job(self, job_id: str) -> TrainingJob | None:
        """Get training job by ID."""
        return self._training_jobs.get(job_id)

    def list_training_jobs(
        self,
        status: TrainingStatus | None = None,
        limit: int = 100,
    ) -> list[TrainingJob]:
        """List training jobs, optionally filtered by status."""
        jobs = list(self._training_jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]

    # -------------------------------------------------------------------------
    # Model Registry
    # -------------------------------------------------------------------------

    async def register_model_version(
        self,
        model_name: str,
        version: str,
        training_job_id: str | None = None,
        metrics: dict[str, float] | None = None,
        parameters: dict[str, Any] | None = None,
        description: str = "",
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_name: Name of the model.
            version: Semantic version string.
            training_job_id: Associated training job.
            metrics: Training/validation metrics.
            parameters: Model parameters.
            description: Version description.

        Returns:
            Created model version.
        """
        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            training_job_id=training_job_id,
            metrics=metrics or {},
            parameters=parameters or {},
            description=description,
        )

        self._model_versions[model_name].append(model_version)
        _logger.info("Registered model version: %s v%s", model_name, version)

        return model_version

    async def promote_model(
        self,
        model_name: str,
        version_id: str,
        target_stage: ModelStage,
        rollout_strategy: RolloutStrategy | None = None,
    ) -> ModelVersion | None:
        """
        Promote a model version to a new stage.

        Args:
            model_name: Name of the model.
            version_id: Version ID to promote.
            target_stage: Target stage (staging/production).
            rollout_strategy: Deployment strategy.

        Returns:
            Updated model version or None if not found.
        """
        rollout_strategy = rollout_strategy or self.config.default_rollout_strategy

        # Find the version
        versions = self._model_versions.get(model_name, [])
        version = next((v for v in versions if v.version_id == version_id), None)

        if not version:
            _logger.warning("Model version not found: %s", version_id)
            return None

        # Check evaluation requirement
        if self.config.require_eval_pass and target_stage == ModelStage.PRODUCTION:
            passed_evals = [
                e for e in self._evaluations if e.model_version_id == version_id and e.passed
            ]
            if not passed_evals:
                _logger.warning(
                    "Cannot promote %s to production - no passing evaluations",
                    version_id,
                )
                return None

        # Update stage
        version.stage = target_stage

        # Update production tracking
        if target_stage == ModelStage.PRODUCTION:
            old_prod = self._production_models.get(model_name)
            if old_prod:
                old_prod.stage = ModelStage.ARCHIVED
            self._production_models[model_name] = version

        _logger.info(
            "Promoted %s v%s to %s via %s",
            model_name,
            version.version,
            target_stage.value,
            rollout_strategy.value,
        )

        # Audit the promotion
        if self._governance_hook:
            await self._governance_hook.audit(
                AuditRecord(
                    engine_id=self.config.engine_id,
                    operation="promote_model",
                    context={
                        "model_name": model_name,
                        "version_id": version_id,
                        "target_stage": target_stage.value,
                        "rollout_strategy": rollout_strategy.value,
                    },
                    result={"success": True},
                    duration_ms=0,
                )
            )

        return version

    def get_production_model(self, model_name: str) -> ModelVersion | None:
        """Get the current production model version."""
        return self._production_models.get(model_name)

    def list_model_versions(
        self,
        model_name: str,
        stage: ModelStage | None = None,
    ) -> list[ModelVersion]:
        """List versions of a model, optionally filtered by stage."""
        versions = self._model_versions.get(model_name, [])
        if stage:
            versions = [v for v in versions if v.stage == stage]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    async def evaluate_model(
        self,
        version_id: str,
        benchmark_name: str,
        metrics: dict[str, float],
        thresholds: dict[str, float],
    ) -> EvaluationResult:
        """
        Evaluate a model version against a benchmark.

        Args:
            version_id: Model version to evaluate.
            benchmark_name: Name of the benchmark.
            metrics: Evaluation metrics.
            thresholds: Required thresholds to pass.

        Returns:
            Evaluation result.
        """
        # Check if metrics meet thresholds
        passed = all(
            metrics.get(k, 0) >= v * self.config.eval_benchmark_threshold
            for k, v in thresholds.items()
        )

        result = EvaluationResult(
            model_version_id=version_id,
            benchmark_name=benchmark_name,
            passed=passed,
            metrics=metrics,
            thresholds=thresholds,
        )

        self._evaluations.append(result)

        _logger.info(
            "Evaluation %s for %s: %s",
            result.eval_id,
            version_id,
            "PASSED" if passed else "FAILED",
        )

        return result

    def get_evaluations(
        self,
        version_id: str | None = None,
        benchmark_name: str | None = None,
    ) -> list[EvaluationResult]:
        """Get evaluation results with optional filters."""
        results = self._evaluations
        if version_id:
            results = [e for e in results if e.model_version_id == version_id]
        if benchmark_name:
            results = [e for e in results if e.benchmark_name == benchmark_name]
        return results

    # -------------------------------------------------------------------------
    # Drift Detection
    # -------------------------------------------------------------------------

    def set_baseline(self, model_name: str, metrics: dict[str, float]) -> None:
        """Set baseline metrics for drift detection."""
        self._baseline_metrics[model_name] = metrics
        _logger.info("Set baseline for %s: %s", model_name, metrics)

    def check_drift(
        self,
        model_name: str,
        current_metrics: dict[str, float],
    ) -> list[DriftAlert]:
        """
        Check for drift against baseline.

        Args:
            model_name: Model to check.
            current_metrics: Current metric values.

        Returns:
            List of drift alerts (empty if no drift).
        """
        baseline = self._baseline_metrics.get(model_name)
        if not baseline:
            return []

        alerts: list[DriftAlert] = []
        threshold = self.config.drift_threshold_pct

        for metric_name, current_value in current_metrics.items():
            baseline_value = baseline.get(metric_name)
            if baseline_value is None:
                continue

            if baseline_value != 0:
                change_pct = abs(current_value - baseline_value) / abs(baseline_value)
            else:
                change_pct = abs(current_value)

            if change_pct > threshold:
                severity = "critical" if change_pct > threshold * 2 else "warning"
                alert = DriftAlert(
                    model_name=model_name,
                    drift_type="performance_drift",
                    severity=severity,
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    threshold=threshold,
                )
                alerts.append(alert)
                self._drift_alerts.append(alert)
                _logger.warning(
                    "Drift detected for %s.%s: %.2f%% change",
                    model_name,
                    metric_name,
                    change_pct * 100,
                )

        return alerts

    def get_drift_alerts(
        self,
        model_name: str | None = None,
        severity: str | None = None,
    ) -> list[DriftAlert]:
        """Get drift alerts with optional filters."""
        alerts = self._drift_alerts
        if model_name:
            alerts = [a for a in alerts if a.model_name == model_name]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "events_buffered": len(self._events),
            "events_by_type": {k.value: len(v) for k, v in self._events_by_type.items()},
            "training_jobs": len(self._training_jobs),
            "active_jobs": len(self._active_jobs),
            "model_versions": sum(len(v) for v in self._model_versions.values()),
            "production_models": len(self._production_models),
            "evaluations": len(self._evaluations),
            "drift_alerts": len(self._drift_alerts),
        }
