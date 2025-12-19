"""Shared fixtures for LearningEngine tests.

This module provides mock implementations and fixtures for testing
the LearningEngine and its components.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ordinis.engines.base import (
    BaseGovernanceHook,
    Decision,
    PreflightContext,
    PreflightResult,
)
from ordinis.engines.learning.core.config import LearningEngineConfig
from ordinis.engines.learning.core.engine import LearningEngine
from ordinis.engines.learning.core.models import (
    DriftAlert,
    EvaluationResult,
    EventType,
    LearningEvent,
    ModelStage,
    ModelVersion,
    TrainingJob,
    TrainingStatus,
)


class MockGovernanceHook(BaseGovernanceHook):
    """Mock governance hook for testing.

    Tracks all calls and can be configured to return specific decisions.
    """

    def __init__(
        self,
        engine_name: str,
        preflight_decision: Decision = Decision.ALLOW,
        preflight_reason: str = "Test policy",
    ) -> None:
        """Initialize the mock governance hook."""
        super().__init__(engine_name)
        self.preflight_decision = preflight_decision
        self.preflight_reason = preflight_reason

        # Call tracking
        self.preflight_calls: list[PreflightContext] = []
        self.audit_calls: list[Any] = []
        self.error_calls: list[Any] = []

    async def preflight(self, context: PreflightContext) -> PreflightResult:
        """Mock preflight that tracks calls."""
        self.preflight_calls.append(context)
        return PreflightResult(
            decision=self.preflight_decision,
            reason=self.preflight_reason,
            policy_id="MOCK-001",
            policy_version=self.policy_version,
        )

    async def audit(self, record: Any) -> None:
        """Mock audit that tracks calls."""
        self.audit_calls.append(record)

    async def on_error(self, error: Any) -> None:
        """Mock error handler that tracks calls."""
        self.error_calls.append(error)


@pytest.fixture
def temp_data_dir() -> Path:
    """Provide a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def learning_config(temp_data_dir: Path) -> LearningEngineConfig:
    """Provide a default learning engine configuration."""
    return LearningEngineConfig(
        engine_id="test_learning",
        engine_name="TestLearningEngine",
        data_dir=temp_data_dir,
        max_events_memory=100,
        flush_interval_seconds=10.0,
        min_samples_for_training=10,
        max_concurrent_jobs=2,
        enable_governance=True,
    )


@pytest.fixture
def learning_config_minimal() -> LearningEngineConfig:
    """Provide a minimal learning engine configuration."""
    return LearningEngineConfig(
        engine_id="minimal_learning",
        enable_governance=False,
    )


@pytest.fixture
def mock_governance_hook() -> MockGovernanceHook:
    """Provide a mock governance hook that allows all operations."""
    return MockGovernanceHook("TestLearningEngine")


@pytest.fixture
def learning_engine(learning_config: LearningEngineConfig) -> LearningEngine:
    """Provide a learning engine with default configuration."""
    return LearningEngine(learning_config)


@pytest.fixture
def learning_engine_with_hook(
    learning_config: LearningEngineConfig,
    mock_governance_hook: MockGovernanceHook,
) -> LearningEngine:
    """Provide a learning engine with governance hook."""
    return LearningEngine(learning_config, mock_governance_hook)


@pytest.fixture
async def initialized_learning_engine(learning_engine: LearningEngine) -> LearningEngine:
    """Provide an initialized learning engine."""
    await learning_engine.initialize()
    yield learning_engine
    # Cleanup
    if learning_engine.is_running:
        await learning_engine.shutdown()


# Sample events for testing
@pytest.fixture
def sample_signal_event() -> LearningEvent:
    """Provide a sample signal event."""
    return LearningEvent(
        event_type=EventType.SIGNAL_GENERATED,
        source_engine="signalcore",
        symbol="AAPL",
        payload={
            "signal_id": "SIG-001",
            "probability": 0.75,
            "direction": "long",
        },
    )


@pytest.fixture
def sample_execution_event() -> LearningEvent:
    """Provide a sample execution event."""
    return LearningEvent(
        event_type=EventType.ORDER_FILLED,
        source_engine="flowroute",
        symbol="MSFT",
        payload={
            "order_id": "ORD-001",
            "quantity": 100,
            "price": 350.25,
        },
        outcome=1.5,  # 1.5% profit
    )


@pytest.fixture
def sample_portfolio_event() -> LearningEvent:
    """Provide a sample portfolio event."""
    return LearningEvent(
        event_type=EventType.REBALANCE_EXECUTED,
        source_engine="portfolio",
        payload={
            "rebalance_id": "REB-001",
            "trades_executed": 5,
        },
    )


@pytest.fixture
def sample_risk_event() -> LearningEvent:
    """Provide a sample risk event."""
    return LearningEvent(
        event_type=EventType.RISK_BREACH,
        source_engine="riskguard",
        symbol="TSLA",
        payload={
            "breach_type": "position_limit",
            "limit": 10000,
            "actual": 12000,
        },
    )


@pytest.fixture
def sample_drift_event() -> LearningEvent:
    """Provide a sample drift detection event."""
    return LearningEvent(
        event_type=EventType.DRIFT_DETECTED,
        source_engine="learning",
        payload={
            "model_name": "signal_predictor",
            "drift_type": "performance_drift",
            "severity": "warning",
        },
    )


# Training job fixtures
@pytest.fixture
def sample_training_job() -> TrainingJob:
    """Provide a sample training job."""
    return TrainingJob(
        model_name="signal_predictor",
        model_type="signal",
        config={
            "batch_size": 256,
            "epochs": 10,
            "learning_rate": 0.001,
        },
    )


@pytest.fixture
def completed_training_job() -> TrainingJob:
    """Provide a completed training job."""
    job = TrainingJob(
        model_name="signal_predictor",
        model_type="signal",
        status=TrainingStatus.COMPLETED,
        config={"batch_size": 256},
        metrics={
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
        },
    )
    job.started_at = datetime.now(UTC) - timedelta(hours=1)
    job.completed_at = datetime.now(UTC)
    return job


# Model version fixtures
@pytest.fixture
def sample_model_version() -> ModelVersion:
    """Provide a sample model version."""
    return ModelVersion(
        model_name="signal_predictor",
        version="1.0.0",
        stage=ModelStage.DEVELOPMENT,
        metrics={
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
        },
        parameters={
            "n_estimators": 100,
            "max_depth": 10,
        },
        description="Initial model version",
    )


@pytest.fixture
def production_model_version() -> ModelVersion:
    """Provide a production model version."""
    return ModelVersion(
        model_name="signal_predictor",
        version="2.0.0",
        stage=ModelStage.PRODUCTION,
        metrics={
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.92,
        },
    )


# Evaluation fixtures
@pytest.fixture
def sample_evaluation_result() -> EvaluationResult:
    """Provide a sample evaluation result."""
    return EvaluationResult(
        model_version_id="VER-001",
        benchmark_name="baseline_v1",
        passed=True,
        metrics={
            "accuracy": 0.90,
            "precision": 0.88,
        },
        thresholds={
            "accuracy": 0.85,
            "precision": 0.80,
        },
    )


@pytest.fixture
def failed_evaluation_result() -> EvaluationResult:
    """Provide a failed evaluation result."""
    return EvaluationResult(
        model_version_id="VER-002",
        benchmark_name="baseline_v1",
        passed=False,
        metrics={
            "accuracy": 0.75,
            "precision": 0.70,
        },
        thresholds={
            "accuracy": 0.85,
            "precision": 0.80,
        },
    )


# Drift alert fixtures
@pytest.fixture
def sample_drift_alert() -> DriftAlert:
    """Provide a sample drift alert."""
    return DriftAlert(
        model_name="signal_predictor",
        drift_type="performance_drift",
        severity="warning",
        metric_name="accuracy",
        current_value=0.75,
        baseline_value=0.85,
        threshold=0.10,
    )


@pytest.fixture
def critical_drift_alert() -> DriftAlert:
    """Provide a critical drift alert."""
    return DriftAlert(
        model_name="risk_estimator",
        drift_type="concept_drift",
        severity="critical",
        metric_name="precision",
        current_value=0.60,
        baseline_value=0.90,
        threshold=0.10,
    )


# Mock ML components
@pytest.fixture
def mock_ml_trainer() -> MagicMock:
    """Provide a mock ML trainer."""
    trainer = MagicMock()
    trainer.train = AsyncMock(
        return_value={
            "accuracy": 0.85,
            "loss": 0.15,
        }
    )
    return trainer


@pytest.fixture
def mock_ml_evaluator() -> MagicMock:
    """Provide a mock ML evaluator."""
    evaluator = MagicMock()
    evaluator.evaluate = AsyncMock(
        return_value={
            "accuracy": 0.87,
            "precision": 0.84,
            "recall": 0.90,
        }
    )
    return evaluator
