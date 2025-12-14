"""Tests for LearningEngine data models.

This module tests the data structures used throughout the learning pipeline:
EventType, LearningEvent, TrainingJob, ModelVersion, EvaluationResult, and DriftAlert.
"""

from datetime import datetime

from ordinis.engines.learning.core.models import (
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


class TestEventType:
    """Test EventType enum."""

    def test_all_event_types_exist(self) -> None:
        """Test all expected event types are defined."""
        expected_types = [
            "SIGNAL_GENERATED",
            "SIGNAL_ACCURACY",
            "ORDER_SUBMITTED",
            "ORDER_FILLED",
            "ORDER_REJECTED",
            "REBALANCE_EXECUTED",
            "POSITION_OPENED",
            "POSITION_CLOSED",
            "RISK_BREACH",
            "DRAWDOWN_EVENT",
            "PNL_SNAPSHOT",
            "METRIC_RECORDED",
            "MODEL_PREDICTION",
            "DRIFT_DETECTED",
        ]

        for event_type in expected_types:
            assert hasattr(EventType, event_type)

    def test_event_type_values(self) -> None:
        """Test event type enum values."""
        assert EventType.SIGNAL_GENERATED.value == "signal_generated"
        assert EventType.ORDER_FILLED.value == "order_filled"
        assert EventType.DRIFT_DETECTED.value == "drift_detected"

    def test_event_type_uniqueness(self) -> None:
        """Test all event types have unique values."""
        values = [e.value for e in EventType]
        assert len(values) == len(set(values))


class TestTrainingStatus:
    """Test TrainingStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Test all expected statuses are defined."""
        expected_statuses = ["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]

        for status in expected_statuses:
            assert hasattr(TrainingStatus, status)

    def test_status_values(self) -> None:
        """Test status enum values."""
        assert TrainingStatus.PENDING.value == "pending"
        assert TrainingStatus.RUNNING.value == "running"
        assert TrainingStatus.COMPLETED.value == "completed"
        assert TrainingStatus.FAILED.value == "failed"
        assert TrainingStatus.CANCELLED.value == "cancelled"


class TestModelStage:
    """Test ModelStage enum."""

    def test_all_stages_exist(self) -> None:
        """Test all expected stages are defined."""
        expected_stages = ["DEVELOPMENT", "STAGING", "PRODUCTION", "ARCHIVED"]

        for stage in expected_stages:
            assert hasattr(ModelStage, stage)

    def test_stage_values(self) -> None:
        """Test stage enum values."""
        assert ModelStage.DEVELOPMENT.value == "development"
        assert ModelStage.STAGING.value == "staging"
        assert ModelStage.PRODUCTION.value == "production"
        assert ModelStage.ARCHIVED.value == "archived"


class TestRolloutStrategy:
    """Test RolloutStrategy enum."""

    def test_all_strategies_exist(self) -> None:
        """Test all expected strategies are defined."""
        expected_strategies = ["IMMEDIATE", "CANARY", "BLUE_GREEN", "SHADOW"]

        for strategy in expected_strategies:
            assert hasattr(RolloutStrategy, strategy)

    def test_strategy_values(self) -> None:
        """Test strategy enum values."""
        assert RolloutStrategy.IMMEDIATE.value == "immediate"
        assert RolloutStrategy.CANARY.value == "canary"
        assert RolloutStrategy.BLUE_GREEN.value == "blue_green"
        assert RolloutStrategy.SHADOW.value == "shadow"


class TestLearningEvent:
    """Test LearningEvent dataclass."""

    def test_default_creation(self) -> None:
        """Test creating an event with defaults."""
        event = LearningEvent()

        assert event.event_id.startswith("EVT-")
        assert event.event_type == EventType.METRIC_RECORDED
        assert event.source_engine == ""
        assert event.symbol is None
        assert isinstance(event.timestamp, datetime)
        assert event.payload == {}
        assert event.labels == {}
        assert event.outcome is None

    def test_signal_event_creation(self, sample_signal_event: LearningEvent) -> None:
        """Test creating a signal event."""
        assert sample_signal_event.event_type == EventType.SIGNAL_GENERATED
        assert sample_signal_event.source_engine == "signalcore"
        assert sample_signal_event.symbol == "AAPL"
        assert "signal_id" in sample_signal_event.payload
        assert sample_signal_event.payload["probability"] == 0.75

    def test_execution_event_with_outcome(self, sample_execution_event: LearningEvent) -> None:
        """Test creating an execution event with outcome."""
        assert sample_execution_event.event_type == EventType.ORDER_FILLED
        assert sample_execution_event.outcome == 1.5
        assert sample_execution_event.payload["quantity"] == 100

    def test_event_id_uniqueness(self) -> None:
        """Test event IDs are unique."""
        event1 = LearningEvent()
        event2 = LearningEvent()

        assert event1.event_id != event2.event_id

    def test_event_to_dict(self, sample_signal_event: LearningEvent) -> None:
        """Test converting event to dictionary."""
        event_dict = sample_signal_event.to_dict()

        assert isinstance(event_dict, dict)
        assert event_dict["event_id"] == sample_signal_event.event_id
        assert event_dict["event_type"] == "signal_generated"
        assert event_dict["source_engine"] == "signalcore"
        assert event_dict["symbol"] == "AAPL"
        assert isinstance(event_dict["timestamp"], str)
        assert event_dict["payload"] == sample_signal_event.payload

    def test_event_with_labels(self) -> None:
        """Test event with custom labels."""
        event = LearningEvent(
            event_type=EventType.MODEL_PREDICTION,
            labels={"model": "v2.0", "env": "production"},
        )

        assert event.labels["model"] == "v2.0"
        assert event.labels["env"] == "production"


class TestTrainingJob:
    """Test TrainingJob dataclass."""

    def test_default_creation(self) -> None:
        """Test creating a training job with defaults."""
        job = TrainingJob()

        assert job.job_id.startswith("JOB-")
        assert job.model_name == ""
        assert job.model_type == ""
        assert job.status == TrainingStatus.PENDING
        assert isinstance(job.created_at, datetime)
        assert job.started_at is None
        assert job.completed_at is None
        assert job.config == {}
        assert job.metrics == {}
        assert job.error_message is None
        assert job.artifacts_path is None

    def test_training_job_creation(self, sample_training_job: TrainingJob) -> None:
        """Test creating a training job."""
        assert sample_training_job.model_name == "signal_predictor"
        assert sample_training_job.model_type == "signal"
        assert sample_training_job.status == TrainingStatus.PENDING
        assert "batch_size" in sample_training_job.config

    def test_completed_job(self, completed_training_job: TrainingJob) -> None:
        """Test a completed training job."""
        assert completed_training_job.status == TrainingStatus.COMPLETED
        assert completed_training_job.started_at is not None
        assert completed_training_job.completed_at is not None
        assert "accuracy" in completed_training_job.metrics
        assert completed_training_job.metrics["accuracy"] == 0.85

    def test_job_id_uniqueness(self) -> None:
        """Test job IDs are unique."""
        job1 = TrainingJob()
        job2 = TrainingJob()

        assert job1.job_id != job2.job_id

    def test_job_to_dict(self, sample_training_job: TrainingJob) -> None:
        """Test converting job to dictionary."""
        job_dict = sample_training_job.to_dict()

        assert isinstance(job_dict, dict)
        assert job_dict["job_id"] == sample_training_job.job_id
        assert job_dict["model_name"] == "signal_predictor"
        assert job_dict["model_type"] == "signal"
        assert job_dict["status"] == "pending"
        assert isinstance(job_dict["created_at"], str)

    def test_job_with_error(self) -> None:
        """Test job with error message."""
        job = TrainingJob(
            model_name="test_model",
            status=TrainingStatus.FAILED,
            error_message="Out of memory",
        )

        assert job.status == TrainingStatus.FAILED
        assert job.error_message == "Out of memory"


class TestModelVersion:
    """Test ModelVersion dataclass."""

    def test_default_creation(self) -> None:
        """Test creating a model version with defaults."""
        version = ModelVersion()

        assert version.version_id.startswith("VER-")
        assert version.model_name == ""
        assert version.version == "1.0.0"
        assert version.stage == ModelStage.DEVELOPMENT
        assert isinstance(version.created_at, datetime)
        assert version.training_job_id is None
        assert version.metrics == {}
        assert version.parameters == {}
        assert version.artifacts_path is None
        assert version.description == ""

    def test_model_version_creation(self, sample_model_version: ModelVersion) -> None:
        """Test creating a model version."""
        assert sample_model_version.model_name == "signal_predictor"
        assert sample_model_version.version == "1.0.0"
        assert sample_model_version.stage == ModelStage.DEVELOPMENT
        assert "accuracy" in sample_model_version.metrics
        assert "n_estimators" in sample_model_version.parameters
        assert sample_model_version.description == "Initial model version"

    def test_production_model(self, production_model_version: ModelVersion) -> None:
        """Test production model version."""
        assert production_model_version.stage == ModelStage.PRODUCTION
        assert production_model_version.version == "2.0.0"
        assert production_model_version.metrics["accuracy"] == 0.90

    def test_version_id_uniqueness(self) -> None:
        """Test version IDs are unique."""
        version1 = ModelVersion()
        version2 = ModelVersion()

        assert version1.version_id != version2.version_id

    def test_version_to_dict(self, sample_model_version: ModelVersion) -> None:
        """Test converting version to dictionary."""
        version_dict = sample_model_version.to_dict()

        assert isinstance(version_dict, dict)
        assert version_dict["version_id"] == sample_model_version.version_id
        assert version_dict["model_name"] == "signal_predictor"
        assert version_dict["version"] == "1.0.0"
        assert version_dict["stage"] == "development"
        assert isinstance(version_dict["created_at"], str)

    def test_version_with_training_job(self) -> None:
        """Test version linked to training job."""
        version = ModelVersion(
            model_name="test_model",
            training_job_id="JOB-123",
        )

        assert version.training_job_id == "JOB-123"


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""

    def test_default_creation(self) -> None:
        """Test creating an evaluation result with defaults."""
        result = EvaluationResult()

        assert result.eval_id.startswith("EVAL-")
        assert result.model_version_id == ""
        assert result.benchmark_name == ""
        assert result.passed is False
        assert isinstance(result.timestamp, datetime)
        assert result.metrics == {}
        assert result.thresholds == {}
        assert result.details == {}

    def test_passed_evaluation(self, sample_evaluation_result: EvaluationResult) -> None:
        """Test a passed evaluation."""
        assert sample_evaluation_result.passed is True
        assert sample_evaluation_result.model_version_id == "VER-001"
        assert sample_evaluation_result.benchmark_name == "baseline_v1"
        assert sample_evaluation_result.metrics["accuracy"] == 0.90
        assert sample_evaluation_result.thresholds["accuracy"] == 0.85

    def test_failed_evaluation(self, failed_evaluation_result: EvaluationResult) -> None:
        """Test a failed evaluation."""
        assert failed_evaluation_result.passed is False
        assert failed_evaluation_result.metrics["accuracy"] == 0.75
        assert (
            failed_evaluation_result.metrics["accuracy"]
            < failed_evaluation_result.thresholds["accuracy"]
        )

    def test_eval_id_uniqueness(self) -> None:
        """Test evaluation IDs are unique."""
        eval1 = EvaluationResult()
        eval2 = EvaluationResult()

        assert eval1.eval_id != eval2.eval_id

    def test_evaluation_to_dict(self, sample_evaluation_result: EvaluationResult) -> None:
        """Test converting evaluation to dictionary."""
        eval_dict = sample_evaluation_result.to_dict()

        assert isinstance(eval_dict, dict)
        assert eval_dict["eval_id"] == sample_evaluation_result.eval_id
        assert eval_dict["model_version_id"] == "VER-001"
        assert eval_dict["benchmark_name"] == "baseline_v1"
        assert eval_dict["passed"] is True
        assert isinstance(eval_dict["timestamp"], str)

    def test_evaluation_with_details(self) -> None:
        """Test evaluation with additional details."""
        result = EvaluationResult(
            model_version_id="VER-001",
            benchmark_name="test",
            passed=True,
            details={
                "confusion_matrix": [[10, 2], [1, 15]],
                "test_samples": 28,
            },
        )

        assert "confusion_matrix" in result.details
        assert result.details["test_samples"] == 28


class TestDriftAlert:
    """Test DriftAlert dataclass."""

    def test_default_creation(self) -> None:
        """Test creating a drift alert with defaults."""
        alert = DriftAlert()

        assert alert.alert_id.startswith("DRIFT-")
        assert alert.model_name == ""
        assert alert.drift_type == ""
        assert alert.severity == "warning"
        assert isinstance(alert.timestamp, datetime)
        assert alert.metric_name == ""
        assert alert.current_value == 0.0
        assert alert.baseline_value == 0.0
        assert alert.threshold == 0.0
        assert alert.details == {}

    def test_warning_alert(self, sample_drift_alert: DriftAlert) -> None:
        """Test a warning drift alert."""
        assert sample_drift_alert.severity == "warning"
        assert sample_drift_alert.model_name == "signal_predictor"
        assert sample_drift_alert.drift_type == "performance_drift"
        assert sample_drift_alert.metric_name == "accuracy"
        assert sample_drift_alert.current_value == 0.75
        assert sample_drift_alert.baseline_value == 0.85

    def test_critical_alert(self, critical_drift_alert: DriftAlert) -> None:
        """Test a critical drift alert."""
        assert critical_drift_alert.severity == "critical"
        assert critical_drift_alert.model_name == "risk_estimator"
        assert critical_drift_alert.drift_type == "concept_drift"

    def test_alert_id_uniqueness(self) -> None:
        """Test alert IDs are unique."""
        alert1 = DriftAlert()
        alert2 = DriftAlert()

        assert alert1.alert_id != alert2.alert_id

    def test_alert_to_dict(self, sample_drift_alert: DriftAlert) -> None:
        """Test converting alert to dictionary."""
        alert_dict = sample_drift_alert.to_dict()

        assert isinstance(alert_dict, dict)
        assert alert_dict["alert_id"] == sample_drift_alert.alert_id
        assert alert_dict["model_name"] == "signal_predictor"
        assert alert_dict["drift_type"] == "performance_drift"
        assert alert_dict["severity"] == "warning"
        assert isinstance(alert_dict["timestamp"], str)
        assert alert_dict["current_value"] == 0.75
        assert alert_dict["baseline_value"] == 0.85

    def test_alert_drift_calculation(self) -> None:
        """Test drift alert with calculated values."""
        baseline = 0.90
        current = 0.75
        threshold = 0.10

        drift_pct = abs(current - baseline) / abs(baseline)

        alert = DriftAlert(
            model_name="test_model",
            drift_type="performance_drift",
            severity="critical" if drift_pct > threshold * 2 else "warning",
            metric_name="accuracy",
            current_value=current,
            baseline_value=baseline,
            threshold=threshold,
        )

        assert alert.severity == "warning"
        assert abs(alert.current_value - alert.baseline_value) / alert.baseline_value > threshold


class TestDataModelSerializationDeserialization:
    """Test serialization and deserialization of all data models."""

    def test_learning_event_round_trip(self, sample_signal_event: LearningEvent) -> None:
        """Test event can be serialized and deserialized."""
        event_dict = sample_signal_event.to_dict()

        assert event_dict["event_id"] == sample_signal_event.event_id
        assert event_dict["event_type"] == sample_signal_event.event_type.value

    def test_training_job_round_trip(self, sample_training_job: TrainingJob) -> None:
        """Test training job can be serialized and deserialized."""
        job_dict = sample_training_job.to_dict()

        assert job_dict["job_id"] == sample_training_job.job_id
        assert job_dict["status"] == sample_training_job.status.value

    def test_model_version_round_trip(self, sample_model_version: ModelVersion) -> None:
        """Test model version can be serialized and deserialized."""
        version_dict = sample_model_version.to_dict()

        assert version_dict["version_id"] == sample_model_version.version_id
        assert version_dict["stage"] == sample_model_version.stage.value

    def test_evaluation_result_round_trip(self, sample_evaluation_result: EvaluationResult) -> None:
        """Test evaluation result can be serialized and deserialized."""
        eval_dict = sample_evaluation_result.to_dict()

        assert eval_dict["eval_id"] == sample_evaluation_result.eval_id
        assert eval_dict["passed"] == sample_evaluation_result.passed

    def test_drift_alert_round_trip(self, sample_drift_alert: DriftAlert) -> None:
        """Test drift alert can be serialized and deserialized."""
        alert_dict = sample_drift_alert.to_dict()

        assert alert_dict["alert_id"] == sample_drift_alert.alert_id
        assert alert_dict["severity"] == sample_drift_alert.severity
