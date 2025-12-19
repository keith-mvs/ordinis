"""Tests for orchestration engine data models.

This module tests the CycleResult, CycleStatus, PipelineMetrics,
StageResult, and PipelineStage models.
"""

from datetime import UTC, datetime

import pytest

from ordinis.engines.orchestration.core.models import (
    CycleResult,
    CycleStatus,
    PipelineMetrics,
    PipelineStage,
    StageResult,
)


class TestCycleStatus:
    """Test CycleStatus enum."""

    def test_all_status_values(self) -> None:
        """Test all cycle status values are defined."""
        assert CycleStatus.PENDING.value == "pending"
        assert CycleStatus.RUNNING.value == "running"
        assert CycleStatus.COMPLETED.value == "completed"
        assert CycleStatus.FAILED.value == "failed"
        assert CycleStatus.SKIPPED.value == "skipped"

    def test_status_enum_count(self) -> None:
        """Test expected number of status values."""
        assert len(CycleStatus) == 5


class TestPipelineStage:
    """Test PipelineStage enum."""

    def test_all_stage_values(self) -> None:
        """Test all pipeline stages are defined."""
        assert PipelineStage.DATA_FETCH.value == "data_fetch"
        assert PipelineStage.ANOMALY_DETECTION.value == "anomaly_detection"
        assert PipelineStage.FEATURE_ENGINEERING.value == "feature_engineering"
        assert PipelineStage.SIGNAL_GENERATION.value == "signal_generation"
        assert PipelineStage.RISK_EVALUATION.value == "risk_evaluation"
        assert PipelineStage.ORDER_EXECUTION.value == "order_execution"
        assert PipelineStage.ANALYTICS_RECORDING.value == "analytics_recording"

    def test_stage_enum_count(self) -> None:
        """Test expected number of pipeline stages."""
        assert len(PipelineStage) == 7


class TestStageResult:
    """Test StageResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values for stage result."""
        result = StageResult(
            stage=PipelineStage.DATA_FETCH,
            success=True,
            duration_ms=10.5,
        )

        assert result.stage == PipelineStage.DATA_FETCH
        assert result.success is True
        assert result.duration_ms == 10.5
        assert result.output_count == 0
        assert result.error is None
        assert result.details == {}

    def test_with_output_count(self) -> None:
        """Test stage result with output count."""
        result = StageResult(
            stage=PipelineStage.SIGNAL_GENERATION,
            success=True,
            duration_ms=20.0,
            output_count=5,
        )

        assert result.output_count == 5

    def test_with_error(self) -> None:
        """Test stage result with error."""
        result = StageResult(
            stage=PipelineStage.RISK_EVALUATION,
            success=False,
            duration_ms=5.0,
            error="Risk check failed",
        )

        assert result.success is False
        assert result.error == "Risk check failed"

    def test_with_details(self) -> None:
        """Test stage result with details."""
        details = {"rejections": ["signal1", "signal2"], "reason": "high_risk"}
        result = StageResult(
            stage=PipelineStage.RISK_EVALUATION,
            success=True,
            duration_ms=15.0,
            details=details,
        )

        assert result.details == details
        assert "rejections" in result.details
        assert len(result.details["rejections"]) == 2


class TestCycleResult:
    """Test CycleResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values for cycle result."""
        result = CycleResult()

        assert result.status == CycleStatus.PENDING
        assert result.cycle_id.startswith("CYC-")
        assert len(result.cycle_id) == 16  # "CYC-" + 12 hex chars
        assert isinstance(result.started_at, datetime)
        assert result.completed_at is None
        assert result.total_duration_ms == 0.0
        assert result.stages == []
        assert result.signals_generated == 0
        assert result.signals_approved == 0
        assert result.orders_submitted == 0
        assert result.orders_filled == 0
        assert result.orders_rejected == 0
        assert result.data_latency_ms == 0.0
        assert result.signal_latency_ms == 0.0
        assert result.risk_latency_ms == 0.0
        assert result.execution_latency_ms == 0.0
        assert result.analytics_latency_ms == 0.0
        assert result.errors == []

    def test_unique_cycle_ids(self) -> None:
        """Test cycle IDs are unique."""
        result1 = CycleResult()
        result2 = CycleResult()

        assert result1.cycle_id != result2.cycle_id

    def test_add_stage_data_fetch(self) -> None:
        """Test adding DATA_FETCH stage updates data_latency_ms."""
        result = CycleResult()
        stage = StageResult(
            stage=PipelineStage.DATA_FETCH,
            success=True,
            duration_ms=50.0,
        )

        result.add_stage(stage)

        assert len(result.stages) == 1
        assert result.stages[0] == stage
        assert result.data_latency_ms == 50.0

    def test_add_stage_signal_generation(self) -> None:
        """Test adding SIGNAL_GENERATION stage updates signal_latency_ms."""
        result = CycleResult()
        stage = StageResult(
            stage=PipelineStage.SIGNAL_GENERATION,
            success=True,
            duration_ms=75.0,
        )

        result.add_stage(stage)

        assert len(result.stages) == 1
        assert result.signal_latency_ms == 75.0

    def test_add_stage_risk_evaluation(self) -> None:
        """Test adding RISK_EVALUATION stage updates risk_latency_ms."""
        result = CycleResult()
        stage = StageResult(
            stage=PipelineStage.RISK_EVALUATION,
            success=True,
            duration_ms=25.0,
        )

        result.add_stage(stage)

        assert len(result.stages) == 1
        assert result.risk_latency_ms == 25.0

    def test_add_stage_order_execution(self) -> None:
        """Test adding ORDER_EXECUTION stage updates execution_latency_ms."""
        result = CycleResult()
        stage = StageResult(
            stage=PipelineStage.ORDER_EXECUTION,
            success=True,
            duration_ms=100.0,
        )

        result.add_stage(stage)

        assert len(result.stages) == 1
        assert result.execution_latency_ms == 100.0

    def test_add_stage_analytics_recording(self) -> None:
        """Test adding ANALYTICS_RECORDING stage updates analytics_latency_ms."""
        result = CycleResult()
        stage = StageResult(
            stage=PipelineStage.ANALYTICS_RECORDING,
            success=True,
            duration_ms=30.0,
        )

        result.add_stage(stage)

        assert len(result.stages) == 1
        assert result.analytics_latency_ms == 30.0

    def test_add_multiple_stages(self) -> None:
        """Test adding multiple stages accumulates correctly."""
        result = CycleResult()

        result.add_stage(
            StageResult(stage=PipelineStage.DATA_FETCH, success=True, duration_ms=50.0)
        )
        result.add_stage(
            StageResult(stage=PipelineStage.SIGNAL_GENERATION, success=True, duration_ms=75.0)
        )
        result.add_stage(
            StageResult(stage=PipelineStage.RISK_EVALUATION, success=True, duration_ms=25.0)
        )

        assert len(result.stages) == 3
        assert result.data_latency_ms == 50.0
        assert result.signal_latency_ms == 75.0
        assert result.risk_latency_ms == 25.0

    def test_to_dict(self) -> None:
        """Test converting cycle result to dictionary."""
        result = CycleResult()
        result.status = CycleStatus.COMPLETED
        result.signals_generated = 5
        result.signals_approved = 3
        result.orders_submitted = 3
        result.orders_filled = 2
        result.data_latency_ms = 50.0
        result.signal_latency_ms = 75.0
        result.risk_latency_ms = 25.0
        result.execution_latency_ms = 100.0
        result.analytics_latency_ms = 30.0
        result.completed_at = datetime.now(UTC)
        result.total_duration_ms = 280.0

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["cycle_id"] == result.cycle_id
        assert result_dict["status"] == "completed"
        assert result_dict["signals_generated"] == 5
        assert result_dict["signals_approved"] == 3
        assert result_dict["orders_submitted"] == 3
        assert result_dict["orders_filled"] == 2
        assert result_dict["total_duration_ms"] == 280.0
        assert "latency" in result_dict
        assert result_dict["latency"]["data_ms"] == 50.0
        assert result_dict["latency"]["signal_ms"] == 75.0
        assert result_dict["latency"]["risk_ms"] == 25.0
        assert result_dict["latency"]["execution_ms"] == 100.0
        assert result_dict["latency"]["analytics_ms"] == 30.0
        assert result_dict["errors"] == []

    def test_to_dict_with_errors(self) -> None:
        """Test to_dict includes errors."""
        result = CycleResult()
        result.errors = ["Error 1", "Error 2"]

        result_dict = result.to_dict()

        assert result_dict["errors"] == ["Error 1", "Error 2"]

    def test_to_dict_completed_at_none(self) -> None:
        """Test to_dict handles None completed_at."""
        result = CycleResult()
        result.completed_at = None

        result_dict = result.to_dict()

        assert result_dict["completed_at"] is None


class TestPipelineMetrics:
    """Test PipelineMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default values for pipeline metrics."""
        metrics = PipelineMetrics()

        assert metrics.total_cycles == 0
        assert metrics.successful_cycles == 0
        assert metrics.failed_cycles == 0
        assert metrics.total_signals == 0
        assert metrics.approved_signals == 0
        assert metrics.rejected_signals == 0
        assert metrics.total_orders == 0
        assert metrics.filled_orders == 0
        assert metrics.rejected_orders == 0
        assert metrics.avg_cycle_duration_ms == 0.0
        assert metrics.avg_signal_latency_ms == 0.0
        assert metrics.avg_risk_latency_ms == 0.0
        assert metrics.avg_execution_latency_ms == 0.0
        assert metrics.p50_cycle_duration_ms == 0.0
        assert metrics.p95_cycle_duration_ms == 0.0
        assert metrics.p99_cycle_duration_ms == 0.0

    def test_update_from_completed_cycle(self) -> None:
        """Test updating metrics from a completed cycle."""
        metrics = PipelineMetrics()
        result = CycleResult()
        result.status = CycleStatus.COMPLETED
        result.signals_generated = 5
        result.signals_approved = 3
        result.orders_submitted = 3
        result.orders_filled = 2
        result.orders_rejected = 1
        result.total_duration_ms = 100.0
        result.signal_latency_ms = 50.0
        result.risk_latency_ms = 20.0
        result.execution_latency_ms = 30.0

        metrics.update_from_cycle(result)

        assert metrics.total_cycles == 1
        assert metrics.successful_cycles == 1
        assert metrics.failed_cycles == 0
        assert metrics.total_signals == 5
        assert metrics.approved_signals == 3
        assert metrics.rejected_signals == 2
        assert metrics.total_orders == 3
        assert metrics.filled_orders == 2
        assert metrics.rejected_orders == 1

    def test_update_from_failed_cycle(self) -> None:
        """Test updating metrics from a failed cycle."""
        metrics = PipelineMetrics()
        result = CycleResult()
        result.status = CycleStatus.FAILED
        result.signals_generated = 0

        metrics.update_from_cycle(result)

        assert metrics.total_cycles == 1
        assert metrics.successful_cycles == 0
        assert metrics.failed_cycles == 1

    def test_update_latency_averages(self) -> None:
        """Test latency averages are calculated correctly."""
        metrics = PipelineMetrics()
        result = CycleResult()
        result.status = CycleStatus.COMPLETED
        result.total_duration_ms = 100.0
        result.signal_latency_ms = 50.0
        result.risk_latency_ms = 20.0
        result.execution_latency_ms = 30.0

        metrics.update_from_cycle(result)

        # First cycle: exponential average = alpha * value + (1-alpha) * 0
        # With alpha = 0.1
        assert metrics.avg_cycle_duration_ms == pytest.approx(10.0)  # 0.1 * 100
        assert metrics.avg_signal_latency_ms == pytest.approx(5.0)  # 0.1 * 50
        assert metrics.avg_risk_latency_ms == pytest.approx(2.0)  # 0.1 * 20
        assert metrics.avg_execution_latency_ms == pytest.approx(3.0)  # 0.1 * 30

    def test_update_multiple_cycles(self) -> None:
        """Test updating metrics from multiple cycles."""
        metrics = PipelineMetrics()

        # Cycle 1
        result1 = CycleResult()
        result1.status = CycleStatus.COMPLETED
        result1.signals_generated = 5
        result1.signals_approved = 3
        metrics.update_from_cycle(result1)

        # Cycle 2
        result2 = CycleResult()
        result2.status = CycleStatus.COMPLETED
        result2.signals_generated = 3
        result2.signals_approved = 2
        metrics.update_from_cycle(result2)

        assert metrics.total_cycles == 2
        assert metrics.successful_cycles == 2
        assert metrics.total_signals == 8
        assert metrics.approved_signals == 5
        assert metrics.rejected_signals == 3

    def test_to_dict(self) -> None:
        """Test converting metrics to dictionary."""
        metrics = PipelineMetrics()
        metrics.total_cycles = 10
        metrics.successful_cycles = 8
        metrics.failed_cycles = 2
        metrics.total_signals = 50
        metrics.approved_signals = 35
        metrics.rejected_signals = 15
        metrics.total_orders = 30
        metrics.filled_orders = 25
        metrics.rejected_orders = 5
        metrics.avg_cycle_duration_ms = 100.0
        metrics.avg_signal_latency_ms = 50.0
        metrics.avg_risk_latency_ms = 20.0
        metrics.avg_execution_latency_ms = 30.0

        metrics_dict = metrics.to_dict()

        assert isinstance(metrics_dict, dict)
        assert metrics_dict["cycles"]["total"] == 10
        assert metrics_dict["cycles"]["successful"] == 8
        assert metrics_dict["cycles"]["failed"] == 2
        assert metrics_dict["signals"]["total"] == 50
        assert metrics_dict["signals"]["approved"] == 35
        assert metrics_dict["signals"]["rejected"] == 15
        assert metrics_dict["orders"]["total"] == 30
        assert metrics_dict["orders"]["filled"] == 25
        assert metrics_dict["orders"]["rejected"] == 5
        assert metrics_dict["latency_avg_ms"]["cycle"] == 100.0
        assert metrics_dict["latency_avg_ms"]["signal"] == 50.0
        assert metrics_dict["latency_avg_ms"]["risk"] == 20.0
        assert metrics_dict["latency_avg_ms"]["execution"] == 30.0
