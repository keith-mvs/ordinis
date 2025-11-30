"""Tests for monitoring.metrics module."""

from datetime import datetime

import pytest

from monitoring.metrics import (
    MetricsCollector,
    PerformanceMetrics,
    get_metrics_collector,
    reset_metrics,
)


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_performance_metrics_initialization(self):
        """Test PerformanceMetrics initialization with defaults."""
        metrics = PerformanceMetrics()

        assert metrics.total_operations == 0
        assert metrics.successful_operations == 0
        assert metrics.failed_operations == 0
        assert metrics.avg_execution_time == 0.0
        assert metrics.min_execution_time == float("inf")
        assert metrics.max_execution_time == 0.0
        assert metrics.signals_generated == 0
        assert metrics.signals_executed == 0
        assert metrics.signals_rejected == 0
        assert metrics.api_calls == 0
        assert metrics.api_errors == 0
        assert metrics.api_avg_response_time == 0.0
        assert isinstance(metrics.custom_metrics, dict)
        assert isinstance(metrics.timestamp, datetime)

    def test_success_rate_calculation(self):
        """Test success rate property calculation."""
        metrics = PerformanceMetrics()
        assert metrics.success_rate == 0.0

        metrics.total_operations = 100
        metrics.successful_operations = 75
        assert metrics.success_rate == 0.75

        metrics.successful_operations = 100
        assert metrics.success_rate == 1.0

    def test_error_rate_calculation(self):
        """Test error rate property calculation."""
        metrics = PerformanceMetrics()
        assert metrics.error_rate == 0.0

        metrics.total_operations = 100
        metrics.failed_operations = 25
        assert metrics.error_rate == 0.25

        metrics.failed_operations = 0
        assert metrics.error_rate == 0.0

    def test_signal_execution_rate_calculation(self):
        """Test signal execution rate property calculation."""
        metrics = PerformanceMetrics()
        assert metrics.signal_execution_rate == 0.0

        metrics.signals_generated = 50
        metrics.signals_executed = 30
        assert metrics.signal_execution_rate == 0.6

        metrics.signals_executed = 50
        assert metrics.signal_execution_rate == 1.0

    def test_to_dict_conversion(self):
        """Test conversion of metrics to dictionary."""
        metrics = PerformanceMetrics(
            total_operations=100,
            successful_operations=75,
            failed_operations=25,
            signals_generated=50,
            signals_executed=40,
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["total_operations"] == 100
        assert result["successful_operations"] == 75
        assert result["failed_operations"] == 25
        assert result["success_rate"] == 0.75
        assert result["error_rate"] == 0.25
        assert result["signals_generated"] == 50
        assert result["signals_executed"] == 40
        assert result["signal_execution_rate"] == 0.8
        assert "timestamp" in result
        assert "custom_metrics" in result

    def test_to_dict_with_infinite_min_execution_time(self):
        """Test to_dict handles infinite min_execution_time."""
        metrics = PerformanceMetrics()
        result = metrics.to_dict()

        assert result["min_execution_time"] == 0


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()

        assert isinstance(collector.metrics, PerformanceMetrics)
        assert collector.metrics.total_operations == 0
        assert len(collector._execution_times) == 0
        assert len(collector._api_response_times) == 0
        assert len(collector._counters) == 0

    def test_record_successful_operation(self):
        """Test recording successful operations."""
        collector = MetricsCollector()

        collector.record_operation(success=True)

        assert collector.metrics.total_operations == 1
        assert collector.metrics.successful_operations == 1
        assert collector.metrics.failed_operations == 0

    def test_record_failed_operation(self):
        """Test recording failed operations."""
        collector = MetricsCollector()

        collector.record_operation(success=False)

        assert collector.metrics.total_operations == 1
        assert collector.metrics.successful_operations == 0
        assert collector.metrics.failed_operations == 1

    def test_record_operation_with_execution_time(self):
        """Test recording operation with execution time."""
        collector = MetricsCollector()

        collector.record_operation(success=True, execution_time=0.5)
        collector.record_operation(success=True, execution_time=1.5)
        collector.record_operation(success=True, execution_time=1.0)

        assert collector.metrics.total_operations == 3
        assert collector.metrics.min_execution_time == 0.5
        assert collector.metrics.max_execution_time == 1.5
        assert collector.metrics.avg_execution_time == 1.0

    def test_record_signal_generated(self):
        """Test recording signal generation."""
        collector = MetricsCollector()

        collector.record_signal(generated=True)

        assert collector.metrics.signals_generated == 1
        assert collector.metrics.signals_executed == 0
        assert collector.metrics.signals_rejected == 0

    def test_record_signal_executed(self):
        """Test recording signal execution."""
        collector = MetricsCollector()

        collector.record_signal(executed=True)

        assert collector.metrics.signals_generated == 0
        assert collector.metrics.signals_executed == 1
        assert collector.metrics.signals_rejected == 0

    def test_record_signal_rejected(self):
        """Test recording signal rejection."""
        collector = MetricsCollector()

        collector.record_signal(rejected=True)

        assert collector.metrics.signals_generated == 0
        assert collector.metrics.signals_executed == 0
        assert collector.metrics.signals_rejected == 1

    def test_record_signal_multiple_flags(self):
        """Test recording signal with multiple flags."""
        collector = MetricsCollector()

        collector.record_signal(generated=True, executed=True)

        assert collector.metrics.signals_generated == 1
        assert collector.metrics.signals_executed == 1
        assert collector.metrics.signals_rejected == 0

    def test_record_api_call_successful(self):
        """Test recording successful API call."""
        collector = MetricsCollector()

        collector.record_api_call(success=True)

        assert collector.metrics.api_calls == 1
        assert collector.metrics.api_errors == 0

    def test_record_api_call_failed(self):
        """Test recording failed API call."""
        collector = MetricsCollector()

        collector.record_api_call(success=False)

        assert collector.metrics.api_calls == 1
        assert collector.metrics.api_errors == 1

    def test_record_api_call_with_response_time(self):
        """Test recording API call with response time."""
        collector = MetricsCollector()

        collector.record_api_call(success=True, response_time=0.2)
        collector.record_api_call(success=True, response_time=0.4)
        collector.record_api_call(success=True, response_time=0.3)

        assert collector.metrics.api_calls == 3
        assert collector.metrics.api_avg_response_time == pytest.approx(0.3)

    def test_increment_counter(self):
        """Test incrementing custom counter."""
        collector = MetricsCollector()

        collector.increment_counter("trades")
        collector.increment_counter("trades")
        collector.increment_counter("trades", value=3)

        assert collector._counters["trades"] == 5
        assert collector.metrics.custom_metrics["trades"] == 5

    def test_set_gauge(self):
        """Test setting gauge metric."""
        collector = MetricsCollector()

        collector.set_gauge("portfolio_value", 100000.0)

        assert collector.metrics.custom_metrics["portfolio_value"] == 100000.0

        collector.set_gauge("portfolio_value", 105000.0)

        assert collector.metrics.custom_metrics["portfolio_value"] == 105000.0

    def test_get_metrics(self):
        """Test getting metrics snapshot."""
        collector = MetricsCollector()

        collector.record_operation(success=True)
        collector.record_signal(generated=True, executed=True)

        metrics = collector.get_metrics()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_operations == 1
        assert metrics.signals_generated == 1

    def test_get_summary(self):
        """Test getting metrics summary."""
        collector = MetricsCollector()

        collector.record_operation(success=True, execution_time=0.5)
        collector.record_signal(generated=True)

        summary = collector.get_summary()

        assert isinstance(summary, dict)
        assert summary["total_operations"] == 1
        assert summary["signals_generated"] == 1
        assert "success_rate" in summary
        assert "timestamp" in summary

    def test_reset(self):
        """Test resetting metrics."""
        collector = MetricsCollector()

        collector.record_operation(success=True, execution_time=0.5)
        collector.record_signal(generated=True)
        collector.increment_counter("trades")

        collector.reset()

        assert collector.metrics.total_operations == 0
        assert collector.metrics.signals_generated == 0
        assert len(collector._execution_times) == 0
        assert len(collector._counters) == 0


class TestGlobalCollector:
    """Tests for global metrics collector functions."""

    def test_get_metrics_collector_singleton(self):
        """Test that get_metrics_collector returns singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        assert collector1 is collector2

    def test_get_metrics_collector_returns_collector(self):
        """Test that get_metrics_collector returns MetricsCollector."""
        collector = get_metrics_collector()

        assert isinstance(collector, MetricsCollector)

    def test_reset_metrics_function(self):
        """Test global reset_metrics function."""
        collector = get_metrics_collector()
        collector.record_operation(success=True)

        assert collector.metrics.total_operations == 1

        reset_metrics()

        assert collector.metrics.total_operations == 0

    def test_global_collector_persistence(self):
        """Test that global collector persists data."""
        collector = get_metrics_collector()
        collector.record_operation(success=True)

        # Get collector again
        collector2 = get_metrics_collector()

        assert collector2.metrics.total_operations == 1


class TestMetricsCollectorIntegration:
    """Integration tests for MetricsCollector."""

    def test_complete_workflow(self):
        """Test complete metrics collection workflow."""
        reset_metrics()
        collector = get_metrics_collector()

        # Simulate backtest operations
        for i in range(100):
            success = i % 5 != 0  # 80% success rate
            exec_time = 0.1 + (i % 10) * 0.05
            collector.record_operation(success=success, execution_time=exec_time)

        # Simulate signals
        for i in range(50):
            collector.record_signal(generated=True)
            if i % 2 == 0:
                collector.record_signal(executed=True)
            else:
                collector.record_signal(rejected=True)

        # Simulate API calls
        for i in range(30):
            success = i % 10 != 0  # 90% success rate
            response_time = 0.2 + (i % 5) * 0.1
            collector.record_api_call(success=success, response_time=response_time)

        # Custom metrics
        collector.increment_counter("profitable_trades", 45)
        collector.set_gauge("current_drawdown", -0.05)

        summary = collector.get_summary()

        assert summary["total_operations"] == 100
        assert summary["success_rate"] == 0.8
        assert summary["signals_generated"] == 50
        assert summary["signals_executed"] == 25
        assert summary["signals_rejected"] == 25
        assert summary["signal_execution_rate"] == 0.5
        assert summary["api_calls"] == 30
        assert summary["api_errors"] == 3
        assert summary["custom_metrics"]["profitable_trades"] == 45
        assert summary["custom_metrics"]["current_drawdown"] == -0.05
