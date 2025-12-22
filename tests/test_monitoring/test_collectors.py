"""Tests for MetricsCollector."""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from unittest.mock import MagicMock, patch

import pytest


class TestCollectorState:
    """Tests for CollectorState dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        from ordinis.monitoring.collectors import CollectorState

        state = CollectorState()
        assert state.last_cycle_count == 0
        assert state.last_signal_count == 0
        assert state.last_order_count == 0
        assert state.last_fill_count == 0
        assert state.last_collection_time > 0

    def test_custom_values(self):
        """Test custom values."""
        from ordinis.monitoring.collectors import CollectorState

        state = CollectorState(
            last_cycle_count=10,
            last_signal_count=5,
            last_order_count=3,
        )
        assert state.last_cycle_count == 10
        assert state.last_signal_count == 5
        assert state.last_order_count == 3


class TestMetricsCollectorInit:
    """Tests for MetricsCollector initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()
        assert collector._collection_interval == 1.0
        assert collector._orchestration_engine is None
        assert collector._flowroute_engine is None
        assert collector._running is False
        assert collector._thread is None

    def test_custom_interval(self):
        """Test custom collection interval."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector(collection_interval=5.0)
        assert collector._collection_interval == 5.0


class TestMetricsCollectorRegistration:
    """Tests for engine registration."""

    def test_register_orchestration_engine(self):
        """Test registering orchestration engine."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()
        mock_engine = MagicMock()

        collector.register_orchestration_engine(mock_engine)
        assert collector._orchestration_engine is mock_engine

    def test_register_flowroute_engine(self):
        """Test registering flowroute engine."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()
        mock_engine = MagicMock()

        collector.register_flowroute_engine(mock_engine)
        assert collector._flowroute_engine is mock_engine


class TestMetricsCollectorStartStop:
    """Tests for start/stop functionality."""

    def test_start_creates_thread(self):
        """Test start creates a daemon thread."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector(collection_interval=0.1)
        try:
            collector.start()
            assert collector._running is True
            assert collector._thread is not None
            assert collector._thread.daemon is True
            assert collector._thread.name == "metrics-collector"
        finally:
            collector.stop()

    def test_start_idempotent(self):
        """Test calling start twice doesn't create two threads."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector(collection_interval=0.1)
        try:
            collector.start()
            thread1 = collector._thread
            collector.start()
            assert collector._thread is thread1
        finally:
            collector.stop()

    def test_stop(self):
        """Test stop terminates thread."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector(collection_interval=0.1)
        collector.start()
        time.sleep(0.05)
        collector.stop()

        assert collector._running is False
        assert collector._thread is None

    def test_stop_without_start(self):
        """Test stop when not started."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()
        collector.stop()  # Should not raise
        assert collector._running is False


class TestMetricsCollectorCollection:
    """Tests for metrics collection."""

    def test_collect_metrics_with_no_engines(self):
        """Test collection when no engines registered."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()
        collector._collect_metrics()  # Should not raise

    def test_collect_orchestration_metrics(self):
        """Test collecting orchestration engine metrics."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()

        # Mock engine with metrics
        mock_metrics = MagicMock()
        mock_metrics.successful_cycles = 10
        mock_metrics.failed_cycles = 2
        mock_metrics.approved_signals = 8
        mock_metrics.rejected_signals = 5
        mock_metrics.total_orders = 6
        mock_metrics.filled_orders = 4
        mock_metrics.rejected_orders = 1

        mock_engine = MagicMock()
        mock_engine.get_metrics.return_value = mock_metrics
        mock_engine.get_last_cycle.return_value = None

        collector.register_orchestration_engine(mock_engine)
        collector._collect_orchestration_metrics()

        # Verify engine methods were called
        mock_engine.get_metrics.assert_called_once()

    def test_collect_orchestration_metrics_with_cycle(self):
        """Test collecting metrics with cycle data."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()

        # Mock cycle result
        mock_cycle = MagicMock()
        mock_cycle.total_duration_ms = 150.0
        mock_cycle.data_latency_ms = 30.0
        mock_cycle.signal_latency_ms = 50.0
        mock_cycle.risk_latency_ms = 20.0
        mock_cycle.execution_latency_ms = 40.0
        mock_cycle.analytics_latency_ms = 10.0

        # Mock engine
        mock_metrics = MagicMock()
        mock_metrics.successful_cycles = 10
        mock_metrics.failed_cycles = 0
        mock_metrics.approved_signals = 5
        mock_metrics.rejected_signals = 2
        mock_metrics.total_orders = 4
        mock_metrics.filled_orders = 3
        mock_metrics.rejected_orders = 0

        mock_engine = MagicMock()
        mock_engine.get_metrics.return_value = mock_metrics
        mock_engine.get_last_cycle.return_value = mock_cycle

        collector.register_orchestration_engine(mock_engine)
        collector._collect_orchestration_metrics()

        mock_engine.get_last_cycle.assert_called_once()

    def test_collect_orchestration_metrics_none_metrics(self):
        """Test collecting when get_metrics returns None."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()
        mock_engine = MagicMock()
        mock_engine.get_metrics.return_value = None

        collector.register_orchestration_engine(mock_engine)
        collector._collect_orchestration_metrics()  # Should not raise

    def test_collect_orchestration_metrics_error_handling(self):
        """Test error handling in metrics collection."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()
        mock_engine = MagicMock()
        mock_engine.get_metrics.side_effect = Exception("Test error")

        collector.register_orchestration_engine(mock_engine)
        collector._collect_orchestration_metrics()  # Should not raise

    def test_collect_flowroute_metrics(self):
        """Test collecting flowroute engine metrics."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()

        # Mock circuit breaker
        mock_cb = MagicMock()
        mock_cb.state.value = "closed"
        mock_cb.metrics.current_error_rate = 0.05

        # Mock account
        mock_account = MagicMock()
        mock_account.equity = 100000.0
        mock_account.cash = 50000.0
        mock_account.buying_power = 200000.0

        # Mock engine
        mock_engine = MagicMock()
        mock_engine._circuit_breaker = mock_cb
        mock_engine._account_state = mock_account
        mock_engine._positions = {"AAPL": {}, "MSFT": {}}

        collector.register_flowroute_engine(mock_engine)
        collector._collect_flowroute_metrics()  # Should not raise

    def test_collect_flowroute_metrics_no_circuit_breaker(self):
        """Test collecting when no circuit breaker."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()
        mock_engine = MagicMock()
        mock_engine._circuit_breaker = None
        mock_engine._account_state = None
        mock_engine._positions = {}

        collector.register_flowroute_engine(mock_engine)
        collector._collect_flowroute_metrics()  # Should not raise


class TestMetricsCollectorRecordMethods:
    """Tests for record helper methods."""

    def test_record_paper_fill(self):
        """Test recording paper trade fill."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()
        collector.record_paper_fill(slippage_bps=5.0, commission=1.50)
        # Should not raise - metrics are updated internally

    def test_update_portfolio(self):
        """Test updating portfolio metrics."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()
        collector.update_portfolio(
            equity=100000.0,
            cash=50000.0,
            buying_power=200000.0,
            position_count=5,
        )
        # Should not raise

    def test_set_circuit_breaker_state(self):
        """Test setting circuit breaker state."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()

        collector.set_circuit_breaker_state("flowroute", "closed")
        collector.set_circuit_breaker_state("flowroute", "open")
        collector.set_circuit_breaker_state("flowroute", "half_open")
        collector.set_circuit_breaker_state("flowroute", "unknown")

    def test_set_error_rate(self):
        """Test setting error rate."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()
        collector.set_error_rate("execution", 0.05)
        collector.set_error_rate("data", 0.01)


class TestMetricsCollectorRecordCycle:
    """Tests for record_cycle method."""

    def test_record_cycle_completed(self):
        """Test recording a completed cycle."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()

        # Mock cycle result
        mock_result = MagicMock()
        mock_result.status.value = "completed"
        mock_result.total_duration_ms = 100.0
        mock_result.signals_approved = 3
        mock_result.signals_generated = 5
        mock_result.orders_submitted = 2
        mock_result.orders_filled = 1
        mock_result.orders_rejected = 0
        mock_result.stages = []

        collector.record_cycle(mock_result)  # Should not raise

    def test_record_cycle_failed(self):
        """Test recording a failed cycle."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()

        mock_result = MagicMock()
        mock_result.status.value = "failed"
        mock_result.total_duration_ms = 50.0
        mock_result.signals_approved = 0
        mock_result.signals_generated = 2
        mock_result.orders_submitted = 0
        mock_result.orders_filled = 0
        mock_result.orders_rejected = 1
        mock_result.stages = []

        collector.record_cycle(mock_result)

    def test_record_cycle_with_stages(self):
        """Test recording cycle with stage durations."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()

        # Mock stage
        mock_stage = MagicMock()
        mock_stage.stage.value = "data_fetch"
        mock_stage.duration_ms = 30.0

        mock_result = MagicMock()
        mock_result.status.value = "completed"
        mock_result.total_duration_ms = 100.0
        mock_result.signals_approved = 1
        mock_result.signals_generated = 1
        mock_result.orders_submitted = 1
        mock_result.orders_filled = 1
        mock_result.orders_rejected = 0
        mock_result.stages = [mock_stage]

        collector.record_cycle(mock_result)

    def test_record_cycle_error_handling(self):
        """Test error handling in record_cycle."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector()
        mock_result = MagicMock()
        mock_result.status = None  # Will cause AttributeError

        collector.record_cycle(mock_result)  # Should not raise


class TestCollectionLoop:
    """Tests for the collection loop."""

    def test_collection_loop_runs(self):
        """Test that collection loop runs and collects metrics."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector(collection_interval=0.05)

        mock_engine = MagicMock()
        mock_metrics = MagicMock()
        mock_metrics.successful_cycles = 1
        mock_metrics.failed_cycles = 0
        mock_metrics.approved_signals = 1
        mock_metrics.rejected_signals = 0
        mock_metrics.total_orders = 1
        mock_metrics.filled_orders = 1
        mock_metrics.rejected_orders = 0
        mock_engine.get_metrics.return_value = mock_metrics
        mock_engine.get_last_cycle.return_value = None

        collector.register_orchestration_engine(mock_engine)
        collector.start()

        time.sleep(0.15)  # Allow 2-3 collections
        collector.stop()

        # Verify get_metrics was called multiple times
        assert mock_engine.get_metrics.call_count >= 2

    def test_collection_loop_handles_errors(self):
        """Test that collection loop handles errors gracefully."""
        from ordinis.monitoring.collectors import MetricsCollector

        collector = MetricsCollector(collection_interval=0.05)

        mock_engine = MagicMock()
        mock_engine.get_metrics.side_effect = Exception("Test error")

        collector.register_orchestration_engine(mock_engine)
        collector.start()

        time.sleep(0.15)
        collector.stop()

        # Loop should continue despite errors
        assert mock_engine.get_metrics.call_count >= 2
