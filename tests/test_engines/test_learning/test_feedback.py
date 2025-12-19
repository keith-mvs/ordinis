"""
Tests for FeedbackCollector and CircuitBreakerMonitor.

Tests the closed-loop feedback system that was missing on 12/17,
which would have prevented 6,120+ execution failures.
"""

from __future__ import annotations

import pytest

from ordinis.engines.learning.collectors.feedback import (
    CircuitBreakerMonitor,
    CircuitBreakerState,
    ErrorWindow,
    FeedbackCollector,
    FeedbackPriority,
    FeedbackRecord,
    FeedbackType,
)


class TestErrorWindow:
    """Tests for sliding window error tracking."""

    def test_record_error_increments_count(self) -> None:
        """Error recording should increment count."""
        window = ErrorWindow(window_seconds=60, max_errors=5)
        count = window.record_error()
        assert count == 1

    def test_multiple_errors_accumulate(self) -> None:
        """Multiple errors should accumulate in window."""
        window = ErrorWindow(window_seconds=60, max_errors=5)
        for i in range(5):
            count = window.record_error()
            assert count == i + 1

    def test_threshold_exceeded(self) -> None:
        """Threshold should be detected when errors exceed max."""
        window = ErrorWindow(window_seconds=60, max_errors=3)
        window.record_error()
        window.record_error()
        assert not window.is_threshold_exceeded()
        window.record_error()
        assert window.is_threshold_exceeded()

    def test_rate_per_minute_calculation(self) -> None:
        """Rate per minute should be calculated correctly."""
        window = ErrorWindow(window_seconds=60, max_errors=10)
        for _ in range(6):
            window.record_error()
        rate = window.get_rate_per_minute()
        assert rate == 6.0


class TestCircuitBreakerMonitor:
    """Tests for real-time circuit breaker behavior."""

    def test_initial_state_closed(self) -> None:
        """All engines should start with closed circuit breaker."""
        monitor = CircuitBreakerMonitor()
        for engine in ["signal_engine", "risk_engine", "execution_engine", "portfolio_engine"]:
            assert monitor.engine_states[engine] == CircuitBreakerState.CLOSED

    def test_record_error_returns_trip_status(self) -> None:
        """record_error should return (tripped, reason) tuple."""
        monitor = CircuitBreakerMonitor()
        tripped, reason = monitor.record_error("execution_failure", "execution_engine")
        assert isinstance(tripped, bool)
        assert isinstance(reason, str)

    def test_insufficient_capital_trips_at_threshold_3(self) -> None:
        """Insufficient capital should trip after 3 errors (12/17 prevention)."""
        monitor = CircuitBreakerMonitor()

        # First two errors don't trip
        tripped1, _ = monitor.record_error("insufficient_capital", "risk_engine")
        tripped2, _ = monitor.record_error("insufficient_capital", "risk_engine")
        assert not tripped1
        assert not tripped2
        assert monitor.engine_states["risk_engine"] == CircuitBreakerState.CLOSED

        # Third error trips
        tripped3, reason = monitor.record_error("insufficient_capital", "risk_engine")
        assert tripped3
        assert "insufficient_capital" in reason
        assert monitor.engine_states["risk_engine"] == CircuitBreakerState.OPEN

    def test_position_mismatch_trips_immediately(self) -> None:
        """Position mismatch should trip after just 1 error (critical)."""
        monitor = CircuitBreakerMonitor()
        tripped, reason = monitor.record_error("position_mismatch", "portfolio_engine")
        assert tripped
        assert "position_mismatch" in reason
        assert monitor.engine_states["portfolio_engine"] == CircuitBreakerState.OPEN

    def test_execution_failure_trips_at_threshold_10(self) -> None:
        """Execution failures should trip after 10 errors."""
        monitor = CircuitBreakerMonitor()

        for i in range(9):
            tripped, _ = monitor.record_error("execution_failure", "execution_engine")
            assert not tripped

        tripped10, reason = monitor.record_error("execution_failure", "execution_engine")
        assert tripped10
        assert "execution_failure" in reason

    def test_should_allow_signal_when_closed(self) -> None:
        """Signals should be allowed when circuit breaker is closed."""
        monitor = CircuitBreakerMonitor()
        allowed, reason = monitor.should_allow_signal("signal_engine")
        assert allowed
        assert reason == ""

    def test_should_block_signal_when_open(self) -> None:
        """Signals should be blocked when circuit breaker is open."""
        monitor = CircuitBreakerMonitor()
        # Trip the breaker
        monitor.record_error("position_mismatch", "signal_engine")

        allowed, reason = monitor.should_allow_signal("signal_engine")
        assert not allowed
        assert "circuit_breaker_open" in reason

    def test_get_status_returns_all_engines(self) -> None:
        """get_status should return status for all engines."""
        monitor = CircuitBreakerMonitor()
        status = monitor.get_status()

        assert "signal_engine" in status
        assert "risk_engine" in status
        assert "execution_engine" in status
        assert "portfolio_engine" in status

        for engine_status in status.values():
            assert "state" in engine_status
            assert "error_windows" in engine_status


class TestFeedbackCollector:
    """Tests for the FeedbackCollector integration hub."""

    @pytest.fixture
    def collector(self) -> FeedbackCollector:
        """Create a fresh FeedbackCollector for each test."""
        return FeedbackCollector()

    def test_initialization(self, collector: FeedbackCollector) -> None:
        """FeedbackCollector should initialize with circuit breaker."""
        assert collector.circuit_breaker is not None
        assert isinstance(collector.circuit_breaker, CircuitBreakerMonitor)

    @pytest.mark.asyncio
    async def test_record_insufficient_capital_triggers_breaker(
        self, collector: FeedbackCollector
    ) -> None:
        """Recording insufficient capital should trigger circuit breaker after threshold."""
        # Record 3 errors
        for _ in range(3):
            await collector.record_insufficient_capital(
                required_capital=10000.0,
                available_capital=0.0,
                buying_power=0.0,
                signal_count_blocked=100,
            )

        # Should now block signals
        allowed, reason = collector.should_allow_signals()
        assert not allowed
        assert "risk_engine" in reason

    @pytest.mark.asyncio
    async def test_record_position_mismatch_triggers_breaker(
        self, collector: FeedbackCollector
    ) -> None:
        """Position mismatch should immediately trigger circuit breaker."""
        await collector.record_position_mismatch(
            symbol="AAPL",
            internal_quantity=0,
            broker_quantity=21,
            internal_cost=0.0,
            broker_cost=3150.0,
        )

        allowed, reason = collector.should_allow_execution()
        assert not allowed
        assert "portfolio_engine" in reason

    @pytest.mark.asyncio
    async def test_record_execution_failure_returns_record_and_trip_status(
        self, collector: FeedbackCollector
    ) -> None:
        """Execution failure should return both record and trip status."""
        record, tripped = await collector.record_execution_failure(
            order_id="ORD-001",
            symbol="AAPL",
            error_type="insufficient_buying_power",
            error_message="Insufficient buying power",
            order_details={"quantity": 100, "side": "buy"},
        )

        assert isinstance(record, FeedbackRecord)
        assert isinstance(tripped, bool)
        assert record.feedback_type == FeedbackType.EXECUTION_FAILURE

    @pytest.mark.asyncio
    async def test_should_allow_signals_checks_all_engines(
        self, collector: FeedbackCollector
    ) -> None:
        """should_allow_signals should check ALL engine states."""
        # Initially allowed
        allowed1, _ = collector.should_allow_signals()
        assert allowed1

        # Trip just the risk_engine
        for _ in range(3):
            await collector.record_insufficient_capital(
                required_capital=1000.0,
                available_capital=0.0,
                buying_power=0.0,
                signal_count_blocked=10,
            )

        # Now blocked because risk_engine tripped
        allowed2, reason = collector.should_allow_signals()
        assert not allowed2
        assert "risk_engine" in reason

    @pytest.mark.asyncio
    async def test_should_allow_execution_checks_all_engines(
        self, collector: FeedbackCollector
    ) -> None:
        """should_allow_execution should check ALL engine states."""
        # Trip portfolio_engine via position mismatch
        await collector.record_position_mismatch(
            symbol="TSLA",
            internal_quantity=10,
            broker_quantity=0,
            internal_cost=1500.0,
            broker_cost=0.0,
        )

        # Execution should be blocked
        allowed, reason = collector.should_allow_execution()
        assert not allowed
        assert "portfolio_engine" in reason

    def test_reset_circuit_breaker(self, collector: FeedbackCollector) -> None:
        """Manual reset should re-enable trading."""
        # Trip breaker
        collector.circuit_breaker.record_error("position_mismatch", "portfolio_engine")
        allowed1, _ = collector.should_allow_execution()
        assert not allowed1

        # Reset
        collector.reset_circuit_breaker("portfolio_engine")

        # Now allowed
        allowed2, _ = collector.should_allow_execution()
        assert allowed2

    def test_get_circuit_breaker_status(self, collector: FeedbackCollector) -> None:
        """get_circuit_breaker_status should return comprehensive status."""
        status = collector.get_circuit_breaker_status()

        assert isinstance(status, dict)
        assert len(status) >= 4  # At least 4 engines


class TestFeedbackRecord:
    """Tests for FeedbackRecord data structure."""

    def test_record_creation(self) -> None:
        """FeedbackRecord should be created with required fields."""
        record = FeedbackRecord(
            feedback_type=FeedbackType.EXECUTION_FAILURE,
            priority=FeedbackPriority.HIGH,
            source="execution_engine",
            summary="Test failure",
        )

        assert record.feedback_type == FeedbackType.EXECUTION_FAILURE
        assert record.priority == FeedbackPriority.HIGH
        assert record.source == "execution_engine"
        assert record.feedback_id.startswith("FB-")

    def test_record_timestamp_is_utc(self) -> None:
        """FeedbackRecord timestamp should be in UTC."""
        record = FeedbackRecord(
            feedback_type=FeedbackType.RISK_EVENT,
            priority=FeedbackPriority.CRITICAL,
            source="risk_engine",
            summary="Risk breach",
        )

        assert record.timestamp.tzinfo is not None


class TestScenarioDisasterPrevention:
    """
    Integration tests simulating the 12/17 trading day disaster.

    On 12/17, the system generated 6,123 signals with 0 successful executions
    because buying power was exhausted. The circuit breaker should have
    prevented this after just 3 failures.
    """

    @pytest.mark.asyncio
    async def test_12_17_would_have_been_prevented(self) -> None:
        """Simulate 12/17 scenario - should stop after 3 failures."""
        collector = FeedbackCollector()
        failures_allowed = 0

        # Simulate rapid-fire execution failures
        for i in range(100):  # Simulate 100 attempts
            allowed, reason = collector.should_allow_execution()
            if not allowed:
                # Circuit breaker tripped - this is what we wanted!
                break

            # Record the failure (simulating broker rejection)
            await collector.record_insufficient_capital(
                required_capital=5000.0,
                available_capital=0.0,
                buying_power=0.0,
                signal_count_blocked=1,
            )
            failures_allowed += 1

        # Should have stopped after exactly 3 failures (threshold)
        assert failures_allowed == 3, (
            f"Circuit breaker should have tripped after 3 failures, "
            f"but allowed {failures_allowed}"
        )

    @pytest.mark.asyncio
    async def test_multiple_error_types_compound(self) -> None:
        """Multiple error types should independently trip their engines."""
        collector = FeedbackCollector()

        # Trip risk_engine with capital errors
        for _ in range(3):
            await collector.record_insufficient_capital(
                required_capital=1000.0,
                available_capital=0.0,
                buying_power=0.0,
                signal_count_blocked=10,
            )

        # Trip portfolio_engine with mismatch
        await collector.record_position_mismatch(
            symbol="NVDA",
            internal_quantity=0,
            broker_quantity=50,
            internal_cost=0.0,
            broker_cost=5000.0,
        )

        status = collector.get_circuit_breaker_status()

        # Both should be open
        assert status["risk_engine"]["state"] == "open"
        assert status["portfolio_engine"]["state"] == "open"
        # But others should be closed
        assert status["signal_engine"]["state"] == "closed"
