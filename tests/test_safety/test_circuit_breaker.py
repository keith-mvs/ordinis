"""
Tests for circuit breaker module.

Tests cover:
- CircuitState enum values
- CircuitStats dataclass
- CircuitBreaker initialization
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Success/failure recording
- Force open/close
- Status reporting
"""

import asyncio

import pytest

from ordinis.safety.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    CircuitStats,
)


class TestCircuitState:
    """Test CircuitState enum."""

    @pytest.mark.unit
    def test_circuit_state_values(self):
        """Test CircuitState enum values."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitStats:
    """Test CircuitStats dataclass."""

    @pytest.mark.unit
    def test_circuit_stats_defaults(self):
        """Test CircuitStats default values."""
        stats = CircuitStats()
        assert stats.total_calls == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 0
        assert stats.last_failure_time is None
        assert stats.last_success_time is None
        assert stats.state_changes == []

    @pytest.mark.unit
    def test_circuit_stats_custom_values(self):
        """Test CircuitStats with custom values."""
        stats = CircuitStats(
            total_calls=10,
            successful_calls=8,
            failed_calls=2,
            consecutive_failures=1,
        )
        assert stats.total_calls == 10
        assert stats.successful_calls == 8
        assert stats.failed_calls == 2
        assert stats.consecutive_failures == 1


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    @pytest.mark.unit
    def test_circuit_breaker_initialization(self):
        """Test CircuitBreaker initialization with defaults."""
        cb = CircuitBreaker(name="test")
        assert cb.name == "test"
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed is True
        assert cb.is_open is False

    @pytest.mark.unit
    def test_circuit_breaker_custom_thresholds(self):
        """Test CircuitBreaker with custom thresholds."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=3,
            success_threshold=2,
            recovery_timeout_seconds=10.0,
            half_open_max_calls=1,
        )
        assert cb._failure_threshold == 3
        assert cb._success_threshold == 2
        assert cb._recovery_timeout.total_seconds() == 10.0
        assert cb._half_open_max_calls == 1

    @pytest.mark.unit
    def test_circuit_breaker_stats_property(self):
        """Test CircuitBreaker stats property."""
        cb = CircuitBreaker(name="test")
        stats = cb.stats
        assert isinstance(stats, CircuitStats)
        assert stats.total_calls == 0


class TestCircuitBreakerCalls:
    """Test CircuitBreaker call functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Test successful function call through circuit breaker."""
        cb = CircuitBreaker(name="test")

        async def success_func():
            return "success"

        result = await cb.call(success_func)
        assert result == "success"
        assert cb.stats.total_calls == 1
        assert cb.stats.successful_calls == 1
        assert cb.stats.consecutive_successes == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sync_function_call(self):
        """Test sync function call through circuit breaker."""
        cb = CircuitBreaker(name="test")

        def sync_func():
            return "sync_result"

        result = await cb.call(sync_func)
        assert result == "sync_result"
        assert cb.stats.successful_calls == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_failed_call(self):
        """Test failed function call through circuit breaker."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        async def fail_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await cb.call(fail_func)

        assert cb.stats.total_calls == 1
        assert cb.stats.failed_calls == 1
        assert cb.stats.consecutive_failures == 1
        assert cb.state == CircuitState.CLOSED  # Not enough failures yet

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker(name="test", failure_threshold=3)

        async def fail_func():
            raise ValueError("fail")

        # Trigger failures until circuit opens
        for _ in range(3):
            with pytest.raises(ValueError, match="fail"):
                await cb.call(fail_func)

        assert cb.state == CircuitState.OPEN
        assert cb.is_open is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_circuit_blocks_when_open(self):
        """Test circuit blocks calls when open."""
        cb = CircuitBreaker(name="test", failure_threshold=1)

        async def fail_func():
            raise ValueError("fail")

        # Open the circuit
        with pytest.raises(ValueError, match="fail"):
            await cb.call(fail_func)

        assert cb.state == CircuitState.OPEN

        # Now it should block
        with pytest.raises(CircuitOpenError) as exc_info:
            await cb.call(fail_func)

        assert "is open" in str(exc_info.value)


class TestCircuitBreakerStateTransitions:
    """Test CircuitBreaker state transitions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_force_open(self):
        """Test force opening circuit."""
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED

        await cb.force_open("Test reason")
        assert cb.state == CircuitState.OPEN

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_force_close(self):
        """Test force closing circuit."""
        cb = CircuitBreaker(name="test")
        await cb.force_open()
        assert cb.state == CircuitState.OPEN

        await cb.force_close()
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_state_change_callback(self):
        """Test state change callback is invoked."""
        state_changes = []

        def on_change(state):
            state_changes.append(state)

        cb = CircuitBreaker(name="test", on_state_change=on_change)
        await cb.force_open()
        await cb.force_close()

        assert CircuitState.OPEN in state_changes
        assert CircuitState.CLOSED in state_changes

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_half_open_transition(self):
        """Test transition to half-open after timeout."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_timeout_seconds=0.01,  # Very short for testing
        )

        async def fail_func():
            raise ValueError("fail")

        # Open the circuit
        with pytest.raises(ValueError, match="fail"):
            await cb.call(fail_func)

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # Next call should transition to half-open (then fail)
        with pytest.raises(ValueError, match="fail"):
            await cb.call(fail_func)

        # After failure in half-open, should go back to open
        assert cb.state == CircuitState.OPEN

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_half_open_to_closed(self):
        """Test transition from half-open to closed on success."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            success_threshold=1,
            recovery_timeout_seconds=0.01,
        )

        async def fail_func():
            raise ValueError("fail")

        async def success_func():
            return "ok"

        # Open the circuit
        with pytest.raises(ValueError, match="fail"):
            await cb.call(fail_func)

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # Successful call should close circuit
        result = await cb.call(success_func)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerStatus:
    """Test CircuitBreaker status reporting."""

    @pytest.mark.unit
    def test_get_status(self):
        """Test get_status returns correct information."""
        cb = CircuitBreaker(name="test_breaker")
        status = cb.get_status()

        assert status["name"] == "test_breaker"
        assert status["state"] == "closed"
        assert status["total_calls"] == 0
        assert "failure_threshold" in status

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_status_after_calls(self):
        """Test status reflects call history."""
        cb = CircuitBreaker(name="test")

        async def success():
            return True

        await cb.call(success)
        await cb.call(success)

        status = cb.get_status()
        assert status["total_calls"] == 2
        assert status["successful_calls"] == 2


class TestCircuitBreakerHalfOpenMaxCalls:
    """Test half-open max calls limit."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_half_open_max_calls_exceeded(self):
        """Test half-open state limits concurrent calls."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            half_open_max_calls=1,
            recovery_timeout_seconds=0.01,
        )

        async def fail_func():
            raise ValueError("fail")

        # Open the circuit
        with pytest.raises(ValueError, match="fail"):
            await cb.call(fail_func)

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # First call transitions to half-open
        with pytest.raises(ValueError, match="fail"):
            await cb.call(fail_func)

        # Circuit should be open again after failure in half-open
        assert cb.state == CircuitState.OPEN
