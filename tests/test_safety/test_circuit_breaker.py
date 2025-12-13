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
    BrokerCircuitBreaker,
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

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_half_open_blocks_after_max_calls(self):
        """Test half-open state blocks calls after max reached."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            half_open_max_calls=2,
            recovery_timeout_seconds=0.01,
        )

        call_count = 0

        async def slow_success():
            """Slow async function to test concurrent calls in half-open."""
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return "ok"

        async def fail_func():
            raise ValueError("fail")

        # Open the circuit
        with pytest.raises(ValueError, match="fail"):
            await cb.call(fail_func)

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.02)

        # Start two async calls (max allowed)
        task1 = asyncio.create_task(cb.call(slow_success))
        await asyncio.sleep(0.001)  # Small delay to ensure first call enters
        task2 = asyncio.create_task(cb.call(slow_success))

        # Third call should be blocked
        await asyncio.sleep(0.001)
        with pytest.raises(CircuitOpenError, match="max calls reached"):
            await cb.call(slow_success)

        # Wait for tasks to complete
        await task1
        await task2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_half_open_multiple_successes_close_circuit(self):
        """Test half-open closes after multiple successes."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            success_threshold=3,
            recovery_timeout_seconds=0.01,
            half_open_max_calls=5,
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

        # First success transitions to half-open
        result = await cb.call(success_func)
        assert result == "ok"
        assert cb.state == CircuitState.HALF_OPEN

        # Second success
        result = await cb.call(success_func)
        assert result == "ok"
        assert cb.state == CircuitState.HALF_OPEN

        # Third success should close circuit
        result = await cb.call(success_func)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED
        assert cb.stats.consecutive_successes == 3


class TestCircuitBreakerCallbackErrors:
    """Test error handling in callbacks."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_state_change_callback_exception(self):
        """Test that exceptions in state change callback are handled."""

        def failing_callback(state):
            raise RuntimeError("Callback error")

        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            on_state_change=failing_callback,
        )

        async def fail_func():
            raise ValueError("fail")

        # Should not raise even though callback fails
        with pytest.raises(ValueError, match="fail"):
            await cb.call(fail_func)

        # Circuit should still be open
        assert cb.state == CircuitState.OPEN

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_state_change_callback_none(self):
        """Test state changes work without callback."""
        cb = CircuitBreaker(name="test", failure_threshold=1, on_state_change=None)

        async def fail_func():
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            await cb.call(fail_func)

        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerAsyncEdgeCases:
    """Test async edge cases and concurrent behavior."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function_with_args_and_kwargs(self):
        """Test calling function with positional and keyword arguments."""
        cb = CircuitBreaker(name="test")

        async def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = await cb.call(func_with_args, "x", "y", c="z")
        assert result == "x-y-z"
        assert cb.stats.successful_calls == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sync_function_with_args(self):
        """Test sync function with arguments."""
        cb = CircuitBreaker(name="test")

        def sync_func(a, b):
            return a + b

        result = await cb.call(sync_func, 10, 20)
        assert result == 30
        assert cb.stats.successful_calls == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        """Test that original exception is propagated."""
        cb = CircuitBreaker(name="test", failure_threshold=5)

        class CustomError(Exception):
            pass

        async def custom_fail():
            raise CustomError("custom error message")

        with pytest.raises(CustomError, match="custom error message"):
            await cb.call(custom_fail)

        assert cb.stats.failed_calls == 1


class TestCircuitBreakerRecovery:
    """Test recovery timeout and state transitions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_recovery_timeout_not_elapsed(self):
        """Test circuit stays open if recovery timeout not elapsed."""
        cb = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_timeout_seconds=1.0,  # 1 second
        )

        async def fail_func():
            raise ValueError("fail")

        async def success_func():
            return "ok"

        # Open the circuit
        with pytest.raises(ValueError, match="fail"):
            await cb.call(fail_func)

        assert cb.state == CircuitState.OPEN

        # Try immediately (before timeout)
        with pytest.raises(CircuitOpenError):
            await cb.call(success_func)

        # Should still be open
        assert cb.state == CircuitState.OPEN

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_opened_at_timestamp_set(self):
        """Test that opened_at timestamp is set when circuit opens."""
        cb = CircuitBreaker(name="test", failure_threshold=1)

        async def fail_func():
            raise ValueError("fail")

        assert cb._opened_at is None

        # Open the circuit
        with pytest.raises(ValueError, match="fail"):
            await cb.call(fail_func)

        assert cb.state == CircuitState.OPEN
        assert cb._opened_at is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_opened_at_cleared_on_close(self):
        """Test that opened_at is cleared when circuit closes."""
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

        assert cb._opened_at is not None

        # Wait for recovery
        await asyncio.sleep(0.02)

        # Close the circuit
        await cb.call(success_func)

        assert cb.state == CircuitState.CLOSED
        assert cb._opened_at is None


class TestCircuitBreakerForceOperations:
    """Test force open/close operations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_force_open_with_reason(self):
        """Test force open with custom reason."""
        cb = CircuitBreaker(name="test")

        await cb.force_open(reason="Testing manual override")
        assert cb.state == CircuitState.OPEN

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_force_close_resets_counters(self):
        """Test force close resets failure/success counters."""
        cb = CircuitBreaker(name="test", failure_threshold=5)

        async def fail_func():
            raise ValueError("fail")

        # Accumulate some failures
        for _ in range(3):
            with pytest.raises(ValueError, match="fail"):
                await cb.call(fail_func)

        assert cb.stats.consecutive_failures == 3

        # Force close
        await cb.force_close()

        assert cb.state == CircuitState.CLOSED
        assert cb.stats.consecutive_failures == 0
        assert cb.stats.consecutive_successes == 0


class TestBrokerCircuitBreaker:
    """Test BrokerCircuitBreaker specialized class."""

    @pytest.mark.unit
    def test_broker_circuit_breaker_initialization(self):
        """Test BrokerCircuitBreaker initialization with defaults."""
        bcb = BrokerCircuitBreaker()
        assert bcb.name == "broker_api"
        assert bcb.state == CircuitState.CLOSED
        assert bcb._failure_threshold == 3
        assert bcb._success_threshold == 2
        assert bcb._recovery_timeout.total_seconds() == 60.0
        assert bcb._order_failures == 0
        assert bcb._quote_failures == 0
        assert bcb._account_failures == 0

    @pytest.mark.unit
    def test_broker_circuit_breaker_custom_params(self):
        """Test BrokerCircuitBreaker with custom parameters."""
        bcb = BrokerCircuitBreaker(
            name="custom_broker",
            failure_threshold=5,
            success_threshold=3,
            recovery_timeout_seconds=120.0,
        )
        assert bcb.name == "custom_broker"
        assert bcb._failure_threshold == 5
        assert bcb._success_threshold == 3
        assert bcb._recovery_timeout.total_seconds() == 120.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_record_order_failure(self):
        """Test recording order submission failures."""
        bcb = BrokerCircuitBreaker(failure_threshold=3)

        error = ValueError("Order submission failed")
        await bcb.record_order_failure(error)

        assert bcb._order_failures == 1
        assert bcb.stats.failed_calls == 1
        assert bcb.stats.consecutive_failures == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_record_quote_failure(self):
        """Test recording quote request failures."""
        bcb = BrokerCircuitBreaker(failure_threshold=3)

        error = ValueError("Quote request failed")
        await bcb.record_quote_failure(error)

        assert bcb._quote_failures == 1
        assert bcb.stats.failed_calls == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_record_account_failure(self):
        """Test recording account query failures."""
        bcb = BrokerCircuitBreaker(failure_threshold=3)

        error = ValueError("Account query failed")
        await bcb.record_account_failure(error)

        assert bcb._account_failures == 1
        assert bcb.stats.failed_calls == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_broker_circuit_opens_after_failures(self):
        """Test broker circuit opens after threshold."""
        bcb = BrokerCircuitBreaker(failure_threshold=3)

        error = ValueError("API error")

        # Record multiple failures
        await bcb.record_order_failure(error)
        await bcb.record_quote_failure(error)
        await bcb.record_account_failure(error)

        assert bcb.state == CircuitState.OPEN
        assert bcb._order_failures == 1
        assert bcb._quote_failures == 1
        assert bcb._account_failures == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_on_open_callback_triggered(self):
        """Test on_open callback is triggered when circuit opens."""
        callback_triggered = []

        def on_open_callback():
            callback_triggered.append(True)

        bcb = BrokerCircuitBreaker(
            failure_threshold=1,
            on_open=on_open_callback,
        )

        async def fail_func():
            raise ValueError("fail")

        # Trigger failure to open circuit
        with pytest.raises(ValueError, match="fail"):
            await bcb.call(fail_func)

        assert bcb.state == CircuitState.OPEN
        assert len(callback_triggered) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_on_open_callback_exception_handled(self):
        """Test exceptions in on_open callback are handled."""

        def failing_on_open():
            raise RuntimeError("Callback error")

        bcb = BrokerCircuitBreaker(
            failure_threshold=1,
            on_open=failing_on_open,
        )

        async def fail_func():
            raise ValueError("fail")

        # Should not raise even though callback fails
        with pytest.raises(ValueError, match="fail"):
            await bcb.call(fail_func)

        # Circuit should still be open
        assert bcb.state == CircuitState.OPEN

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_on_open_callback_none(self):
        """Test on_open callback works when None."""
        bcb = BrokerCircuitBreaker(
            failure_threshold=1,
            on_open=None,
        )

        async def fail_func():
            raise ValueError("fail")

        with pytest.raises(ValueError, match="fail"):
            await bcb.call(fail_func)

        assert bcb.state == CircuitState.OPEN

    @pytest.mark.unit
    def test_broker_get_status(self):
        """Test broker circuit breaker status includes failure counts."""
        bcb = BrokerCircuitBreaker()
        status = bcb.get_status()

        assert status["name"] == "broker_api"
        assert status["state"] == "closed"
        assert "order_failures" in status
        assert "quote_failures" in status
        assert "account_failures" in status
        assert status["order_failures"] == 0
        assert status["quote_failures"] == 0
        assert status["account_failures"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_broker_status_after_failures(self):
        """Test broker status reflects failure counts."""
        bcb = BrokerCircuitBreaker(failure_threshold=10)

        error = ValueError("API error")

        # Record various failures
        await bcb.record_order_failure(error)
        await bcb.record_order_failure(error)
        await bcb.record_quote_failure(error)
        await bcb.record_account_failure(error)

        status = bcb.get_status()
        assert status["order_failures"] == 2
        assert status["quote_failures"] == 1
        assert status["account_failures"] == 1
        assert status["failed_calls"] == 4

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_broker_state_change_callback(self):
        """Test broker circuit breaker with state change callback."""
        state_changes = []

        def on_state_change(state):
            state_changes.append(state)

        bcb = BrokerCircuitBreaker(
            failure_threshold=1,
            on_state_change=on_state_change,
        )

        error = ValueError("fail")
        await bcb.record_order_failure(error)

        assert CircuitState.OPEN in state_changes

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_broker_on_open_only_on_open_state(self):
        """Test on_open callback only triggered when transitioning to OPEN."""
        callback_count = []

        def on_open_callback():
            callback_count.append(1)

        bcb = BrokerCircuitBreaker(
            failure_threshold=1,
            success_threshold=1,
            recovery_timeout_seconds=0.01,
            on_open=on_open_callback,
        )

        async def fail_func():
            raise ValueError("fail")

        async def success_func():
            return "ok"

        # Open circuit
        with pytest.raises(ValueError, match="fail"):
            await bcb.call(fail_func)

        assert len(callback_count) == 1

        # Wait for recovery
        await asyncio.sleep(0.02)

        # Success should close circuit
        await bcb.call(success_func)

        # Callback should only be called once (on OPEN)
        assert len(callback_count) == 1
