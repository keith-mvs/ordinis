"""Tests for Enhanced Circuit Breaker and Error Recovery System."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ordinis.safety.error_recovery import (
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    EnhancedCircuitBreaker,
    ErrorCategory,
    ErrorClassifier,
    ErrorEvent,
    ErrorSeverity,
    ProtectedCall,
    RateLimitError,
    RecoveryConfig,
    RecoveryState,
)


class TestErrorSeverity:
    """Tests for ErrorSeverity enum."""

    def test_severity_levels_exist(self):
        """Test all severity levels exist."""
        assert ErrorSeverity.LOW is not None
        assert ErrorSeverity.MEDIUM is not None
        assert ErrorSeverity.HIGH is not None
        assert ErrorSeverity.CRITICAL is not None


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_network_categories(self):
        """Test network error categories exist."""
        assert ErrorCategory.CONNECTION_TIMEOUT is not None
        assert ErrorCategory.CONNECTION_REFUSED is not None
        assert ErrorCategory.DNS_FAILURE is not None
        assert ErrorCategory.SSL_ERROR is not None

    def test_api_categories(self):
        """Test API error categories exist."""
        assert ErrorCategory.RATE_LIMITED is not None
        assert ErrorCategory.AUTHENTICATION_FAILED is not None
        assert ErrorCategory.INSUFFICIENT_FUNDS is not None
        assert ErrorCategory.ORDER_REJECTED is not None

    def test_data_categories(self):
        """Test data error categories exist."""
        assert ErrorCategory.INVALID_RESPONSE is not None
        assert ErrorCategory.DATA_CORRUPTION is not None
        assert ErrorCategory.STALE_DATA is not None

    def test_system_categories(self):
        """Test system error categories exist."""
        assert ErrorCategory.OUT_OF_MEMORY is not None
        assert ErrorCategory.DISK_FULL is not None
        assert ErrorCategory.THREAD_POOL_EXHAUSTED is not None


class TestRecoveryState:
    """Tests for RecoveryState enum."""

    def test_recovery_states_exist(self):
        """Test all recovery states exist."""
        assert RecoveryState.NORMAL is not None
        assert RecoveryState.DEGRADED is not None
        assert RecoveryState.RECOVERING is not None
        assert RecoveryState.HALTED is not None


class TestErrorEvent:
    """Tests for ErrorEvent dataclass."""

    def test_create_error_event(self):
        """Test creating an error event."""
        error = ValueError("Test error")
        event = ErrorEvent(
            error=error,
            category=ErrorCategory.INVALID_RESPONSE,
            severity=ErrorSeverity.MEDIUM,
            operation="get_quote",
            engine="MarketData",
        )

        assert event.error == error
        assert event.category == ErrorCategory.INVALID_RESPONSE
        assert event.severity == ErrorSeverity.MEDIUM
        assert event.operation == "get_quote"
        assert event.engine == "MarketData"
        assert event.timestamp is not None
        assert event.recoverable is True

    def test_error_event_str(self):
        """Test error event string representation."""
        error = ValueError("Test error")
        event = ErrorEvent(
            error=error,
            category=ErrorCategory.CONNECTION_TIMEOUT,
            severity=ErrorSeverity.HIGH,
            operation="submit_order",
            engine="Broker",
        )

        str_repr = str(event)
        assert "HIGH" in str_repr
        assert "CONNECTION_TIMEOUT" in str_repr
        assert "Broker" in str_repr
        assert "submit_order" in str_repr


class TestCircuitState:
    """Tests for CircuitState dataclass."""

    def test_default_circuit_state(self):
        """Test default circuit state."""
        state = CircuitState(name="test_circuit")

        assert state.name == "test_circuit"
        assert state.state == RecoveryState.NORMAL
        assert state.failure_count == 0
        assert state.success_count == 0
        assert state.consecutive_failures == 0
        assert state.consecutive_successes == 0

    def test_reset_circuit_state(self):
        """Test resetting circuit state."""
        state = CircuitState(
            name="test_circuit",
            failure_count=5,
            success_count=10,
            consecutive_failures=3,
            consecutive_successes=2,
            half_open_successes=1,
        )

        state.reset()

        assert state.failure_count == 0
        assert state.success_count == 0
        assert state.consecutive_failures == 0
        assert state.consecutive_successes == 0
        assert state.half_open_successes == 0


class TestRecoveryConfig:
    """Tests for RecoveryConfig dataclass."""

    def test_default_config(self):
        """Test default recovery config."""
        config = RecoveryConfig()

        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.base_timeout_seconds == 30.0
        assert config.max_timeout_seconds == 300.0
        assert config.timeout_multiplier == 2.0

    def test_custom_config(self):
        """Test custom recovery config."""
        config = RecoveryConfig(
            failure_threshold=10,
            success_threshold=5,
            base_timeout_seconds=60.0,
            max_timeout_seconds=600.0,
        )

        assert config.failure_threshold == 10
        assert config.success_threshold == 5
        assert config.base_timeout_seconds == 60.0
        assert config.max_timeout_seconds == 600.0

    def test_get_timeout_exponential_backoff(self):
        """Test exponential backoff timeout calculation."""
        config = RecoveryConfig(
            base_timeout_seconds=10.0,
            timeout_multiplier=2.0,
            max_timeout_seconds=300.0,
        )

        assert config.get_timeout(0) == 10.0
        assert config.get_timeout(1) == 20.0
        assert config.get_timeout(2) == 40.0
        assert config.get_timeout(3) == 80.0

    def test_get_timeout_respects_max(self):
        """Test timeout respects maximum."""
        config = RecoveryConfig(
            base_timeout_seconds=100.0,
            timeout_multiplier=2.0,
            max_timeout_seconds=300.0,
        )

        # Should cap at max
        assert config.get_timeout(5) == 300.0


class TestErrorClassifier:
    """Tests for ErrorClassifier."""

    def test_classify_timeout(self):
        """Test classifying timeout errors."""
        error = Exception("Connection timed out")
        category, severity, recoverable = ErrorClassifier.classify(error)

        assert category == ErrorCategory.CONNECTION_TIMEOUT
        assert severity == ErrorSeverity.MEDIUM
        assert recoverable is True

    def test_classify_rate_limited(self):
        """Test classifying rate limit errors."""
        error = Exception("Rate limit exceeded - 429 Too Many Requests")
        category, severity, recoverable = ErrorClassifier.classify(error)

        assert category == ErrorCategory.RATE_LIMITED
        assert severity == ErrorSeverity.MEDIUM
        assert recoverable is True

    def test_classify_authentication_failed(self):
        """Test classifying auth errors."""
        error = Exception("401 Unauthorized")
        category, severity, recoverable = ErrorClassifier.classify(error)

        assert category == ErrorCategory.AUTHENTICATION_FAILED
        assert severity == ErrorSeverity.CRITICAL
        assert recoverable is False

    def test_classify_insufficient_funds(self):
        """Test classifying insufficient funds errors."""
        error = Exception("Insufficient buying power")
        category, severity, recoverable = ErrorClassifier.classify(error)

        assert category == ErrorCategory.INSUFFICIENT_FUNDS
        assert severity == ErrorSeverity.HIGH
        assert recoverable is True

    def test_classify_market_closed(self):
        """Test classifying market closed errors."""
        error = Exception("Market closed for trading")
        category, severity, recoverable = ErrorClassifier.classify(error)

        assert category == ErrorCategory.MARKET_CLOSED
        assert severity == ErrorSeverity.LOW
        assert recoverable is True

    def test_classify_unknown(self):
        """Test classifying unknown errors."""
        error = Exception("Some random error")
        category, severity, recoverable = ErrorClassifier.classify(error)

        assert category == ErrorCategory.UNKNOWN
        assert severity == ErrorSeverity.MEDIUM
        assert recoverable is True

    def test_classify_ssl_error(self):
        """Test classifying SSL errors."""
        error = Exception("SSL certificate verification failed")
        category, severity, recoverable = ErrorClassifier.classify(error)

        assert category == ErrorCategory.SSL_ERROR
        assert severity == ErrorSeverity.HIGH
        assert recoverable is False

    def test_classify_memory_error(self):
        """Test classifying memory errors."""
        error = Exception("Out of memory")
        category, severity, recoverable = ErrorClassifier.classify(error)

        assert category == ErrorCategory.OUT_OF_MEMORY
        assert severity == ErrorSeverity.CRITICAL
        assert recoverable is False


class TestEnhancedCircuitBreaker:
    """Tests for EnhancedCircuitBreaker."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker for testing."""
        config = RecoveryConfig(
            failure_threshold=3,
            success_threshold=2,
            base_timeout_seconds=1.0,
        )
        return EnhancedCircuitBreaker(name="test_breaker", config=config)

    def test_init(self, circuit_breaker):
        """Test circuit breaker initialization."""
        assert circuit_breaker.name == "test_breaker"
        assert circuit_breaker.is_healthy is True
        assert circuit_breaker.is_open is False

    def test_state_property(self, circuit_breaker):
        """Test state property."""
        assert circuit_breaker.state == RecoveryState.NORMAL

    def test_internal_state(self, circuit_breaker):
        """Test internal state structure."""
        assert circuit_breaker._state.name == "test_breaker"
        assert circuit_breaker._state.state == RecoveryState.NORMAL


class TestEnhancedCircuitBreakerAsync:
    """Async tests for EnhancedCircuitBreaker."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker for testing."""
        config = RecoveryConfig(
            failure_threshold=3,
            success_threshold=2,
            base_timeout_seconds=0.1,
        )
        return EnhancedCircuitBreaker(name="test_breaker", config=config)

    @pytest.mark.asyncio
    async def test_call_success(self, circuit_breaker):
        """Test calling function through circuit breaker."""
        async def successful_func():
            return "success"

        result = await circuit_breaker.call(successful_func)

        assert result == "success"

    @pytest.mark.asyncio
    async def test_call_failure(self, circuit_breaker):
        """Test calling failing function."""
        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_func)


class TestErrorRecoveryIntegration:
    """Integration tests for error recovery components."""

    def test_error_flow(self):
        """Test complete error classification and handling flow."""
        # Simulate network error
        error = Exception("Connection timed out after 30 seconds")

        # Classify
        category, severity, recoverable = ErrorClassifier.classify(error)

        # Create event
        event = ErrorEvent(
            error=error,
            category=category,
            severity=severity,
            operation="fetch_data",
            engine="MarketData",
            recoverable=recoverable,
        )

        # Verify classification
        assert event.category == ErrorCategory.CONNECTION_TIMEOUT
        assert event.severity == ErrorSeverity.MEDIUM
        assert event.recoverable is True

    def test_config_with_circuit_breaker(self):
        """Test config usage with circuit breaker."""
        config = RecoveryConfig(failure_threshold=2)
        breaker = EnhancedCircuitBreaker(name="test", config=config)

        assert breaker.config.failure_threshold == 2
        assert breaker.is_healthy is True


class TestEnhancedCircuitBreakerForceOperations:
    """Tests for force operations on circuit breaker."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker for testing."""
        return EnhancedCircuitBreaker(name="test_breaker")

    @pytest.mark.asyncio
    async def test_force_open(self, breaker):
        """Test force opening circuit."""
        await breaker.force_open(reason="Test")

        assert breaker.state == RecoveryState.HALTED
        assert breaker.is_open is True
        assert breaker.is_healthy is False

    @pytest.mark.asyncio
    async def test_force_close(self, breaker):
        """Test force closing circuit."""
        # First force open
        await breaker.force_open()
        assert breaker.state == RecoveryState.HALTED

        # Then force close
        await breaker.force_close()

        assert breaker.state == RecoveryState.NORMAL
        assert breaker.is_healthy is True

    @pytest.mark.asyncio
    async def test_force_open_blocks_calls(self, breaker):
        """Test force open blocks subsequent calls."""
        await breaker.force_open()

        async def test_func():
            return "success"

        with pytest.raises(CircuitOpenError):
            await breaker.call(test_func, operation="test")


class TestEnhancedCircuitBreakerStatus:
    """Tests for circuit breaker status reporting."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker for testing."""
        return EnhancedCircuitBreaker(name="test_breaker")

    def test_get_status(self, breaker):
        """Test getting circuit breaker status."""
        status = breaker.get_status()

        assert status["name"] == "test_breaker"
        assert status["state"] == "NORMAL"
        assert status["recovery_level"] == 1.0
        assert status["failure_count"] == 0
        assert status["success_count"] == 0
        assert status["consecutive_failures"] == 0
        assert status["consecutive_successes"] == 0

    @pytest.mark.asyncio
    async def test_status_after_calls(self, breaker):
        """Test status updates after calls."""
        async def success_func():
            return "success"

        await breaker.call(success_func, operation="test")
        await breaker.call(success_func, operation="test")

        status = breaker.get_status()
        assert status["success_count"] == 2
        assert status["consecutive_successes"] == 2


class TestEnhancedCircuitBreakerRateLimiting:
    """Tests for rate limiting functionality."""

    @pytest.fixture
    def breaker(self):
        """Create breaker with low rate limit."""
        config = RecoveryConfig(max_calls_per_minute=3)
        return EnhancedCircuitBreaker(name="test", config=config)

    @pytest.mark.asyncio
    async def test_rate_limit_allows_under_limit(self, breaker):
        """Test rate limit allows calls under limit."""
        async def test_func():
            return "success"

        # 3 calls should work
        for _ in range(3):
            await breaker.call(test_func, operation="test")

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_over_limit(self, breaker):
        """Test rate limit blocks calls over limit."""
        async def test_func():
            return "success"

        # 3 calls should work
        for _ in range(3):
            await breaker.call(test_func, operation="test")

        # 4th should fail
        with pytest.raises(RateLimitError):
            await breaker.call(test_func, operation="test")


class TestEnhancedCircuitBreakerCallbacks:
    """Tests for circuit breaker callbacks."""

    @pytest.mark.asyncio
    async def test_state_change_callback(self):
        """Test state change callback is called."""
        callback = MagicMock()
        breaker = EnhancedCircuitBreaker(
            name="test",
            on_state_change=callback,
        )

        await breaker.force_open()

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "test"
        assert args[1] == RecoveryState.NORMAL
        assert args[2] == RecoveryState.HALTED

    @pytest.mark.asyncio
    async def test_error_callback(self):
        """Test error callback is called."""
        callback = MagicMock()
        breaker = EnhancedCircuitBreaker(
            name="test",
            on_error=callback,
        )

        async def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            await breaker.call(failing_func, operation="test")

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert isinstance(event, ErrorEvent)
        assert event.operation == "test"


class TestEnhancedCircuitBreakerStateTransitions:
    """Tests for circuit breaker state transitions."""

    @pytest.fixture
    def breaker(self):
        """Create breaker with low thresholds."""
        config = RecoveryConfig(
            failure_threshold=2,
            success_threshold=2,
        )
        return EnhancedCircuitBreaker(name="test", config=config)

    @pytest.mark.asyncio
    async def test_transitions_to_degraded(self, breaker):
        """Test breaker transitions to degraded after failures."""
        async def failing_func():
            raise TimeoutError("timeout")

        # Fail twice to reach threshold
        for _ in range(2):
            with pytest.raises(TimeoutError):
                await breaker.call(failing_func, operation="test")

        assert breaker.state == RecoveryState.DEGRADED

    @pytest.mark.asyncio
    async def test_critical_error_halts(self, breaker):
        """Test critical error halts circuit."""
        async def critical_func():
            raise MemoryError("out of memory")

        with pytest.raises(MemoryError):
            await breaker.call(critical_func, operation="test")

        assert breaker.state == RecoveryState.HALTED


class TestEnhancedCircuitBreakerSyncFunction:
    """Tests for calling sync functions through breaker."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker."""
        return EnhancedCircuitBreaker(name="test")

    @pytest.mark.asyncio
    async def test_sync_function_success(self, breaker):
        """Test calling sync function."""
        def sync_func():
            return "sync result"

        result = await breaker.call(sync_func, operation="test")

        assert result == "sync result"
        assert breaker._state.success_count == 1


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_get_or_create_new(self):
        """Test creating new breaker."""
        registry = CircuitBreakerRegistry()

        breaker = registry.get_or_create("test")

        assert breaker is not None
        assert breaker.name == "test"

    def test_get_or_create_existing(self):
        """Test getting existing breaker."""
        registry = CircuitBreakerRegistry()

        breaker1 = registry.get_or_create("test")
        breaker2 = registry.get_or_create("test")

        assert breaker1 is breaker2

    def test_get_nonexistent(self):
        """Test getting nonexistent breaker."""
        registry = CircuitBreakerRegistry()

        breaker = registry.get("nonexistent")

        assert breaker is None

    def test_get_existing(self):
        """Test getting existing breaker."""
        registry = CircuitBreakerRegistry()

        created = registry.get_or_create("test")
        retrieved = registry.get("test")

        assert created is retrieved

    def test_get_global_status(self):
        """Test getting global status."""
        registry = CircuitBreakerRegistry()
        registry.get_or_create("breaker1")
        registry.get_or_create("breaker2")

        status = registry.get_global_status()

        assert "breaker1" in status
        assert "breaker2" in status
        assert status["breaker1"]["name"] == "breaker1"

    def test_get_unhealthy_all_healthy(self):
        """Test get unhealthy when all healthy."""
        registry = CircuitBreakerRegistry()
        registry.get_or_create("test1")
        registry.get_or_create("test2")

        unhealthy = registry.get_unhealthy()

        assert unhealthy == []

    def test_get_unhealthy_some_unhealthy(self):
        """Test get unhealthy when some are unhealthy."""
        registry = CircuitBreakerRegistry()
        healthy = registry.get_or_create("healthy")
        unhealthy_breaker = registry.get_or_create("unhealthy")

        # Force to degraded
        unhealthy_breaker._state.state = RecoveryState.DEGRADED

        unhealthy = registry.get_unhealthy()

        assert "unhealthy" in unhealthy
        assert "healthy" not in unhealthy

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """Test resetting all breakers."""
        registry = CircuitBreakerRegistry()
        breaker1 = registry.get_or_create("test1")
        breaker2 = registry.get_or_create("test2")

        # Force to degraded
        breaker1._state.state = RecoveryState.DEGRADED
        breaker2._state.state = RecoveryState.DEGRADED

        await registry.reset_all()

        assert breaker1.state == RecoveryState.NORMAL
        assert breaker2.state == RecoveryState.NORMAL


class TestCircuitBreakerRegistryWithKillSwitch:
    """Tests for registry with kill switch integration."""

    def test_create_with_kill_switch(self):
        """Test creating registry with kill switch."""
        mock_kill_switch = MagicMock()
        registry = CircuitBreakerRegistry(kill_switch=mock_kill_switch)

        breaker = registry.get_or_create("test")

        assert breaker._kill_switch is mock_kill_switch


class TestProtectedCall:
    """Tests for ProtectedCall context manager."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker."""
        return EnhancedCircuitBreaker(name="test")

    @pytest.mark.asyncio
    async def test_protected_success(self, breaker):
        """Test protected call on success."""
        async with breaker.protected(operation="test"):
            pass

        assert breaker._state.success_count == 1
        assert breaker._state.consecutive_successes == 1

    @pytest.mark.asyncio
    async def test_protected_failure(self, breaker):
        """Test protected call on failure."""
        with pytest.raises(ValueError):
            async with breaker.protected(operation="test"):
                raise ValueError("test error")

        assert breaker._state.failure_count == 1
        assert breaker._state.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_protected_blocked_when_open(self, breaker):
        """Test protected call blocked when circuit open."""
        await breaker.force_open()

        with pytest.raises(CircuitOpenError):
            async with breaker.protected(operation="test"):
                pass

    @pytest.mark.asyncio
    async def test_protected_default_operation(self, breaker):
        """Test protected call with default operation name."""
        protected = breaker.protected()

        assert protected._operation == "unknown"


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_circuit_open_error_message(self):
        """Test CircuitOpenError has message."""
        error = CircuitOpenError("test message")

        assert str(error) == "test message"

    def test_circuit_open_error_inheritance(self):
        """Test CircuitOpenError inherits from Exception."""
        error = CircuitOpenError("test")

        assert isinstance(error, Exception)


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_rate_limit_error_message(self):
        """Test RateLimitError has message."""
        error = RateLimitError("rate limit exceeded")

        assert str(error) == "rate limit exceeded"

    def test_rate_limit_error_inheritance(self):
        """Test RateLimitError inherits from Exception."""
        error = RateLimitError("test")

        assert isinstance(error, Exception)
