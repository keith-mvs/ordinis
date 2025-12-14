"""Tests for BaseEngine class.

This module tests the core engine lifecycle, health checks,
governance integration, and metrics collection.
"""

import pytest

from ordinis.engines.base import (
    Decision,
    EngineState,
    HealthLevel,
)
from tests.test_engines.test_base.conftest import (
    MockConfig,
    MockEngine,
    MockGovernanceHook,
)


class TestEngineLifecycle:
    """Test engine lifecycle state transitions."""

    @pytest.mark.asyncio
    async def test_initial_state(self, mock_engine: MockEngine) -> None:
        """Test engine starts in UNINITIALIZED state."""
        assert mock_engine.state == EngineState.UNINITIALIZED
        assert not mock_engine.is_running
        assert not mock_engine.initialized

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_engine: MockEngine) -> None:
        """Test successful engine initialization."""
        await mock_engine.initialize()

        assert mock_engine.state == EngineState.READY
        assert mock_engine.is_running
        assert mock_engine.initialized

        await mock_engine.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_failure(self) -> None:
        """Test engine initialization failure transitions to ERROR state."""
        config = MockConfig(name="TestEngine", fail_initialize=True)
        engine = MockEngine(config)

        with pytest.raises(RuntimeError, match="Simulated initialization failure"):
            await engine.initialize()

        assert engine.state == EngineState.ERROR
        assert not engine.is_running
        assert not engine.initialized

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, initialized_engine: MockEngine) -> None:
        """Test initializing an already initialized engine raises error."""
        with pytest.raises(RuntimeError, match="already initialized"):
            await initialized_engine.initialize()

    @pytest.mark.asyncio
    async def test_shutdown_success(self, initialized_engine: MockEngine) -> None:
        """Test successful engine shutdown."""
        await initialized_engine.shutdown()

        assert initialized_engine.state == EngineState.STOPPED
        assert not initialized_engine.is_running
        assert initialized_engine.shutdown_called
        assert not initialized_engine.initialized

    @pytest.mark.asyncio
    async def test_shutdown_with_error(self) -> None:
        """Test engine shutdown handles errors gracefully."""
        config = MockConfig(name="TestEngine", fail_shutdown=True)
        engine = MockEngine(config)
        await engine.initialize()

        await engine.shutdown()

        assert engine.state == EngineState.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_already_stopped(self, mock_engine: MockEngine) -> None:
        """Test shutting down an already stopped engine is idempotent."""
        await mock_engine.shutdown()
        assert mock_engine.state == EngineState.STOPPED

        await mock_engine.shutdown()
        assert mock_engine.state == EngineState.STOPPED

    @pytest.mark.asyncio
    async def test_state_transitions(self, mock_engine: MockEngine) -> None:
        """Test state transitions through full lifecycle."""
        assert mock_engine.state == EngineState.UNINITIALIZED

        await mock_engine.initialize()
        assert mock_engine.state == EngineState.READY

        await mock_engine.shutdown()
        assert mock_engine.state == EngineState.STOPPED


class TestEngineHealthChecks:
    """Test engine health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_not_running(self, mock_engine: MockEngine) -> None:
        """Test health check on uninitialized engine returns UNHEALTHY."""
        status = await mock_engine.health_check()

        assert status.level == HealthLevel.UNHEALTHY
        assert "not running" in status.message.lower()
        assert mock_engine.health_check_count == 0

    @pytest.mark.asyncio
    async def test_health_check_success(self, initialized_engine: MockEngine) -> None:
        """Test successful health check on running engine."""
        status = await initialized_engine.health_check()

        assert status.level == HealthLevel.HEALTHY
        assert "health check #1" in status.message.lower()
        assert status.latency_ms is not None
        assert status.latency_ms >= 0
        assert initialized_engine.health_check_count == 1

    @pytest.mark.asyncio
    async def test_health_check_degraded(self) -> None:
        """Test health check with degraded status."""
        config = MockConfig(name="TestEngine", health_level=HealthLevel.DEGRADED)
        engine = MockEngine(config)
        await engine.initialize()

        status = await engine.health_check()

        assert status.level == HealthLevel.DEGRADED
        assert status.is_healthy  # Degraded is still operational

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_failure(self) -> None:
        """Test health check that raises exception."""
        config = MockConfig(name="TestEngine", fail_health_check=True)
        engine = MockEngine(config)
        await engine.initialize()

        status = await engine.health_check()

        assert status.level == HealthLevel.UNHEALTHY
        assert "health check failed" in status.message.lower()

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_multiple_calls(self, initialized_engine: MockEngine) -> None:
        """Test multiple health checks increment counter."""
        status1 = await initialized_engine.health_check()
        assert "health check #1" in status1.message.lower()

        status2 = await initialized_engine.health_check()
        assert "health check #2" in status2.message.lower()

        status3 = await initialized_engine.health_check()
        assert "health check #3" in status3.message.lower()

        assert initialized_engine.health_check_count == 3


class TestEngineMetrics:
    """Test engine metrics collection."""

    @pytest.mark.asyncio
    async def test_initial_metrics(self, mock_engine: MockEngine) -> None:
        """Test metrics before any operations."""
        metrics = mock_engine.get_metrics()

        assert metrics.requests_total == 0
        assert metrics.requests_failed == 0
        assert metrics.latency_p50_ms == 0.0
        assert metrics.latency_p95_ms == 0.0
        assert metrics.latency_p99_ms == 0.0
        assert metrics.uptime_seconds == 0.0
        assert metrics.error_rate == 0.0

    @pytest.mark.asyncio
    async def test_metrics_after_initialization(self, initialized_engine: MockEngine) -> None:
        """Test metrics after engine initialization."""
        metrics = initialized_engine.get_metrics()

        assert metrics.uptime_seconds >= 0.0
        assert initialized_engine._started_at is not None

    @pytest.mark.asyncio
    async def test_metrics_track_operations(self, initialized_engine: MockEngine) -> None:
        """Test metrics update during operations."""
        async with initialized_engine.track_operation("test_operation", {"input": "data"}) as ctx:
            ctx["outputs"] = {"result": "success"}

        metrics = initialized_engine.get_metrics()

        assert metrics.requests_total == 1
        assert metrics.requests_failed == 0
        assert metrics.latency_p50_ms >= 0.0
        assert metrics.error_rate == 0.0

    @pytest.mark.asyncio
    async def test_metrics_track_failures(self, initialized_engine: MockEngine) -> None:
        """Test metrics track failed operations."""
        with pytest.raises(RuntimeError):
            async with initialized_engine.track_operation(
                "test_operation", {"input": "data"}
            ) as ctx:
                raise RuntimeError("Test error")

        metrics = initialized_engine.get_metrics()

        assert metrics.requests_total == 1
        assert metrics.requests_failed == 1
        assert metrics.error_rate == 100.0

    @pytest.mark.asyncio
    async def test_metrics_latency_percentiles(self, initialized_engine: MockEngine) -> None:
        """Test latency percentile calculations."""
        for i in range(10):
            async with initialized_engine.track_operation(
                "test_operation", {"iteration": i}
            ) as ctx:
                ctx["outputs"] = {"result": i}

        metrics = initialized_engine.get_metrics()

        assert metrics.requests_total == 10
        assert metrics.latency_p50_ms >= 0.0
        assert metrics.latency_p95_ms >= metrics.latency_p50_ms
        assert metrics.latency_p99_ms >= metrics.latency_p95_ms


class TestEngineGovernance:
    """Test engine governance integration."""

    @pytest.mark.asyncio
    async def test_preflight_allow(
        self, mock_engine_with_hook: MockEngine, mock_governance_hook: MockGovernanceHook
    ) -> None:
        """Test preflight check that allows operation."""
        result = await mock_engine_with_hook.preflight(
            action="test_action", inputs={"key": "value"}
        )

        assert result.decision == Decision.ALLOW
        assert result.allowed
        assert not result.blocked
        assert len(mock_governance_hook.preflight_calls) == 1

    @pytest.mark.asyncio
    async def test_preflight_deny(
        self, mock_config: MockConfig, mock_governance_hook_deny: MockGovernanceHook
    ) -> None:
        """Test preflight check that denies operation."""
        engine = MockEngine(mock_config, mock_governance_hook_deny)

        result = await engine.preflight(action="test_action")

        assert result.decision == Decision.DENY
        assert not result.allowed
        assert result.blocked
        assert result.reason == "Test denial"

    @pytest.mark.asyncio
    async def test_preflight_disabled(
        self, mock_config_with_governance_disabled: MockConfig
    ) -> None:
        """Test preflight with governance disabled."""
        engine = MockEngine(mock_config_with_governance_disabled)

        result = await engine.preflight(action="test_action")

        assert result.decision == Decision.ALLOW
        assert result.reason == "Governance disabled"

    @pytest.mark.asyncio
    async def test_audit_record_created(
        self,
        mock_engine_with_hook: MockEngine,
        mock_governance_hook: MockGovernanceHook,
    ) -> None:
        """Test audit record is created for operations."""
        await mock_engine_with_hook.audit(
            action="test_action",
            inputs={"key": "value"},
            outputs={"result": "success"},
            model_used="test-model",
            latency_ms=10.5,
        )

        assert len(mock_governance_hook.audit_calls) == 1
        record = mock_governance_hook.audit_calls[0]
        assert record.engine == "TestEngine"
        assert record.action == "test_action"
        assert record.inputs == {"key": "value"}
        assert record.outputs == {"result": "success"}
        assert record.model_used == "test-model"
        assert record.latency_ms == 10.5

    @pytest.mark.asyncio
    async def test_audit_disabled(self, mock_config_with_governance_disabled: MockConfig) -> None:
        """Test audit with auditing disabled."""
        hook = MockGovernanceHook("TestEngine")
        engine = MockEngine(mock_config_with_governance_disabled, hook)

        await engine.audit(action="test_action")

        assert len(hook.audit_calls) == 0

    @pytest.mark.asyncio
    async def test_error_handling(
        self, mock_engine_with_hook: MockEngine, mock_governance_hook: MockGovernanceHook
    ) -> None:
        """Test error handling invokes governance hook."""
        await mock_engine_with_hook.initialize()

        with pytest.raises(RuntimeError):
            async with mock_engine_with_hook.track_operation("test_operation"):
                raise RuntimeError("Test error")

        assert len(mock_governance_hook.error_calls) == 1
        error = mock_governance_hook.error_calls[0]
        assert error.code == "OPERATION_FAILED"
        assert error.engine == "TestEngine"

        await mock_engine_with_hook.shutdown()


class TestEngineOperationTracking:
    """Test operation tracking context manager."""

    @pytest.mark.asyncio
    async def test_track_operation_success(self, initialized_engine: MockEngine) -> None:
        """Test successful operation tracking."""
        async with initialized_engine.track_operation("test_action", {"input": "data"}) as ctx:
            ctx["outputs"] = {"result": "success"}
            ctx["model_used"] = "test-model"

        metrics = initialized_engine.get_metrics()
        assert metrics.requests_total == 1
        assert metrics.requests_failed == 0

    @pytest.mark.asyncio
    async def test_track_operation_failure(self, initialized_engine: MockEngine) -> None:
        """Test failed operation tracking."""
        with pytest.raises(ValueError, match="Test error"):
            async with initialized_engine.track_operation("test_action", {"input": "data"}) as ctx:
                raise ValueError("Test error")

        metrics = initialized_engine.get_metrics()
        assert metrics.requests_total == 1
        assert metrics.requests_failed == 1

    @pytest.mark.asyncio
    async def test_track_operation_preflight_denied(
        self, mock_config: MockConfig, mock_governance_hook_deny: MockGovernanceHook
    ) -> None:
        """Test operation tracking with denied preflight."""
        engine = MockEngine(mock_config, mock_governance_hook_deny)
        await engine.initialize()

        with pytest.raises(PermissionError, match="Operation denied"):
            async with engine.track_operation("test_action"):
                pass

        metrics = engine.get_metrics()
        assert metrics.requests_total == 0

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_track_operation_latency(self, initialized_engine: MockEngine) -> None:
        """Test operation latency is tracked."""
        async with initialized_engine.track_operation("test_action", {"input": "data"}) as ctx:
            ctx["outputs"] = {"result": "success"}

        metrics = initialized_engine.get_metrics()
        assert metrics.latency_p50_ms >= 0.0
        assert len(initialized_engine._latencies) == 1

    @pytest.mark.asyncio
    async def test_track_operation_limits_latencies(self, initialized_engine: MockEngine) -> None:
        """Test latency history is limited to 1000 entries."""
        for i in range(1100):
            async with initialized_engine.track_operation("test_action", {"iteration": i}) as ctx:
                ctx["outputs"] = {"result": i}

        assert len(initialized_engine._latencies) == 1000


class TestEngineProperties:
    """Test engine property accessors."""

    def test_name_from_config(self, mock_engine: MockEngine) -> None:
        """Test engine name comes from config."""
        assert mock_engine.name == "TestEngine"

    def test_name_from_class(self) -> None:
        """Test engine name defaults to class name."""
        config = MockConfig()
        config.name = ""
        engine = MockEngine(config)

        assert "MockEngine" in engine.name

    def test_config_property(self, mock_engine: MockEngine) -> None:
        """Test config property accessor."""
        assert mock_engine.config.name == "TestEngine"
        assert mock_engine.config.test_setting == "default"

    def test_state_property(self, mock_engine: MockEngine) -> None:
        """Test state property accessor."""
        assert mock_engine.state == EngineState.UNINITIALIZED

    def test_is_running_property(self, mock_engine: MockEngine) -> None:
        """Test is_running property."""
        assert not mock_engine.is_running

        mock_engine._state = EngineState.READY
        assert mock_engine.is_running

        mock_engine._state = EngineState.RUNNING
        assert mock_engine.is_running

        mock_engine._state = EngineState.STOPPED
        assert not mock_engine.is_running

    def test_requirements_property(self, mock_engine: MockEngine) -> None:
        """Test requirements property accessor."""
        assert mock_engine.requirements is not None
        assert mock_engine.requirements.engine_name == "TESTENGINE"

    def test_governance_property(self, mock_engine: MockEngine) -> None:
        """Test governance property accessor."""
        assert mock_engine.governance is not None
        assert isinstance(mock_engine.governance, MockGovernanceHook) or hasattr(
            mock_engine.governance, "preflight"
        )


class TestEngineRepresentation:
    """Test engine string representation."""

    def test_repr(self, mock_engine: MockEngine) -> None:
        """Test __repr__ method."""
        repr_str = repr(mock_engine)

        assert "MockEngine" in repr_str
        assert "TestEngine" in repr_str
        assert "uninitialized" in repr_str.lower()

    @pytest.mark.asyncio
    async def test_repr_after_initialize(self, initialized_engine: MockEngine) -> None:
        """Test __repr__ after initialization."""
        repr_str = repr(initialized_engine)

        assert "MockEngine" in repr_str
        assert "ready" in repr_str.lower() or "running" in repr_str.lower()
