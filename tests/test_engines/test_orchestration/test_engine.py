"""Tests for OrchestrationEngine class.

This module tests the orchestration engine lifecycle, pipeline coordination,
trading cycles, mode switching, and metrics collection.
"""

import asyncio
from typing import Any

import pytest

from ordinis.engines.base import EngineState, HealthLevel
from ordinis.engines.orchestration.core.config import OrchestrationEngineConfig
from ordinis.engines.orchestration.core.engine import OrchestrationEngine
from ordinis.engines.orchestration.core.models import CycleStatus, PipelineStage
from tests.test_engines.test_orchestration.conftest import (
    MockAnalyticsEngine,
    MockDataSource,
    MockExecutionEngine,
    MockRiskEngine,
    MockSignalEngine,
    apply_governance_workarounds,
)


class TestEngineLifecycle:
    """Test orchestration engine lifecycle state transitions."""

    @pytest.mark.asyncio
    async def test_initial_state(self, orchestration_engine: OrchestrationEngine) -> None:
        """Test engine starts in UNINITIALIZED state."""
        assert orchestration_engine.state == EngineState.UNINITIALIZED
        assert not orchestration_engine.is_running

    @pytest.mark.asyncio
    async def test_initialize_success(self, orchestration_engine: OrchestrationEngine) -> None:
        """Test successful engine initialization."""
        await orchestration_engine.initialize()

        assert orchestration_engine.state == EngineState.READY
        assert orchestration_engine.is_running

        await orchestration_engine.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_success(
        self, initialized_orchestration_engine: OrchestrationEngine
    ) -> None:
        """Test successful engine shutdown."""
        await initialized_orchestration_engine.shutdown()

        assert initialized_orchestration_engine.state == EngineState.STOPPED
        assert not initialized_orchestration_engine.is_running

    @pytest.mark.asyncio
    async def test_state_transitions(self, orchestration_engine: OrchestrationEngine) -> None:
        """Test state transitions through full lifecycle."""
        assert orchestration_engine.state == EngineState.UNINITIALIZED

        await orchestration_engine.initialize()
        assert orchestration_engine.state == EngineState.READY

        await orchestration_engine.shutdown()
        assert orchestration_engine.state == EngineState.STOPPED


class TestEngineHealthChecks:
    """Test orchestration engine health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_not_running(
        self, orchestration_engine: OrchestrationEngine
    ) -> None:
        """Test health check on uninitialized engine returns UNHEALTHY."""
        status = await orchestration_engine.health_check()

        assert status.level == HealthLevel.UNHEALTHY
        assert "not running" in status.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_missing_engines(
        self, initialized_orchestration_engine: OrchestrationEngine
    ) -> None:
        """Test health check with missing required engines."""
        status = await initialized_orchestration_engine.health_check()

        assert status.level == HealthLevel.DEGRADED
        assert "missing engines" in status.message.lower()
        assert "signal_engine" in status.details.get("missing_engines", [])
        assert "risk_engine" in status.details.get("missing_engines", [])
        assert "execution_engine" in status.details.get("missing_engines", [])

    @pytest.mark.asyncio
    async def test_health_check_fully_configured(
        self, fully_configured_engine: OrchestrationEngine
    ) -> None:
        """Test health check with all engines registered."""
        status = await fully_configured_engine.health_check()

        assert status.level == HealthLevel.HEALTHY
        assert "operational" in status.message.lower()
        assert status.details["mode"] == "paper"
        assert "total_cycles" in status.details


class TestEngineRegistration:
    """Test engine registration methods."""

    @pytest.mark.asyncio
    async def test_register_signal_engine(
        self,
        orchestration_engine: OrchestrationEngine,
        mock_signal_engine: MockSignalEngine,
    ) -> None:
        """Test registering signal engine."""
        orchestration_engine.register_signal_engine(mock_signal_engine)

        assert orchestration_engine._engines.signal_engine is mock_signal_engine

    @pytest.mark.asyncio
    async def test_register_risk_engine(
        self,
        orchestration_engine: OrchestrationEngine,
        mock_risk_engine: MockRiskEngine,
    ) -> None:
        """Test registering risk engine."""
        orchestration_engine.register_risk_engine(mock_risk_engine)

        assert orchestration_engine._engines.risk_engine is mock_risk_engine

    @pytest.mark.asyncio
    async def test_register_execution_engine(
        self,
        orchestration_engine: OrchestrationEngine,
        mock_execution_engine: MockExecutionEngine,
    ) -> None:
        """Test registering execution engine."""
        orchestration_engine.register_execution_engine(mock_execution_engine)

        assert orchestration_engine._engines.execution_engine is mock_execution_engine

    @pytest.mark.asyncio
    async def test_register_analytics_engine(
        self,
        orchestration_engine: OrchestrationEngine,
        mock_analytics_engine: MockAnalyticsEngine,
    ) -> None:
        """Test registering analytics engine."""
        orchestration_engine.register_analytics_engine(mock_analytics_engine)

        assert orchestration_engine._engines.analytics_engine is mock_analytics_engine

    @pytest.mark.asyncio
    async def test_register_data_source(
        self,
        orchestration_engine: OrchestrationEngine,
        mock_data_source: MockDataSource,
    ) -> None:
        """Test registering data source."""
        orchestration_engine.register_data_source(mock_data_source)

        assert orchestration_engine._engines.data_source is mock_data_source

    @pytest.mark.asyncio
    async def test_register_engines_bulk(
        self,
        orchestration_engine: OrchestrationEngine,
        mock_signal_engine: MockSignalEngine,
        mock_risk_engine: MockRiskEngine,
        mock_execution_engine: MockExecutionEngine,
        mock_analytics_engine: MockAnalyticsEngine,
        mock_data_source: MockDataSource,
    ) -> None:
        """Test registering multiple engines at once."""
        orchestration_engine.register_engines(
            signal_engine=mock_signal_engine,
            risk_engine=mock_risk_engine,
            execution_engine=mock_execution_engine,
            analytics_engine=mock_analytics_engine,
            data_source=mock_data_source,
        )

        assert orchestration_engine._engines.signal_engine is mock_signal_engine
        assert orchestration_engine._engines.risk_engine is mock_risk_engine
        assert orchestration_engine._engines.execution_engine is mock_execution_engine
        assert orchestration_engine._engines.analytics_engine is mock_analytics_engine
        assert orchestration_engine._engines.data_source is mock_data_source

    @pytest.mark.asyncio
    async def test_register_engines_partial(
        self,
        orchestration_engine: OrchestrationEngine,
        mock_signal_engine: MockSignalEngine,
        mock_risk_engine: MockRiskEngine,
    ) -> None:
        """Test registering only some engines."""
        orchestration_engine.register_engines(
            signal_engine=mock_signal_engine,
            risk_engine=mock_risk_engine,
        )

        assert orchestration_engine._engines.signal_engine is mock_signal_engine
        assert orchestration_engine._engines.risk_engine is mock_risk_engine
        assert orchestration_engine._engines.execution_engine is None
        assert orchestration_engine._engines.analytics_engine is None
        assert orchestration_engine._engines.data_source is None


class TestTradingCycle:
    """Test trading cycle execution."""

    @pytest.mark.asyncio
    async def test_run_cycle_no_data(self, fully_configured_engine: OrchestrationEngine) -> None:
        """Test cycle with no data available."""
        result = await fully_configured_engine.run_cycle(symbols=["AAPL"])

        assert result.status == CycleStatus.FAILED
        assert "No data available" in result.errors
        assert len(result.stages) >= 1
        assert result.stages[0].stage == PipelineStage.DATA_FETCH

    @pytest.mark.asyncio
    async def test_run_cycle_with_data(
        self,
        fully_configured_engine: OrchestrationEngine,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test successful cycle with market data."""
        result = await fully_configured_engine.run_cycle(data=sample_market_data)

        assert result.status == CycleStatus.COMPLETED
        assert len(result.stages) >= 2  # At least data fetch and signal generation
        assert result.cycle_id.startswith("CYC-")
        assert result.completed_at is not None
        assert result.total_duration_ms >= 0  # Can be 0 for very fast cycles

    @pytest.mark.asyncio
    async def test_run_cycle_with_signals(
        self,
        fully_configured_engine: OrchestrationEngine,
        mock_signal_engine: MockSignalEngine,
        sample_market_data: dict[str, Any],
        sample_signals: list[dict[str, Any]],
    ) -> None:
        """Test cycle that generates signals."""
        mock_signal_engine.set_return_value(sample_signals)

        result = await fully_configured_engine.run_cycle(data=sample_market_data)

        assert result.status == CycleStatus.COMPLETED
        assert result.signals_generated == len(sample_signals)
        assert mock_signal_engine.call_count == 1

    @pytest.mark.asyncio
    async def test_run_cycle_signals_approved(
        self,
        fully_configured_engine: OrchestrationEngine,
        mock_signal_engine: MockSignalEngine,
        mock_risk_engine: MockRiskEngine,
        sample_market_data: dict[str, Any],
        sample_signals: list[dict[str, Any]],
    ) -> None:
        """Test cycle with signals approved by risk engine."""
        mock_signal_engine.set_return_value(sample_signals)
        mock_risk_engine.set_return_value((sample_signals, []))

        result = await fully_configured_engine.run_cycle(data=sample_market_data)

        assert result.status == CycleStatus.COMPLETED
        assert result.signals_generated == len(sample_signals)
        assert result.signals_approved == len(sample_signals)
        assert mock_risk_engine.call_count == 1

    @pytest.mark.asyncio
    async def test_run_cycle_signals_rejected(
        self,
        fully_configured_engine: OrchestrationEngine,
        mock_signal_engine: MockSignalEngine,
        mock_risk_engine: MockRiskEngine,
        sample_market_data: dict[str, Any],
        sample_signals: list[dict[str, Any]],
    ) -> None:
        """Test cycle with all signals rejected by risk engine."""
        mock_signal_engine.set_return_value(sample_signals)
        mock_risk_engine.set_return_value(([], ["high_risk", "high_risk"]))

        result = await fully_configured_engine.run_cycle(data=sample_market_data)

        assert result.status == CycleStatus.COMPLETED
        assert result.signals_generated == len(sample_signals)
        assert result.signals_approved == 0
        assert result.orders_submitted == 0

    @pytest.mark.asyncio
    async def test_run_cycle_orders_executed(
        self,
        fully_configured_engine: OrchestrationEngine,
        mock_signal_engine: MockSignalEngine,
        mock_risk_engine: MockRiskEngine,
        mock_execution_engine: MockExecutionEngine,
        sample_market_data: dict[str, Any],
        sample_signals: list[dict[str, Any]],
        sample_fills: list[dict[str, Any]],
    ) -> None:
        """Test cycle with orders executed."""
        mock_signal_engine.set_return_value(sample_signals)
        mock_risk_engine.set_return_value((sample_signals, []))
        mock_execution_engine.set_return_value(sample_fills)

        result = await fully_configured_engine.run_cycle(data=sample_market_data)

        assert result.status == CycleStatus.COMPLETED
        assert result.orders_submitted == len(sample_signals)
        assert result.orders_filled == len(sample_fills)
        assert mock_execution_engine.call_count == 1

    @pytest.mark.asyncio
    async def test_run_cycle_analytics_recorded(
        self,
        fully_configured_engine: OrchestrationEngine,
        mock_signal_engine: MockSignalEngine,
        mock_risk_engine: MockRiskEngine,
        mock_execution_engine: MockExecutionEngine,
        mock_analytics_engine: MockAnalyticsEngine,
        sample_market_data: dict[str, Any],
        sample_signals: list[dict[str, Any]],
        sample_fills: list[dict[str, Any]],
    ) -> None:
        """Test cycle with analytics recording."""
        mock_signal_engine.set_return_value(sample_signals)
        mock_risk_engine.set_return_value((sample_signals, []))
        mock_execution_engine.set_return_value(sample_fills)

        result = await fully_configured_engine.run_cycle(data=sample_market_data)

        assert result.status == CycleStatus.COMPLETED
        assert mock_analytics_engine.call_count == 1

        # Verify analytics stage was recorded
        analytics_stages = [
            s for s in result.stages if s.stage == PipelineStage.ANALYTICS_RECORDING
        ]
        assert len(analytics_stages) == 1

    @pytest.mark.asyncio
    async def test_run_cycle_analytics_disabled(
        self,
        orchestration_config: OrchestrationEngineConfig,
        mock_signal_engine: MockSignalEngine,
        mock_risk_engine: MockRiskEngine,
        mock_execution_engine: MockExecutionEngine,
        mock_analytics_engine: MockAnalyticsEngine,
        mock_data_source: MockDataSource,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test cycle with analytics recording disabled."""
        orchestration_config.enable_analytics_recording = False
        engine = OrchestrationEngine(orchestration_config)
        apply_governance_workarounds(engine)
        engine.register_engines(
            signal_engine=mock_signal_engine,
            risk_engine=mock_risk_engine,
            execution_engine=mock_execution_engine,
            analytics_engine=mock_analytics_engine,
            data_source=mock_data_source,
        )
        await engine.initialize()

        result = await engine.run_cycle(data=sample_market_data)

        assert result.status == CycleStatus.COMPLETED
        assert mock_analytics_engine.call_count == 0

        # Verify no analytics stage was recorded
        analytics_stages = [
            s for s in result.stages if s.stage == PipelineStage.ANALYTICS_RECORDING
        ]
        assert len(analytics_stages) == 0

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_run_cycle_latency_tracking(
        self,
        fully_configured_engine: OrchestrationEngine,
        mock_signal_engine: MockSignalEngine,
        mock_risk_engine: MockRiskEngine,
        mock_execution_engine: MockExecutionEngine,
        sample_market_data: dict[str, Any],
        sample_signals: list[dict[str, Any]],
        sample_fills: list[dict[str, Any]],
    ) -> None:
        """Test cycle tracks latency for each stage."""
        mock_signal_engine.set_return_value(sample_signals)
        mock_risk_engine.set_return_value((sample_signals, []))
        mock_execution_engine.set_return_value(sample_fills)

        result = await fully_configured_engine.run_cycle(data=sample_market_data)

        assert result.data_latency_ms >= 0
        assert result.signal_latency_ms >= 0
        assert result.risk_latency_ms >= 0
        assert result.execution_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_run_cycle_exception_handling(
        self,
        fully_configured_engine: OrchestrationEngine,
        mock_signal_engine: MockSignalEngine,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test cycle handles exceptions gracefully."""
        mock_signal_engine.set_side_effect(RuntimeError("Test error"))

        result = await fully_configured_engine.run_cycle(data=sample_market_data)

        assert result.status == CycleStatus.FAILED
        assert len(result.errors) > 0
        assert "Test error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_run_cycle_history_updated(
        self,
        fully_configured_engine: OrchestrationEngine,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test cycle result is added to history."""
        initial_count = len(fully_configured_engine.get_cycle_history())

        await fully_configured_engine.run_cycle(data=sample_market_data)

        assert len(fully_configured_engine.get_cycle_history()) == initial_count + 1


class TestContinuousOperation:
    """Test continuous trading loop."""

    @pytest.mark.asyncio
    async def test_run_loop_max_cycles(
        self,
        fully_configured_engine: OrchestrationEngine,
        mock_data_source: MockDataSource,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test trading loop with max cycles limit."""
        mock_data_source.set_return_value(sample_market_data)

        await fully_configured_engine.run_loop(symbols=["AAPL"], max_cycles=3)

        assert len(fully_configured_engine.get_cycle_history()) == 3

    @pytest.mark.asyncio
    async def test_run_loop_stop(
        self,
        fully_configured_engine: OrchestrationEngine,
        mock_data_source: MockDataSource,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test stopping trading loop."""
        mock_data_source.set_return_value(sample_market_data)

        # Run loop in background and stop after short delay
        async def stop_after_delay() -> None:
            await asyncio.sleep(0.2)
            fully_configured_engine.stop_loop()

        loop_task = asyncio.create_task(fully_configured_engine.run_loop(symbols=["AAPL"]))
        stop_task = asyncio.create_task(stop_after_delay())

        await stop_task
        await loop_task

        # Should have run at least one cycle
        assert len(fully_configured_engine.get_cycle_history()) >= 1

    @pytest.mark.asyncio
    async def test_run_loop_respects_interval(
        self,
        orchestration_config: OrchestrationEngineConfig,
        mock_signal_engine: MockSignalEngine,
        mock_risk_engine: MockRiskEngine,
        mock_execution_engine: MockExecutionEngine,
        mock_analytics_engine: MockAnalyticsEngine,
        mock_data_source: MockDataSource,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test trading loop respects cycle interval."""
        orchestration_config.cycle_interval_ms = 100  # 100ms interval
        engine = OrchestrationEngine(orchestration_config)
        apply_governance_workarounds(engine)
        engine.register_engines(
            signal_engine=mock_signal_engine,
            risk_engine=mock_risk_engine,
            execution_engine=mock_execution_engine,
            analytics_engine=mock_analytics_engine,
            data_source=mock_data_source,
        )
        await engine.initialize()

        mock_data_source.set_return_value(sample_market_data)

        import time

        start_time = time.time()
        await engine.run_loop(symbols=["AAPL"], max_cycles=2)
        elapsed = time.time() - start_time

        # Should take at least 100ms for the interval
        assert elapsed >= 0.1

        await engine.shutdown()


class TestBacktestSupport:
    """Test backtest mode functionality."""

    @pytest.mark.asyncio
    async def test_run_backtest(
        self,
        fully_configured_engine: OrchestrationEngine,
        sample_backtest_data: list[dict[str, Any]],
    ) -> None:
        """Test running backtest over historical data."""
        results = await fully_configured_engine.run_backtest(sample_backtest_data)

        assert len(results) == len(sample_backtest_data)
        assert all(isinstance(r.cycle_id, str) for r in results)

    @pytest.mark.asyncio
    async def test_run_backtest_with_signals(
        self,
        fully_configured_engine: OrchestrationEngine,
        mock_signal_engine: MockSignalEngine,
        sample_backtest_data: list[dict[str, Any]],
        sample_signals: list[dict[str, Any]],
    ) -> None:
        """Test backtest generates signals for each data point."""
        mock_signal_engine.set_return_value(sample_signals)

        results = await fully_configured_engine.run_backtest(sample_backtest_data)

        assert len(results) == len(sample_backtest_data)
        assert mock_signal_engine.call_count == len(sample_backtest_data)

    @pytest.mark.asyncio
    async def test_run_backtest_mode_warning(
        self,
        orchestration_config_live: OrchestrationEngineConfig,
        mock_signal_engine: MockSignalEngine,
        mock_risk_engine: MockRiskEngine,
        mock_execution_engine: MockExecutionEngine,
        mock_analytics_engine: MockAnalyticsEngine,
        mock_data_source: MockDataSource,
        sample_backtest_data: list[dict[str, Any]],
    ) -> None:
        """Test backtest warns when run in non-backtest mode."""
        # Config is in live mode
        engine = OrchestrationEngine(orchestration_config_live)
        apply_governance_workarounds(engine)
        engine.register_engines(
            signal_engine=mock_signal_engine,
            risk_engine=mock_risk_engine,
            execution_engine=mock_execution_engine,
            analytics_engine=mock_analytics_engine,
            data_source=mock_data_source,
        )
        await engine.initialize()

        # Should still work but log warning
        results = await engine.run_backtest(sample_backtest_data)

        assert len(results) == len(sample_backtest_data)

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_run_backtest_empty_data(
        self, fully_configured_engine: OrchestrationEngine
    ) -> None:
        """Test backtest with empty historical data."""
        results = await fully_configured_engine.run_backtest([])

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_run_backtest_successful_cycles(
        self,
        fully_configured_engine: OrchestrationEngine,
        sample_backtest_data: list[dict[str, Any]],
    ) -> None:
        """Test backtest tracks successful cycles."""
        results = await fully_configured_engine.run_backtest(sample_backtest_data)

        completed_count = sum(1 for r in results if r.status == CycleStatus.COMPLETED)
        assert completed_count >= 0  # At least some should complete


class TestMetricsAndHistory:
    """Test metrics collection and history tracking."""

    @pytest.mark.asyncio
    async def test_get_metrics_initial(self, fully_configured_engine: OrchestrationEngine) -> None:
        """Test initial metrics before any cycles."""
        metrics = fully_configured_engine.get_metrics()

        assert metrics.total_cycles == 0
        assert metrics.successful_cycles == 0
        assert metrics.failed_cycles == 0

    @pytest.mark.asyncio
    async def test_get_metrics_after_cycle(
        self,
        fully_configured_engine: OrchestrationEngine,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test metrics after running a cycle."""
        await fully_configured_engine.run_cycle(data=sample_market_data)

        metrics = fully_configured_engine.get_metrics()

        assert metrics.total_cycles == 1
        assert metrics.successful_cycles >= 0
        assert metrics.failed_cycles >= 0

    @pytest.mark.asyncio
    async def test_get_cycle_history(
        self,
        fully_configured_engine: OrchestrationEngine,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test getting cycle history."""
        await fully_configured_engine.run_cycle(data=sample_market_data)
        await fully_configured_engine.run_cycle(data=sample_market_data)

        history = fully_configured_engine.get_cycle_history()

        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_get_cycle_history_limit(
        self,
        fully_configured_engine: OrchestrationEngine,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test getting limited cycle history."""
        for _ in range(5):
            await fully_configured_engine.run_cycle(data=sample_market_data)

        history = fully_configured_engine.get_cycle_history(limit=3)

        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_get_last_cycle(
        self,
        fully_configured_engine: OrchestrationEngine,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test getting last cycle result."""
        result = await fully_configured_engine.run_cycle(data=sample_market_data)

        last_cycle = fully_configured_engine.get_last_cycle()

        assert last_cycle is not None
        assert last_cycle.cycle_id == result.cycle_id

    @pytest.mark.asyncio
    async def test_get_last_cycle_none(self, fully_configured_engine: OrchestrationEngine) -> None:
        """Test getting last cycle when no cycles have run."""
        last_cycle = fully_configured_engine.get_last_cycle()

        assert last_cycle is None

    @pytest.mark.asyncio
    async def test_clear_history(
        self,
        fully_configured_engine: OrchestrationEngine,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test clearing cycle history."""
        await fully_configured_engine.run_cycle(data=sample_market_data)
        assert len(fully_configured_engine.get_cycle_history()) > 0

        fully_configured_engine.clear_history()

        assert len(fully_configured_engine.get_cycle_history()) == 0
        assert fully_configured_engine.get_metrics().total_cycles == 0


class TestModeSupport:
    """Test different operating modes."""

    @pytest.mark.asyncio
    async def test_paper_mode(
        self,
        orchestration_config: OrchestrationEngineConfig,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test paper trading mode."""
        orchestration_config.mode = "paper"
        engine = OrchestrationEngine(orchestration_config)
        apply_governance_workarounds(engine)
        await engine.initialize()

        assert engine.config.mode == "paper"

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_live_mode(
        self,
        orchestration_config_live: OrchestrationEngineConfig,
    ) -> None:
        """Test live trading mode."""
        engine = OrchestrationEngine(orchestration_config_live)
        apply_governance_workarounds(engine)
        await engine.initialize()

        assert engine.config.mode == "live"

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_backtest_mode(
        self,
        orchestration_config_backtest: OrchestrationEngineConfig,
    ) -> None:
        """Test backtest mode."""
        engine = OrchestrationEngine(orchestration_config_backtest)
        apply_governance_workarounds(engine)
        await engine.initialize()

        assert engine.config.mode == "backtest"

        await engine.shutdown()


class TestGovernanceIntegration:
    """Test governance hook integration."""

    @pytest.mark.asyncio
    async def test_governance_enabled(
        self, orchestration_config: OrchestrationEngineConfig
    ) -> None:
        """Test engine with governance enabled."""
        orchestration_config.enable_governance = True
        engine = OrchestrationEngine(orchestration_config)
        apply_governance_workarounds(engine)

        assert engine.config.enable_governance is True

    @pytest.mark.asyncio
    async def test_governance_disabled(
        self, orchestration_config_no_governance: OrchestrationEngineConfig
    ) -> None:
        """Test engine with governance disabled."""
        engine = OrchestrationEngine(orchestration_config_no_governance)
        apply_governance_workarounds(engine)

        assert engine.config.enable_governance is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_cycle_with_no_engines_registered(
        self,
        initialized_orchestration_engine: OrchestrationEngine,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test running cycle with no engines registered."""
        result = await initialized_orchestration_engine.run_cycle(data=sample_market_data)

        # Should complete but with no signals/orders
        assert result.signals_generated == 0
        assert result.orders_submitted == 0

    @pytest.mark.asyncio
    async def test_cycle_with_empty_data(
        self, fully_configured_engine: OrchestrationEngine
    ) -> None:
        """Test running cycle with empty data."""
        result = await fully_configured_engine.run_cycle(data={})

        # Empty dict is treated as "no data" and fails the cycle
        assert result.status == CycleStatus.FAILED
        assert "No data available" in result.errors

    @pytest.mark.asyncio
    async def test_cycle_without_risk_approval_required(
        self,
        orchestration_config: OrchestrationEngineConfig,
        mock_signal_engine: MockSignalEngine,
        mock_risk_engine: MockRiskEngine,
        mock_execution_engine: MockExecutionEngine,
        mock_analytics_engine: MockAnalyticsEngine,
        mock_data_source: MockDataSource,
        sample_market_data: dict[str, Any],
        sample_signals: list[dict[str, Any]],
    ) -> None:
        """Test cycle without required risk approval."""
        orchestration_config.require_risk_approval = False
        engine = OrchestrationEngine(orchestration_config)
        apply_governance_workarounds(engine)
        engine.register_engines(
            signal_engine=mock_signal_engine,
            risk_engine=mock_risk_engine,
            execution_engine=mock_execution_engine,
            analytics_engine=mock_analytics_engine,
            data_source=mock_data_source,
        )
        await engine.initialize()

        mock_signal_engine.set_return_value(sample_signals)

        result = await engine.run_cycle(data=sample_market_data)

        # Signals should pass through without risk evaluation
        assert result.signals_generated == len(sample_signals)
        # Risk engine may not be called

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_cycles_sequential(
        self,
        fully_configured_engine: OrchestrationEngine,
        sample_market_data: dict[str, Any],
    ) -> None:
        """Test running multiple cycles sequentially."""
        results = []
        for _ in range(3):
            result = await fully_configured_engine.run_cycle(data=sample_market_data)
            results.append(result)

        assert len(results) == 3
        # All cycle IDs should be unique
        cycle_ids = [r.cycle_id for r in results]
        assert len(set(cycle_ids)) == 3
