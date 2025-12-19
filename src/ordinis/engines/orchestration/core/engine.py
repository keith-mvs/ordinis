"""
Orchestration Engine - Trading Pipeline Coordinator.

Implements the trading cycle per design doc:
1. Data fetch from StreamingBus
2. Signal generation via SignalCore
3. Risk evaluation
4. Order execution
5. Analytics recording

Supports live, paper, and backtest modes.

Phase 2 enhancements (2025-12-17):
- FeedbackCollector integration for cycle-level feedback
- Circuit breaker check before signal generation
- Error rate tracking across cycles
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from typing import TYPE_CHECKING, Any, Protocol
import uuid

from ordinis.engines.base import (
    AuditRecord,
    BaseEngine,
    HealthLevel,
    HealthStatus,
    PreflightContext,
)

from .config import OrchestrationEngineConfig
from .models import CycleResult, CycleStatus, PipelineMetrics, PipelineStage, StageResult

if TYPE_CHECKING:
    from ordinis.engines.base import GovernanceHook
    from ordinis.engines.learning.collectors.feedback import FeedbackCollector

_logger = logging.getLogger(__name__)


# Protocol definitions for engine interfaces
class SignalEngineProtocol(Protocol):
    """Protocol for signal engine interface."""

    async def generate_signals(self, data: Any) -> list[Any]:
        """Generate trading signals from market data."""
        ...


class RiskEngineProtocol(Protocol):
    """Protocol for risk engine interface."""

    async def evaluate(self, signals: list[Any]) -> tuple[list[Any], list[str]]:
        """Evaluate signals and return approved signals with rejection reasons."""
        ...


class ExecutionEngineProtocol(Protocol):
    """Protocol for execution engine interface."""

    async def execute(self, orders: list[Any]) -> list[Any]:
        """Execute orders and return fill results."""
        ...


class AnalyticsEngineProtocol(Protocol):
    """Protocol for analytics engine interface."""

    async def record(self, results: list[Any]) -> None:
        """Record execution results."""
        ...


class PortfolioEngineProtocol(Protocol):
    """Protocol for portfolio engine interface."""

    async def update(self, fills: list[Any]) -> None:
        """Update portfolio with trade fills."""
        ...

    async def get_state(self) -> Any:
        """Get current portfolio state."""
        ...


class LearningEngineProtocol(Protocol):
    """Protocol for learning engine interface."""

    async def update(self, results: list[Any]) -> None:
        """Update models based on results."""
        ...


class DataSourceProtocol(Protocol):
    """Protocol for data source (StreamingBus) interface."""

    async def get_latest(self, symbols: list[str] | None = None) -> dict[str, Any]:
        """Get latest market data for symbols."""
        ...


@dataclass
class PipelineEngines:
    """Container for pipeline engine references."""

    signal_engine: SignalEngineProtocol | None = None
    risk_engine: RiskEngineProtocol | None = None
    execution_engine: ExecutionEngineProtocol | None = None
    analytics_engine: AnalyticsEngineProtocol | None = None
    portfolio_engine: PortfolioEngineProtocol | None = None
    learning_engine: LearningEngineProtocol | None = None
    data_source: DataSourceProtocol | None = None


class OrchestrationEngine(BaseEngine[OrchestrationEngineConfig]):
    """
    Trading pipeline orchestration engine.

    Coordinates the flow of data through the trading pipeline:
    Data → Signals → Risk → Execution → Analytics (ProofBench)

    Supports three operating modes:
    - live: Real-time trading with broker integration
    - paper: Simulated trading with real market data
    - backtest: Historical simulation

    Example:
        >>> config = OrchestrationEngineConfig(mode="paper")
        >>> engine = OrchestrationEngine(config)
        >>> await engine.initialize()
        >>> engine.register_engines(signal_engine, risk_engine, execution_engine)
        >>> result = await engine.run_cycle(symbols=["AAPL", "MSFT"])
    """

    def __init__(
        self,
        config: OrchestrationEngineConfig | None = None,
        governance_hook: GovernanceHook | None = None,
        feedback_collector: FeedbackCollector | None = None,
    ) -> None:
        """Initialize the OrchestrationEngine."""
        config = config or OrchestrationEngineConfig()
        super().__init__(config, governance_hook)

        self._engines = PipelineEngines()
        self._metrics = PipelineMetrics()
        self._cycle_history: list[CycleResult] = []
        self._running = False
        self._last_cycle_time: datetime | None = None
        self._feedback = feedback_collector

    async def _do_initialize(self) -> None:
        """Initialize engine resources."""
        _logger.info(
            "OrchestrationEngine initialized in %s mode",
            self.config.mode,
        )

    async def _do_shutdown(self) -> None:
        """Shutdown engine resources."""
        self._running = False
        _logger.info("OrchestrationEngine shutdown complete")

    async def _do_health_check(self) -> HealthStatus:
        """Check engine health."""
        missing_engines: list[str] = []
        if not self._engines.signal_engine:
            missing_engines.append("signal_engine")
        if not self._engines.risk_engine:
            missing_engines.append("risk_engine")
        if not self._engines.execution_engine:
            missing_engines.append("execution_engine")
        if not self._engines.portfolio_engine:
            missing_engines.append("portfolio_engine")

        if missing_engines:
            return HealthStatus(
                level=HealthLevel.DEGRADED,
                message=f"Missing engines: {', '.join(missing_engines)}",
                details={"missing_engines": missing_engines},
            )

        return HealthStatus(
            level=HealthLevel.HEALTHY,
            message="OrchestrationEngine operational",
            details={
                "mode": self.config.mode,
                "total_cycles": self._metrics.total_cycles,
                "success_rate": (
                    self._metrics.successful_cycles / self._metrics.total_cycles
                    if self._metrics.total_cycles > 0
                    else 0
                ),
            },
        )

    # -------------------------------------------------------------------------
    # Engine Registration
    # -------------------------------------------------------------------------

    def register_signal_engine(self, engine: SignalEngineProtocol) -> None:
        """Register the signal generation engine."""
        self._engines.signal_engine = engine
        _logger.info("Signal engine registered")

    def register_risk_engine(self, engine: RiskEngineProtocol) -> None:
        """Register the risk evaluation engine."""
        self._engines.risk_engine = engine
        _logger.info("Risk engine registered")

    def register_execution_engine(self, engine: ExecutionEngineProtocol) -> None:
        """Register the order execution engine."""
        self._engines.execution_engine = engine
        _logger.info("Execution engine registered")

    def register_analytics_engine(self, engine: AnalyticsEngineProtocol) -> None:
        """Register the analytics recording engine."""
        self._engines.analytics_engine = engine
        _logger.info("Analytics engine registered")

    def register_portfolio_engine(self, engine: PortfolioEngineProtocol) -> None:
        """Register the portfolio management engine."""
        self._engines.portfolio_engine = engine
        _logger.info("Portfolio engine registered")

    def register_learning_engine(self, engine: LearningEngineProtocol) -> None:
        """Register the learning engine."""
        self._engines.learning_engine = engine
        _logger.info("Learning engine registered")

    def register_data_source(self, source: DataSourceProtocol) -> None:
        """Register the data source (e.g., StreamingBus)."""
        self._engines.data_source = source
        _logger.info("Data source registered")

    def register_engines(
        self,
        signal_engine: SignalEngineProtocol | None = None,
        risk_engine: RiskEngineProtocol | None = None,
        execution_engine: ExecutionEngineProtocol | None = None,
        analytics_engine: AnalyticsEngineProtocol | None = None,
        portfolio_engine: PortfolioEngineProtocol | None = None,
        learning_engine: LearningEngineProtocol | None = None,
        data_source: DataSourceProtocol | None = None,
    ) -> None:
        """Register multiple engines at once."""
        if signal_engine:
            self.register_signal_engine(signal_engine)
        if risk_engine:
            self.register_risk_engine(risk_engine)
        if execution_engine:
            self.register_execution_engine(execution_engine)
        if analytics_engine:
            self.register_analytics_engine(analytics_engine)
        if portfolio_engine:
            self.register_portfolio_engine(portfolio_engine)
        if learning_engine:
            self.register_learning_engine(learning_engine)
        if data_source:
            self.register_data_source(data_source)

    # -------------------------------------------------------------------------
    # Trading Cycle
    # -------------------------------------------------------------------------

    async def run_cycle(
        self,
        symbols: list[str] | None = None,
        data: dict[str, Any] | None = None,
    ) -> CycleResult:
        """
        Execute a single trading cycle.

        Per design doc:
        1. enriched = bus.get_latest(symbol)
        2. signals = signal_engine.generate(enriched)
        3. approved = risk_engine.evaluate(signals)
        4. results = execution_engine.execute(approved)
        5. analytics_engine.record(results)

        Args:
            symbols: Symbols to process (optional if data provided).
            data: Pre-fetched market data (optional).

        Returns:
            CycleResult with timing and outcome details.
        """
        result = CycleResult(status=CycleStatus.RUNNING)
        cycle_start = datetime.now(UTC)

        context = {
            "operation": "run_cycle",
            "mode": self.config.mode,
            "symbols": symbols,
        }

        try:
            async with self.track_operation("run_cycle", context):
                # Phase 2: Circuit breaker check before any processing
                # This prevents the 12/17 issue where 6,000+ signals were generated
                # despite zero buying power
                if self._feedback:
                    allowed, reason = self._feedback.should_allow_signals()
                    if not allowed:
                        _logger.warning(f"Cycle blocked by circuit breaker: {reason}")
                        result.status = CycleStatus.SKIPPED
                        result.errors.append(f"Circuit breaker: {reason}")
                        return result

                # Governance preflight
                if self.config.enable_governance and self._governance:
                    preflight_ctx = PreflightContext(
                        engine=self.config.engine_id,
                        action="run_cycle",
                        inputs=context,
                        trace_id=str(uuid.uuid4()),
                    )
                    preflight = await self._governance.preflight(preflight_ctx)
                    if not preflight.allowed:
                        result.status = CycleStatus.SKIPPED
                        result.errors.append(f"Governance blocked: {preflight.reason}")
                        return result

                # Stage 1: Data Fetch
                stage_start = datetime.now(UTC)
                if data is None:
                    data = await self._fetch_data(symbols)
                stage_duration = (datetime.now(UTC) - stage_start).total_seconds() * 1000
                result.add_stage(
                    StageResult(
                        stage=PipelineStage.DATA_FETCH,
                        success=bool(data),
                        duration_ms=stage_duration,
                        output_count=len(data) if data else 0,
                    )
                )

                if not data:
                    result.status = CycleStatus.FAILED
                    result.errors.append("No data available")
                    return result

                # Stage 2: Signal Generation
                stage_start = datetime.now(UTC)
                signals = await self._generate_signals(data)
                stage_duration = (datetime.now(UTC) - stage_start).total_seconds() * 1000
                result.add_stage(
                    StageResult(
                        stage=PipelineStage.SIGNAL_GENERATION,
                        success=True,
                        duration_ms=stage_duration,
                        output_count=len(signals),
                    )
                )
                result.signals_generated = len(signals)

                if not signals:
                    result.status = CycleStatus.COMPLETED
                    result.completed_at = datetime.now(UTC)
                    result.total_duration_ms = (
                        result.completed_at - cycle_start
                    ).total_seconds() * 1000
                    return result

                # Stage 3: Risk Evaluation
                stage_start = datetime.now(UTC)
                approved, rejections = await self._evaluate_risk(signals)
                stage_duration = (datetime.now(UTC) - stage_start).total_seconds() * 1000
                result.add_stage(
                    StageResult(
                        stage=PipelineStage.RISK_EVALUATION,
                        success=True,
                        duration_ms=stage_duration,
                        output_count=len(approved),
                        details={"rejections": rejections},
                    )
                )
                result.signals_approved = len(approved)

                if not approved:
                    result.status = CycleStatus.COMPLETED
                    result.completed_at = datetime.now(UTC)
                    result.total_duration_ms = (
                        result.completed_at - cycle_start
                    ).total_seconds() * 1000
                    return result

                # Stage 4: Order Execution
                stage_start = datetime.now(UTC)
                fills = await self._execute_orders(approved)
                stage_duration = (datetime.now(UTC) - stage_start).total_seconds() * 1000
                result.add_stage(
                    StageResult(
                        stage=PipelineStage.ORDER_EXECUTION,
                        success=True,
                        duration_ms=stage_duration,
                        output_count=len(fills),
                    )
                )
                result.orders_submitted = len(approved)
                result.orders_filled = len([f for f in fills if f.get("filled", False)])
                result.orders_rejected = len([f for f in fills if f.get("rejected", False)])

                # Stage 5: Portfolio Update
                if self._engines.portfolio_engine and fills:
                    stage_start = datetime.now(UTC)
                    await self._engines.portfolio_engine.update(fills)
                    stage_duration = (datetime.now(UTC) - stage_start).total_seconds() * 1000
                    # Note: We might want to add a PORTFOLIO_UPDATE stage to PipelineStage enum later
                    # For now, we just log it or track it implicitly

                # Stage 6: Analytics Recording
                if self.config.enable_analytics_recording:
                    stage_start = datetime.now(UTC)
                    await self._record_analytics(fills)
                    stage_duration = (datetime.now(UTC) - stage_start).total_seconds() * 1000
                    result.add_stage(
                        StageResult(
                            stage=PipelineStage.ANALYTICS_RECORDING,
                            success=True,
                            duration_ms=stage_duration,
                            output_count=len(fills),
                        )
                    )

                # Stage 7: Learning Update
                if self._engines.learning_engine:
                    # We can pass the entire cycle result or specific parts
                    # For now, let's assume we pass the fills and signals for learning
                    learning_data = {"signals": signals, "fills": fills, "cycle_result": result}
                    # The protocol expects a list, so we wrap it or adapt it
                    await self._engines.learning_engine.update([learning_data])

                result.status = CycleStatus.COMPLETED

        except Exception as e:
            _logger.exception("Cycle failed")
            result.status = CycleStatus.FAILED
            result.errors.append(str(e))

        finally:
            result.completed_at = datetime.now(UTC)
            result.total_duration_ms = (result.completed_at - cycle_start).total_seconds() * 1000
            self._cycle_history.append(result)
            self._metrics.update_from_cycle(result)
            self._last_cycle_time = result.completed_at

            # Phase 2: Record trading cycle to FeedbackCollector
            if self._feedback:
                try:
                    await self._feedback.record_trading_cycle(
                        cycle_id=str(uuid.uuid4()),
                        duration_ms=result.total_duration_ms,
                        signals_generated=result.signals_generated,
                        orders_submitted=result.orders_submitted,
                        orders_filled=result.orders_filled,
                        orders_rejected=result.orders_rejected,
                        errors=[{"error": e} for e in result.errors] if result.errors else None,
                    )
                except Exception as feedback_err:
                    _logger.error(f"Failed to record trading cycle: {feedback_err}")

            # Audit the cycle
            if self._governance:
                await self._governance.audit(
                    AuditRecord(
                        engine=self.config.engine_id,
                        action="run_cycle",
                        inputs=context,
                        outputs=result.to_dict(),
                        latency_ms=result.total_duration_ms,
                    )
                )

        return result

    # -------------------------------------------------------------------------
    # Pipeline Stage Implementations
    # -------------------------------------------------------------------------

    async def _fetch_data(self, symbols: list[str] | None) -> dict[str, Any]:
        """Fetch latest market data."""
        if self._engines.data_source:
            return await self._engines.data_source.get_latest(symbols)
        # Placeholder if no data source registered
        return {}

    async def _generate_signals(self, data: dict[str, Any]) -> list[Any]:
        """Generate trading signals from market data."""
        if self._engines.signal_engine:
            return await self._engines.signal_engine.generate_signals(data)
        return []

    async def _evaluate_risk(self, signals: list[Any]) -> tuple[list[Any], list[str]]:
        """Evaluate signals through risk engine."""
        if self._engines.risk_engine and self.config.require_risk_approval:
            return await self._engines.risk_engine.evaluate(signals)
        # Pass through if no risk engine or approval not required
        return signals, []

    async def _execute_orders(self, orders: list[Any]) -> list[Any]:
        """Execute approved orders."""
        if self._engines.execution_engine:
            return await self._engines.execution_engine.execute(orders)
        return []

    async def _record_analytics(self, results: list[Any]) -> None:
        """Record results to analytics engine."""
        if self._engines.analytics_engine:
            await self._engines.analytics_engine.record(results)

    # -------------------------------------------------------------------------
    # Continuous Operation
    # -------------------------------------------------------------------------

    async def run_loop(
        self,
        symbols: list[str] | None = None,
        max_cycles: int | None = None,
    ) -> None:
        """
        Run continuous trading loop.

        Args:
            symbols: Symbols to trade.
            max_cycles: Maximum cycles to run (None = infinite).
        """
        self._running = True
        cycle_count = 0
        min_interval = self.config.cycle_interval_ms / 1000

        _logger.info("Starting trading loop")

        while self._running:
            if max_cycles and cycle_count >= max_cycles:
                _logger.info("Max cycles reached: %d", max_cycles)
                break

            cycle_start = datetime.now(UTC)

            # Run cycle
            await self.run_cycle(symbols=symbols)
            cycle_count += 1

            # Enforce minimum interval
            elapsed = (datetime.now(UTC) - cycle_start).total_seconds()
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

        _logger.info("Trading loop stopped after %d cycles", cycle_count)

    def stop_loop(self) -> None:
        """Signal the trading loop to stop."""
        self._running = False

    # -------------------------------------------------------------------------
    # Backtest Support
    # -------------------------------------------------------------------------

    async def run_backtest(
        self,
        historical_data: list[dict[str, Any]],
    ) -> list[CycleResult]:
        """
        Run backtest over historical data.

        Args:
            historical_data: List of market data snapshots in chronological order.

        Returns:
            List of cycle results for analysis.
        """
        if self.config.mode != "backtest":
            _logger.warning("Running backtest in %s mode", self.config.mode)

        results: list[CycleResult] = []

        for i, data_snapshot in enumerate(historical_data):
            _logger.debug("Backtest cycle %d/%d", i + 1, len(historical_data))
            result = await self.run_cycle(data=data_snapshot)
            results.append(result)

        _logger.info(
            "Backtest completed: %d cycles, %d successful",
            len(results),
            sum(1 for r in results if r.status == CycleStatus.COMPLETED),
        )

        return results

    # -------------------------------------------------------------------------
    # Metrics and History
    # -------------------------------------------------------------------------

    def get_metrics(self) -> PipelineMetrics:
        """Get aggregated pipeline metrics."""
        return self._metrics

    def get_cycle_history(self, limit: int = 100) -> list[CycleResult]:
        """Get recent cycle history."""
        return self._cycle_history[-limit:]

    def get_last_cycle(self) -> CycleResult | None:
        """Get the most recent cycle result."""
        return self._cycle_history[-1] if self._cycle_history else None

    def clear_history(self) -> None:
        """Clear cycle history (for testing)."""
        self._cycle_history.clear()
        self._metrics = PipelineMetrics()
