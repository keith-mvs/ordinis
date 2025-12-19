"""
Orchestration Engine - Trading Pipeline Coordinator.

Implements the trading cycle workflow:
1. Data fetch from StreamingBus
2. Signal generation via SignalCore
3. Risk evaluation
4. Order execution
5. Analytics recording

Supports live, paper, and backtest modes.

Example:
    >>> from ordinis.engines.orchestration import (
    ...     OrchestrationEngine,
    ...     OrchestrationEngineConfig,
    ... )
    >>> config = OrchestrationEngineConfig(mode="paper")
    >>> engine = OrchestrationEngine(config)
    >>> await engine.initialize()
    >>> engine.register_engines(signal_engine, risk_engine, execution_engine)
    >>> result = await engine.run_cycle(symbols=["AAPL", "MSFT"])
"""

from .core import (
    AnalyticsEngineProtocol,
    CycleResult,
    CycleStatus,
    DataSourceProtocol,
    ExecutionEngineProtocol,
    OrchestrationEngine,
    OrchestrationEngineConfig,
    PipelineEngines,
    PipelineMetrics,
    PipelineStage,
    RiskEngineProtocol,
    SignalEngineProtocol,
    StageResult,
)
from .hooks import (
    LatencyBudgetRule,
    OrchestrationGovernanceHook,
    TradingModeRule,
)

__all__ = [
    # Core
    "AnalyticsEngineProtocol",
    "CycleResult",
    "CycleStatus",
    "DataSourceProtocol",
    "ExecutionEngineProtocol",
    # Hooks
    "LatencyBudgetRule",
    "OrchestrationEngine",
    "OrchestrationEngineConfig",
    "OrchestrationGovernanceHook",
    "PipelineEngines",
    "PipelineMetrics",
    "PipelineStage",
    "RiskEngineProtocol",
    "SignalEngineProtocol",
    "StageResult",
    "TradingModeRule",
]
