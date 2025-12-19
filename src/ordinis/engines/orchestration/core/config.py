"""
Orchestration Engine Configuration.

Defines configuration for the trading pipeline orchestrator
that coordinates signals, risk, execution, and analytics.
"""

from dataclasses import dataclass
from typing import Literal

from ordinis.engines.base import BaseEngineConfig


@dataclass
class OrchestrationEngineConfig(BaseEngineConfig):
    """
    Configuration for the OrchestrationEngine.

    Attributes:
        engine_id: Unique identifier for the engine.
        engine_name: Display name for the engine.
        mode: Operating mode (live, paper, backtest).
        cycle_interval_ms: Minimum interval between cycles in milliseconds.
        max_signals_per_cycle: Maximum signals to process per cycle.
        enable_governance: Whether to enable governance hooks.
        require_risk_approval: Require risk engine approval for all orders.
        enable_analytics_recording: Record all results to analytics engine.
    """

    engine_id: str = "orchestration"
    engine_name: str = "Orchestration Engine"

    # Operating mode
    mode: Literal["live", "paper", "backtest"] = "paper"

    # Pipeline settings
    cycle_interval_ms: int = 100  # 100ms minimum between cycles
    max_signals_per_cycle: int = 100
    signal_batch_enabled: bool = True
    parallel_execution: bool = False

    # Risk settings
    require_risk_approval: bool = True
    risk_timeout_ms: int = 50

    # Execution settings
    execution_timeout_ms: int = 500
    max_orders_per_cycle: int = 10

    # Analytics settings
    enable_analytics_recording: bool = True
    analytics_batch_size: int = 100

    # Governance
    enable_governance: bool = True

    # Latency budgets (from design doc)
    data_pipeline_budget_ms: int = 100
    anomaly_detection_budget_ms: int = 50
    feature_engineering_budget_ms: int = 20
    signal_generation_budget_ms: int = 100
    risk_checks_budget_ms: int = 10
    order_routing_budget_ms: int = 60
    total_budget_ms: int = 300  # Sum of above

    def validate(self) -> list[str]:
        """Validate configuration parameters."""
        errors = super().validate()

        if self.cycle_interval_ms < 10:
            errors.append("cycle_interval_ms should be at least 10ms")
        if self.max_signals_per_cycle < 1:
            errors.append("max_signals_per_cycle must be positive")
        if self.risk_timeout_ms < 1:
            errors.append("risk_timeout_ms must be positive")
        if self.execution_timeout_ms < 1:
            errors.append("execution_timeout_ms must be positive")
        if self.total_budget_ms < self.cycle_interval_ms:
            errors.append("total_budget_ms should be >= cycle_interval_ms")

        return errors
