"""
ProofBench engine configuration.

Provides ProofBenchEngineConfig extending BaseEngineConfig for standardized
engine configuration with backtesting-specific settings.
"""

from dataclasses import dataclass, field

from ordinis.engines.base import BaseEngineConfig
from ordinis.engines.proofbench.core.execution import ExecutionConfig


@dataclass
class ProofBenchEngineConfig(BaseEngineConfig):
    """Configuration for ProofBench backtesting engine.

    Extends BaseEngineConfig with backtesting-specific settings including
    simulation parameters, execution modeling, and governance controls.

    Attributes:
        engine_id: Unique identifier (default: "proofbench")
        engine_name: Display name (default: "ProofBench Backtesting Engine")
        initial_capital: Starting capital for simulations
        execution_config: Execution simulator configuration
        bar_frequency: Bar frequency (e.g., '1min', '1h', '1d')
        record_equity_frequency: How often to record equity (bars)
        risk_free_rate: Annual risk-free rate for metrics
        enable_logging: Enable detailed simulation logging
        enable_governance: Enable governance hooks for all operations
    """

    engine_id: str = "proofbench"
    engine_name: str = "ProofBench Backtesting Engine"

    # Simulation settings
    initial_capital: float = 100000.0
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    bar_frequency: str = "1d"
    record_equity_frequency: int = 1
    risk_free_rate: float = 0.02
    enable_logging: bool = False

    # Governance
    enable_governance: bool = True

    # Analytics Configuration
    storage_path: str = "data/analytics"
    metrics_retention_days: int = 30
    real_time_calculation: bool = True
    performance_metrics: list[str] = field(default_factory=lambda: ["pnl", "win_rate", "drawdown"])

    # Tracking
    loaded_symbols: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate proofbench engine configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = super().validate()

        if self.initial_capital <= 0:
            errors.append("initial_capital must be positive")

        if self.record_equity_frequency <= 0:
            errors.append("record_equity_frequency must be positive")

        if self.risk_free_rate < 0:
            errors.append("risk_free_rate cannot be negative")

        return errors
