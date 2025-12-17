"""
Portfolio engine configuration.

Provides PortfolioEngineConfig extending BaseEngineConfig for standardized
engine configuration with portfolio-specific settings.
"""

from dataclasses import dataclass, field

from ordinis.engines.base import BaseEngineConfig
from ordinis.engines.portfolio.core.models import StrategyType


@dataclass
class PortfolioEngineConfig(BaseEngineConfig):
    """Configuration for portfolio rebalancing engine.

    Extends BaseEngineConfig with portfolio-specific settings including
    strategy selection, history tracking, and execution parameters.

    Attributes:
        engine_id: Unique identifier (default: "portfolio")
        engine_name: Display name (default: "Portfolio Rebalancing Engine")
        default_strategy: Default rebalancing strategy to use
        track_history: Whether to maintain rebalancing history
        max_history_entries: Maximum history entries to retain (0 = unlimited)
        min_trade_value: Minimum dollar value for execution
        max_position_pct: Maximum single position as % of portfolio
        enable_governance: Enable governance hooks for all operations
    """

    engine_id: str = "portfolio"
    engine_name: str = "Portfolio Rebalancing Engine"

    # Strategy settings
    default_strategy: StrategyType = StrategyType.TARGET_ALLOCATION
    track_history: bool = True
    max_history_entries: int = 1000

    # Execution settings
    initial_capital: float = 100000.0
    min_trade_value: float = 10.0
    max_position_pct: float = 0.25

    # Governance
    enable_governance: bool = True

    # Registered strategies (populated at runtime)
    registered_strategies: list[StrategyType] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate portfolio engine configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = super().validate()

        if self.min_trade_value < 0:
            errors.append("min_trade_value must be non-negative")

        if not 0 < self.max_position_pct <= 1.0:
            errors.append("max_position_pct must be between 0 and 1")

        if self.max_history_entries < 0:
            errors.append("max_history_entries must be non-negative")

        return errors
