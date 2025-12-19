"""
Portfolio engine configuration.

Provides PortfolioEngineConfig extending BaseEngineConfig for standardized
engine configuration with portfolio-specific settings.

Phase 2 enhancements (2025-12-19):
- PortfolioOptAdapter integration settings (drift bands, calendar rebalancing)
- Transaction cost model configuration
- Execution feedback settings
"""

from dataclasses import dataclass, field
from enum import Enum

from ordinis.engines.base import BaseEngineConfig
from ordinis.engines.portfolio.core.models import StrategyType


class DriftBandType(Enum):
    """Types of drift measurement for rebalancing triggers."""

    ABSOLUTE = "absolute"  # Percentage point deviation (5% = 25% → 30%)
    RELATIVE = "relative"  # Relative deviation (5% of 25% = 1.25%)


class CalendarPeriod(Enum):
    """Calendar rebalancing frequencies."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class DriftBandSettings:
    """Settings for drift-band rebalancing (Alpaca-style).

    Attributes:
        enabled: Whether drift-triggered rebalancing is enabled
        drift_type: Absolute or relative drift measurement
        threshold_pct: Drift threshold percentage to trigger rebalance
        cooldown_days: Minimum days between drift-triggered rebalances
    """

    enabled: bool = True
    drift_type: DriftBandType = DriftBandType.ABSOLUTE
    threshold_pct: float = 5.0  # 5% default threshold
    cooldown_days: int = 7  # 7-day default cooldown


@dataclass
class CalendarSettings:
    """Settings for calendar-based rebalancing.

    Attributes:
        enabled: Whether calendar rebalancing is enabled
        period: Rebalancing frequency
        day_of_week: Day for weekly rebalance (0=Monday)
        day_of_month: Day for monthly/quarterly rebalance
    """

    enabled: bool = False
    period: CalendarPeriod = CalendarPeriod.MONTHLY
    day_of_week: int = 0  # Monday
    day_of_month: int = 1  # First of month


@dataclass
class TransactionCostSettings:
    """Settings for transaction cost modeling.

    Attributes:
        model_type: Cost model type ('simple', 'almgren_chriss', 'adaptive')
        fixed_cost_bps: Fixed cost in basis points (for simple model)
        market_impact_coeff: Almgren-Chriss gamma coefficient
        adaptive_alpha: Exponential smoothing factor for adaptive model
        min_spread_bps: Minimum bid-ask spread assumption
    """

    model_type: str = "adaptive"
    fixed_cost_bps: float = 10.0  # 10 bps default
    market_impact_coeff: float = 0.1  # Almgren-Chriss gamma
    adaptive_alpha: float = 0.3  # Exponential smoothing
    min_spread_bps: float = 1.0  # 1 bp minimum spread


@dataclass
class ExecutionFeedbackSettings:
    """Settings for execution feedback collection.

    Attributes:
        enabled: Whether execution feedback is collected
        slippage_warning_bps: Slippage threshold for warnings
        slippage_critical_bps: Slippage threshold for sizing reduction
        sizing_reduction_factor: Factor to reduce sizing on poor execution
        feedback_window_hours: Rolling window for feedback analysis
    """

    enabled: bool = True
    slippage_warning_bps: float = 15.0  # 15 bps warning
    slippage_critical_bps: float = 30.0  # 30 bps triggers sizing reduction
    sizing_reduction_factor: float = 0.8  # Reduce to 80% on poor execution
    feedback_window_hours: int = 24  # 24-hour rolling window


@dataclass
class PortfolioEngineConfig(BaseEngineConfig):
    """Configuration for portfolio rebalancing engine.

    Extends BaseEngineConfig with portfolio-specific settings including
    strategy selection, history tracking, execution parameters, and
    integrated PortfolioOpt adapter configuration.

    Attributes:
        engine_id: Unique identifier (default: "portfolio")
        engine_name: Display name (default: "Portfolio Rebalancing Engine")
        default_strategy: Default rebalancing strategy to use
        track_history: Whether to maintain rebalancing history
        max_history_entries: Maximum history entries to retain (0 = unlimited)
        min_trade_value: Minimum dollar value for execution
        max_position_pct: Maximum single position as % of portfolio
        enable_governance: Enable governance hooks for all operations

        # PortfolioOpt Adapter Settings
        cash_reserve_pct: Target cash reserve as percentage (0-100)
        min_weight_pct: Minimum weight to include in allocation
        drift_band: Drift band rebalancing settings
        calendar: Calendar-based rebalancing settings
        transaction_costs: Transaction cost model settings
        execution_feedback: Execution feedback collection settings
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

    # PortfolioOpt Adapter Settings (Phase 2)
    cash_reserve_pct: float = 5.0  # 5% cash reserve
    min_weight_pct: float = 0.1  # 0.1% minimum weight

    # Rebalancing trigger settings
    drift_band: DriftBandSettings = field(default_factory=DriftBandSettings)
    calendar: CalendarSettings = field(default_factory=CalendarSettings)

    # Cost and feedback settings
    transaction_costs: TransactionCostSettings = field(
        default_factory=TransactionCostSettings
    )
    execution_feedback: ExecutionFeedbackSettings = field(
        default_factory=ExecutionFeedbackSettings
    )

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

        # Validate PortfolioOpt settings
        if not 0 <= self.cash_reserve_pct <= 50:
            errors.append("cash_reserve_pct must be between 0 and 50")

        if not 0 <= self.min_weight_pct <= 10:
            errors.append("min_weight_pct must be between 0 and 10")

        if self.drift_band.threshold_pct <= 0:
            errors.append("drift_band.threshold_pct must be positive")

        if self.drift_band.cooldown_days < 0:
            errors.append("drift_band.cooldown_days must be non-negative")

        if self.transaction_costs.fixed_cost_bps < 0:
            errors.append("transaction_costs.fixed_cost_bps must be non-negative")

        if not 0 < self.execution_feedback.sizing_reduction_factor <= 1:
            errors.append(
                "execution_feedback.sizing_reduction_factor must be between 0 and 1"
            )

        return errors
