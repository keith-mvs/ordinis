"""
Threshold-Based Rebalancing Strategy.

Combines multiple threshold types to determine optimal rebalancing timing:
- Drift bands (corridor rebalancing)
- Time-based constraints (minimum days between rebalances)
- Transaction cost thresholds
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd


@dataclass
class ThresholdConfig:
    """Configuration for threshold-based rebalancing.

    Attributes:
        symbol: Ticker symbol
        target_weight: Target allocation as decimal (0.0 to 1.0)
        lower_band: Lower drift threshold (e.g., -0.05 = -5%)
        upper_band: Upper drift threshold (e.g., +0.05 = +5%)
    """

    symbol: str
    target_weight: float
    lower_band: float = -0.05
    upper_band: float = 0.05

    def __post_init__(self) -> None:
        """Validate threshold configuration."""
        if not 0.0 <= self.target_weight <= 1.0:
            raise ValueError(f"target_weight must be in [0, 1], got {self.target_weight}")
        if self.lower_band >= 0:
            raise ValueError(f"lower_band must be negative, got {self.lower_band}")
        if self.upper_band <= 0:
            raise ValueError(f"upper_band must be positive, got {self.upper_band}")
        if abs(self.lower_band) > 1.0 or self.upper_band > 1.0:
            raise ValueError("Bands cannot exceed ±1.0 (±100%)")


@dataclass
class ThresholdStatus:
    """Current status of threshold conditions.

    Attributes:
        symbol: Ticker symbol
        current_weight: Current allocation
        target_weight: Target allocation
        drift: Current drift from target
        breaches_lower: True if drift below lower band
        breaches_upper: True if drift above upper band
        days_since_rebalance: Days since last rebalance
        exceeds_time_threshold: True if time threshold exceeded
        should_rebalance: True if ANY threshold is breached
    """

    symbol: str
    current_weight: float
    target_weight: float
    drift: float
    breaches_lower: bool
    breaches_upper: bool
    days_since_rebalance: int
    exceeds_time_threshold: bool
    should_rebalance: bool


@dataclass
class ThresholdDecision:
    """Rebalancing decision for threshold-based strategy.

    Attributes:
        symbol: Ticker symbol
        current_weight: Current allocation
        target_weight: Target allocation
        drift: Drift from target
        adjustment_shares: Number of shares to buy (+) or sell (-)
        adjustment_value: Dollar value of adjustment
        trigger_reason: Why rebalancing was triggered
        timestamp: When the decision was generated
    """

    symbol: str
    current_weight: float
    target_weight: float
    drift: float
    adjustment_shares: float
    adjustment_value: float
    trigger_reason: str
    timestamp: datetime


class ThresholdBasedRebalancer:
    """Rebalances portfolio using sophisticated threshold logic.

    Combines multiple threshold types:
    - Drift bands (corridor rebalancing): Only rebalance when outside bands
    - Time-based: Minimum days between rebalances
    - Transaction cost: Minimum trade size to execute

    Example:
        >>> configs = [
        ...     ThresholdConfig("AAPL", 0.40, lower_band=-0.05, upper_band=0.05),
        ...     ThresholdConfig("MSFT", 0.30, lower_band=-0.05, upper_band=0.05),
        ...     ThresholdConfig("GOOGL", 0.30, lower_band=-0.05, upper_band=0.05),
        ... ]
        >>> rebalancer = ThresholdBasedRebalancer(
        ...     threshold_configs=configs,
        ...     min_days_between_rebalance=30,
        ...     min_trade_value=100.0,
        ... )
        >>> positions = {"AAPL": 10, "MSFT\": 5, "GOOGL": 3}
        >>> prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 100.0}
        >>> last_rebalance = datetime(2023, 11, 1)
        >>> decisions = rebalancer.generate_rebalance_orders(positions, prices, last_rebalance)
    """

    def __init__(
        self,
        threshold_configs: list[ThresholdConfig],
        min_days_between_rebalance: int = 0,
        min_trade_value: float = 0.0,
    ) -> None:
        """Initialize threshold-based rebalancer.

        Args:
            threshold_configs: List of ThresholdConfig objects
            min_days_between_rebalance: Minimum days between rebalances (0 = no constraint)
            min_trade_value: Minimum trade value to execute (0 = no constraint)

        Raises:
            ValueError: If configuration is invalid
        """
        if min_days_between_rebalance < 0:
            raise ValueError(
                f"min_days_between_rebalance must be >= 0, got {min_days_between_rebalance}"
            )
        if min_trade_value < 0:
            raise ValueError(f"min_trade_value must be >= 0, got {min_trade_value}")

        self.configs = {c.symbol: c for c in threshold_configs}
        self.min_days_between_rebalance = min_days_between_rebalance
        self.min_trade_value = min_trade_value

        # Validate weights sum to 1.0
        total_weight = sum(c.target_weight for c in threshold_configs)
        if not 0.999 <= total_weight <= 1.001:
            raise ValueError(f"Target weights must sum to 1.0, got {total_weight:.4f}")

    def check_thresholds(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        last_rebalance_date: datetime | None = None,
    ) -> list[ThresholdStatus]:
        """Check threshold status for all symbols.

        Args:
            positions: Current shares held per symbol
            prices: Current price per symbol
            last_rebalance_date: Date of last rebalance (None = never rebalanced)

        Returns:
            List of ThresholdStatus objects
        """
        # Calculate current weights
        total_value = sum(positions.get(sym, 0.0) * prices.get(sym, 0.0) for sym in self.configs)

        if total_value == 0.0:
            # Empty portfolio - all symbols need rebalancing
            return [
                ThresholdStatus(
                    symbol=sym,
                    current_weight=0.0,
                    target_weight=config.target_weight,
                    drift=-config.target_weight,
                    breaches_lower=True,
                    breaches_upper=False,
                    days_since_rebalance=999999,
                    exceeds_time_threshold=True,
                    should_rebalance=True,
                )
                for sym, config in self.configs.items()
            ]

        current_weights = {
            sym: (positions.get(sym, 0.0) * prices.get(sym, 0.0)) / total_value
            for sym in self.configs
        }

        # Calculate days since last rebalance
        if last_rebalance_date is None:
            days_since = 999999  # Never rebalanced
        else:
            days_since = (datetime.now(tz=UTC) - last_rebalance_date).days

        exceeds_time_threshold = days_since >= self.min_days_between_rebalance

        # Check each symbol
        statuses: list[ThresholdStatus] = []
        for sym, config in self.configs.items():
            current_weight = current_weights[sym]
            drift = current_weight - config.target_weight

            breaches_lower = drift < config.lower_band
            breaches_upper = drift > config.upper_band

            # Should rebalance if EITHER drift OR time threshold exceeded
            should_rebalance = (breaches_lower or breaches_upper) and exceeds_time_threshold

            statuses.append(
                ThresholdStatus(
                    symbol=sym,
                    current_weight=current_weight,
                    target_weight=config.target_weight,
                    drift=drift,
                    breaches_lower=breaches_lower,
                    breaches_upper=breaches_upper,
                    days_since_rebalance=days_since,
                    exceeds_time_threshold=exceeds_time_threshold,
                    should_rebalance=should_rebalance,
                )
            )

        return statuses

    def should_rebalance(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        last_rebalance_date: datetime | None = None,
    ) -> bool:
        """Check if portfolio should be rebalanced.

        Args:
            positions: Current shares held per symbol
            prices: Current price per symbol
            last_rebalance_date: Date of last rebalance

        Returns:
            True if ANY symbol breaches thresholds and time constraint is met
        """
        statuses = self.check_thresholds(positions, prices, last_rebalance_date)
        return any(s.should_rebalance for s in statuses)

    def generate_rebalance_orders(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        last_rebalance_date: datetime | None = None,
        cash: float = 0.0,
    ) -> list[ThresholdDecision]:
        """Generate rebalancing orders based on threshold breaches.

        Args:
            positions: Current shares held per symbol
            prices: Current price per symbol
            last_rebalance_date: Date of last rebalance
            cash: Available cash for rebalancing

        Returns:
            List of ThresholdDecision objects (only for symbols that should rebalance)
        """
        # Check if we should rebalance
        statuses = self.check_thresholds(positions, prices, last_rebalance_date)

        if not any(s.should_rebalance for s in statuses):
            return []

        # Calculate total portfolio value
        equity_value = sum(positions.get(sym, 0.0) * prices.get(sym, 0.0) for sym in self.configs)
        total_value = equity_value + cash

        if total_value == 0.0:
            return []

        # Generate rebalancing decisions for ALL symbols (not just breached ones)
        # This ensures portfolio stays balanced
        decisions: list[ThresholdDecision] = []
        timestamp = datetime.now(tz=UTC)

        for status in statuses:
            sym = status.symbol
            config = self.configs[sym]

            # Calculate target dollar value
            target_value = total_value * config.target_weight
            current_value = positions.get(sym, 0.0) * prices.get(sym, 0.0)

            # Calculate adjustment
            adjustment_value = target_value - current_value
            adjustment_shares = adjustment_value / prices[sym] if prices[sym] > 0 else 0.0

            # Skip if trade value is below minimum
            if abs(adjustment_value) < self.min_trade_value:
                continue

            # Determine trigger reason
            if status.breaches_lower:
                trigger = f"Below lower band ({config.lower_band:.1%})"
            elif status.breaches_upper:
                trigger = f"Above upper band (+{config.upper_band:.1%})"
            elif status.exceeds_time_threshold and status.drift != 0:
                trigger = f"Time threshold ({self.min_days_between_rebalance} days)"
            else:
                trigger = "Rebalance for portfolio alignment"

            decisions.append(
                ThresholdDecision(
                    symbol=sym,
                    current_weight=status.current_weight,
                    target_weight=status.target_weight,
                    drift=status.drift,
                    adjustment_shares=adjustment_shares,
                    adjustment_value=adjustment_value,
                    trigger_reason=trigger,
                    timestamp=timestamp,
                )
            )

        return decisions

    def get_threshold_status_summary(
        self,
        statuses: list[ThresholdStatus],
    ) -> pd.DataFrame:
        """Convert threshold statuses to a summary DataFrame.

        Args:
            statuses: List of ThresholdStatus objects

        Returns:
            DataFrame with threshold status details
        """
        data = [
            {
                "symbol": s.symbol,
                "current_weight": s.current_weight,
                "target_weight": s.target_weight,
                "drift": s.drift,
                "breaches_lower": s.breaches_lower,
                "breaches_upper": s.breaches_upper,
                "days_since_rebalance": s.days_since_rebalance,
                "exceeds_time_threshold": s.exceeds_time_threshold,
                "should_rebalance": s.should_rebalance,
            }
            for s in statuses
        ]

        return pd.DataFrame(data)

    def get_rebalance_summary(
        self,
        decisions: list[ThresholdDecision],
    ) -> pd.DataFrame:
        """Convert rebalancing decisions to a summary DataFrame.

        Args:
            decisions: List of ThresholdDecision objects

        Returns:
            DataFrame with columns: symbol, current_weight, target_weight, drift,
                                   adjustment_shares, adjustment_value, trigger_reason
        """
        data = [
            {
                "symbol": d.symbol,
                "current_weight": d.current_weight,
                "target_weight": d.target_weight,
                "drift": d.drift,
                "adjustment_shares": d.adjustment_shares,
                "adjustment_value": d.adjustment_value,
                "trigger_reason": d.trigger_reason,
            }
            for d in decisions
        ]

        return pd.DataFrame(data)
