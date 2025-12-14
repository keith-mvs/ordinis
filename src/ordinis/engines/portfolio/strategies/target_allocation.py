"""
Target Allocation Rebalancing Strategy.

Maintains fixed percentage weights for each symbol in the portfolio.
Designed for stocks and options positions.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd


@dataclass
class TargetAllocation:
    """Target weight configuration for a symbol.

    Attributes:
        symbol: Ticker symbol (e.g., "AAPL", "SPY")
        target_weight: Desired allocation as decimal (0.0 to 1.0)
    """

    symbol: str
    target_weight: float

    def __post_init__(self) -> None:
        """Validate target weight is between 0 and 1."""
        if not 0.0 <= self.target_weight <= 1.0:
            raise ValueError(f"target_weight must be in [0, 1], got {self.target_weight}")


@dataclass
class RebalanceDecision:
    """Rebalancing action for a single symbol.

    Attributes:
        symbol: Ticker symbol
        current_weight: Current allocation as decimal (0.0 to 1.0)
        target_weight: Desired allocation as decimal (0.0 to 1.0)
        adjustment_shares: Number of shares to buy (+) or sell (-)
        adjustment_value: Dollar value of adjustment (positive = buy, negative = sell)
        timestamp: When the decision was generated
    """

    symbol: str
    current_weight: float
    target_weight: float
    adjustment_shares: float
    adjustment_value: float
    timestamp: datetime


class TargetAllocationRebalancer:
    """Rebalances portfolio to maintain fixed target weights.

    Supports:
    - Stock positions (long only for now)
    - Configurable drift tolerance before rebalancing
    - Dollar-based or share-based rebalancing

    Example:
        >>> targets = [
        ...     TargetAllocation("AAPL", 0.40),
        ...     TargetAllocation("MSFT", 0.30),
        ...     TargetAllocation("GOOGL", 0.30),
        ... ]
        >>> rebalancer = TargetAllocationRebalancer(targets, drift_threshold=0.05)
        >>> positions = {"AAPL": 10, "MSFT": 5, "GOOGL": 3}
        >>> prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 100.0}
        >>> decisions = rebalancer.generate_rebalance_orders(positions, prices, 10000.0)
    """

    def __init__(
        self,
        target_allocations: list[TargetAllocation],
        drift_threshold: float = 0.05,
    ) -> None:
        """Initialize the rebalancer.

        Args:
            target_allocations: List of target weight configurations
            drift_threshold: Maximum acceptable drift from target (e.g., 0.05 = 5%)

        Raises:
            ValueError: If target weights don't sum to 1.0 (within tolerance)
        """
        self.targets = {t.symbol: t.target_weight for t in target_allocations}
        self.drift_threshold = drift_threshold

        # Validate weights sum to 1.0 (with small tolerance for floating point)
        total_weight = sum(self.targets.values())
        if not 0.999 <= total_weight <= 1.001:
            raise ValueError(f"Target weights must sum to 1.0, got {total_weight:.4f}")

    def calculate_drift(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
    ) -> dict[str, float]:
        """Calculate drift from target weights for each symbol.

        Args:
            positions: Current shares held per symbol
            prices: Current price per symbol

        Returns:
            Dictionary of {symbol: drift} where drift is (current_weight - target_weight)
        """
        # Calculate total portfolio value
        total_value = sum(positions.get(sym, 0.0) * prices.get(sym, 0.0) for sym in self.targets)

        if total_value == 0.0:
            # Empty portfolio - all symbols need full allocation
            return {sym: -weight for sym, weight in self.targets.items()}

        # Calculate current weights
        current_weights = {
            sym: (positions.get(sym, 0.0) * prices.get(sym, 0.0)) / total_value
            for sym in self.targets
        }

        # Calculate drift
        drift = {sym: current_weights[sym] - self.targets[sym] for sym in self.targets}

        return drift

    def should_rebalance(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
    ) -> bool:
        """Check if any symbol exceeds drift threshold.

        Args:
            positions: Current shares held per symbol
            prices: Current price per symbol

        Returns:
            True if rebalancing is needed, False otherwise
        """
        drift = self.calculate_drift(positions, prices)
        max_drift = max(abs(d) for d in drift.values())
        return max_drift > self.drift_threshold

    def generate_rebalance_orders(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        cash: float = 0.0,
    ) -> list[RebalanceDecision]:
        """Generate rebalancing orders to restore target weights.

        Args:
            positions: Current shares held per symbol
            prices: Current price per symbol
            cash: Available cash for rebalancing (default: 0 = rebalance within portfolio)

        Returns:
            List of RebalanceDecision objects with buy/sell instructions

        Notes:
            - Positive adjustment_shares means BUY
            - Negative adjustment_shares means SELL
            - If cash=0, this is a portfolio-internal rebalance (sell overweight, buy underweight)
            - If cash>0, includes cash in total portfolio value calculation
        """
        # Calculate total portfolio value including cash
        equity_value = sum(positions.get(sym, 0.0) * prices.get(sym, 0.0) for sym in self.targets)
        total_value = equity_value + cash

        if total_value == 0.0:
            # Cannot rebalance empty portfolio with no cash
            return []

        # Calculate current weights
        current_weights = {
            sym: (positions.get(sym, 0.0) * prices.get(sym, 0.0)) / total_value
            for sym in self.targets
        }

        # Generate rebalancing decisions
        decisions: list[RebalanceDecision] = []
        timestamp = datetime.now(tz=UTC)

        for sym in self.targets:
            target_weight = self.targets[sym]
            current_weight = current_weights[sym]

            # Calculate target dollar value for this symbol
            target_value = total_value * target_weight
            current_value = positions.get(sym, 0.0) * prices.get(sym, 0.0)

            # Calculate adjustment needed
            adjustment_value = target_value - current_value
            adjustment_shares = adjustment_value / prices[sym] if prices[sym] > 0 else 0.0

            decision = RebalanceDecision(
                symbol=sym,
                current_weight=current_weight,
                target_weight=target_weight,
                adjustment_shares=adjustment_shares,
                adjustment_value=adjustment_value,
                timestamp=timestamp,
            )
            decisions.append(decision)

        return decisions

    def get_rebalance_summary(
        self,
        decisions: list[RebalanceDecision],
    ) -> pd.DataFrame:
        """Convert rebalancing decisions to a summary DataFrame.

        Args:
            decisions: List of RebalanceDecision objects

        Returns:
            DataFrame with columns: symbol, current_weight, target_weight, drift,
                                   adjustment_shares, adjustment_value
        """
        data = [
            {
                "symbol": d.symbol,
                "current_weight": d.current_weight,
                "target_weight": d.target_weight,
                "drift": d.current_weight - d.target_weight,
                "adjustment_shares": d.adjustment_shares,
                "adjustment_value": d.adjustment_value,
            }
            for d in decisions
        ]

        return pd.DataFrame(data)
