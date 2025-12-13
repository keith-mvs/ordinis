"""
Risk Parity Rebalancing Strategy.

Allocates portfolio weights based on equal risk contribution.
Each asset contributes equally to total portfolio risk (volatility).
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd


@dataclass
class RiskParityWeights:
    """Calculated risk parity weights for symbols.

    Attributes:
        weights: Dictionary of {symbol: weight}
        volatilities: Dictionary of {symbol: annualized_volatility}
        risk_contributions: Dictionary of {symbol: risk_contribution}
        timestamp: When weights were calculated
    """

    weights: dict[str, float]
    volatilities: dict[str, float]
    risk_contributions: dict[str, float]
    timestamp: datetime


@dataclass
class RiskParityDecision:
    """Rebalancing action for risk parity strategy.

    Attributes:
        symbol: Ticker symbol
        current_weight: Current allocation as decimal (0.0 to 1.0)
        target_weight: Risk parity weight as decimal (0.0 to 1.0)
        current_volatility: Annualized volatility of the asset
        risk_contribution: Percentage of total portfolio risk from this asset
        adjustment_shares: Number of shares to buy (+) or sell (-)
        adjustment_value: Dollar value of adjustment
        timestamp: When the decision was generated
    """

    symbol: str
    current_weight: float
    target_weight: float
    current_volatility: float
    risk_contribution: float
    adjustment_shares: float
    adjustment_value: float
    timestamp: datetime


class RiskParityRebalancer:
    """Rebalances portfolio using risk parity methodology.

    Risk parity allocates capital such that each asset contributes equally
    to total portfolio risk. Uses inverse-volatility weighting as the base method.

    Supports:
    - Inverse volatility weighting (simpler, no correlation)
    - Configurable lookback period for volatility calculation
    - Minimum weight constraints to avoid extreme allocations

    Example:
        >>> # Calculate risk parity weights from historical returns
        >>> returns = pd.DataFrame({
        ...     "AAPL": [0.01, -0.02, 0.015, ...],
        ...     "MSFT": [0.005, 0.01, -0.005, ...],
        ...     "GOOGL": [0.02, -0.01, 0.025, ...],
        ... })
        >>> rebalancer = RiskParityRebalancer(lookback_days=252, min_weight=0.05)
        >>> weights = rebalancer.calculate_weights(returns)
        >>> positions = {"AAPL": 10, "MSFT": 20, "GOOGL": 15}
        >>> prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 100.0}
        >>> decisions = rebalancer.generate_rebalance_orders(returns, positions, prices)
    """

    def __init__(
        self,
        lookback_days: int = 252,
        min_weight: float = 0.01,
        max_weight: float = 0.50,
        drift_threshold: float = 0.05,
    ) -> None:
        """Initialize risk parity rebalancer.

        Args:
            lookback_days: Number of days to use for volatility calculation (default: 252 = 1 year)
            min_weight: Minimum weight per asset (default: 0.01 = 1%)
            max_weight: Maximum weight per asset (default: 0.50 = 50%)
            drift_threshold: Maximum acceptable drift from target (default: 0.05 = 5%)

        Raises:
            ValueError: If parameters are invalid
        """
        if lookback_days < 20:
            raise ValueError(f"lookback_days must be >= 20, got {lookback_days}")
        if not 0.0 <= min_weight < max_weight <= 1.0:
            raise ValueError(
                f"Must have 0 <= min_weight < max_weight <= 1, got {min_weight}, {max_weight}"
            )
        if not 0.0 < drift_threshold <= 1.0:
            raise ValueError(f"drift_threshold must be in (0, 1], got {drift_threshold}")

        self.lookback_days = lookback_days
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.drift_threshold = drift_threshold

    def calculate_weights(
        self,
        returns: pd.DataFrame,
    ) -> RiskParityWeights:
        """Calculate risk parity weights from historical returns.

        Uses inverse-volatility weighting: weight = (1/volatility) / sum(1/volatility)

        Args:
            returns: DataFrame with returns for each symbol (columns = symbols, rows = time periods)

        Returns:
            RiskParityWeights object with calculated weights, volatilities, and risk contributions

        Raises:
            ValueError: If returns DataFrame is empty or has insufficient history
        """
        if returns.empty:
            raise ValueError("Returns DataFrame cannot be empty")

        # Use last N days for volatility calculation
        recent_returns = returns.tail(self.lookback_days)

        if len(recent_returns) < 20:
            raise ValueError(f"Insufficient return history: {len(recent_returns)} days, need >= 20")

        # Calculate annualized volatility for each symbol
        # std() gives daily volatility, multiply by sqrt(252) for annual
        volatilities = recent_returns.std() * np.sqrt(252)
        volatilities_dict = volatilities.to_dict()

        # Inverse volatility weighting
        inverse_vols = 1.0 / volatilities
        raw_weights = inverse_vols / inverse_vols.sum()

        # Apply min/max constraints
        constrained_weights = raw_weights.clip(lower=self.min_weight, upper=self.max_weight)

        # Renormalize to sum to 1.0 after constraints
        final_weights = constrained_weights / constrained_weights.sum()
        weights_dict = final_weights.to_dict()

        # Calculate risk contribution for each asset
        # Risk contribution = weight * volatility
        portfolio_volatility = sum(
            weights_dict[sym] * volatilities_dict[sym] for sym in weights_dict
        )
        risk_contributions = {
            sym: (weights_dict[sym] * volatilities_dict[sym]) / portfolio_volatility
            for sym in weights_dict
        }

        return RiskParityWeights(
            weights=weights_dict,
            volatilities=volatilities_dict,
            risk_contributions=risk_contributions,
            timestamp=datetime.now(tz=UTC),
        )

    def should_rebalance(
        self,
        returns: pd.DataFrame,
        positions: dict[str, float],
        prices: dict[str, float],
    ) -> bool:
        """Check if portfolio exceeds drift threshold from risk parity weights.

        Args:
            returns: Historical returns DataFrame
            positions: Current shares held per symbol
            prices: Current price per symbol

        Returns:
            True if rebalancing is needed, False otherwise
        """
        # Calculate current weights
        total_value = sum(positions.get(sym, 0.0) * prices.get(sym, 0.0) for sym in returns.columns)

        if total_value == 0.0:
            return True  # Empty portfolio needs rebalancing

        current_weights = {
            sym: (positions.get(sym, 0.0) * prices.get(sym, 0.0)) / total_value
            for sym in returns.columns
        }

        # Calculate target weights
        rp_weights = self.calculate_weights(returns)

        # Calculate drift
        max_drift = max(
            abs(current_weights.get(sym, 0.0) - rp_weights.weights.get(sym, 0.0))
            for sym in returns.columns
        )

        return max_drift > self.drift_threshold

    def generate_rebalance_orders(
        self,
        returns: pd.DataFrame,
        positions: dict[str, float],
        prices: dict[str, float],
        cash: float = 0.0,
    ) -> list[RiskParityDecision]:
        """Generate rebalancing orders based on risk parity weights.

        Args:
            returns: Historical returns DataFrame for volatility calculation
            positions: Current shares held per symbol
            prices: Current price per symbol
            cash: Available cash for rebalancing (default: 0 = rebalance within portfolio)

        Returns:
            List of RiskParityDecision objects with buy/sell instructions
        """
        # Calculate risk parity weights
        rp_weights = self.calculate_weights(returns)

        # Calculate total portfolio value including cash
        equity_value = sum(
            positions.get(sym, 0.0) * prices.get(sym, 0.0) for sym in returns.columns
        )
        total_value = equity_value + cash

        if total_value == 0.0:
            return []

        # Calculate current weights
        current_weights = {
            sym: (positions.get(sym, 0.0) * prices.get(sym, 0.0)) / total_value
            for sym in returns.columns
        }

        # Generate rebalancing decisions
        decisions: list[RiskParityDecision] = []
        timestamp = datetime.now(tz=UTC)

        for sym in returns.columns:
            target_weight = rp_weights.weights[sym]
            current_weight = current_weights[sym]

            # Calculate target dollar value for this symbol
            target_value = total_value * target_weight
            current_value = positions.get(sym, 0.0) * prices.get(sym, 0.0)

            # Calculate adjustment needed
            adjustment_value = target_value - current_value
            adjustment_shares = adjustment_value / prices[sym] if prices[sym] > 0 else 0.0

            decision = RiskParityDecision(
                symbol=sym,
                current_weight=current_weight,
                target_weight=target_weight,
                current_volatility=rp_weights.volatilities[sym],
                risk_contribution=rp_weights.risk_contributions[sym],
                adjustment_shares=adjustment_shares,
                adjustment_value=adjustment_value,
                timestamp=timestamp,
            )
            decisions.append(decision)

        return decisions

    def get_rebalance_summary(
        self,
        decisions: list[RiskParityDecision],
    ) -> pd.DataFrame:
        """Convert rebalancing decisions to a summary DataFrame.

        Args:
            decisions: List of RiskParityDecision objects

        Returns:
            DataFrame with columns: symbol, current_weight, target_weight, volatility,
                                   risk_contribution, drift, adjustment_shares, adjustment_value
        """
        data = [
            {
                "symbol": d.symbol,
                "current_weight": d.current_weight,
                "target_weight": d.target_weight,
                "volatility": d.current_volatility,
                "risk_contribution": d.risk_contribution,
                "drift": d.current_weight - d.target_weight,
                "adjustment_shares": d.adjustment_shares,
                "adjustment_value": d.adjustment_value,
            }
            for d in decisions
        ]

        return pd.DataFrame(data)
