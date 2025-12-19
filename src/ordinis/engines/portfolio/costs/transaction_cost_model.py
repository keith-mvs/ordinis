"""
Transaction Cost Model - Production-grade cost estimation.

Provides accurate transaction cost modeling including:
- Spread costs (bid-ask)
- Market impact (temporary and permanent)
- Commission fees (broker-specific)
- Slippage estimation based on order size and liquidity

Gap Addressed: Previously sizing.py used a rough 0.1% estimate.
This provides a sophisticated model for cost-aware optimization.

Reference: Alpaca positions API provides cost_basis and unrealized_pl
which can be used to validate actual vs estimated costs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum, auto
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order execution types."""

    MARKET = auto()
    LIMIT = auto()
    MOC = auto()  # Market-on-close
    TWAP = auto()  # Time-weighted average
    VWAP = auto()  # Volume-weighted average


class AssetClass(Enum):
    """Asset class for cost modeling."""

    US_EQUITY = auto()
    ETF = auto()
    CRYPTO = auto()
    FUTURES = auto()
    OPTIONS = auto()


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for cost estimation.

    Attributes:
        avg_daily_volume: Average daily trading volume
        avg_spread_bps: Average bid-ask spread in basis points
        volatility: 20-day realized volatility
        market_cap: Market capitalization (for equities)
    """

    avg_daily_volume: float
    avg_spread_bps: float = 5.0  # Default 5 bps
    volatility: float = 0.20  # Default 20% annual vol
    market_cap: float | None = None

    @property
    def liquidity_score(self) -> float:
        """Calculate liquidity score (0-100, higher = more liquid)."""
        # Based on volume and spread
        vol_score = min(50, np.log10(max(self.avg_daily_volume, 1)) * 5)
        spread_score = max(0, 50 - self.avg_spread_bps)
        return vol_score + spread_score


@dataclass
class TransactionCostEstimate:
    """Detailed transaction cost breakdown.

    Attributes:
        symbol: Asset symbol
        order_size: Order size in shares
        notional_value: Order value in dollars
        spread_cost: Estimated spread cost
        market_impact: Estimated market impact cost
        commission: Broker commission
        total_cost: Total estimated cost
        total_bps: Total cost in basis points
        confidence: Estimation confidence (0-1)
    """

    symbol: str
    order_size: float
    notional_value: Decimal
    spread_cost: Decimal
    market_impact: Decimal
    commission: Decimal
    total_cost: Decimal
    total_bps: float
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_material(self) -> bool:
        """Check if cost is material (> 10 bps)."""
        return self.total_bps > 10.0


class TransactionCostModel(ABC):
    """Abstract base class for transaction cost models.

    Provides interface for estimating trading costs across different
    asset classes and execution strategies.
    """

    @abstractmethod
    def estimate_cost(
        self,
        symbol: str,
        order_size: float,
        price: float,
        side: str,
        order_type: OrderType = OrderType.MARKET,
        liquidity: LiquidityMetrics | None = None,
    ) -> TransactionCostEstimate:
        """Estimate transaction cost for an order.

        Args:
            symbol: Asset symbol
            order_size: Number of shares/units
            price: Current price
            side: 'buy' or 'sell'
            order_type: Execution type
            liquidity: Liquidity metrics (optional)

        Returns:
            TransactionCostEstimate with cost breakdown
        """
        ...

    @abstractmethod
    def estimate_portfolio_cost(
        self,
        trades: list[dict[str, Any]],
    ) -> tuple[Decimal, list[TransactionCostEstimate]]:
        """Estimate total cost for a portfolio rebalance.

        Args:
            trades: List of trades with symbol, size, price, side

        Returns:
            Tuple of (total_cost, individual_estimates)
        """
        ...


class AlmgrenChrissModel(TransactionCostModel):
    """
    Almgren-Chriss market impact model.

    Industry-standard model for estimating market impact as a function of:
    - Order size relative to daily volume
    - Asset volatility
    - Time horizon for execution

    References:
    - Almgren, Chriss (2000): "Optimal Execution of Portfolio Transactions"
    - Used by major institutional investors
    """

    def __init__(
        self,
        permanent_impact_coef: float = 0.1,
        temporary_impact_coef: float = 0.01,
        base_commission_per_share: float = 0.0,  # Alpaca: $0 commission
        min_commission: float = 0.0,
        default_spread_bps: float = 5.0,
    ) -> None:
        """Initialize Almgren-Chriss model.

        Args:
            permanent_impact_coef: Coefficient for permanent impact (eta)
            temporary_impact_coef: Coefficient for temporary impact (gamma)
            base_commission_per_share: Per-share commission
            min_commission: Minimum commission per trade
            default_spread_bps: Default spread if not provided
        """
        self.eta = permanent_impact_coef
        self.gamma = temporary_impact_coef
        self.commission_per_share = base_commission_per_share
        self.min_commission = min_commission
        self.default_spread_bps = default_spread_bps

    def estimate_cost(
        self,
        symbol: str,
        order_size: float,
        price: float,
        side: str,
        order_type: OrderType = OrderType.MARKET,
        liquidity: LiquidityMetrics | None = None,
    ) -> TransactionCostEstimate:
        """Estimate transaction cost using Almgren-Chriss model.

        Cost components:
        1. Spread cost: Half the bid-ask spread
        2. Permanent impact: Price change that persists (proportional to sqrt(volume))
        3. Temporary impact: Additional cost during execution
        4. Commission: Broker fees

        Args:
            symbol: Asset symbol
            order_size: Number of shares
            price: Current price
            side: 'buy' or 'sell'
            order_type: Order execution type
            liquidity: Optional liquidity metrics

        Returns:
            TransactionCostEstimate with breakdown
        """
        notional = Decimal(str(abs(order_size) * price))
        order_size = abs(order_size)

        # Use provided liquidity or defaults
        if liquidity:
            adv = liquidity.avg_daily_volume
            spread_bps = liquidity.avg_spread_bps
            volatility = liquidity.volatility
        else:
            adv = 1_000_000  # Default 1M ADV
            spread_bps = self.default_spread_bps
            volatility = 0.20

        # 1. Spread cost (half the spread for crossing)
        spread_cost = float(notional) * (spread_bps / 10000.0) * 0.5

        # 2. Market impact using Almgren-Chriss square-root formula
        # Impact = sigma * sqrt(order_size / ADV) * impact_coefficient
        if adv > 0:
            participation_rate = order_size / adv
            # Temporary impact (execution cost)
            temp_impact = (
                self.gamma * volatility * np.sqrt(participation_rate) * float(notional)
            )
            # Permanent impact (information leakage)
            perm_impact = self.eta * volatility * participation_rate * float(notional)
            market_impact = temp_impact + perm_impact
        else:
            market_impact = 0.0

        # 3. Commission
        commission = max(
            self.min_commission, order_size * self.commission_per_share
        )

        # Adjust for order type
        if order_type == OrderType.LIMIT:
            # Limit orders may not fill, reduce impact estimate
            market_impact *= 0.5
        elif order_type in (OrderType.TWAP, OrderType.VWAP):
            # Algorithmic execution reduces impact
            market_impact *= 0.7

        total_cost = Decimal(str(spread_cost + market_impact + commission))
        total_bps = float(total_cost) / float(notional) * 10000.0 if notional > 0 else 0

        # Confidence based on data quality
        confidence = 0.8 if liquidity else 0.5

        return TransactionCostEstimate(
            symbol=symbol,
            order_size=order_size,
            notional_value=notional,
            spread_cost=Decimal(str(spread_cost)),
            market_impact=Decimal(str(market_impact)),
            commission=Decimal(str(commission)),
            total_cost=total_cost,
            total_bps=total_bps,
            confidence=confidence,
            metadata={
                "model": "almgren_chriss",
                "participation_rate": participation_rate if adv > 0 else 0,
                "volatility": volatility,
                "adv": adv,
                "order_type": order_type.name,
            },
        )

    def estimate_portfolio_cost(
        self,
        trades: list[dict[str, Any]],
    ) -> tuple[Decimal, list[TransactionCostEstimate]]:
        """Estimate total cost for portfolio rebalance.

        Args:
            trades: List of trades with symbol, shares, price, side

        Returns:
            Tuple of (total_cost, individual_estimates)
        """
        estimates = []
        total_cost = Decimal("0")

        for trade in trades:
            estimate = self.estimate_cost(
                symbol=trade.get("symbol", ""),
                order_size=trade.get("shares", trade.get("qty", 0)),
                price=trade.get("price", 0),
                side=trade.get("side", "buy"),
                order_type=OrderType[trade.get("order_type", "MARKET").upper()],
                liquidity=trade.get("liquidity"),
            )
            estimates.append(estimate)
            total_cost += estimate.total_cost

        return total_cost, estimates


class SimpleCostModel(TransactionCostModel):
    """
    Simple fixed-rate transaction cost model.

    Suitable for quick estimates or when detailed liquidity data unavailable.
    Uses fixed basis point charges per trade.
    """

    def __init__(
        self,
        spread_bps: float = 5.0,
        impact_bps: float = 5.0,
        commission_per_trade: float = 0.0,
    ) -> None:
        """Initialize simple cost model.

        Args:
            spread_bps: Fixed spread cost in basis points
            impact_bps: Fixed impact cost in basis points
            commission_per_trade: Fixed commission per trade
        """
        self.spread_bps = spread_bps
        self.impact_bps = impact_bps
        self.commission = commission_per_trade

    def estimate_cost(
        self,
        symbol: str,
        order_size: float,
        price: float,
        side: str,
        order_type: OrderType = OrderType.MARKET,
        liquidity: LiquidityMetrics | None = None,
    ) -> TransactionCostEstimate:
        """Estimate cost using simple fixed rates."""
        notional = Decimal(str(abs(order_size) * price))

        spread_cost = float(notional) * (self.spread_bps / 10000.0)
        impact_cost = float(notional) * (self.impact_bps / 10000.0)
        commission = self.commission

        total_cost = Decimal(str(spread_cost + impact_cost + commission))
        total_bps = (self.spread_bps + self.impact_bps) + (
            float(commission) / float(notional) * 10000.0 if notional > 0 else 0
        )

        return TransactionCostEstimate(
            symbol=symbol,
            order_size=abs(order_size),
            notional_value=notional,
            spread_cost=Decimal(str(spread_cost)),
            market_impact=Decimal(str(impact_cost)),
            commission=Decimal(str(commission)),
            total_cost=total_cost,
            total_bps=total_bps,
            confidence=0.5,  # Low confidence for simple model
            metadata={"model": "simple"},
        )

    def estimate_portfolio_cost(
        self,
        trades: list[dict[str, Any]],
    ) -> tuple[Decimal, list[TransactionCostEstimate]]:
        """Estimate total cost for portfolio rebalance."""
        estimates = []
        total_cost = Decimal("0")

        for trade in trades:
            estimate = self.estimate_cost(
                symbol=trade.get("symbol", ""),
                order_size=trade.get("shares", trade.get("qty", 0)),
                price=trade.get("price", 0),
                side=trade.get("side", "buy"),
            )
            estimates.append(estimate)
            total_cost += estimate.total_cost

        return total_cost, estimates


class AdaptiveCostModel(TransactionCostModel):
    """
    Adaptive transaction cost model that learns from execution data.

    Combines model-based estimates with historical execution feedback
    to improve accuracy over time. Integrates with LearningEngine.
    """

    def __init__(
        self,
        base_model: TransactionCostModel | None = None,
        learning_rate: float = 0.1,
    ) -> None:
        """Initialize adaptive cost model.

        Args:
            base_model: Underlying model for initial estimates
            learning_rate: Rate of adaptation to new data
        """
        self.base_model = base_model or AlmgrenChrissModel()
        self.learning_rate = learning_rate

        # Symbol-specific adjustments learned from execution
        self._symbol_adjustments: dict[str, float] = {}
        self._execution_history: list[dict[str, Any]] = []

    def estimate_cost(
        self,
        symbol: str,
        order_size: float,
        price: float,
        side: str,
        order_type: OrderType = OrderType.MARKET,
        liquidity: LiquidityMetrics | None = None,
    ) -> TransactionCostEstimate:
        """Estimate cost with learned adjustments."""
        base_estimate = self.base_model.estimate_cost(
            symbol, order_size, price, side, order_type, liquidity
        )

        # Apply learned adjustment
        adjustment = self._symbol_adjustments.get(symbol, 1.0)
        adjusted_cost = base_estimate.total_cost * Decimal(str(adjustment))
        adjusted_bps = base_estimate.total_bps * adjustment

        return TransactionCostEstimate(
            symbol=symbol,
            order_size=base_estimate.order_size,
            notional_value=base_estimate.notional_value,
            spread_cost=base_estimate.spread_cost * Decimal(str(adjustment)),
            market_impact=base_estimate.market_impact * Decimal(str(adjustment)),
            commission=base_estimate.commission,
            total_cost=adjusted_cost,
            total_bps=adjusted_bps,
            confidence=min(0.9, base_estimate.confidence + 0.1),  # Higher after learning
            metadata={
                **base_estimate.metadata,
                "adaptive_adjustment": adjustment,
                "model": "adaptive",
            },
        )

    def estimate_portfolio_cost(
        self,
        trades: list[dict[str, Any]],
    ) -> tuple[Decimal, list[TransactionCostEstimate]]:
        """Estimate total cost for portfolio rebalance."""
        estimates = []
        total_cost = Decimal("0")

        for trade in trades:
            estimate = self.estimate_cost(
                symbol=trade.get("symbol", ""),
                order_size=trade.get("shares", trade.get("qty", 0)),
                price=trade.get("price", 0),
                side=trade.get("side", "buy"),
                order_type=OrderType[trade.get("order_type", "MARKET").upper()],
                liquidity=trade.get("liquidity"),
            )
            estimates.append(estimate)
            total_cost += estimate.total_cost

        return total_cost, estimates

    def record_execution(
        self,
        symbol: str,
        estimated_cost_bps: float,
        actual_cost_bps: float,
        notional: float,
    ) -> None:
        """Record actual execution for learning.

        Args:
            symbol: Asset symbol
            estimated_cost_bps: What we estimated
            actual_cost_bps: What we observed
            notional: Order notional value
        """
        # Calculate adjustment ratio
        if estimated_cost_bps > 0:
            ratio = actual_cost_bps / estimated_cost_bps
        else:
            ratio = 1.0

        # Update with exponential smoothing
        current = self._symbol_adjustments.get(symbol, 1.0)
        updated = current * (1 - self.learning_rate) + ratio * self.learning_rate
        self._symbol_adjustments[symbol] = updated

        # Record for analysis
        self._execution_history.append(
            {
                "symbol": symbol,
                "estimated_bps": estimated_cost_bps,
                "actual_bps": actual_cost_bps,
                "adjustment": ratio,
                "new_adjustment": updated,
                "notional": notional,
                "timestamp": datetime.now(UTC),
            }
        )

        logger.debug(
            f"Updated cost adjustment for {symbol}: {current:.3f} -> {updated:.3f}"
        )

    def get_model_accuracy(self) -> dict[str, Any]:
        """Get model accuracy statistics.

        Returns:
            Dictionary with RMSE, bias, and per-symbol stats
        """
        if not self._execution_history:
            return {"rmse": 0.0, "bias": 0.0, "n_observations": 0}

        errors = [
            h["actual_bps"] - h["estimated_bps"] for h in self._execution_history
        ]

        return {
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "bias": float(np.mean(errors)),
            "n_observations": len(self._execution_history),
            "symbol_adjustments": dict(self._symbol_adjustments),
        }


# Factory function for common cost model configurations
def create_cost_model(
    model_type: str = "almgren_chriss",
    **kwargs: Any,
) -> TransactionCostModel:
    """Factory function to create transaction cost models.

    Args:
        model_type: 'almgren_chriss', 'simple', or 'adaptive'
        **kwargs: Model-specific parameters

    Returns:
        Configured TransactionCostModel instance
    """
    if model_type == "almgren_chriss":
        return AlmgrenChrissModel(**kwargs)
    elif model_type == "simple":
        return SimpleCostModel(**kwargs)
    elif model_type == "adaptive":
        base = AlmgrenChrissModel()
        return AdaptiveCostModel(base_model=base, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
