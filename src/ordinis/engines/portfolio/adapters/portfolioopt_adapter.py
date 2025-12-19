"""
PortfolioOpt Adapter - Bridge between PortfolioOptEngine and PortfolioEngine.

Integrates GPU-accelerated Mean-CVaR optimization from PortfolioOptEngine
with the rebalancing workflows in PortfolioEngine. Inspired by Alpaca's
portfolio rebalancing API patterns:
- Target weight allocation with drift detection
- Cooldown periods between rebalances
- Cash/asset weight specifications

Gap Addressed: Previously PortfolioOptEngine and PortfolioEngine were disconnected.
This adapter bridges optimized weights to rebalancing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum, auto
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ordinis.engines.portfolio.core.engine import PortfolioEngine
    from ordinis.engines.portfolioopt.core.engine import (
        OptimizationResult,
        PortfolioOptEngine,
    )


class DriftType(Enum):
    """Types of portfolio drift measurement (Alpaca-inspired)."""

    ABSOLUTE = auto()  # Deviation in percentage points
    RELATIVE = auto()  # Deviation relative to target weight


class RebalanceCondition(Enum):
    """Rebalance trigger conditions (Alpaca-inspired)."""

    DRIFT_BAND = auto()  # Weight drift exceeded threshold
    CALENDAR = auto()  # Scheduled rebalance (daily/weekly/monthly)
    CASH_INFLOW = auto()  # New cash to invest
    MANUAL = auto()  # Manually triggered


@dataclass
class DriftBandConfig:
    """Configuration for drift-band rebalancing (Alpaca-style).

    Attributes:
        drift_type: Absolute or relative drift measurement
        threshold_pct: Drift threshold to trigger rebalance
        cooldown_days: Minimum days between rebalances
    """

    drift_type: DriftType = DriftType.ABSOLUTE
    threshold_pct: float = 5.0  # 5% absolute drift
    cooldown_days: int = 7


@dataclass
class CalendarConfig:
    """Configuration for calendar-based rebalancing.

    Attributes:
        frequency: Rebalance frequency ('daily', 'weekly', 'monthly', 'quarterly')
        day_of_week: Day for weekly rebalance (0=Monday, 6=Sunday)
        day_of_month: Day for monthly/quarterly rebalance
    """

    frequency: str = "monthly"
    day_of_week: int = 0  # Monday
    day_of_month: int = 1  # First of month


@dataclass
class PortfolioWeight:
    """Portfolio weight specification (Alpaca-style).

    Attributes:
        symbol: Asset symbol or 'CASH' for cash allocation
        weight_type: 'asset' or 'cash'
        target_pct: Target weight percentage (0-100)
        current_pct: Current weight percentage
    """

    symbol: str
    weight_type: str  # 'asset' or 'cash'
    target_pct: float
    current_pct: float = 0.0

    @property
    def drift(self) -> float:
        """Calculate absolute drift from target."""
        return self.current_pct - self.target_pct

    @property
    def relative_drift(self) -> float:
        """Calculate relative drift from target."""
        if self.target_pct == 0:
            return 0.0 if self.current_pct == 0 else float("inf")
        return (self.current_pct - self.target_pct) / self.target_pct * 100


@dataclass
class DriftAnalysis:
    """Analysis of portfolio drift from target weights.

    Attributes:
        weights: Current vs target weight analysis
        max_drift: Maximum absolute drift across all positions
        max_relative_drift: Maximum relative drift
        drift_triggered: Whether any drift exceeded threshold
        cash_available: Available cash for investment
        invest_cash_eligible: Whether cash investment is warranted
    """

    weights: list[PortfolioWeight]
    max_drift: float
    max_relative_drift: float
    drift_triggered: bool
    cash_available: float
    invest_cash_eligible: bool  # Alpaca: >= $10 cash triggers invest_cash run
    cooldown_remaining: timedelta | None = None

    def get_rebalance_actions(self) -> list[dict[str, Any]]:
        """Get required rebalance actions to return to target.

        Returns:
            List of actions with symbol, direction, and amount.
        """
        actions = []
        for weight in self.weights:
            if weight.weight_type == "cash":
                continue
            if abs(weight.drift) > 0.1:  # Ignore tiny drifts
                direction = "sell" if weight.drift > 0 else "buy"
                actions.append(
                    {
                        "symbol": weight.symbol,
                        "direction": direction,
                        "weight_delta": abs(weight.drift),
                        "current_pct": weight.current_pct,
                        "target_pct": weight.target_pct,
                    }
                )
        return actions


@dataclass
class RebalanceRun:
    """Record of a rebalance run (Alpaca-inspired).

    Attributes:
        run_id: Unique run identifier
        run_type: Type of rebalance ('full_rebalance' or 'invest_cash')
        condition: What triggered the rebalance
        status: Run status ('pending', 'in_progress', 'completed', 'failed')
        weights: Target weights used
        orders: Orders generated/executed
        timestamp: When run was initiated
    """

    run_id: str
    run_type: str  # 'full_rebalance' or 'invest_cash'
    condition: RebalanceCondition
    status: str
    weights: list[PortfolioWeight]
    orders: list[dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    error_message: str | None = None


class PortfolioOptAdapter:
    """
    Adapter bridging PortfolioOptEngine optimization to PortfolioEngine rebalancing.

    Key Features:
    - Converts GPU-optimized weights to target allocations
    - Implements Alpaca-style drift detection (absolute/relative)
    - Supports cooldown periods and calendar-based rebalancing
    - Cash investment runs for new deposits
    - Transaction cost awareness in weight optimization

    Example:
        >>> from ordinis.engines.portfolio import PortfolioEngine
        >>> from ordinis.engines.portfolioopt import PortfolioOptEngine
        >>> from ordinis.engines.portfolio.adapters import PortfolioOptAdapter

        >>> adapter = PortfolioOptAdapter(
        ...     drift_config=DriftBandConfig(threshold_pct=5.0),
        ...     cash_reserve_pct=5.0,
        ... )

        >>> # Get optimized weights from PortfolioOptEngine
        >>> opt_result = await portfolioopt_engine.optimize(returns_df)

        >>> # Convert to target allocations for PortfolioEngine
        >>> targets = adapter.convert_to_targets(opt_result, portfolio_engine)

        >>> # Check if rebalance is needed
        >>> drift = adapter.analyze_drift(portfolio_engine, targets)
        >>> if drift.drift_triggered:
        ...     decisions = await portfolio_engine.generate_rebalancing_decisions(...)
    """

    # Alpaca constants
    MIN_CASH_FOR_INVEST = 10.0  # $10 minimum for invest_cash run
    MIN_ORDER_VALUE = 1.0  # $1 minimum per order

    def __init__(
        self,
        drift_config: DriftBandConfig | None = None,
        calendar_config: CalendarConfig | None = None,
        cash_reserve_pct: float = 5.0,
        min_weight_pct: float = 0.1,
        transaction_cost_bps: float = 10.0,  # 10 basis points
    ) -> None:
        """Initialize PortfolioOpt adapter.

        Args:
            drift_config: Drift band configuration
            calendar_config: Calendar rebalance configuration
            cash_reserve_pct: Target cash reserve percentage (0-100)
            min_weight_pct: Minimum weight to include in allocation
            transaction_cost_bps: Estimated transaction cost in basis points
        """
        self.drift_config = drift_config or DriftBandConfig()
        self.calendar_config = calendar_config
        self.cash_reserve_pct = cash_reserve_pct
        self.min_weight_pct = min_weight_pct
        self.transaction_cost_bps = transaction_cost_bps

        self._last_rebalance: datetime | None = None
        self._run_history: list[RebalanceRun] = []

    def convert_to_targets(
        self,
        optimization_result: "OptimizationResult",
        total_equity: float | None = None,
    ) -> list[PortfolioWeight]:
        """Convert optimization result to target portfolio weights.

        Args:
            optimization_result: Result from PortfolioOptEngine
            total_equity: Total portfolio equity (for cash calculation)

        Returns:
            List of PortfolioWeight with target allocations
        """
        weights = []

        # Add cash reserve first
        if self.cash_reserve_pct > 0:
            weights.append(
                PortfolioWeight(
                    symbol="CASH",
                    weight_type="cash",
                    target_pct=self.cash_reserve_pct,
                )
            )

        # Scale remaining weights to account for cash reserve
        remaining_pct = 100.0 - self.cash_reserve_pct
        scale_factor = remaining_pct / 100.0

        # Add asset weights from optimization
        for symbol, weight in optimization_result.weights.items():
            weight_pct = weight * 100.0 * scale_factor
            if weight_pct >= self.min_weight_pct:
                weights.append(
                    PortfolioWeight(
                        symbol=symbol,
                        weight_type="asset",
                        target_pct=weight_pct,
                    )
                )

        return weights

    def calculate_current_weights(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        cash: float,
    ) -> dict[str, float]:
        """Calculate current portfolio weights.

        Args:
            positions: Position quantities by symbol
            prices: Current prices by symbol
            cash: Current cash balance

        Returns:
            Current weight percentages by symbol (plus 'CASH')
        """
        # Calculate total portfolio value
        position_values = {
            symbol: qty * prices.get(symbol, 0.0) for symbol, qty in positions.items()
        }
        total_equity = sum(position_values.values()) + cash

        if total_equity <= 0:
            return {"CASH": 100.0}

        weights = {}
        for symbol, value in position_values.items():
            weights[symbol] = (value / total_equity) * 100.0
        weights["CASH"] = (cash / total_equity) * 100.0

        return weights

    def analyze_drift(
        self,
        current_weights: dict[str, float],
        target_weights: list[PortfolioWeight],
        cash: float = 0.0,
    ) -> DriftAnalysis:
        """Analyze portfolio drift from target weights.

        Implements Alpaca-style drift detection:
        - Absolute drift: Simple percentage point difference
        - Relative drift: Percentage of target weight

        Args:
            current_weights: Current weight percentages
            target_weights: Target portfolio weights
            cash: Available cash balance

        Returns:
            DriftAnalysis with drift metrics and actions
        """
        # Update current weights in target list
        weight_analysis = []
        max_drift = 0.0
        max_relative_drift = 0.0

        for target in target_weights:
            current_pct = current_weights.get(target.symbol, 0.0)
            updated = PortfolioWeight(
                symbol=target.symbol,
                weight_type=target.weight_type,
                target_pct=target.target_pct,
                current_pct=current_pct,
            )
            weight_analysis.append(updated)

            abs_drift = abs(updated.drift)
            if abs_drift > max_drift:
                max_drift = abs_drift

            rel_drift = abs(updated.relative_drift)
            if rel_drift > max_relative_drift and rel_drift != float("inf"):
                max_relative_drift = rel_drift

        # Check if drift threshold exceeded
        if self.drift_config.drift_type == DriftType.ABSOLUTE:
            drift_triggered = max_drift >= self.drift_config.threshold_pct
        else:
            drift_triggered = max_relative_drift >= self.drift_config.threshold_pct

        # Check cooldown
        cooldown_remaining = None
        if self._last_rebalance and self.drift_config.cooldown_days > 0:
            cooldown_end = self._last_rebalance + timedelta(
                days=self.drift_config.cooldown_days
            )
            if datetime.now(UTC) < cooldown_end:
                drift_triggered = False  # Don't trigger during cooldown
                cooldown_remaining = cooldown_end - datetime.now(UTC)

        return DriftAnalysis(
            weights=weight_analysis,
            max_drift=max_drift,
            max_relative_drift=max_relative_drift,
            drift_triggered=drift_triggered,
            cash_available=cash,
            invest_cash_eligible=cash >= self.MIN_CASH_FOR_INVEST,
            cooldown_remaining=cooldown_remaining,
        )

    def calculate_rebalance_trades(
        self,
        current_weights: dict[str, float],
        target_weights: list[PortfolioWeight],
        total_equity: float,
        prices: dict[str, float],
        min_order_value: float | None = None,
    ) -> list[dict[str, Any]]:
        """Calculate trades needed to achieve target weights.

        Considers transaction costs and minimum order sizes (Alpaca: $1 per asset).

        Args:
            current_weights: Current weight percentages
            target_weights: Target portfolio weights
            total_equity: Total portfolio value
            prices: Current prices
            min_order_value: Minimum order value (default: $1)

        Returns:
            List of trade instructions
        """
        min_order = min_order_value or self.MIN_ORDER_VALUE
        trades: list[dict[str, Any]] = []

        for target in target_weights:
            if target.weight_type == "cash":
                continue

            current_pct = current_weights.get(target.symbol, 0.0)
            delta_pct = target.target_pct - current_pct
            delta_value = (delta_pct / 100.0) * total_equity

            # Account for transaction costs
            if delta_value > 0:  # Buy
                # Reduce buy amount by expected transaction cost
                cost_reduction = delta_value * (self.transaction_cost_bps / 10000.0)
                delta_value -= cost_reduction

            if abs(delta_value) < min_order:
                continue

            price = prices.get(target.symbol, 0.0)
            if price <= 0:
                continue

            shares = delta_value / price
            side = "buy" if shares > 0 else "sell"

            trades.append(
                {
                    "symbol": target.symbol,
                    "side": side,
                    "notional": abs(delta_value),
                    "shares": abs(shares),
                    "price": price,
                    "weight_delta": delta_pct,
                    "order_type": "market",
                }
            )

        # Sort: sells first, then buys (to free up cash)
        trades.sort(key=lambda t: (0 if t["side"] == "sell" else 1, -float(t["notional"])))

        return trades

    def create_invest_cash_trades(
        self,
        cash_amount: float,
        target_weights: list[PortfolioWeight],
        prices: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Create trades to invest available cash proportionally.

        Alpaca-style invest_cash: Only buys, no sells.

        Args:
            cash_amount: Cash available to invest
            target_weights: Target portfolio weights
            prices: Current prices

        Returns:
            List of buy orders
        """
        if cash_amount < self.MIN_CASH_FOR_INVEST:
            return []

        # Reserve some cash based on configuration
        investable = cash_amount * (1 - self.cash_reserve_pct / 100.0)

        # Allocate proportionally to target weights
        total_asset_weight = sum(
            w.target_pct for w in target_weights if w.weight_type == "asset"
        )

        trades = []
        for target in target_weights:
            if target.weight_type == "cash":
                continue

            if total_asset_weight <= 0:
                continue

            allocation = (target.target_pct / total_asset_weight) * investable
            if allocation < self.MIN_ORDER_VALUE:
                continue

            price = prices.get(target.symbol, 0.0)
            if price <= 0:
                continue

            shares = allocation / price

            trades.append(
                {
                    "symbol": target.symbol,
                    "side": "buy",
                    "notional": allocation,
                    "shares": shares,
                    "price": price,
                    "order_type": "market",
                    "source": "invest_cash",
                }
            )

        return trades

    def record_rebalance(
        self,
        run_type: str,
        condition: RebalanceCondition,
        weights: list[PortfolioWeight],
        orders: list[dict[str, Any]],
        success: bool = True,
        error: str | None = None,
    ) -> RebalanceRun:
        """Record a completed rebalance run.

        Args:
            run_type: 'full_rebalance' or 'invest_cash'
            condition: What triggered the rebalance
            weights: Target weights used
            orders: Orders executed
            success: Whether run completed successfully
            error: Error message if failed

        Returns:
            RebalanceRun record
        """
        run = RebalanceRun(
            run_id=f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            run_type=run_type,
            condition=condition,
            status="completed" if success else "failed",
            weights=weights,
            orders=orders,
            completed_at=datetime.now(UTC) if success else None,
            error_message=error,
        )

        self._run_history.append(run)

        if success:
            self._last_rebalance = datetime.now(UTC)

        return run

    def get_run_history(self, limit: int | None = None) -> list[RebalanceRun]:
        """Get rebalance run history.

        Args:
            limit: Maximum runs to return

        Returns:
            List of RebalanceRun records
        """
        if limit:
            return self._run_history[-limit:]
        return list(self._run_history)

    def should_calendar_rebalance(self, now: datetime | None = None) -> bool:
        """Check if calendar-based rebalance is due.

        Args:
            now: Current timestamp (default: now)

        Returns:
            True if calendar rebalance should occur
        """
        if not self.calendar_config:
            return False

        now = now or datetime.now(UTC)
        freq = self.calendar_config.frequency

        if freq == "daily":
            return True
        elif freq == "weekly":
            return now.weekday() == self.calendar_config.day_of_week
        elif freq == "monthly":
            return now.day == self.calendar_config.day_of_month
        elif freq == "quarterly":
            return now.day == self.calendar_config.day_of_month and now.month in (
                1,
                4,
                7,
                10,
            )

        return False

    def estimate_rebalance_cost(
        self,
        trades: list[dict[str, Any]],
    ) -> float:
        """Estimate total transaction cost for rebalance.

        Args:
            trades: Proposed trades

        Returns:
            Estimated cost in dollars
        """
        total_notional = sum(t.get("notional", 0) for t in trades)
        return total_notional * (self.transaction_cost_bps / 10000.0)

    def is_in_cooldown(self) -> bool:
        """Check if we're in cooldown period."""
        if not self._last_rebalance or self.drift_config.cooldown_days <= 0:
            return False

        cooldown_end = self._last_rebalance + timedelta(
            days=self.drift_config.cooldown_days
        )
        return datetime.now(UTC) < cooldown_end
