# Fixed Fractional Position Sizing

**Category**: Position Sizing
**Complexity**: Basic to Intermediate
**Prerequisites**: Risk management fundamentals

---

## Overview

Fixed fractional position sizing risks a constant percentage of capital on each trade, automatically adjusting position size as account equity changes. This approach provides geometric growth potential while maintaining consistent risk.

---

## Core Methods

### 1. Percent Risk Model

Risk a fixed percentage of equity per trade:

```
Position Size = (Account * Risk%) / (Entry - Stop)
```

### 2. Percent Volatility Model

Size based on ATR or volatility:

```
Position Size = (Account * Risk%) / (ATR * Multiplier)
```

### 3. Fixed Ratio Model (Ryan Jones)

Increases position at fixed equity intervals:

```
Contracts = floor(sqrt(2 * Equity / Delta + 0.25) - 0.5)
```

---

## Python Implementation

```python
"""
Fixed Fractional Position Sizing Methods
Production-ready implementations for trading systems.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum
import numpy as np


class SizingMethod(Enum):
    """Position sizing methods."""
    PERCENT_RISK = "percent_risk"
    PERCENT_EQUITY = "percent_equity"
    PERCENT_VOLATILITY = "percent_volatility"
    FIXED_RATIO = "fixed_ratio"
    OPTIMAL_F = "optimal_f"


@dataclass
class PositionSizeResult:
    """Position sizing calculation result."""
    shares: float
    notional_value: float
    risk_amount: float
    risk_percent: float
    method: SizingMethod


class FixedFractionalSizing:
    """
    Fixed fractional position sizing calculator.

    Implements multiple sizing methods with risk controls.
    """

    def __init__(
        self,
        risk_percent: float = 0.01,      # 1% risk per trade
        max_position_percent: float = 0.20,  # 20% max position
        min_position_value: float = 100.0,   # Minimum $100
        round_to_lots: bool = True,
        lot_size: int = 100
    ):
        self.risk_percent = risk_percent
        self.max_position_percent = max_position_percent
        self.min_position_value = min_position_value
        self.round_to_lots = round_to_lots
        self.lot_size = lot_size

    def percent_risk(
        self,
        account_value: float,
        entry_price: float,
        stop_price: float
    ) -> PositionSizeResult:
        """
        Calculate position size based on percent risk per trade.

        This is the most common method - risks fixed % of equity.

        Args:
            account_value: Current account value
            entry_price: Planned entry price
            stop_price: Stop loss price

        Returns:
            PositionSizeResult with sizing details
        """
        if entry_price <= 0 or stop_price <= 0:
            raise ValueError("Prices must be positive")

        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share == 0:
            raise ValueError("Entry and stop cannot be equal")

        risk_amount = account_value * self.risk_percent
        raw_shares = risk_amount / risk_per_share

        # Apply position limits
        max_shares = (account_value * self.max_position_percent) / entry_price
        shares = min(raw_shares, max_shares)

        # Round to lot size
        if self.round_to_lots:
            shares = (shares // self.lot_size) * self.lot_size

        # Minimum position check
        if shares * entry_price < self.min_position_value:
            shares = 0

        notional = shares * entry_price
        actual_risk = shares * risk_per_share

        return PositionSizeResult(
            shares=shares,
            notional_value=notional,
            risk_amount=actual_risk,
            risk_percent=actual_risk / account_value if account_value > 0 else 0,
            method=SizingMethod.PERCENT_RISK
        )

    def percent_equity(
        self,
        account_value: float,
        entry_price: float,
        allocation_percent: float = 0.10
    ) -> PositionSizeResult:
        """
        Allocate fixed percentage of equity to position.

        Simpler than percent risk - just allocates % of capital.

        Args:
            account_value: Current account value
            entry_price: Entry price
            allocation_percent: Percent of equity to allocate

        Returns:
            PositionSizeResult
        """
        allocation = min(allocation_percent, self.max_position_percent)
        notional = account_value * allocation
        raw_shares = notional / entry_price

        if self.round_to_lots:
            shares = (raw_shares // self.lot_size) * self.lot_size
        else:
            shares = raw_shares

        actual_notional = shares * entry_price

        return PositionSizeResult(
            shares=shares,
            notional_value=actual_notional,
            risk_amount=actual_notional,  # Full position at risk
            risk_percent=allocation,
            method=SizingMethod.PERCENT_EQUITY
        )

    def percent_volatility(
        self,
        account_value: float,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 2.0
    ) -> PositionSizeResult:
        """
        Size position based on ATR/volatility.

        Adjusts size for current market volatility.

        Args:
            account_value: Current account value
            entry_price: Entry price
            atr: Average True Range
            atr_multiplier: ATR multiplier for stop distance

        Returns:
            PositionSizeResult
        """
        if atr <= 0:
            raise ValueError("ATR must be positive")

        stop_distance = atr * atr_multiplier
        risk_amount = account_value * self.risk_percent
        raw_shares = risk_amount / stop_distance

        # Apply limits
        max_shares = (account_value * self.max_position_percent) / entry_price
        shares = min(raw_shares, max_shares)

        if self.round_to_lots:
            shares = (shares // self.lot_size) * self.lot_size

        notional = shares * entry_price
        actual_risk = shares * stop_distance

        return PositionSizeResult(
            shares=shares,
            notional_value=notional,
            risk_amount=actual_risk,
            risk_percent=actual_risk / account_value if account_value > 0 else 0,
            method=SizingMethod.PERCENT_VOLATILITY
        )


class FixedRatioSizing:
    """
    Ryan Jones Fixed Ratio position sizing.

    Increases position size at fixed equity intervals (delta).
    More conservative than fixed fractional for smaller accounts.
    """

    def __init__(
        self,
        delta: float = 5000,      # Equity per contract increase
        starting_units: int = 1,   # Initial contracts/lots
        max_units: int = 100       # Maximum position
    ):
        self.delta = delta
        self.starting_units = starting_units
        self.max_units = max_units

    def calculate_units(self, account_value: float) -> int:
        """
        Calculate number of units based on Fixed Ratio formula.

        Formula: N = floor(sqrt(2*E/D + 0.25) - 0.5)
        Where E = equity, D = delta

        Args:
            account_value: Current account value

        Returns:
            Number of units to trade
        """
        if account_value <= 0:
            return 0

        # Fixed ratio formula
        units = int(np.floor(
            np.sqrt(2 * account_value / self.delta + 0.25) - 0.5
        ))

        units = max(units, self.starting_units)
        units = min(units, self.max_units)

        return units

    def calculate_position(
        self,
        account_value: float,
        entry_price: float,
        contract_value: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate full position sizing.

        Args:
            account_value: Current account value
            entry_price: Entry price
            contract_value: Value per contract/lot

        Returns:
            Dictionary with position details
        """
        units = self.calculate_units(account_value)
        shares = units * self.lot_size if hasattr(self, 'lot_size') else units

        return {
            'units': units,
            'shares': shares,
            'notional': shares * entry_price,
            'next_level': self._next_level_equity(units),
            'prev_level': self._prev_level_equity(units)
        }

    def _next_level_equity(self, current_units: int) -> float:
        """Calculate equity needed for next unit increase."""
        n = current_units + 1
        return self.delta * n * (n + 1) / 2

    def _prev_level_equity(self, current_units: int) -> float:
        """Calculate equity threshold for current units."""
        n = current_units
        return self.delta * n * (n - 1) / 2


class AntiMartingale:
    """
    Anti-Martingale position sizing.

    Increases position size after wins, decreases after losses.
    Follows the "let winners run" philosophy.
    """

    def __init__(
        self,
        base_risk: float = 0.01,
        win_increase: float = 0.005,
        loss_decrease: float = 0.003,
        max_risk: float = 0.03,
        min_risk: float = 0.005
    ):
        self.base_risk = base_risk
        self.win_increase = win_increase
        self.loss_decrease = loss_decrease
        self.max_risk = max_risk
        self.min_risk = min_risk
        self.current_risk = base_risk
        self.consecutive_wins = 0
        self.consecutive_losses = 0

    def update(self, trade_result: float) -> float:
        """
        Update risk level based on trade result.

        Args:
            trade_result: P/L of last trade (positive = win)

        Returns:
            Updated risk percentage
        """
        if trade_result > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.current_risk = min(
                self.current_risk + self.win_increase,
                self.max_risk
            )
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.current_risk = max(
                self.current_risk - self.loss_decrease,
                self.min_risk
            )

        return self.current_risk

    def reset(self):
        """Reset to base risk level."""
        self.current_risk = self.base_risk
        self.consecutive_wins = 0
        self.consecutive_losses = 0


class PositionSizingEngine:
    """
    Unified position sizing engine combining multiple methods.

    Production-ready with validation and risk controls.
    """

    def __init__(
        self,
        account_value: float,
        default_method: SizingMethod = SizingMethod.PERCENT_RISK,
        default_risk: float = 0.01
    ):
        self.account_value = account_value
        self.default_method = default_method
        self.default_risk = default_risk

        self.fixed_fractional = FixedFractionalSizing(risk_percent=default_risk)
        self.fixed_ratio = FixedRatioSizing()
        self.anti_martingale = AntiMartingale(base_risk=default_risk)

    def calculate(
        self,
        entry_price: float,
        stop_price: Optional[float] = None,
        atr: Optional[float] = None,
        method: Optional[SizingMethod] = None
    ) -> PositionSizeResult:
        """
        Calculate position size using specified method.

        Args:
            entry_price: Entry price
            stop_price: Stop loss price (for percent risk)
            atr: ATR value (for volatility sizing)
            method: Sizing method to use

        Returns:
            PositionSizeResult
        """
        method = method or self.default_method

        if method == SizingMethod.PERCENT_RISK:
            if stop_price is None:
                raise ValueError("Stop price required for percent risk method")
            return self.fixed_fractional.percent_risk(
                self.account_value, entry_price, stop_price
            )

        elif method == SizingMethod.PERCENT_EQUITY:
            return self.fixed_fractional.percent_equity(
                self.account_value, entry_price
            )

        elif method == SizingMethod.PERCENT_VOLATILITY:
            if atr is None:
                raise ValueError("ATR required for volatility method")
            return self.fixed_fractional.percent_volatility(
                self.account_value, entry_price, atr
            )

        else:
            raise ValueError(f"Unknown method: {method}")

    def update_account(self, new_value: float):
        """Update account value after trade."""
        self.account_value = new_value
```

---

## Method Comparison

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| Percent Risk | Most traders | Consistent risk | Requires stop |
| Percent Equity | Long-term | Simple | No stop control |
| Percent Volatility | Adaptive | Vol-adjusted | ATR lag |
| Fixed Ratio | Small accounts | Conservative | Complex |
| Anti-Martingale | Trending | Rides winners | Streak dependent |

---

## Integration with Ordinis

```python
from ordinis.risk.position_sizing import PositionSizingEngine, SizingMethod

engine = PositionSizingEngine(
    account_value=100000,
    default_risk=0.01  # 1% per trade
)

# Calculate position
result = engine.calculate(
    entry_price=150.00,
    stop_price=145.00,
    method=SizingMethod.PERCENT_RISK
)

print(f"Shares: {result.shares}")
print(f"Risk: ${result.risk_amount:.2f} ({result.risk_percent:.2%})")
```

---

## References

1. Van Tharp (2006). "Trade Your Way to Financial Freedom"
2. Ryan Jones (1999). "The Trading Game"
3. Ralph Vince (1990). "Portfolio Management Formulas"
