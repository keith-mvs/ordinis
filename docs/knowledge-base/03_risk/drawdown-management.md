# Drawdown Management

**Category**: Risk Management
**Complexity**: Advanced
**Prerequisites**: Portfolio theory, volatility concepts, behavioral finance

---

## Overview

Drawdown management encompasses strategies to limit, recover from, and adapt to equity declines. Effective drawdown control is essential for capital preservation and psychological sustainability.

---

## Key Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Drawdown | (Peak - Current) / Peak | Current decline from peak |
| Max Drawdown | Max(Drawdown_t) | Worst peak-to-trough |
| Calmar Ratio | CAGR / MaxDD | Return per unit drawdown |
| Ulcer Index | sqrt(mean(DD²)) | Severity-weighted drawdown |

---

## Python Implementation

```python
"""
Drawdown Management System
Production-ready drawdown monitoring, circuit breakers, and recovery.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class DrawdownState(Enum):
    """Current drawdown state."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    CIRCUIT_BREAKER = "circuit_breaker"
    RECOVERY = "recovery"


class CircuitBreakerAction(Enum):
    """Actions when circuit breaker triggers."""
    REDUCE_SIZE = "reduce_size"
    FLATTEN_ALL = "flatten_all"
    STOP_NEW_TRADES = "stop_new_trades"
    HALT_TRADING = "halt_trading"


@dataclass
class DrawdownMetrics:
    """Comprehensive drawdown metrics."""
    current_drawdown: float
    max_drawdown: float
    drawdown_duration: int  # Days in current drawdown
    time_to_recovery: Optional[int] = None
    ulcer_index: float = 0.0
    pain_index: float = 0.0
    calmar_ratio: float = 0.0
    state: DrawdownState = DrawdownState.NORMAL


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    warning_threshold: float = 0.05     # 5% - enter warning state
    critical_threshold: float = 0.10    # 10% - enter critical state
    halt_threshold: float = 0.15        # 15% - halt trading
    daily_loss_limit: float = 0.02      # 2% daily loss limit
    weekly_loss_limit: float = 0.05     # 5% weekly loss limit
    cooldown_period: int = 1            # Days after halt
    position_reduction: float = 0.50    # Reduce by 50% in critical


class DrawdownCalculator:
    """
    Calculate various drawdown metrics from equity curve.
    """

    @staticmethod
    def calculate_drawdown_series(equity: np.ndarray) -> np.ndarray:
        """Calculate running drawdown series."""
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return drawdown

    @staticmethod
    def max_drawdown(equity: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        dd_series = DrawdownCalculator.calculate_drawdown_series(equity)
        return abs(np.min(dd_series))

    @staticmethod
    def drawdown_duration(equity: np.ndarray) -> int:
        """Calculate current drawdown duration in periods."""
        running_max = np.maximum.accumulate(equity)
        in_drawdown = equity < running_max

        if not in_drawdown[-1]:
            return 0

        # Count consecutive periods in drawdown from end
        duration = 0
        for i in range(len(equity) - 1, -1, -1):
            if in_drawdown[i]:
                duration += 1
            else:
                break
        return duration

    @staticmethod
    def ulcer_index(equity: np.ndarray) -> float:
        """
        Calculate Ulcer Index - severity-weighted drawdown measure.

        UI = sqrt(mean(DD²)) where DD is drawdown percentage
        """
        dd_series = DrawdownCalculator.calculate_drawdown_series(equity)
        return np.sqrt(np.mean(dd_series ** 2))

    @staticmethod
    def pain_index(equity: np.ndarray) -> float:
        """
        Calculate Pain Index - average drawdown depth.

        Pain = mean(|DD|)
        """
        dd_series = DrawdownCalculator.calculate_drawdown_series(equity)
        return np.mean(np.abs(dd_series))

    @staticmethod
    def calmar_ratio(
        equity: np.ndarray,
        periods_per_year: int = 252
    ) -> float:
        """Calculate Calmar Ratio (CAGR / Max Drawdown)."""
        if len(equity) < 2:
            return 0.0

        # Calculate CAGR
        total_return = equity[-1] / equity[0]
        years = len(equity) / periods_per_year
        cagr = total_return ** (1 / years) - 1

        # Max drawdown
        max_dd = DrawdownCalculator.max_drawdown(equity)

        if max_dd == 0:
            return float('inf') if cagr > 0 else 0.0

        return cagr / max_dd

    @staticmethod
    def recovery_time_distribution(equity: np.ndarray) -> Dict[str, float]:
        """
        Analyze historical recovery times from drawdowns.

        Returns statistics on how long recoveries typically take.
        """
        running_max = np.maximum.accumulate(equity)
        at_high = equity >= running_max

        recovery_times = []
        in_dd_start = None

        for i, is_high in enumerate(at_high):
            if not is_high and in_dd_start is None:
                in_dd_start = i
            elif is_high and in_dd_start is not None:
                recovery_times.append(i - in_dd_start)
                in_dd_start = None

        if len(recovery_times) == 0:
            return {'mean': 0, 'median': 0, 'max': 0, 'count': 0}

        return {
            'mean': np.mean(recovery_times),
            'median': np.median(recovery_times),
            'max': np.max(recovery_times),
            'count': len(recovery_times)
        }


class CircuitBreaker:
    """
    Trading circuit breaker with configurable thresholds.

    Implements automatic risk reduction when drawdowns exceed limits.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = DrawdownState.NORMAL
        self.halt_until: Optional[datetime] = None
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self._callbacks: List[Callable] = []

    def check(
        self,
        current_drawdown: float,
        daily_return: float = 0.0
    ) -> Dict[str, any]:
        """
        Check drawdown levels and update state.

        Args:
            current_drawdown: Current drawdown (positive value)
            daily_return: Today's return

        Returns:
            Dictionary with state and recommended action
        """
        self.daily_pnl = daily_return
        self.weekly_pnl += daily_return

        # Check if in cooldown
        if self.halt_until and datetime.now() < self.halt_until:
            return {
                'state': DrawdownState.CIRCUIT_BREAKER,
                'action': CircuitBreakerAction.HALT_TRADING,
                'message': f'In cooldown until {self.halt_until}',
                'can_trade': False
            }

        # Check daily loss limit
        if daily_return < -self.config.daily_loss_limit:
            self._trigger_halt("Daily loss limit exceeded")
            return self._build_response(
                CircuitBreakerAction.HALT_TRADING,
                "Daily loss limit exceeded"
            )

        # Check weekly loss limit
        if self.weekly_pnl < -self.config.weekly_loss_limit:
            self._trigger_halt("Weekly loss limit exceeded")
            return self._build_response(
                CircuitBreakerAction.HALT_TRADING,
                "Weekly loss limit exceeded"
            )

        # Check drawdown thresholds
        dd = abs(current_drawdown)

        if dd >= self.config.halt_threshold:
            self._trigger_halt("Max drawdown threshold exceeded")
            return self._build_response(
                CircuitBreakerAction.HALT_TRADING,
                f"Drawdown {dd:.1%} exceeds halt threshold"
            )

        elif dd >= self.config.critical_threshold:
            self.state = DrawdownState.CRITICAL
            return self._build_response(
                CircuitBreakerAction.REDUCE_SIZE,
                f"Critical drawdown: {dd:.1%}",
                position_scalar=1 - self.config.position_reduction
            )

        elif dd >= self.config.warning_threshold:
            self.state = DrawdownState.WARNING
            return self._build_response(
                CircuitBreakerAction.STOP_NEW_TRADES,
                f"Warning drawdown: {dd:.1%}"
            )

        else:
            self.state = DrawdownState.NORMAL
            return {
                'state': DrawdownState.NORMAL,
                'action': None,
                'message': 'Normal operation',
                'can_trade': True,
                'position_scalar': 1.0
            }

    def _trigger_halt(self, reason: str):
        """Trigger trading halt."""
        self.state = DrawdownState.CIRCUIT_BREAKER
        self.halt_until = datetime.now() + timedelta(days=self.config.cooldown_period)
        self._notify_callbacks(reason)

    def _build_response(
        self,
        action: CircuitBreakerAction,
        message: str,
        position_scalar: float = 0.0
    ) -> Dict:
        return {
            'state': self.state,
            'action': action,
            'message': message,
            'can_trade': action not in [
                CircuitBreakerAction.HALT_TRADING,
                CircuitBreakerAction.FLATTEN_ALL
            ],
            'position_scalar': position_scalar
        }

    def _notify_callbacks(self, reason: str):
        for callback in self._callbacks:
            callback(self.state, reason)

    def register_callback(self, callback: Callable):
        """Register callback for state changes."""
        self._callbacks.append(callback)

    def reset_weekly(self):
        """Reset weekly P/L counter (call on week start)."""
        self.weekly_pnl = 0.0

    def manual_reset(self):
        """Manually reset circuit breaker."""
        self.state = DrawdownState.NORMAL
        self.halt_until = None
        self.daily_pnl = 0.0


class DrawdownRecoveryManager:
    """
    Manages position sizing during drawdown recovery.

    Implements gradual size increase as equity recovers.
    """

    def __init__(
        self,
        base_size: float = 1.0,
        recovery_threshold: float = 0.5,  # Start recovery at 50% DD recovered
        full_recovery_threshold: float = 0.9,  # Full size at 90% recovered
        min_size: float = 0.25
    ):
        self.base_size = base_size
        self.recovery_threshold = recovery_threshold
        self.full_recovery_threshold = full_recovery_threshold
        self.min_size = min_size
        self.max_drawdown = 0.0
        self.peak_equity = 0.0

    def update(self, equity: float) -> float:
        """
        Update state and return position size scalar.

        Args:
            equity: Current equity value

        Returns:
            Position size scalar (0-1)
        """
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.max_drawdown = 0.0
            return self.base_size

        current_dd = (self.peak_equity - equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, current_dd)

        if self.max_drawdown == 0:
            return self.base_size

        # Calculate recovery percentage
        recovery_pct = 1 - (current_dd / self.max_drawdown)

        if recovery_pct >= self.full_recovery_threshold:
            return self.base_size
        elif recovery_pct >= self.recovery_threshold:
            # Linear interpolation
            progress = (recovery_pct - self.recovery_threshold) / \
                      (self.full_recovery_threshold - self.recovery_threshold)
            return self.min_size + progress * (self.base_size - self.min_size)
        else:
            return self.min_size


class EquityCurveTrading:
    """
    Trade the equity curve - reduce/increase size based on performance.

    When equity is above its moving average, trade full size.
    When below, reduce or pause trading.
    """

    def __init__(
        self,
        lookback: int = 20,
        below_ma_scalar: float = 0.5,
        significantly_below_scalar: float = 0.0,
        significance_threshold: float = 0.02
    ):
        self.lookback = lookback
        self.below_ma_scalar = below_ma_scalar
        self.significantly_below_scalar = significantly_below_scalar
        self.significance_threshold = significance_threshold
        self.equity_history: List[float] = []

    def update(self, equity: float) -> float:
        """
        Update equity history and return position scalar.

        Args:
            equity: Current equity

        Returns:
            Position scalar (0-1)
        """
        self.equity_history.append(equity)

        if len(self.equity_history) < self.lookback:
            return 1.0

        # Calculate moving average
        recent = self.equity_history[-self.lookback:]
        ma = np.mean(recent)

        # Position relative to MA
        deviation = (equity - ma) / ma

        if deviation < -self.significance_threshold:
            return self.significantly_below_scalar
        elif equity < ma:
            return self.below_ma_scalar
        else:
            return 1.0
```

---

## Recovery Strategies

### 1. Gradual Size Increase
Start small after drawdown, increase as equity recovers

### 2. Equity Curve Filter
Trade full size only when equity above MA

### 3. Time-Based Recovery
Wait fixed period after hitting drawdown threshold

### 4. Win-Streak Recovery
Require consecutive wins before increasing size

---

## Integration with Ordinis

```python
from ordinis.risk.drawdown import (
    CircuitBreaker,
    CircuitBreakerConfig,
    DrawdownRecoveryManager
)

# Configure circuit breaker
config = CircuitBreakerConfig(
    warning_threshold=0.05,
    critical_threshold=0.10,
    halt_threshold=0.15,
    daily_loss_limit=0.02
)

breaker = CircuitBreaker(config)
recovery = DrawdownRecoveryManager()

# Check on each trade
result = breaker.check(current_drawdown=0.08, daily_return=-0.01)
if result['can_trade']:
    size_scalar = recovery.update(current_equity)
    position_size = base_size * size_scalar * result.get('position_scalar', 1.0)
```

---

## References

1. Bacon, C. (2008). "Practical Portfolio Performance"
2. Magdon-Ismail & Atiya (2004). "Maximum Drawdown"
3. Chekhlov et al. (2005). "Drawdown Measure in Portfolio Optimization"
