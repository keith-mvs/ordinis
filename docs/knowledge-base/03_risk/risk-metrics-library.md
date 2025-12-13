# Risk Metrics Library

**Category**: Risk Management
**Complexity**: Intermediate
**Prerequisites**: Statistics, portfolio theory

---

## Overview

Comprehensive library of risk and performance metrics for evaluating trading strategies and portfolio performance.

---

## Python Implementation

```python
"""
Risk Metrics Library
Production-ready performance and risk metric calculations.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PerformanceMetrics:
    """Complete performance metrics snapshot."""
    # Returns
    total_return: float
    annualized_return: float
    cagr: float

    # Risk
    volatility: float
    downside_deviation: float
    max_drawdown: float
    var_95: float
    cvar_95: float

    # Risk-Adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float

    # Trade Statistics
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float


class RiskMetrics:
    """
    Comprehensive risk and performance metric calculations.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ):
        self.rf = risk_free_rate
        self.periods = periods_per_year
        self.sqrt_periods = np.sqrt(periods_per_year)

    # ==================== RETURN METRICS ====================

    def total_return(self, returns: np.ndarray) -> float:
        """Total cumulative return."""
        return np.prod(1 + returns) - 1

    def annualized_return(self, returns: np.ndarray) -> float:
        """Annualized return from period returns."""
        total = self.total_return(returns)
        years = len(returns) / self.periods
        if years <= 0:
            return 0.0
        return (1 + total) ** (1 / years) - 1

    def cagr(self, equity: np.ndarray) -> float:
        """Compound Annual Growth Rate."""
        if len(equity) < 2 or equity[0] <= 0:
            return 0.0
        years = len(equity) / self.periods
        return (equity[-1] / equity[0]) ** (1 / years) - 1

    # ==================== VOLATILITY METRICS ====================

    def volatility(self, returns: np.ndarray) -> float:
        """Annualized volatility (standard deviation)."""
        return np.std(returns, ddof=1) * self.sqrt_periods

    def downside_deviation(
        self,
        returns: np.ndarray,
        mar: float = 0.0
    ) -> float:
        """
        Downside deviation (semi-deviation below MAR).

        Args:
            returns: Return series
            mar: Minimum Acceptable Return (default 0)
        """
        downside = returns[returns < mar]
        if len(downside) == 0:
            return 0.0
        return np.std(downside, ddof=1) * self.sqrt_periods

    def upside_deviation(self, returns: np.ndarray, mar: float = 0.0) -> float:
        """Upside deviation above MAR."""
        upside = returns[returns > mar]
        if len(upside) == 0:
            return 0.0
        return np.std(upside, ddof=1) * self.sqrt_periods

    # ==================== DRAWDOWN METRICS ====================

    def max_drawdown(self, returns: np.ndarray) -> float:
        """Maximum drawdown from return series."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return abs(np.min(drawdowns))

    def max_drawdown_duration(self, returns: np.ndarray) -> int:
        """Longest drawdown duration in periods."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        in_drawdown = cumulative < running_max

        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def ulcer_index(self, returns: np.ndarray) -> float:
        """Ulcer Index - pain-weighted drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return np.sqrt(np.mean(drawdowns ** 2))

    # ==================== VALUE AT RISK ====================

    def var_historical(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Historical Value at Risk."""
        return -np.percentile(returns, (1 - confidence) * 100)

    def var_parametric(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Parametric (Gaussian) VaR."""
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        z = stats.norm.ppf(1 - confidence)
        return -(mu + z * sigma)

    def cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Conditional VaR (Expected Shortfall)."""
        var = self.var_historical(returns, confidence)
        tail_returns = returns[returns <= -var]
        if len(tail_returns) == 0:
            return var
        return -np.mean(tail_returns)

    # ==================== RISK-ADJUSTED RETURNS ====================

    def sharpe_ratio(self, returns: np.ndarray) -> float:
        """Sharpe Ratio (excess return / volatility)."""
        excess = returns - self.rf / self.periods
        vol = np.std(excess, ddof=1)
        if vol == 0:
            return 0.0
        return np.mean(excess) / vol * self.sqrt_periods

    def sortino_ratio(
        self,
        returns: np.ndarray,
        mar: float = 0.0
    ) -> float:
        """Sortino Ratio (excess return / downside deviation)."""
        excess = np.mean(returns) - mar / self.periods
        downside_dev = self.downside_deviation(returns, mar / self.periods)
        if downside_dev == 0:
            return 0.0
        return excess * self.periods / downside_dev

    def calmar_ratio(self, returns: np.ndarray) -> float:
        """Calmar Ratio (annualized return / max drawdown)."""
        ann_ret = self.annualized_return(returns)
        max_dd = self.max_drawdown(returns)
        if max_dd == 0:
            return float('inf') if ann_ret > 0 else 0.0
        return ann_ret / max_dd

    def omega_ratio(
        self,
        returns: np.ndarray,
        threshold: float = 0.0
    ) -> float:
        """Omega Ratio (probability weighted return ratio)."""
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())
        if losses == 0:
            return float('inf') if gains > 0 else 1.0
        return gains / losses

    def information_ratio(
        self,
        returns: np.ndarray,
        benchmark: np.ndarray
    ) -> float:
        """Information Ratio (excess return / tracking error)."""
        active = returns - benchmark
        tracking_error = np.std(active, ddof=1)
        if tracking_error == 0:
            return 0.0
        return np.mean(active) / tracking_error * self.sqrt_periods

    def treynor_ratio(
        self,
        returns: np.ndarray,
        benchmark: np.ndarray
    ) -> float:
        """Treynor Ratio (excess return / beta)."""
        beta = self.beta(returns, benchmark)
        if beta == 0:
            return 0.0
        excess_return = self.annualized_return(returns) - self.rf
        return excess_return / beta

    # ==================== REGRESSION METRICS ====================

    def beta(
        self,
        returns: np.ndarray,
        benchmark: np.ndarray
    ) -> float:
        """Beta to benchmark."""
        cov = np.cov(returns, benchmark)[0, 1]
        var = np.var(benchmark, ddof=1)
        if var == 0:
            return 0.0
        return cov / var

    def alpha(
        self,
        returns: np.ndarray,
        benchmark: np.ndarray
    ) -> float:
        """Jensen's Alpha (annualized)."""
        b = self.beta(returns, benchmark)
        r_p = self.annualized_return(returns)
        r_m = self.annualized_return(benchmark)
        return r_p - (self.rf + b * (r_m - self.rf))

    def r_squared(
        self,
        returns: np.ndarray,
        benchmark: np.ndarray
    ) -> float:
        """R-squared of returns vs benchmark."""
        corr = np.corrcoef(returns, benchmark)[0, 1]
        return corr ** 2

    # ==================== TRADE STATISTICS ====================

    def win_rate(self, trades: np.ndarray) -> float:
        """Percentage of winning trades."""
        if len(trades) == 0:
            return 0.0
        return np.sum(trades > 0) / len(trades)

    def profit_factor(self, trades: np.ndarray) -> float:
        """Gross profit / gross loss."""
        gains = np.sum(trades[trades > 0])
        losses = abs(np.sum(trades[trades < 0]))
        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        return gains / losses

    def expectancy(self, trades: np.ndarray) -> float:
        """Expected value per trade."""
        if len(trades) == 0:
            return 0.0
        return np.mean(trades)

    def payoff_ratio(self, trades: np.ndarray) -> float:
        """Average win / average loss."""
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        if len(losses) == 0 or len(wins) == 0:
            return 0.0
        return np.mean(wins) / abs(np.mean(losses))

    def kelly_criterion(self, trades: np.ndarray) -> float:
        """Optimal Kelly fraction from trade history."""
        wr = self.win_rate(trades)
        pr = self.payoff_ratio(trades)
        if pr == 0:
            return 0.0
        return wr - (1 - wr) / pr

    # ==================== COMPREHENSIVE REPORT ====================

    def calculate_all(
        self,
        returns: np.ndarray,
        trades: Optional[np.ndarray] = None,
        benchmark: Optional[np.ndarray] = None
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            returns: Period returns
            trades: Individual trade P/L (optional)
            benchmark: Benchmark returns (optional)

        Returns:
            PerformanceMetrics dataclass
        """
        if trades is None:
            trades = returns

        equity = np.cumprod(1 + returns)

        return PerformanceMetrics(
            total_return=self.total_return(returns),
            annualized_return=self.annualized_return(returns),
            cagr=self.cagr(equity),
            volatility=self.volatility(returns),
            downside_deviation=self.downside_deviation(returns),
            max_drawdown=self.max_drawdown(returns),
            var_95=self.var_historical(returns, 0.95),
            cvar_95=self.cvar(returns, 0.95),
            sharpe_ratio=self.sharpe_ratio(returns),
            sortino_ratio=self.sortino_ratio(returns),
            calmar_ratio=self.calmar_ratio(returns),
            omega_ratio=self.omega_ratio(returns),
            win_rate=self.win_rate(trades),
            profit_factor=self.profit_factor(trades),
            expectancy=self.expectancy(trades),
            avg_win=np.mean(trades[trades > 0]) if np.any(trades > 0) else 0,
            avg_loss=np.mean(trades[trades < 0]) if np.any(trades < 0) else 0
        )
```

---

## Metric Reference Table

| Metric | Good | Excellent | Interpretation |
|--------|------|-----------|----------------|
| Sharpe | > 1.0 | > 2.0 | Risk-adjusted return |
| Sortino | > 1.5 | > 3.0 | Downside-adjusted |
| Calmar | > 1.0 | > 3.0 | Return per drawdown |
| Max DD | < 20% | < 10% | Worst decline |
| Win Rate | > 50% | > 60% | Trade accuracy |
| Profit Factor | > 1.5 | > 2.5 | Gain/loss ratio |

---

## Integration with Ordinis

```python
from ordinis.risk.metrics import RiskMetrics

metrics = RiskMetrics(risk_free_rate=0.05)

# Calculate all metrics
report = metrics.calculate_all(
    returns=strategy_returns,
    trades=trade_pnl
)

print(f"Sharpe: {report.sharpe_ratio:.2f}")
print(f"Max DD: {report.max_drawdown:.1%}")
print(f"Win Rate: {report.win_rate:.1%}")
```

---

## References

1. Bacon (2008). "Practical Portfolio Performance"
2. Sortino & Price (1994). "Performance Measurement"
3. Keating & Shadwick (2002). "Omega Function"
