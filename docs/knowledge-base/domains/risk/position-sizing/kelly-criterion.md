# Kelly Criterion Position Sizing

**Category**: Position Sizing
**Complexity**: Advanced
**Prerequisites**: Probability theory, expected value, logarithmic utility

---

## Overview

The Kelly Criterion determines the optimal fraction of capital to allocate to a bet or investment, maximizing long-term geometric growth rate while avoiding ruin.

### Key Concepts

- **Optimal f**: Fraction maximizing log-utility
- **Edge**: Expected return per unit wagered
- **Odds**: Ratio of win amount to loss amount
- **Fractional Kelly**: Risk-adjusted Kelly for practical use

---

## Mathematical Foundation

### Basic Kelly Formula

For a binary outcome (win/loss):

```
f* = (p * b - q) / b = (p * (b + 1) - 1) / b
```

Where:
- f* = Optimal fraction of capital
- p = Probability of winning
- q = Probability of losing (1 - p)
- b = Odds received (win/loss ratio)

### Continuous Kelly (Merton)

For continuous returns with known mean and variance:

```
f* = μ / σ²
```

Where:
- μ = Expected excess return
- σ² = Variance of returns

---

## Python Implementation

```python
"""
Kelly Criterion Position Sizing Implementation
Production-ready with risk controls and validation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar


class KellyVariant(Enum):
    """Kelly calculation variants."""
    BASIC = "basic"           # Simple win/loss
    CONTINUOUS = "continuous"  # Gaussian returns
    EMPIRICAL = "empirical"    # From historical data
    MULTI_ASSET = "multi_asset"  # Portfolio optimization


@dataclass
class KellyResult:
    """Kelly criterion calculation result."""
    optimal_fraction: float
    adjusted_fraction: float
    expected_growth: float
    ruin_probability: float
    kelly_variant: KellyVariant
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        # Validate results
        if not -1.0 <= self.optimal_fraction <= 10.0:
            raise ValueError(f"Invalid Kelly fraction: {self.optimal_fraction}")


@dataclass
class TradeStatistics:
    """Statistics for Kelly calculation."""
    win_rate: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float = field(init=False)
    expectancy: float = field(init=False)

    def __post_init__(self):
        if self.avg_loss == 0:
            raise ValueError("Average loss cannot be zero")
        self.win_loss_ratio = abs(self.avg_win / self.avg_loss)
        self.expectancy = (self.win_rate * self.avg_win -
                          (1 - self.win_rate) * abs(self.avg_loss))


class KellyCriterion:
    """
    Kelly Criterion calculator with multiple variants and risk controls.

    Supports:
    - Basic binary Kelly
    - Continuous (Gaussian) Kelly
    - Empirical Kelly from returns
    - Multi-asset Kelly optimization
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,  # Fractional Kelly multiplier
        max_position: float = 0.25,    # Maximum position size
        min_position: float = 0.01,    # Minimum position size
        confidence_level: float = 0.95
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.min_position = min_position
        self.confidence_level = confidence_level

    def basic_kelly(
        self,
        win_probability: float,
        win_loss_ratio: float
    ) -> KellyResult:
        """
        Calculate basic Kelly for binary outcomes.

        Args:
            win_probability: Probability of winning (0-1)
            win_loss_ratio: Ratio of average win to average loss

        Returns:
            KellyResult with optimal and adjusted fractions
        """
        if not 0 < win_probability < 1:
            raise ValueError("Win probability must be between 0 and 1")
        if win_loss_ratio <= 0:
            raise ValueError("Win/loss ratio must be positive")

        p = win_probability
        q = 1 - p
        b = win_loss_ratio

        # Kelly formula
        kelly = (p * b - q) / b

        # Adjust for fractional Kelly
        adjusted = kelly * self.kelly_fraction
        adjusted = np.clip(adjusted, -self.max_position, self.max_position)

        # Expected log growth
        if kelly > 0:
            growth = p * np.log(1 + kelly * b) + q * np.log(1 - kelly)
        else:
            growth = 0.0

        # Approximate ruin probability (simplified)
        if kelly > 0 and kelly < 1:
            ruin_prob = ((q / p) ** (1 / kelly)) if p > q else 1.0
        else:
            ruin_prob = 1.0 if kelly <= 0 else 0.0

        return KellyResult(
            optimal_fraction=kelly,
            adjusted_fraction=adjusted,
            expected_growth=growth,
            ruin_probability=min(ruin_prob, 1.0),
            kelly_variant=KellyVariant.BASIC
        )

    def continuous_kelly(
        self,
        expected_return: float,
        volatility: float,
        risk_free_rate: float = 0.0
    ) -> KellyResult:
        """
        Calculate Kelly for continuous (Gaussian) returns.

        This is the Merton optimal portfolio formula.

        Args:
            expected_return: Expected annual return
            volatility: Annual standard deviation
            risk_free_rate: Risk-free rate

        Returns:
            KellyResult with optimal leverage
        """
        if volatility <= 0:
            raise ValueError("Volatility must be positive")

        excess_return = expected_return - risk_free_rate
        variance = volatility ** 2

        # Merton formula
        kelly = excess_return / variance

        # Adjust for fractional Kelly
        adjusted = kelly * self.kelly_fraction
        adjusted = np.clip(adjusted, -self.max_position * 4, self.max_position * 4)

        # Expected log growth (geometric return approximation)
        growth = excess_return - 0.5 * variance * kelly

        # Ruin probability under geometric Brownian motion
        if kelly > 0 and excess_return > 0:
            sharpe = excess_return / volatility
            ruin_prob = np.exp(-2 * sharpe * kelly)
        else:
            ruin_prob = 1.0

        return KellyResult(
            optimal_fraction=kelly,
            adjusted_fraction=adjusted,
            expected_growth=growth,
            ruin_probability=min(ruin_prob, 1.0),
            kelly_variant=KellyVariant.CONTINUOUS
        )

    def empirical_kelly(
        self,
        returns: np.ndarray,
        bootstrap_samples: int = 1000
    ) -> KellyResult:
        """
        Calculate Kelly from empirical return distribution.

        Uses numerical optimization to find fraction maximizing
        expected log growth.

        Args:
            returns: Array of historical returns
            bootstrap_samples: Number of bootstrap iterations

        Returns:
            KellyResult with confidence intervals
        """
        returns = np.asarray(returns)
        if len(returns) < 30:
            raise ValueError("Need at least 30 observations")

        def neg_expected_log_growth(f: float) -> float:
            """Negative expected log growth for minimization."""
            if f <= -1 or f >= 10:
                return 1e10
            growth = np.mean(np.log(1 + f * returns))
            return -growth

        # Find optimal f
        result = minimize_scalar(
            neg_expected_log_growth,
            bounds=(-0.5, 5.0),
            method='bounded'
        )
        kelly = result.x
        growth = -result.fun

        # Bootstrap confidence interval
        kelly_samples = []
        for _ in range(bootstrap_samples):
            sample = np.random.choice(returns, size=len(returns), replace=True)

            def neg_growth(f):
                if f <= -1 or f >= 10:
                    return 1e10
                return -np.mean(np.log(1 + f * sample))

            res = minimize_scalar(neg_growth, bounds=(-0.5, 5.0), method='bounded')
            kelly_samples.append(res.x)

        ci_low = np.percentile(kelly_samples, (1 - self.confidence_level) * 50)
        ci_high = np.percentile(kelly_samples, 50 + self.confidence_level * 50)

        # Adjusted fraction
        adjusted = kelly * self.kelly_fraction
        adjusted = np.clip(adjusted, -self.max_position, self.max_position)

        # Empirical ruin probability (simplified)
        drawdowns = self._calculate_drawdowns(returns * kelly)
        max_dd = np.min(drawdowns)
        ruin_prob = 1 - np.exp(max_dd * 2)  # Heuristic

        return KellyResult(
            optimal_fraction=kelly,
            adjusted_fraction=adjusted,
            expected_growth=growth,
            ruin_probability=min(max(ruin_prob, 0), 1),
            kelly_variant=KellyVariant.EMPIRICAL,
            confidence_interval=(ci_low, ci_high)
        )

    def multi_asset_kelly(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> Tuple[np.ndarray, KellyResult]:
        """
        Calculate Kelly weights for multiple assets.

        Uses mean-variance optimization with Kelly criterion.

        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            risk_free_rate: Risk-free rate

        Returns:
            Tuple of (weight vector, KellyResult summary)
        """
        n_assets = len(expected_returns)
        mu = np.asarray(expected_returns) - risk_free_rate
        cov = np.asarray(covariance_matrix)

        if cov.shape != (n_assets, n_assets):
            raise ValueError("Covariance matrix dimension mismatch")

        # Optimal Kelly weights: f* = Σ^(-1) * μ
        try:
            cov_inv = np.linalg.inv(cov)
            kelly_weights = cov_inv @ mu
        except np.linalg.LinAlgError:
            # Use pseudo-inverse for singular matrices
            cov_inv = np.linalg.pinv(cov)
            kelly_weights = cov_inv @ mu

        # Adjust weights
        adjusted_weights = kelly_weights * self.kelly_fraction

        # Apply position limits
        max_weight = self.max_position * 2  # Allow higher for diversified
        adjusted_weights = np.clip(adjusted_weights, -max_weight, max_weight)

        # Portfolio metrics
        total_leverage = np.sum(np.abs(kelly_weights))
        port_return = kelly_weights @ mu
        port_var = kelly_weights @ cov @ kelly_weights
        port_vol = np.sqrt(port_var)

        growth = port_return - 0.5 * port_var

        return adjusted_weights, KellyResult(
            optimal_fraction=total_leverage,
            adjusted_fraction=np.sum(np.abs(adjusted_weights)),
            expected_growth=growth,
            ruin_probability=0.0,  # Complex to calculate
            kelly_variant=KellyVariant.MULTI_ASSET
        )

    def from_trade_statistics(
        self,
        stats: TradeStatistics
    ) -> KellyResult:
        """
        Calculate Kelly from trade statistics.

        Args:
            stats: TradeStatistics object

        Returns:
            KellyResult
        """
        return self.basic_kelly(
            win_probability=stats.win_rate,
            win_loss_ratio=stats.win_loss_ratio
        )

    def _calculate_drawdowns(self, returns: np.ndarray) -> np.ndarray:
        """Calculate running drawdown series."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns


class FractionalKellyManager:
    """
    Manages fractional Kelly with dynamic adjustment.

    Reduces Kelly fraction during drawdowns and
    increases during favorable conditions.
    """

    def __init__(
        self,
        base_fraction: float = 0.5,
        min_fraction: float = 0.1,
        max_fraction: float = 0.75,
        drawdown_threshold: float = 0.10,
        recovery_rate: float = 0.1
    ):
        self.base_fraction = base_fraction
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.drawdown_threshold = drawdown_threshold
        self.recovery_rate = recovery_rate
        self.current_fraction = base_fraction
        self.peak_equity = 1.0
        self.current_equity = 1.0

    def update(self, equity: float) -> float:
        """
        Update Kelly fraction based on current equity.

        Args:
            equity: Current portfolio equity

        Returns:
            Updated Kelly fraction
        """
        self.current_equity = equity

        if equity > self.peak_equity:
            self.peak_equity = equity
            # Gradually increase fraction
            self.current_fraction = min(
                self.current_fraction + self.recovery_rate * 0.1,
                self.max_fraction
            )
        else:
            # Calculate drawdown
            drawdown = (self.peak_equity - equity) / self.peak_equity

            if drawdown > self.drawdown_threshold:
                # Reduce fraction proportionally to drawdown
                reduction = drawdown / self.drawdown_threshold
                self.current_fraction = max(
                    self.base_fraction * (1 - reduction * 0.5),
                    self.min_fraction
                )

        return self.current_fraction

    def get_position_size(
        self,
        kelly_result: KellyResult,
        account_value: float
    ) -> float:
        """
        Calculate position size with dynamic Kelly adjustment.

        Args:
            kelly_result: Base Kelly calculation
            account_value: Current account value

        Returns:
            Dollar position size
        """
        adjusted_kelly = kelly_result.optimal_fraction * self.current_fraction
        return account_value * adjusted_kelly
```

---

## Practical Considerations

### Why Fractional Kelly?

| Full Kelly | Half Kelly | Quarter Kelly |
|------------|------------|---------------|
| Maximum growth | 75% growth, 50% volatility | 56% growth, 25% volatility |
| High variance | Moderate variance | Low variance |
| Aggressive | Balanced | Conservative |

### Estimation Error Impact

Kelly is highly sensitive to input errors:
- 10% error in win rate -> ~20% error in optimal f
- Always use fractional Kelly (0.25-0.5x) for safety

### When Kelly Breaks Down

1. **Fat tails**: Kelly assumes known distributions
2. **Serial correlation**: Assumes independent bets
3. **Estimation error**: Historical stats may not persist
4. **Leverage constraints**: Practical limits on position size
5. **Liquidity**: Can't always get desired size

---

## Integration with Ordinis

```python
from ordinis.risk.position_sizing import KellyCriterion, TradeStatistics

# Initialize Kelly calculator
kelly = KellyCriterion(
    kelly_fraction=0.5,
    max_position=0.20
)

# From trade statistics
stats = TradeStatistics(
    win_rate=0.55,
    avg_win=0.02,
    avg_loss=0.015
)
result = kelly.from_trade_statistics(stats)
print(f"Optimal position: {result.adjusted_fraction:.1%}")

# From historical returns
returns = np.array([...])  # Historical returns
result = kelly.empirical_kelly(returns)
print(f"Kelly fraction: {result.optimal_fraction:.2f}")
print(f"95% CI: [{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]")
```

---

## References

1. Kelly, J.L. (1956). "A New Interpretation of Information Rate"
2. Thorp, E.O. (2006). "The Kelly Criterion in Blackjack, Sports Betting and the Stock Market"
3. MacLean, Thorp, Ziemba (2011). "The Kelly Capital Growth Investment Criterion"
4. Merton, R.C. (1969). "Lifetime Portfolio Selection under Uncertainty"
