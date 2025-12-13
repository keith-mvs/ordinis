# Mean Reversion Strategies

## Overview

Mean reversion strategies profit from the tendency of prices to return to a historical average. When prices deviate significantly, trade expecting reversion.

**Mathematical Foundation**: See [10_mathematical_foundations - OU Process](../../10_mathematical_foundations/README.md#25-mean-reverting-processes)

---

## Core Concept

```
Price → Deviates from mean → Trade expecting return → Profit on convergence

Entry: Price far from mean (oversold/overbought)
Exit: Price returns to mean
Stop: Deviation continues (trend, not mean-reversion)
```

---

## Detection Methods

### 1. Z-Score Method
```python
def zscore_signal(prices: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Standard deviation from rolling mean.
    """
    mean = prices.rolling(lookback).mean()
    std = prices.rolling(lookback).std()
    return (prices - mean) / std

# Signal rules
OVERSOLD = zscore < -2.0   # Buy
OVERBOUGHT = zscore > 2.0  # Sell/Short
```

### 2. Bollinger Band Method
```python
def bollinger_signal(prices: pd.Series, lookback: int = 20, num_std: float = 2.0):
    """
    Price relative to Bollinger Bands.
    """
    mean = prices.rolling(lookback).mean()
    std = prices.rolling(lookback).std()

    upper = mean + num_std * std
    lower = mean - num_std * std

    pct_b = (prices - lower) / (upper - lower)

    return {
        'upper': upper,
        'lower': lower,
        'pct_b': pct_b
    }

# Signal rules
BUY = prices < lower   # At lower band
SELL = prices > upper  # At upper band
```

### 3. RSI Method
```python
def rsi_mean_reversion(rsi: pd.Series, oversold: int = 30, overbought: int = 70):
    """
    RSI-based mean reversion.
    """
    signals = pd.Series(0, index=rsi.index)
    signals[rsi < oversold] = 1   # Buy when oversold
    signals[rsi > overbought] = -1  # Sell when overbought
    return signals
```

### 4. Hurst Exponent
```python
def hurst_exponent(series: pd.Series, lags: range = range(2, 100)) -> float:
    """
    Estimate Hurst exponent to identify mean-reverting series.

    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    """
    tau = []
    lagvec = []

    for lag in lags:
        # Calculate variance of lagged differences
        pp = series.diff(lag).dropna()
        tau.append(np.sqrt(np.std(pp)))
        lagvec.append(lag)

    # Fit log-log relationship
    m = np.polyfit(np.log(lagvec), np.log(tau), 1)
    return m[0]

# Interpretation
H = hurst_exponent(prices)
if H < 0.5:
    print("Mean-reverting - suitable for mean reversion strategy")
elif H > 0.5:
    print("Trending - use trend-following instead")
```

---

## Strategy Variants

### 1. Simple Z-Score Reversion
```python
class ZScoreReversion:
    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.0
    ):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        zscore = self._calculate_zscore(prices)

        signals = pd.Series(0, index=prices.index)
        position = 0

        for i in range(self.lookback, len(prices)):
            z = zscore.iloc[i]

            # Entry
            if position == 0:
                if z < -self.entry_z:
                    position = 1
                    signals.iloc[i] = 1
                elif z > self.entry_z:
                    position = -1
                    signals.iloc[i] = -1

            # Exit
            elif position == 1:
                if z >= -self.exit_z or z < -self.stop_z:
                    position = 0
                    signals.iloc[i] = 0
            elif position == -1:
                if z <= self.exit_z or z > self.stop_z:
                    position = 0
                    signals.iloc[i] = 0

        return signals
```

### 2. Adaptive Mean Reversion
```python
class AdaptiveMeanReversion:
    """
    Adjusts parameters based on recent volatility regime.
    """
    def __init__(self, base_lookback: int = 20):
        self.base_lookback = base_lookback

    def adaptive_lookback(self, volatility: pd.Series) -> pd.Series:
        """
        Longer lookback in high volatility, shorter in low.
        """
        vol_percentile = volatility.rolling(252).apply(
            lambda x: np.percentile(x, len(x) / 2)
        )

        # Scale lookback inversely with volatility
        lookback = self.base_lookback * (1 + vol_percentile)
        return lookback.clip(10, 60)  # Bounds

    def adaptive_threshold(self, volatility: pd.Series) -> pd.Series:
        """
        Higher threshold (more deviation) in high volatility.
        """
        vol_ratio = volatility / volatility.rolling(60).mean()
        return 2.0 * vol_ratio.clip(0.5, 2.0)
```

### 3. Ornstein-Uhlenbeck Based
```python
class OUMeanReversion:
    """
    Uses fitted OU process parameters for signal generation.
    """
    def __init__(self, window: int = 60):
        self.window = window

    def fit_ou_parameters(self, spread: pd.Series) -> dict:
        """
        Estimate θ (mean reversion speed), μ (long-term mean), σ (volatility).
        """
        # See 10_mathematical_foundations for implementation
        from ordinis.analysis.technical.statistics import estimate_ou_parameters
        return estimate_ou_parameters(spread)

    def expected_value(self, current: float, mu: float, theta: float, dt: float) -> float:
        """
        Expected value at time t + dt under OU process.
        """
        return mu + (current - mu) * np.exp(-theta * dt)

    def signal(self, current: float, params: dict, threshold: float = 0.5) -> int:
        """
        Signal based on expected deviation from current.
        """
        expected = self.expected_value(
            current,
            params['mu'],
            params['theta'],
            dt=params['half_life'] / 2
        )

        deviation = (expected - current) / params['sigma']

        if deviation > threshold:
            return 1  # Expect price to rise
        elif deviation < -threshold:
            return -1  # Expect price to fall
        return 0
```

---

## When Mean Reversion Works

| Condition | Mean Reversion | Why |
|-----------|----------------|-----|
| Range-bound market | Strong | Prices oscillate around mean |
| Low ADX (< 20) | Strong | No trend to override |
| High VIX spike | Strong | Fear overshoots, then reverts |
| Earnings surprise | Weak | Fundamental shift |
| Trend breakout | Weak | New regime, mean shifts |

---

## When Mean Reversion Fails

1. **Regime Change**: Fundamentals shift, old mean irrelevant
2. **Strong Trend**: Momentum overpowers reversion
3. **Structural Break**: M&A, bankruptcy, sector disruption
4. **Black Swan**: Extreme events beyond historical range

---

## Risk Management

```python
MEAN_REVERSION_LIMITS = {
    'max_holding_period': 10,       # Days - exit if no reversion
    'max_z_stop': 3.5,              # Exit if deviation continues
    'profit_target_z': 0.0,         # Exit at mean (or slightly beyond)
    'max_position_size': 0.05,      # 5% of equity per trade
    'max_correlation': 0.7,         # Avoid correlated mean-reversion bets
}

def mean_reversion_stop(entry_z: float, current_z: float, max_z: float) -> bool:
    """
    Stop loss if deviation continues beyond max threshold.
    """
    if entry_z < 0:  # Long position (bought oversold)
        return current_z < -max_z
    else:  # Short position (sold overbought)
        return current_z > max_z
```

---

## Regime Filter

```python
def should_trade_mean_reversion(adx: float, atr_percentile: float) -> bool:
    """
    Only trade mean reversion in appropriate regime.
    """
    # Low ADX = ranging market
    ranging = adx < 25

    # Normal volatility (not extreme)
    normal_vol = 20 < atr_percentile < 80

    return ranging and normal_vol
```

---

## Performance Metrics

```python
def mean_reversion_metrics(trades: list) -> dict:
    """
    Metrics specific to mean reversion strategies.
    """
    return {
        'avg_holding_period': np.mean([t.holding_days for t in trades]),
        'reversion_rate': sum(1 for t in trades if t.hit_target) / len(trades),
        'avg_entry_z': np.mean([t.entry_z for t in trades]),
        'avg_exit_z': np.mean([t.exit_z for t in trades]),
        'stopped_out_pct': sum(1 for t in trades if t.stopped) / len(trades)
    }
```

---

## Academic References

- Poterba & Summers (1988): "Mean Reversion in Stock Prices"
- Lo & MacKinlay (1988): "Stock Market Prices Do Not Follow Random Walks"
- Fama & French (1988): "Permanent and Temporary Components of Stock Prices"
