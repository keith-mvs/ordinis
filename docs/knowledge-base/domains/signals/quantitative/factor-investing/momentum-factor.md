# Momentum Factor

## Overview

Momentum is the tendency for assets that have performed well (poorly) recently to continue performing well (poorly). It's one of the most robust anomalies in finance, documented across asset classes and time periods.

---

## Academic Foundation

### Jegadeesh & Titman (1993)
- Formation period: 3-12 months
- Holding period: 3-12 months
- Skip most recent month (short-term reversal)
- Long winners, short losers

### Carhart (1997)
- Added momentum to FF 3-factor model
- MOM = Return of winners minus losers

---

## Momentum Definitions

### Price Momentum (12-1 Month)
```python
def calculate_momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Classic 12-1 month momentum (skip most recent month).

    Return from 12 months ago to 1 month ago.
    """
    # 12-month return
    ret_12m = prices.pct_change(252)

    # 1-month return (to skip)
    ret_1m = prices.pct_change(21)

    # 12-1 momentum
    momentum = (1 + ret_12m) / (1 + ret_1m) - 1

    return momentum
```

### Alternative Momentum Measures
```python
def momentum_variants(prices: pd.DataFrame) -> dict:
    """
    Various momentum definitions.
    """
    return {
        # Standard 12-1 month
        'mom_12_1': (prices.shift(21) / prices.shift(252)) - 1,

        # 6-1 month (shorter horizon)
        'mom_6_1': (prices.shift(21) / prices.shift(126)) - 1,

        # 12-month (including recent)
        'mom_12': prices.pct_change(252),

        # 3-month
        'mom_3': prices.pct_change(63),

        # 52-week high momentum
        'high_52w': prices / prices.rolling(252).max(),
    }
```

### Residual Momentum
```python
def residual_momentum(returns: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.Series:
    """
    Momentum after controlling for factor exposures.
    Captures stock-specific momentum.
    """
    from sklearn.linear_model import LinearRegression

    residuals = {}
    for stock in returns.columns:
        y = returns[stock].dropna()
        X = factor_returns.loc[y.index]

        model = LinearRegression()
        model.fit(X, y)

        resid = y - model.predict(X)
        residuals[stock] = resid.iloc[-252:-21].sum()  # 12-1 month residual return

    return pd.Series(residuals)
```

---

## Strategy Implementation

### Basic Momentum Strategy
```python
class MomentumStrategy:
    def __init__(
        self,
        formation_period: int = 252,
        skip_period: int = 21,
        holding_period: int = 21,
        n_long: int = 50,
        n_short: int = 50
    ):
        self.formation = formation_period
        self.skip = skip_period
        self.holding = holding_period
        self.n_long = n_long
        self.n_short = n_short

    def rank_stocks(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Rank stocks by momentum.
        """
        # Calculate momentum (skip recent month)
        mom = (prices.shift(self.skip) / prices.shift(self.formation)) - 1

        # Rank (higher = better momentum)
        ranks = mom.rank(axis=1, ascending=False)

        return ranks

    def select_stocks(self, ranks: pd.Series) -> dict:
        """
        Select long and short portfolios.
        """
        sorted_ranks = ranks.sort_values()

        return {
            'long': sorted_ranks.head(self.n_long).index.tolist(),
            'short': sorted_ranks.tail(self.n_short).index.tolist()
        }

    def calculate_weights(self, selection: dict) -> pd.Series:
        """
        Equal-weight within long/short legs.
        """
        weights = pd.Series(0.0)

        for stock in selection['long']:
            weights[stock] = 1 / self.n_long

        for stock in selection['short']:
            weights[stock] = -1 / self.n_short

        return weights
```

### 52-Week High Momentum
```python
def high_52_week_momentum(prices: pd.DataFrame) -> pd.Series:
    """
    Momentum based on proximity to 52-week high.
    Stocks near 52-week high continue to outperform.
    """
    high_52w = prices.rolling(252).max()
    nearness = prices / high_52w

    return nearness
```

### Industry-Adjusted Momentum
```python
def industry_adjusted_momentum(
    momentum: pd.DataFrame,
    industry: pd.Series
) -> pd.DataFrame:
    """
    Momentum relative to industry peers.
    Removes sector effects.
    """
    adjusted = momentum.copy()

    for ind in industry.unique():
        mask = industry == ind
        ind_stocks = momentum.loc[:, mask]
        ind_mean = ind_stocks.mean(axis=1)

        for stock in ind_stocks.columns:
            adjusted[stock] = momentum[stock] - ind_mean

    return adjusted
```

---

## Momentum Crashes

### Risk of Momentum Strategy
```
Momentum crashes occur during sharp market reversals:
- 2009 (post-GFC reversal): -73% in 2 months
- 2020 (COVID reversal): Significant drawdown
- Losers rally, winners fall
```

### Crash Protection
```python
def momentum_crash_protection(
    momentum_returns: pd.Series,
    market_returns: pd.Series,
    lookback: int = 21
) -> float:
    """
    Reduce exposure when crash risk is high.
    """
    # Market volatility
    market_vol = market_returns.rolling(lookback).std() * np.sqrt(252)

    # Recent market drawdown
    market_dd = market_returns.rolling(lookback).sum()

    # Momentum volatility
    mom_vol = momentum_returns.rolling(lookback).std() * np.sqrt(252)

    # High vol + negative market = crash risk
    crash_risk = (market_vol > market_vol.quantile(0.8)) & (market_dd < -0.05)

    # Scale exposure inversely with crash risk
    exposure = 1.0
    if crash_risk.iloc[-1]:
        exposure = 0.3  # Reduce to 30%

    return exposure

def dynamic_momentum_hedge(
    momentum_weights: pd.Series,
    market_condition: str
) -> pd.Series:
    """
    Hedge momentum in adverse conditions.
    """
    if market_condition == 'HIGH_VOL_REVERSAL':
        # Reduce long momentum, increase short
        return momentum_weights * 0.5
    elif market_condition == 'BEAR_MARKET_BOTTOM':
        # Momentum likely to crash on reversal
        return momentum_weights * 0.25
    return momentum_weights
```

---

## Time-Series Momentum (Trend Following)

```python
def time_series_momentum(prices: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """
    Time-series momentum: Long if positive return, short if negative.
    Unlike cross-sectional, this is absolute, not relative.
    """
    returns = prices.pct_change(lookback)

    signals = np.sign(returns)  # +1 or -1

    return signals
```

---

## Factor Attribution

```python
def momentum_factor_return(
    returns: pd.DataFrame,
    momentum_ranks: pd.DataFrame,
    n_quantiles: int = 10
) -> pd.Series:
    """
    Calculate momentum factor return (long-short).
    """
    factor_returns = []

    for date in returns.index:
        if date not in momentum_ranks.index:
            continue

        ranks = momentum_ranks.loc[date]
        rets = returns.loc[date]

        # Top decile (winners)
        winners = ranks[ranks >= ranks.quantile(0.9)].index
        winner_ret = rets[winners].mean()

        # Bottom decile (losers)
        losers = ranks[ranks <= ranks.quantile(0.1)].index
        loser_ret = rets[losers].mean()

        # Factor return
        factor_returns.append(winner_ret - loser_ret)

    return pd.Series(factor_returns, index=returns.index[:len(factor_returns)])
```

---

## Momentum Decay

```python
def momentum_decay_analysis(
    formation_returns: pd.DataFrame,
    forward_returns: pd.DataFrame,
    periods: list = [1, 3, 6, 12, 24]
) -> dict:
    """
    Analyze how momentum effect decays over time.
    """
    decay = {}

    for period in periods:
        # Forward return at each horizon
        fwd_ret = forward_returns.shift(-period * 21)  # Months to days

        # Correlation with formation momentum
        corr = formation_returns.corrwith(fwd_ret, axis=1).mean()
        decay[f'{period}m'] = corr

    return decay

# Momentum typically reverses after 12-18 months
```

---

## Performance Characteristics

| Metric | Typical Value |
|--------|---------------|
| Annual Return | 6-10% |
| Volatility | 15-20% |
| Sharpe Ratio | 0.4-0.6 |
| Max Drawdown | 30-50% (crashes) |
| Turnover | 100-200% annually |
| Market Correlation | 0.0-0.3 |

---

## Implementation Costs

```python
def momentum_cost_analysis(
    turnover: float,
    transaction_cost: float = 0.001,
    market_impact: float = 0.002
) -> float:
    """
    Estimate cost drag on momentum strategy.
    """
    # Total cost per turnover
    cost_per_trade = transaction_cost + market_impact

    # Annual cost
    annual_cost = turnover * cost_per_trade * 2  # Both sides

    return annual_cost

# High turnover (~150%) makes costs significant
# Expected annual cost: 150% * 0.3% * 2 = ~0.9%
```

---

## Academic References

- Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers"
- Carhart (1997): Momentum factor in mutual fund performance
- Moskowitz, Ooi, Pedersen (2012): Time-series momentum
- Daniel & Moskowitz (2016): Momentum crashes
- Asness et al. (2013): "Value and Momentum Everywhere"
