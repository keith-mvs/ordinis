# Implied vs Realized Volatility

## Overview

Comparing implied volatility (IV) from options prices to realized/historical volatility (RV/HV) reveals market expectations versus actual behavior. This relationship drives volatility trading strategies.

---

## Definitions

### Realized Volatility (RV / HV)
```
RV = std(log_returns) × sqrt(252) × 100

Where:
- log_returns = ln(close / close[1])
- 252 = trading days per year (annualization)
```

### Implied Volatility (IV)
- Derived from option prices using Black-Scholes or similar models
- Represents market's expectation of future volatility
- Forward-looking (priced into options)

---

## Standard Parameters

| Metric | Period | Use Case |
|--------|--------|----------|
| RV (Short) | 10-20 days | Recent volatility |
| RV (Medium) | 30-60 days | Standard measurement |
| RV (Long) | 90-252 days | Baseline/normal volatility |
| IV | 30 days (VIX) | Standard expectation |

---

## Key Relationships

### IV vs RV Premium
```python
# Volatility Risk Premium (VRP)
VRP = IV - RV

# Historical average: IV > RV about 85% of the time
NORMAL_VRP = IV > RV

# Interpretation
if VRP > historical_avg:
    # Options "expensive" - favor selling premium
elif VRP < 0:
    # Rare - options "cheap" - favor buying premium
```

### IV Rank / IV Percentile
```python
# IV Rank (0-100)
IV_RANK = (IV - IV_52wk_low) / (IV_52wk_high - IV_52wk_low) × 100

# IV Percentile (% of days IV was lower)
IV_PERCENTILE = percentile_rank(IV, lookback=252)

# Interpretation
HIGH_IV = IV_RANK > 50 OR IV_PERCENTILE > 50
LOW_IV = IV_RANK < 25 OR IV_PERCENTILE < 25
```

---

## Rule Templates

### Volatility State
```python
# Current volatility regime
LOW_VOL = RV_20 < RV_252.quantile(0.25)
NORMAL_VOL = RV_252.quantile(0.25) <= RV_20 <= RV_252.quantile(0.75)
HIGH_VOL = RV_20 > RV_252.quantile(0.75)

# Volatility trend
VOL_EXPANDING = RV_10 > RV_20 > RV_60
VOL_CONTRACTING = RV_10 < RV_20 < RV_60
```

### Premium Signals
```python
# Rich premium (sell strategies)
RICH_PREMIUM = (
    IV > RV_20 * 1.2 AND            # IV 20%+ above RV
    IV_RANK > 50                     # Above median IV
)

# Cheap premium (buy strategies)
CHEAP_PREMIUM = (
    IV < RV_20 * 0.9 AND            # IV below RV
    IV_RANK < 25                     # Low IV rank
)
```

### Volatility Mean Reversion
```python
# IV tends to mean-revert
IV_ELEVATED = IV_RANK > 80
EXPECT_IV_DROP = IV_ELEVATED AND IV < IV[1]

IV_DEPRESSED = IV_RANK < 20
EXPECT_IV_RISE = IV_DEPRESSED AND IV > IV[1]
```

---

## Trading Strategies

### 1. Volatility Risk Premium Harvesting
```python
# Systematically sell premium when IV > RV
SELL_PREMIUM_CONDITIONS = (
    IV > RV_30 AND
    IV_RANK > 30 AND
    days_to_expiration > 30
)

# Strategies: Short straddles, strangles, iron condors
# Edge: Collect premium as IV > subsequent RV
```

### 2. Volatility Crush Plays
```python
# After events (earnings), IV typically drops
PRE_EVENT_HIGH_IV = (
    days_to_earnings < 5 AND
    IV_RANK > 70
)

# Sell premium before event, profit from IV crush
POST_EVENT_IV_CRUSH = (
    IV < IV[1] * 0.8                 # 20%+ IV drop
)
```

### 3. Long Volatility
```python
# Buy options when IV unusually low
LONG_VOL_SETUP = (
    IV_RANK < 20 AND                 # Low IV
    RV_10 > RV_60 AND                # Recent vol pickup
    IV < RV_20                       # IV below realized
)

# Strategies: Long straddles, strangles, calls/puts
```

### 4. Volatility Spread
```python
# Trade IV term structure
CONTANGO = IV_30 < IV_60 < IV_90    # Normal: longer = higher IV
BACKWARDATION = IV_30 > IV_60       # Inverted: fear in near-term

# Trade calendar spreads based on term structure
if BACKWARDATION:
    sell_front_month()
    buy_back_month()
```

---

## Volatility Indicators

### VIX (S&P 500 Implied Volatility)
```python
# VIX levels
LOW_VIX = VIX < 15
NORMAL_VIX = 15 <= VIX <= 25
ELEVATED_VIX = 25 < VIX <= 35
EXTREME_VIX = VIX > 35

# VIX spikes often mark bottoms
VIX_SPIKE = VIX > VIX.rolling(20).mean() + 2 * VIX.rolling(20).std()
POTENTIAL_BOTTOM = VIX_SPIKE AND SPY.close < SPY.SMA(20)
```

### VIX Term Structure
```python
# VIX futures vs spot
VIX_CONTANGO = VIX_FUTURE_1 > VIX_SPOT
VIX_BACKWARDATION = VIX_FUTURE_1 < VIX_SPOT

# Backwardation = fear, often signals bottom
```

---

## Realized Volatility Calculations

### Close-to-Close
```python
def realized_volatility_cc(prices, period=20):
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(period).std() * np.sqrt(252) * 100
```

### Parkinson (High-Low)
```python
def parkinson_volatility(high, low, period=20):
    # More efficient estimator using intraday range
    hl_ratio = np.log(high / low)
    return np.sqrt(hl_ratio.pow(2).rolling(period).mean() / (4 * np.log(2))) * np.sqrt(252) * 100
```

### Yang-Zhang
```python
# Most efficient estimator, uses OHLC
def yang_zhang_volatility(open, high, low, close, period=20):
    # Combines overnight and intraday volatility
    # Implementation in technical indicators library
    pass
```

---

## Applications for Stock Trading

### Position Sizing by Volatility
```python
# Inverse volatility sizing
POSITION_WEIGHT = 1 / RV_20

# Normalized across portfolio
NORMALIZED_WEIGHT = POSITION_WEIGHT / sum(all_position_weights)

# Higher vol = smaller position
```

### Volatility-Adjusted Stops
```python
# Wider stops in high volatility
def get_stop_distance(price, rv):
    base_atr_multiple = 2.0
    vol_adjustment = rv / rv.rolling(252).mean()
    return price * 0.02 * vol_adjustment  # 2% base, adjusted
```

### Entry Timing
```python
# Avoid entries during volatility spikes
STABLE_ENTRY = RV_10 < RV_60 * 1.3

# Or buy the fear
FEAR_ENTRY = (
    VIX > VIX.rolling(20).mean() + 1.5 * VIX.rolling(20).std() AND
    price at support
)
```

---

## Common Pitfalls

1. **IV is Not a Prediction**: Just market expectation, often wrong
2. **Volatility Clustering**: High vol begets high vol
3. **Fat Tails**: Realized vol underestimates extreme moves
4. **Mean Reversion Timing**: IV can stay elevated longer than expected

---

## Implementation

```python
from ordinis.analysis.technical.indicators import VolatilityIndicators

vol = VolatilityIndicators()

# Realized volatility
rv_20 = vol.realized_volatility(data, period=20)
rv_60 = vol.realized_volatility(data, period=60)

# Volatility ratio
vol_ratio = rv_20 / rv_60

# Volatility regime
vol_percentile = rv_20.rolling(252).apply(
    lambda x: percentileofscore(x, x.iloc[-1])
)

# High/low vol states
high_vol = vol_percentile > 75
low_vol = vol_percentile < 25
```

---

## Academic Notes

- **Black-Scholes (1973)**: Foundation for IV calculation
- **Volatility Risk Premium**: Well-documented in literature
- **Key Insight**: IV typically exceeds subsequent RV (insurance premium)

**Best Practice**: Use IV vs RV comparison for options strategies, and RV alone for stock position sizing and risk management.
