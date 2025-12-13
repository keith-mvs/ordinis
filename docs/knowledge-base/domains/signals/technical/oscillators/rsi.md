# Relative Strength Index (RSI)

## Overview

The Relative Strength Index, developed by J. Welles Wilder in 1978, measures the speed and magnitude of price changes to identify overbought and oversold conditions. It's one of the most widely used momentum oscillators.

---

## Formula

```
RS = Average Gain / Average Loss (over N periods)
RSI = 100 - (100 / (1 + RS))
```

**Calculation Steps:**
1. Calculate price changes: Change = Close - Close[1]
2. Separate gains (positive changes) and losses (negative changes)
3. Calculate smoothed averages (Wilder's smoothing)
4. Compute RS ratio and transform to 0-100 scale

---

## Standard Parameters

| Parameter | Default | Alternatives |
|-----------|---------|--------------|
| Period | 14 | 7, 9, 21, 28 |
| Overbought | 70 | 80 (strong trend) |
| Oversold | 30 | 20 (strong trend) |

---

## Key Levels

| Level | Meaning | Action |
|-------|---------|--------|
| > 70 | Overbought | Watch for reversal |
| 50 | Neutral/Centerline | Trend confirmation |
| < 30 | Oversold | Watch for bounce |
| > 80 | Extreme overbought | High reversal probability |
| < 20 | Extreme oversold | High bounce probability |

---

## Rule Templates

### Basic Overbought/Oversold
```python
# Simple OB/OS readings
OVERBOUGHT = RSI(14) > 70
OVERSOLD = RSI(14) < 30

# Extreme readings
EXTREME_OB = RSI(14) > 80
EXTREME_OS = RSI(14) < 20
```

### Mean Reversion Entry
```python
# Oversold reversal (long)
LONG_ENTRY = (
    RSI(14) < 30 AND           # Oversold
    RSI(14) > RSI(14)[1] AND   # Turning up
    close > close[1]           # Price confirmation
)

# Overbought reversal (short)
SHORT_ENTRY = (
    RSI(14) > 70 AND           # Overbought
    RSI(14) < RSI(14)[1] AND   # Turning down
    close < close[1]           # Price confirmation
)
```

### Centerline Crossover
```python
# Bullish momentum shift
BULLISH_MOMENTUM = RSI(14) crosses_above 50

# Bearish momentum shift
BEARISH_MOMENTUM = RSI(14) crosses_below 50

# Trend confirmation
UPTREND_CONFIRMED = RSI(14) > 50 AND close > SMA(50)
DOWNTREND_CONFIRMED = RSI(14) < 50 AND close < SMA(50)
```

### RSI Divergence
```python
# Bullish divergence
def bullish_divergence(price, rsi, lookback=14):
    # Price makes lower low
    price_ll = price.iloc[-1] < price.iloc[-lookback:].min()
    # RSI makes higher low
    rsi_hl = rsi.iloc[-1] > rsi.iloc[-lookback:].min()
    return price_ll AND rsi_hl AND rsi.iloc[-1] < 40

# Bearish divergence
def bearish_divergence(price, rsi, lookback=14):
    # Price makes higher high
    price_hh = price.iloc[-1] > price.iloc[-lookback:].max()
    # RSI makes lower high
    rsi_lh = rsi.iloc[-1] < rsi.iloc[-lookback:].max()
    return price_hh AND rsi_lh AND rsi.iloc[-1] > 60
```

### RSI Range Shift
```python
# Bull market range (40-80)
BULL_MARKET = RSI(14) > 40 AND RSI(14).rolling(20).min() > 35

# Bear market range (20-60)
BEAR_MARKET = RSI(14) < 60 AND RSI(14).rolling(20).max() < 65

# Range identification
RSI_RANGE = "bull" if BULL_MARKET else "bear" if BEAR_MARKET else "neutral"
```

### Failure Swings
```python
# Bullish failure swing
# RSI falls below 30, bounces, fails to make new low, breaks above bounce high
BULLISH_FAILURE_SWING = (
    RSI(14)[n] < 30 AND           # Initial oversold
    RSI(14)[m] > RSI(14)[n] AND   # Bounce
    RSI(14) > RSI(14)[m]          # Break bounce high
)

# Bearish failure swing
BEARISH_FAILURE_SWING = (
    RSI(14)[n] > 70 AND           # Initial overbought
    RSI(14)[m] < RSI(14)[n] AND   # Pullback
    RSI(14) < RSI(14)[m]          # Break pullback low
)
```

---

## Trading Strategies

### 1. RSI Reversal with Trend Filter
```python
# Only take longs in uptrend
UPTREND = close > SMA(200)

LONG_SIGNAL = (
    UPTREND AND
    RSI(14) < 35 AND          # Oversold pullback
    RSI(14) > RSI(14)[1]      # Turning up
)

STOP = lowest(low, 5)
TARGET = SMA(20)
```

### 2. RSI + Bollinger Bands
```python
LONG_SIGNAL = (
    close < lower_bollinger AND
    RSI(14) < 30 AND
    RSI(14) > RSI(14)[1]
)

SHORT_SIGNAL = (
    close > upper_bollinger AND
    RSI(14) > 70 AND
    RSI(14) < RSI(14)[1]
)
```

### 3. Multi-Timeframe RSI
```python
# Daily RSI for direction
DAILY_BULLISH = RSI(14, timeframe='D') > 50

# 4H RSI for entry
ENTRY = (
    DAILY_BULLISH AND
    RSI(14, timeframe='4H') < 40 AND
    RSI(14, timeframe='4H') > RSI(14, timeframe='4H')[1]
)
```

---

## Common Pitfalls

1. **Staying Overbought**: RSI can remain > 70 for weeks in bull trends
2. **False Divergences**: Divergence can persist without reversal
3. **Over-reliance**: RSI alone has poor predictive power
4. **Wrong Context**: Mean reversion fails in trending markets

---

## RSI Variations

### Stochastic RSI
```python
# RSI of RSI with stochastic transformation
STOCH_RSI = (RSI - lowest(RSI, 14)) / (highest(RSI, 14) - lowest(RSI, 14))
```

### Connors RSI
```python
# Combines RSI, streak RSI, and percent rank
CONNORS_RSI = (RSI(3) + RSI_STREAK(2) + PERCENT_RANK(100)) / 3
```

---

## Implementation

```python
from ordinis.analysis.technical.indicators import Oscillators

osc = Oscillators()

# Calculate RSI
rsi = osc.rsi(data['close'], period=14)

# With Stochastic RSI
stoch_rsi = osc.stochastic_rsi(data['close'], period=14)
```

---

## Academic Notes

- **Chong & Ng (2008)**: Found some predictive power for RSI in Asian markets
- **Wong et al. (2003)**: RSI works better with filters than standalone
- **Evidence**: Generally weak as standalone predictor; best used with confluence

**Key Insight**: RSI is most effective as a **confirmation tool** rather than primary signal generator.
