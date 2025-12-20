# MACD (Moving Average Convergence Divergence)

## Overview

MACD, developed by Gerald Appel in the 1970s, is a trend-following momentum indicator that shows the relationship between two exponential moving averages. It's one of the most popular and versatile technical indicators.

---

## Formula

```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line
```

---

## Components

| Component | Description | Use |
|-----------|-------------|-----|
| MACD Line | EMA difference | Trend direction |
| Signal Line | 9-period EMA of MACD | Crossover signals |
| Histogram | MACD - Signal | Momentum visualization |
| Zero Line | Horizontal at 0 | Trend confirmation |

---

## Standard Parameters

| Parameter | Default | Aggressive | Conservative |
|-----------|---------|------------|--------------|
| Fast EMA | 12 | 8 | 17 |
| Slow EMA | 26 | 17 | 35 |
| Signal | 9 | 9 | 9 |

---

## Key Signal Types

### 1. Signal Line Crossovers
```python
# Bullish crossover
BULLISH_CROSS = MACD crosses_above Signal

# Bearish crossover
BEARISH_CROSS = MACD crosses_below Signal

# Quality filter: Position relative to zero
STRONG_BULLISH = BULLISH_CROSS AND MACD > 0
STRONG_BEARISH = BEARISH_CROSS AND MACD < 0
```

### 2. Zero Line Crossovers
```python
# Bullish trend confirmation
BULLISH_TREND = MACD crosses_above 0

# Bearish trend confirmation
BEARISH_TREND = MACD crosses_below 0

# Combined signal
CONFIRMED_UPTREND = MACD > 0 AND MACD > Signal
```

### 3. Histogram Analysis
```python
# Momentum increasing
MOMENTUM_UP = histogram > histogram[1] > histogram[2]

# Momentum decreasing
MOMENTUM_DOWN = histogram < histogram[1] < histogram[2]

# Histogram reversal
HIST_REVERSAL_UP = histogram > histogram[1] AND histogram[1] < histogram[2]
HIST_REVERSAL_DOWN = histogram < histogram[1] AND histogram[1] > histogram[2]
```

### 4. Divergence
```python
# Bullish divergence
BULL_DIV = (
    price < price[lookback] AND     # Price lower low
    MACD > MACD[lookback] AND       # MACD higher low
    MACD < 0                         # Below zero line
)

# Bearish divergence
BEAR_DIV = (
    price > price[lookback] AND     # Price higher high
    MACD < MACD[lookback] AND       # MACD lower high
    MACD > 0                         # Above zero line
)
```

---

## Rule Templates

### Basic Entry System
```python
# Bullish entry
LONG_ENTRY = (
    MACD crosses_above Signal AND
    histogram > 0 AND
    close > SMA(50)                  # Trend filter
)

# Bearish entry
SHORT_ENTRY = (
    MACD crosses_below Signal AND
    histogram < 0 AND
    close < SMA(50)
)
```

### Zero Line Confirmation
```python
# High-quality bullish signal
QUALITY_LONG = (
    MACD crosses_above Signal AND    # Crossover
    MACD > 0 OR                       # Already above zero
    MACD crosses_above 0              # Or crossing zero
)
```

### Histogram Momentum
```python
# Early entry on histogram reversal
EARLY_LONG = (
    MACD < Signal AND                 # Still bearish
    histogram > histogram[1] AND      # But histogram turning
    histogram[1] < histogram[2]       # From decreasing
)

# Confirmation entry
CONFIRMED_LONG = (
    histogram > 0 AND                 # Histogram positive
    histogram > histogram[1]          # And increasing
)
```

### Multi-Timeframe MACD
```python
# Weekly MACD direction
WEEKLY_BULL = MACD_weekly > Signal_weekly

# Daily entry aligned with weekly
ALIGNED_LONG = (
    WEEKLY_BULL AND
    MACD_daily crosses_above Signal_daily
)
```

---

## Trading Strategies

### 1. MACD Crossover Strategy
```python
# Simple crossover system
LONG = MACD crosses_above Signal
SHORT = MACD crosses_below Signal

# With zero line filter
FILTERED_LONG = LONG AND MACD > 0
FILTERED_SHORT = SHORT AND MACD < 0

STOP = 2 * ATR(14)
TARGET = 3 * ATR(14)
```

### 2. MACD Divergence Strategy
```python
# Find divergence
BULL_DIV = price.lower_low(14) AND MACD.higher_low(14)

# Entry on divergence confirmation
ENTRY = (
    BULL_DIV AND
    MACD crosses_above Signal AND
    close > close[1]
)

STOP = recent_swing_low
TARGET = previous_resistance
```

### 3. MACD + RSI Combination
```python
STRONG_LONG = (
    MACD crosses_above Signal AND
    RSI(14) > 50 AND
    RSI(14) < 70 AND
    close > SMA(200)
)
```

### 4. Histogram Fade Strategy
```python
# Fade extreme histogram readings
HISTOGRAM_EXTREME = histogram > percentile(histogram, 95, 252)

FADE_ENTRY = (
    HISTOGRAM_EXTREME AND
    histogram < histogram[1]          # Starting to decline
)
```

---

## MACD Variations

### PPO (Percentage Price Oscillator)
```python
# Normalized MACD for cross-security comparison
PPO = (EMA(12) - EMA(26)) / EMA(26) * 100
Signal = EMA(9) of PPO
```

### MACD-Histogram
```python
# Trading the histogram rate of change
HIST_ROC = histogram - histogram[1]
HIST_ACCELERATING = HIST_ROC > HIST_ROC[1]
```

---

## Common Pitfalls

1. **Lag**: MACD is derived from MAs, inherently lagging
2. **Whipsaws**: False signals in ranging markets
3. **Divergence Persistence**: Divergences can last long without reversal
4. **Over-optimization**: Easy to curve-fit parameters

---

## MACD Histogram Patterns

### Positive/Negative Slope
```python
# Bullish: Histogram rising (even if negative)
BULLISH_SLOPE = histogram > histogram[1]

# Bearish: Histogram falling (even if positive)
BEARISH_SLOPE = histogram < histogram[1]
```

### Peak/Trough Pattern
```python
# Histogram peak (potential top)
HIST_PEAK = histogram[1] > histogram AND histogram[1] > histogram[2]

# Histogram trough (potential bottom)
HIST_TROUGH = histogram[1] < histogram AND histogram[1] < histogram[2]
```

---

## Implementation

```python
from ordinis.analysis.technical.indicators import Oscillators

osc = Oscillators()

# Standard MACD
macd_result = osc.macd(data['close'], fast=12, slow=26, signal=9)
macd_line = macd_result['macd']
signal_line = macd_result['signal']
histogram = macd_result['histogram']

# Custom parameters
macd_fast = osc.macd(data['close'], fast=8, slow=17, signal=9)
```

---

## Academic Notes

- **Appel (2005)**: "Technical Analysis: Power Tools for Active Investors"
- **Chong & Ng (2008)**: Found modest predictive power in Asian markets
- **Evidence**: Works best in trending markets; poor in ranges

**Key Insight**: MACD is most effective when combined with trend filters and used as confirmation rather than primary signal.
