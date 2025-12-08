# Parabolic SAR (Stop and Reverse)

## Overview

Parabolic SAR, developed by J. Welles Wilder in 1978, is a trend-following indicator that provides potential entry and exit points. "SAR" stands for "Stop And Reverse" - when price crosses the indicator, the trend reverses.

---

## Formula

```
SAR(tomorrow) = SAR(today) + AF × (EP - SAR(today))

Where:
- AF = Acceleration Factor (starts at 0.02, increases by 0.02 each new EP, max 0.20)
- EP = Extreme Point (highest high in uptrend, lowest low in downtrend)
```

**On Trend Reversal**:
- SAR resets to previous EP
- AF resets to initial value (0.02)

---

## Standard Parameters

| Parameter | Default | Use Case |
|-----------|---------|----------|
| AF Start | 0.02 | Standard sensitivity |
| AF Increment | 0.02 | Standard acceleration |
| AF Maximum | 0.20 | Prevents over-sensitivity |

**Variations**:
- Lower AF (0.01) = Slower, fewer signals, larger moves
- Higher AF (0.04) = Faster, more signals, tighter stops

---

## Interpretation

| Condition | Meaning |
|-----------|---------|
| SAR below price | Uptrend |
| SAR above price | Downtrend |
| Price crosses SAR | Trend reversal signal |

---

## Rule Templates

### Basic Trend Detection
```python
UPTREND = close > SAR(0.02, 0.02, 0.20)
DOWNTREND = close < SAR(0.02, 0.02, 0.20)

# Trend reversal
REVERSAL_LONG = close crosses_above SAR
REVERSAL_SHORT = close crosses_below SAR
```

### Entry Signals
```python
# Long entry on reversal
LONG_ENTRY = (
    close > SAR AND
    close[1] < SAR[1] AND           # Just crossed
    ADX(14) > 25                     # Trending market
)

# Short entry on reversal
SHORT_ENTRY = (
    close < SAR AND
    close[1] > SAR[1] AND
    ADX(14) > 25
)
```

### Trailing Stop
```python
# Use SAR as trailing stop
LONG_STOP = SAR                      # Exit if close < SAR
SHORT_STOP = SAR                     # Exit if close > SAR

def update_trailing_stop(position, sar):
    if position == "long":
        return sar                   # SAR trails below price
    elif position == "short":
        return sar                   # SAR trails above price
```

### Trend Strength Filter
```python
# SAR distance from price indicates trend strength
SAR_DISTANCE = abs(close - SAR) / ATR(14)

STRONG_TREND = SAR_DISTANCE > 2.0
WEAK_TREND = SAR_DISTANCE < 0.5

# Only take signals in strong trends
FILTERED_LONG = LONG_ENTRY AND STRONG_TREND
```

---

## Trading Strategies

### 1. Pure SAR Reversal
```python
# Simple reversal system (always in market)
LONG = close > SAR
SHORT = close < SAR

# Stop-and-reverse on crossover
IF LONG AND close crosses_below SAR:
    close_long()
    open_short()

IF SHORT AND close crosses_above SAR:
    close_short()
    open_long()
```

### 2. SAR with Trend Filter
```python
# Only trade in direction of higher timeframe trend
WEEKLY_TREND = weekly_close > weekly_SMA(20)

# Only take long signals when weekly uptrend
FILTERED_LONG = (
    close crosses_above SAR AND
    WEEKLY_TREND
)

# Skip short signals in weekly uptrend
```

### 3. SAR + Moving Average
```python
# Use MA for trend direction, SAR for timing
MA_TREND_UP = close > EMA(50)
MA_TREND_DOWN = close < EMA(50)

LONG_ENTRY = (
    MA_TREND_UP AND
    close crosses_above SAR
)

SHORT_ENTRY = (
    MA_TREND_DOWN AND
    close crosses_below SAR
)
```

### 4. SAR for Exit Only
```python
# Use other signals for entry, SAR for exit
def manage_long_position():
    if close < SAR:
        return "EXIT"
    return "HOLD"

def manage_short_position():
    if close > SAR:
        return "EXIT"
    return "HOLD"
```

---

## SAR Optimization

### Adaptive AF
```python
# Adjust AF based on volatility
def adaptive_af(atr_percentile):
    if atr_percentile > 75:
        return (0.01, 0.01, 0.10)    # Slower in high volatility
    elif atr_percentile < 25:
        return (0.03, 0.03, 0.30)    # Faster in low volatility
    else:
        return (0.02, 0.02, 0.20)    # Standard
```

### Multi-SAR
```python
# Multiple SARs with different sensitivities
SAR_FAST = SAR(0.04, 0.04, 0.40)
SAR_STANDARD = SAR(0.02, 0.02, 0.20)
SAR_SLOW = SAR(0.01, 0.01, 0.10)

# Confluence when all agree
STRONG_UPTREND = (
    close > SAR_FAST AND
    close > SAR_STANDARD AND
    close > SAR_SLOW
)
```

---

## Limitations

1. **Whipsaws in Ranges**: Many false reversals in sideways markets
2. **Gaps**: Can cause sudden large jumps in SAR
3. **No Neutral Zone**: Always implies a direction (up or down)
4. **Lag in New Trends**: AF starts slow, catches up over time

---

## Range Detection

```python
# Avoid SAR in ranging markets
RANGING = ADX(14) < 20

# Filter out signals
SAR_SIGNAL = (
    close crosses_above SAR AND
    NOT RANGING
)
```

---

## Combining with Other Indicators

```python
# SAR + RSI
CONFIRMED_LONG = (
    close crosses_above SAR AND
    RSI(14) > 50 AND                 # Momentum confirmation
    RSI(14) < 70                     # Not overbought
)

# SAR + MACD
MACD_CONFIRMED = (
    close crosses_above SAR AND
    MACD_histogram > 0               # MACD positive
)

# SAR + Bollinger
BB_CONFIRMED = (
    close crosses_above SAR AND
    close > BB_MIDDLE                # Above middle band
)
```

---

## Implementation

```python
from src.analysis.technical.indicators import TrendIndicators

trend = TrendIndicators()

# Calculate Parabolic SAR
sar = trend.parabolic_sar(
    data,
    af_start=0.02,
    af_increment=0.02,
    af_max=0.20
)

# Identify signals
uptrend = data['close'] > sar
downtrend = data['close'] < sar
reversal_up = (data['close'] > sar) & (data['close'].shift(1) <= sar.shift(1))
reversal_down = (data['close'] < sar) & (data['close'].shift(1) >= sar.shift(1))
```

---

## Academic Notes

- **Wilder (1978)**: "New Concepts in Technical Trading Systems"
- **Best Use**: Trend-following markets with clear direction
- **Key Insight**: Excellent for trailing stops; less reliable for entries

**Best Practice**: Use SAR primarily as a trailing stop mechanism rather than an entry signal generator. Combine with a trend filter (ADX > 25) to avoid whipsaws.
