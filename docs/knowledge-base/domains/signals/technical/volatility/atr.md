# Average True Range (ATR)

## Overview

ATR, developed by J. Welles Wilder in 1978, measures market volatility by calculating the average of true ranges over a specified period. Unlike standard deviation, ATR accounts for gaps.

---

## Formula

```
True Range = max(
    High - Low,                    # Current bar range
    |High - Close[1]|,             # Gap up from previous close
    |Low - Close[1]|               # Gap down from previous close
)

ATR = SMA or Wilder Smoothing of True Range over N periods
```

---

## Standard Parameters

| Parameter | Default | Use Case |
|-----------|---------|----------|
| Period | 14 | Standard measurement |
| Period | 10 | More responsive |
| Period | 20 | Smoother, less noise |

---

## Key Characteristics

1. **Always Positive**: Measures volatility magnitude, not direction
2. **Gap-Inclusive**: Captures overnight moves
3. **Adaptive**: Adjusts to current volatility
4. **Non-Directional**: Same for up and down moves

---

## Rule Templates

### Volatility State
```python
# Current volatility relative to history
ATR_PERCENTILE = percentile_rank(ATR(14), 252)

LOW_VOLATILITY = ATR_PERCENTILE < 25
NORMAL_VOLATILITY = 25 <= ATR_PERCENTILE <= 75
HIGH_VOLATILITY = ATR_PERCENTILE > 75

# Relative to recent average
VOL_EXPANDING = ATR(14) > ATR(14).rolling(50).mean() * 1.2
VOL_CONTRACTING = ATR(14) < ATR(14).rolling(50).mean() * 0.8
```

### ATR-Based Stops
```python
# Fixed ATR multiple stops
STOP_LOSS_LONG = entry_price - (2.0 * ATR(14))
STOP_LOSS_SHORT = entry_price + (2.0 * ATR(14))

# Chandelier Exit (trailing stop)
CHANDELIER_LONG = highest(high, 22) - (3.0 * ATR(22))
CHANDELIER_SHORT = lowest(low, 22) + (3.0 * ATR(22))

# Kase Stop (adaptive)
KASE_STOP = close - (ATR(14) * (1 + log(N/10)))
```

### Position Sizing
```python
# Risk-based position sizing
RISK_AMOUNT = account_equity * risk_per_trade  # e.g., 1%
STOP_DISTANCE = ATR(14) * atr_multiplier        # e.g., 2.0
POSITION_SIZE = RISK_AMOUNT / STOP_DISTANCE

# Volatility-adjusted sizing
BASE_SIZE = 100  # shares
VOL_ADJUSTED_SIZE = BASE_SIZE * (avg_atr / current_atr)
```

### Breakout Signals
```python
# ATR breakout (volatility expansion)
ATR_BREAKOUT = (
    close > close[1] + (1.5 * ATR(14)) OR
    close < close[1] - (1.5 * ATR(14))
)

# Squeeze breakout (after low volatility)
SQUEEZE = ATR(14) < ATR(14).rolling(126).quantile(0.1)
SQUEEZE_BREAKOUT = SQUEEZE[1:5].any() AND ATR_BREAKOUT
```

### Volatility Filters
```python
# Avoid low volatility (poor R:R)
MIN_VOLATILITY = ATR(14) > 0.005 * close  # 0.5% of price

# Avoid extreme volatility (whipsaws)
MAX_VOLATILITY = ATR(14) < 0.05 * close   # 5% of price

VALID_VOLATILITY = MIN_VOLATILITY AND MAX_VOLATILITY
```

---

## Trading Applications

### 1. Dynamic Stop Placement
```python
# Initial stop
INITIAL_STOP = entry - (2.0 * ATR(14))

# Trailing stop (adjusts with price)
def trailing_stop(position, current_high, atr):
    if position == "long":
        new_stop = current_high - (2.5 * atr)
        return max(existing_stop, new_stop)  # Only move up
```

### 2. Take Profit Targets
```python
# ATR-based targets
TARGET_1 = entry + (2.0 * ATR(14))   # 2R
TARGET_2 = entry + (3.0 * ATR(14))   # 3R
TARGET_3 = entry + (5.0 * ATR(14))   # 5R

# Risk:Reward check
MIN_RR = 2.0
TARGET_DISTANCE = TARGET_1 - entry
STOP_DISTANCE = entry - STOP_LOSS
VALID_RR = (TARGET_DISTANCE / STOP_DISTANCE) >= MIN_RR
```

### 3. Volatility Regime Trading
```python
# High volatility regime
if HIGH_VOLATILITY:
    reduce_position_size()
    widen_stops()
    take_profits_faster()

# Low volatility regime
if LOW_VOLATILITY:
    increase_position_size()
    tighter_stops_ok()
    watch_for_breakout()
```

### 4. Entry Filters
```python
# Only enter on "clean" bars
CLEAN_BAR = (
    (high - low) < 1.5 * ATR(14) AND  # Not too wide
    (high - low) > 0.5 * ATR(14)       # Not too narrow
)
```

---

## ATR Normalization

### ATR Percentage
```python
# Normalize ATR as percentage of price
ATR_PCT = ATR(14) / close * 100

# Useful for comparing volatility across securities
# SPY might have ATR_PCT = 1.2%
# TSLA might have ATR_PCT = 4.5%
```

### Relative ATR
```python
# Compare current ATR to historical
RELATIVE_ATR = ATR(14) / ATR(14).rolling(252).mean()

# > 1.0 = higher than average volatility
# < 1.0 = lower than average volatility
```

---

## Common Pitfalls

1. **Static Stops**: ATR changes; update stop distances
2. **Ignoring Gaps**: TR includes gaps, important for risk
3. **Short Period Issues**: Low period = noisy ATR
4. **Different Assets**: Compare normalized ATR, not absolute

---

## Combining with Other Indicators

```python
# ATR + Bollinger Bands
BOLLINGER_SQUEEZE = bandwidth < bandwidth.rolling(126).quantile(0.2)
ATR_SQUEEZE = ATR(14) < ATR(14).rolling(126).quantile(0.2)
DOUBLE_SQUEEZE = BOLLINGER_SQUEEZE AND ATR_SQUEEZE

# ATR + ADX
TRENDING_WITH_VOL = ADX(14) > 25 AND ATR_PERCENTILE > 50

# ATR for Keltner Channels
KELTNER_UPPER = EMA(20) + (2.0 * ATR(10))
KELTNER_LOWER = EMA(20) - (2.0 * ATR(10))
```

---

## Implementation

```python
from ordinis.analysis.technical.indicators import VolatilityIndicators

vol = VolatilityIndicators()

# Calculate ATR
atr = vol.atr(data, period=14)

# ATR percentage
atr_pct = atr / data['close'] * 100

# Position sizing
def calculate_position_size(equity, risk_pct, atr, multiplier=2.0):
    risk_amount = equity * risk_pct
    stop_distance = atr * multiplier
    return risk_amount / stop_distance
```

---

## Academic Notes

- **Wilder (1978)**: Original ATR introduction
- **Usage**: Widely accepted for risk management
- **Key Insight**: ATR is for risk management, not signal generation

**Best Practice**: Use ATR for position sizing and stops, not as a trading signal.
