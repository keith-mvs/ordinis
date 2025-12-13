# Williams %R

## Overview

Williams %R, developed by Larry Williams in 1973, is a momentum oscillator that measures overbought and oversold levels. It's mathematically similar to Stochastic %K but plotted on an inverted scale.

---

## Formula

```
%R = (Highest High - Close) / (Highest High - Lowest Low) × -100

Where:
- Highest High = highest high over N periods
- Lowest Low = lowest low over N periods
- Close = current closing price
```

**Note**: %R is expressed as a negative value (-100 to 0)

---

## Standard Parameters

| Parameter | Default | Use Case |
|-----------|---------|----------|
| Period | 14 | Standard (Williams' original) |
| Period | 10 | Short-term trading |
| Period | 20 | Swing trading |

---

## Key Levels

| Level | Interpretation |
|-------|----------------|
| -20 to 0 | Overbought zone |
| -50 | Midpoint / Neutral |
| -100 to -80 | Oversold zone |

**Relationship to Stochastic**: %R = Stochastic %K - 100 (inverted)

---

## Rule Templates

### Basic Overbought/Oversold
```python
OVERBOUGHT = WilliamsR(14) > -20
OVERSOLD = WilliamsR(14) < -80

# Extreme levels
EXTREME_OB = WilliamsR(14) > -10
EXTREME_OS = WilliamsR(14) < -90
```

### Failure Swings
```python
# Bullish failure swing (more reliable than simple oversold)
BULLISH_FAILURE = (
    WilliamsR(14)[3] < -80 AND      # First oversold
    WilliamsR(14)[2] > -80 AND      # Rallied above -80
    WilliamsR(14)[1] < WilliamsR(14)[2] AND  # Pulled back
    WilliamsR(14)[1] > -80 AND      # Stayed above -80
    WilliamsR(14) > WilliamsR(14)[1] # Now rising
)

# Bearish failure swing
BEARISH_FAILURE = (
    WilliamsR(14)[3] > -20 AND
    WilliamsR(14)[2] < -20 AND
    WilliamsR(14)[1] > WilliamsR(14)[2] AND
    WilliamsR(14)[1] < -20 AND
    WilliamsR(14) < WilliamsR(14)[1]
)
```

### Midpoint Crossovers
```python
# Crossing -50 line (momentum shift)
BULLISH_MOMENTUM = WilliamsR(14) crosses_above -50
BEARISH_MOMENTUM = WilliamsR(14) crosses_below -50

# With confirmation
STRONG_MOMENTUM = (
    WilliamsR(14) crosses_above -50 AND
    close > SMA(20)                  # Price confirmation
)
```

### Divergence
```python
# Bullish divergence
BULLISH_DIV = (
    close < close[lookback] AND         # Lower price low
    WilliamsR(14) > WilliamsR(14)[lookback]  # Higher %R low
)

# Bearish divergence
BEARISH_DIV = (
    close > close[lookback] AND
    WilliamsR(14) < WilliamsR(14)[lookback]
)
```

---

## Trading Strategies

### 1. Overbought/Oversold Reversal
```python
# Long entry from oversold
LONG_ENTRY = (
    WilliamsR(14) < -80 AND         # Oversold
    WilliamsR(14) > WilliamsR(14)[1] AND  # Turning up
    at_support                       # Price at support
)

# Short entry from overbought
SHORT_ENTRY = (
    WilliamsR(14) > -20 AND
    WilliamsR(14) < WilliamsR(14)[1] AND
    at_resistance
)

# Exit
LONG_EXIT = WilliamsR(14) > -20     # Take profit at overbought
SHORT_EXIT = WilliamsR(14) < -80    # Take profit at oversold
```

### 2. Momentum Breakout
```python
# Long on momentum breakout
MOMENTUM_LONG = (
    WilliamsR(14) crosses_above -50 AND
    WilliamsR(14)[1:5].mean() < -70 AND  # Was deeply oversold
    volume > volume.rolling(20).mean()
)

# Short on momentum breakdown
MOMENTUM_SHORT = (
    WilliamsR(14) crosses_below -50 AND
    WilliamsR(14)[1:5].mean() > -30 AND
    volume > volume.rolling(20).mean()
)
```

### 3. Williams' Ultimate Oscillator Style
```python
# Multi-period confirmation
WR_FAST = WilliamsR(7)
WR_MED = WilliamsR(14)
WR_SLOW = WilliamsR(28)

# All timeframes aligned oversold
MULTI_TF_OVERSOLD = (
    WR_FAST < -80 AND
    WR_MED < -80 AND
    WR_SLOW < -80
)

# Entry when fast turns while others still oversold
ENTRY = (
    WR_FAST crosses_above -80 AND
    WR_MED < -60 AND
    WR_SLOW < -60
)
```

### 4. Range-Bound Trading
```python
# Define range
in_range = ADX(14) < 20

# Trade reversals in range
RANGE_LONG = (
    in_range AND
    WilliamsR(14) < -80 AND
    WilliamsR(14) > WilliamsR(14)[1]
)

RANGE_SHORT = (
    in_range AND
    WilliamsR(14) > -20 AND
    WilliamsR(14) < WilliamsR(14)[1]
)
```

---

## Williams %R vs Stochastic

| Aspect | Williams %R | Stochastic |
|--------|-------------|------------|
| Scale | -100 to 0 | 0 to 100 |
| Overbought | > -20 | > 80 |
| Oversold | < -80 | < 20 |
| Signal Line | None (raw) | %D smoothing |
| Sensitivity | Higher | Lower (smoothed) |

**Conversion**: %R = %K - 100

---

## Trend Context

```python
# Use trend filter to avoid counter-trend trades
TREND = close > SMA(50)

# Only take long signals in uptrend
FILTERED_LONG = (
    TREND AND
    WilliamsR(14) < -80 AND
    WilliamsR(14) > WilliamsR(14)[1]
)

# In downtrend, only short signals
FILTERED_SHORT = (
    NOT TREND AND
    WilliamsR(14) > -20 AND
    WilliamsR(14) < WilliamsR(14)[1]
)
```

---

## Common Pitfalls

1. **Overbought Can Stay Overbought**: In strong trends, %R near 0 for extended periods
2. **No Signal Line**: Raw oscillator produces more noise
3. **Inverted Scale Confusion**: Higher values (closer to 0) = overbought
4. **Whipsaws in Trends**: Mean reversion fails in trending markets

---

## Combining with Other Indicators

```python
# Williams %R + RSI confirmation
DOUBLE_OVERSOLD = (
    WilliamsR(14) < -80 AND
    RSI(14) < 30
)

# Williams %R + Volume
VOLUME_CONFIRMED = (
    WilliamsR(14) < -90 AND         # Extreme oversold
    volume > volume.rolling(20).mean() * 2  # High volume
)

# Williams %R + Bollinger Bands
BB_CONFIRMED = (
    WilliamsR(14) < -80 AND
    close < BB_LOWER                 # At lower band
)
```

---

## Smoothed Williams %R

```python
# Add smoothing to reduce noise (like Stochastic %D)
def smoothed_williams_r(data, period=14, smooth=3):
    wr = williams_r(data, period)
    return wr.rolling(smooth).mean()

# Use smoothed version
WR_SMOOTH = smoothed_williams_r(data, 14, 3)

SMOOTHED_SIGNAL = (
    WR_SMOOTH crosses_above -80 AND
    WR_SMOOTH > WR_SMOOTH[1]
)
```

---

## Implementation

```python
from src.analysis.technical.indicators import Oscillators

osc = Oscillators()

# Calculate Williams %R
wr = osc.williams_r(data, period=14)

# Identify signals
overbought = wr > -20
oversold = wr < -80
turning_up = (wr < -80) & (wr > wr.shift(1))
turning_down = (wr > -20) & (wr < wr.shift(1))
```

---

## Academic Notes

- **Williams (1973)**: "How I Made One Million Dollars Last Year Trading Commodities"
- **Usage**: Simple momentum indicator, best in ranging markets
- **Key Insight**: Effective when combined with support/resistance levels

**Best Practice**: Use %R for timing entries within an established trend direction.
