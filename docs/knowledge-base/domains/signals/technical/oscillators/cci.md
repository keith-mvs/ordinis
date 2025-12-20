# Commodity Channel Index (CCI)

## Overview

CCI, developed by Donald Lambert in 1980, measures the current price level relative to an average price level over a given period. Despite its name, CCI works on any asset class, not just commodities.

---

## Formula

```
Typical Price (TP) = (High + Low + Close) / 3

Mean Deviation = mean(|TP - SMA(TP, N)|)

CCI = (TP - SMA(TP, N)) / (0.015 × Mean Deviation)
```

The 0.015 constant scales values so approximately 70-80% fall between -100 and +100.

---

## Standard Parameters

| Parameter | Default | Use Case |
|-----------|---------|----------|
| Period | 20 | Standard (Lambert's original) |
| Period | 14 | More responsive |
| Period | 50 | Longer-term trends |

---

## Key Levels

| Level | Interpretation |
|-------|----------------|
| > +100 | Overbought / Strong uptrend |
| +100 to -100 | Normal trading range |
| < -100 | Oversold / Strong downtrend |
| > +200 | Extremely overbought |
| < -200 | Extremely oversold |

---

## Rule Templates

### Basic Overbought/Oversold
```python
CCI_OVERBOUGHT = CCI(20) > 100
CCI_OVERSOLD = CCI(20) < -100
CCI_EXTREME_OB = CCI(20) > 200
CCI_EXTREME_OS = CCI(20) < -200
```

### Zero Line Crossovers
```python
# Momentum shift signals
CCI_BULLISH_CROSS = CCI(20) crosses_above 0
CCI_BEARISH_CROSS = CCI(20) crosses_below 0

# With momentum confirmation
STRONG_BULL_CROSS = (
    CCI(20) crosses_above 0 AND
    CCI(20) > CCI(20)[1] > CCI(20)[2]  # Rising CCI
)
```

### Trend Following
```python
# CCI above/below 100 as trend confirmation
UPTREND = CCI(20) > 100
DOWNTREND = CCI(20) < -100

# Entry on pullback within trend
PULLBACK_LONG = (
    CCI(20)[5:20].max() > 100 AND  # Was overbought recently
    CCI(20) < 100 AND               # Pulled back
    CCI(20) > 0 AND                 # Still positive
    CCI(20) > CCI(20)[1]            # Turning up
)
```

### Divergence
```python
# Bullish divergence
BULLISH_DIV = (
    close < close[lookback] AND     # Lower price low
    CCI(20) > CCI(20)[lookback]     # Higher CCI low
)

# Bearish divergence
BEARISH_DIV = (
    close > close[lookback] AND     # Higher price high
    CCI(20) < CCI(20)[lookback]     # Lower CCI high
)
```

---

## Trading Strategies

### 1. Zero Line Rejection
```python
# Bullish: CCI pulls back to zero, then bounces
ZERO_LINE_LONG = (
    CCI(20)[3:7].min() < 20 AND     # Approached zero
    CCI(20)[3:7].min() > -20 AND
    CCI(20) > 50 AND                 # Now rising
    CCI(20) > CCI(20)[1]
)

# Bearish: CCI rallies to zero, then fails
ZERO_LINE_SHORT = (
    CCI(20)[3:7].max() > -20 AND
    CCI(20)[3:7].max() < 20 AND
    CCI(20) < -50 AND
    CCI(20) < CCI(20)[1]
)
```

### 2. CCI Breakout
```python
# Breakout above +100 (trend initiation)
CCI_BREAKOUT_LONG = (
    CCI(20) > 100 AND
    CCI(20)[1] < 100 AND             # Just crossed
    volume > volume.rolling(20).mean()  # Volume confirmation
)

# Breakdown below -100
CCI_BREAKOUT_SHORT = (
    CCI(20) < -100 AND
    CCI(20)[1] > -100 AND
    volume > volume.rolling(20).mean()
)
```

### 3. Mean Reversion
```python
# From extreme oversold
MEAN_REVERSION_LONG = (
    CCI(20) < -200 AND              # Extremely oversold
    CCI(20) > CCI(20)[1] AND        # Turning up
    at_support                       # Additional confirmation
)

# From extreme overbought
MEAN_REVERSION_SHORT = (
    CCI(20) > 200 AND
    CCI(20) < CCI(20)[1] AND
    at_resistance
)
```

### 4. Turnaround Strategy
```python
# Lambert's original concept
# Enter when CCI turns from extreme back toward zero

TURNAROUND_LONG = (
    CCI(20)[1] < -100 AND           # Was oversold
    CCI(20) > CCI(20)[1] AND        # Now rising
    CCI(20) > -100                   # Crossed back above -100
)

TURNAROUND_SHORT = (
    CCI(20)[1] > 100 AND
    CCI(20) < CCI(20)[1] AND
    CCI(20) < 100
)
```

---

## CCI vs Other Oscillators

| Feature | CCI | RSI | Stochastic |
|---------|-----|-----|------------|
| Unbounded | Yes | No (0-100) | No (0-100) |
| Center Line | 0 | 50 | 50 |
| Overbought | >100 | >70 | >80 |
| Oversold | <-100 | <30 | <20 |
| Best For | Trends | Reversals | Range-bound |

---

## Multi-Timeframe CCI

```python
# Alignment across timeframes
CCI_DAILY = CCI(20, daily)
CCI_WEEKLY = CCI(20, weekly)

# Confluence signal
STRONG_LONG = (
    CCI_DAILY > 0 AND
    CCI_WEEKLY > 0 AND
    CCI_DAILY crosses_above 100
)

# Counter-trend warning
CAUTION = (
    CCI_DAILY > 100 AND             # Daily overbought
    CCI_WEEKLY < 0                   # Weekly still bearish
)
```

---

## Common Pitfalls

1. **Unbounded Nature**: CCI can reach extreme values without reverting quickly
2. **Whipsaws in Ranges**: Many false signals in sideways markets
3. **Ignoring Trend**: Overbought in uptrend can stay overbought
4. **No Volume Consideration**: CCI doesn't incorporate volume

---

## Combining with Other Indicators

```python
# CCI + ADX (trend filter)
FILTERED_CCI = (
    CCI(20) crosses_above 100 AND
    ADX(14) > 25                     # Only in trending markets
)

# CCI + Volume
VOLUME_CONFIRMED = (
    CCI(20) > 100 AND
    volume > volume.rolling(20).mean() * 1.5
)

# CCI + Moving Average
MA_ALIGNED = (
    CCI(20) > 0 AND
    close > EMA(50)                  # Price above MA
)
```

---

## Implementation

```python
from ordinis.analysis.technical.indicators import Oscillators

osc = Oscillators()

# Calculate CCI
cci = osc.cci(data, period=20)

# Identify signals
overbought = cci > 100
oversold = cci < -100
zero_cross_up = (cci > 0) & (cci.shift(1) <= 0)
zero_cross_down = (cci < 0) & (cci.shift(1) >= 0)
```

---

## Academic Notes

- **Lambert (1980)**: Original CCI article in Commodities magazine
- **Usage**: Better for trend identification than reversal
- **Key Insight**: CCI identifies when price deviates significantly from its mean

**Best Practice**: Use CCI >100/<-100 for trend confirmation, not reversal signals.
