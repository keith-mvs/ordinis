# Aroon Indicator

## Overview

Aroon, developed by Tushar Chande in 1995, identifies trend strength and direction by measuring the time elapsed since the highest high and lowest low over a given period. "Aroon" is Sanskrit for "dawn's early light," signifying the indicator's ability to detect early trend changes.

---

## Formula

```
Aroon Up = ((N - Days Since N-period High) / N) × 100

Aroon Down = ((N - Days Since N-period Low) / N) × 100

Aroon Oscillator = Aroon Up - Aroon Down
```

**Range**: 0 to 100 for Aroon Up/Down, -100 to +100 for Oscillator

---

## Standard Parameters

| Parameter | Default | Use Case |
|-----------|---------|----------|
| Period | 25 | Standard (Chande's original) |
| Period | 14 | More responsive |
| Period | 50 | Longer-term trends |

---

## Key Levels

### Aroon Up/Down
| Level | Interpretation |
|-------|----------------|
| > 70 | Strong trend |
| 50 | Neutral |
| < 30 | Weak/No trend |
| 100 | New high/low just made |
| 0 | No new high/low in N periods |

### Aroon Oscillator
| Level | Interpretation |
|-------|----------------|
| > +50 | Strong uptrend |
| +50 to -50 | Consolidation/Weak trend |
| < -50 | Strong downtrend |

---

## Rule Templates

### Trend Detection
```python
# Basic trend identification
UPTREND = Aroon_Up(25) > 70 AND Aroon_Down(25) < 30
DOWNTREND = Aroon_Down(25) > 70 AND Aroon_Up(25) < 30
CONSOLIDATION = Aroon_Up(25) < 50 AND Aroon_Down(25) < 50

# Using oscillator
STRONG_UP = Aroon_Oscillator(25) > 50
STRONG_DOWN = Aroon_Oscillator(25) < -50
NO_TREND = abs(Aroon_Oscillator(25)) < 25
```

### Crossover Signals
```python
# Bullish crossover
BULLISH_CROSS = Aroon_Up crosses_above Aroon_Down

# Bearish crossover
BEARISH_CROSS = Aroon_Down crosses_above Aroon_Up

# With threshold filter
STRONG_BULL_CROSS = (
    Aroon_Up crosses_above Aroon_Down AND
    Aroon_Up > 70
)
```

### New High/Low Signals
```python
# New N-period high (Aroon Up = 100)
NEW_HIGH = Aroon_Up(25) == 100

# New N-period low (Aroon Down = 100)
NEW_LOW = Aroon_Down(25) == 100

# Sustained new highs (trend continuation)
SUSTAINED_UPTREND = Aroon_Up(25) > 90 for 3 consecutive days
```

### Trend Exhaustion
```python
# Aroon Up/Down both declining = trend exhaustion
EXHAUSTION = (
    Aroon_Up < Aroon_Up[5] AND
    Aroon_Down < Aroon_Down[5] AND
    Aroon_Up < 50 AND
    Aroon_Down < 50
)
```

---

## Trading Strategies

### 1. Aroon Crossover
```python
# Basic crossover system
LONG_ENTRY = (
    Aroon_Up crosses_above Aroon_Down AND
    Aroon_Up > 50                    # Minimum strength
)

SHORT_ENTRY = (
    Aroon_Down crosses_above Aroon_Up AND
    Aroon_Down > 50
)

# Exit on opposite crossover
LONG_EXIT = Aroon_Down crosses_above Aroon_Up
SHORT_EXIT = Aroon_Up crosses_above Aroon_Down
```

### 2. Aroon Breakout
```python
# Enter on new high breakout
BREAKOUT_LONG = (
    Aroon_Up == 100 AND              # New N-period high
    Aroon_Up[1] < 100 AND            # First new high
    Aroon_Down < 30                  # No recent lows
)

# Enter on new low breakdown
BREAKOUT_SHORT = (
    Aroon_Down == 100 AND
    Aroon_Down[1] < 100 AND
    Aroon_Up < 30
)
```

### 3. Oscillator Trading
```python
# Oscillator trend following
LONG_ENTRY = (
    Aroon_Oscillator crosses_above 50 AND
    Aroon_Oscillator[1] < 50
)

SHORT_ENTRY = (
    Aroon_Oscillator crosses_below -50 AND
    Aroon_Oscillator[1] > -50
)

# Exit when oscillator approaches zero
LONG_EXIT = Aroon_Oscillator < 0
SHORT_EXIT = Aroon_Oscillator > 0
```

### 4. Consolidation Breakout
```python
# Identify consolidation
CONSOLIDATING = (
    Aroon_Up < 50 AND
    Aroon_Down < 50 AND
    abs(Aroon_Oscillator) < 25
)

# Wait for breakout from consolidation
BREAKOUT = (
    CONSOLIDATING[1:5].all() AND     # Was consolidating
    (Aroon_Up > 70 OR Aroon_Down > 70)  # Now trending
)

LONG_BREAKOUT = BREAKOUT AND Aroon_Up > 70
SHORT_BREAKOUT = BREAKOUT AND Aroon_Down > 70
```

---

## Aroon vs ADX

| Aspect | Aroon | ADX |
|--------|-------|-----|
| Direction | Yes (Up/Down) | No (requires +DI/-DI) |
| Trend Strength | Yes | Yes |
| Time-Based | Yes (days since high/low) | No (price movement) |
| Range Detection | Yes (both low) | Yes (ADX < 20) |
| Responsiveness | Faster | Slower (more smoothing) |

---

## Multi-Timeframe Aroon

```python
# Daily and Weekly alignment
AROON_DAILY = Aroon_Oscillator(25, daily)
AROON_WEEKLY = Aroon_Oscillator(25, weekly)

# Strong uptrend confluence
STRONG_UPTREND = (
    AROON_DAILY > 50 AND
    AROON_WEEKLY > 50
)

# Divergence warning
WARNING = (
    AROON_DAILY > 50 AND             # Daily uptrend
    AROON_WEEKLY < 0                 # Weekly downtrend
)
```

---

## Common Pitfalls

1. **Whipsaws in Ranges**: Frequent crossovers when no trend
2. **100/0 Not Sustainable**: Aroon at extremes naturally decays
3. **Period Sensitivity**: Different periods give different signals
4. **Lagging in Fast Moves**: Time-based, not price-magnitude

---

## Combining with Other Indicators

```python
# Aroon + Volume
VOLUME_CONFIRMED = (
    Aroon_Up > 70 AND
    volume > volume.rolling(20).mean()
)

# Aroon + RSI
MOMENTUM_CONFIRMED = (
    Aroon_Up crosses_above Aroon_Down AND
    RSI(14) > 50 AND
    RSI(14) < 70                     # Not overbought
)

# Aroon + Price Action
PRICE_CONFIRMED = (
    Aroon_Up > 70 AND
    close > SMA(50)                  # Above moving average
)

# Aroon + ADX (double confirmation)
STRONG_TREND = (
    Aroon_Oscillator > 50 AND
    ADX(14) > 25
)
```

---

## Implementation

```python
from ordinis.analysis.technical.indicators import TrendIndicators

trend = TrendIndicators()

# Calculate Aroon
aroon = trend.aroon(data, period=25)
aroon_up = aroon['aroon_up']
aroon_down = aroon['aroon_down']
aroon_osc = aroon['aroon_oscillator']

# Identify signals
uptrend = (aroon_up > 70) & (aroon_down < 30)
downtrend = (aroon_down > 70) & (aroon_up < 30)
bull_cross = (aroon_up > aroon_down) & (aroon_up.shift(1) <= aroon_down.shift(1))
```

---

## Academic Notes

- **Chande (1995)**: "The New Technical Trader"
- **Unique Aspect**: Time-based rather than price-based measurement
- **Key Insight**: Measures how recently a trend made a new extreme

**Best Practice**: Use Aroon for early trend detection and as a filter for other signals. Most effective in clearly trending markets.
