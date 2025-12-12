# ADX / DMI (Average Directional Index / Directional Movement)

## Overview

ADX, developed by J. Welles Wilder in 1978, measures trend strength regardless of direction. The Directional Movement System (DMI) includes +DI and -DI to indicate trend direction.

---

## Formula

```
+DM = High - High[1] (if > 0 and > -DM, else 0)
-DM = Low[1] - Low (if > 0 and > +DM, else 0)

+DI = 100 × Smoothed(+DM) / ATR
-DI = 100 × Smoothed(-DM) / ATR

DX = 100 × |+DI - -DI| / (+DI + -DI)
ADX = Smoothed(DX) over N periods
```

---

## Components

| Component | Range | Meaning |
|-----------|-------|---------|
| ADX | 0-100 | Trend strength (not direction) |
| +DI | 0-100 | Bullish directional movement |
| -DI | 0-100 | Bearish directional movement |
| DX | 0-100 | Unsmoothed trend strength |

---

## Key Levels

| ADX Level | Interpretation | Strategy |
|-----------|----------------|----------|
| < 20 | No trend / ranging | Mean reversion |
| 20-25 | Possible trend emerging | Watch for breakout |
| 25-40 | Trending | Trend-following |
| 40-50 | Strong trend | Aggressive trend-following |
| > 50 | Very strong trend | Watch for exhaustion |

---

## Rule Templates

### Trend Strength Assessment
```python
# Basic trend detection
NO_TREND = ADX(14) < 20
TRENDING = ADX(14) > 25
STRONG_TREND = ADX(14) > 40
EXTREME_TREND = ADX(14) > 50
```

### Trend Direction
```python
# Direction from DI lines
BULLISH_TREND = +DI > -DI AND ADX > 25
BEARISH_TREND = -DI > +DI AND ADX > 25

# DI crossovers
BULLISH_CROSS = +DI crosses_above -DI
BEARISH_CROSS = -DI crosses_above +DI
```

### ADX Rising/Falling
```python
# Trend strengthening
ADX_RISING = ADX > ADX[1] > ADX[2]

# Trend weakening
ADX_FALLING = ADX < ADX[1] < ADX[2]

# Trend emerging from range
TREND_EMERGING = ADX < 25 AND ADX > ADX[1] > ADX[2]
```

### Combined Entry Signals
```python
# Strong bullish signal
STRONG_BULL = (
    +DI > -DI AND
    ADX > 25 AND
    ADX > ADX[1] AND            # ADX rising
    +DI > +DI[1]                 # +DI rising
)

# Strong bearish signal
STRONG_BEAR = (
    -DI > +DI AND
    ADX > 25 AND
    ADX > ADX[1] AND
    -DI > -DI[1]
)
```

### Strategy Selection
```python
# Regime-based strategy selection
def select_strategy(adx, plus_di, minus_di):
    if adx < 20:
        return "mean_reversion"
    elif adx > 25:
        if plus_di > minus_di:
            return "trend_following_long"
        else:
            return "trend_following_short"
    else:
        return "wait"
```

---

## Trading Strategies

### 1. ADX + DI Crossover
```python
LONG_ENTRY = (
    +DI crosses_above -DI AND
    ADX > 20 AND
    ADX > ADX[1]                 # ADX rising
)

SHORT_ENTRY = (
    -DI crosses_above +DI AND
    ADX > 20 AND
    ADX > ADX[1]
)

# Exit on opposite crossover or ADX declining
EXIT = DI_crossover_opposite OR (ADX < 20 AND ADX < ADX[1])
```

### 2. ADX Trend Filter
```python
# Only take other signals when trending
TREND_FILTER = ADX(14) > 25

# Apply to MA crossover strategy
LONG = EMA(20) crosses_above EMA(50) AND TREND_FILTER
SHORT = EMA(20) crosses_below EMA(50) AND TREND_FILTER
```

### 3. ADX Breakout
```python
# Enter when ADX breaks above 25 from below
ADX_BREAKOUT = (
    ADX > 25 AND
    ADX[1] < 25 AND              # Was below
    ADX > ADX[1]                  # Rising
)

# Direction from DI
LONG = ADX_BREAKOUT AND +DI > -DI
SHORT = ADX_BREAKOUT AND -DI > +DI
```

### 4. Exhaustion Detection
```python
# Very high ADX may precede reversal
EXHAUSTION_WARNING = (
    ADX > 50 AND
    ADX < ADX[1] AND             # ADX declining
    ADX[1] > ADX[2]              # Was rising
)

# Consider taking profits or tightening stops
```

---

## ADX Patterns

### Hooking Pattern
```python
# ADX hooks down from high level
ADX_HOOK_DOWN = ADX > 40 AND ADX < ADX[1] AND ADX[1] > ADX[2]
# Often precedes trend pause or reversal
```

### Valley Pattern
```python
# ADX rises from low level
ADX_VALLEY = ADX < 20 AND ADX > ADX[1] AND ADX[1] < ADX[2]
# New trend potentially emerging
```

---

## Common Pitfalls

1. **ADX Lag**: ADX is smoothed, slow to react
2. **High ADX ≠ Continue**: Extreme ADX can precede reversals
3. **DI Crossover Whipsaws**: Many false signals in ranges
4. **Ignoring Direction**: ADX alone doesn't show direction

---

## Combining with Other Indicators

```python
# ADX + RSI
ENTRY = (
    ADX > 25 AND                  # Trending
    +DI > -DI AND                 # Bullish
    RSI(14) > 50 AND              # Momentum
    RSI(14) < 70                  # Not overbought
)

# ADX + MA
FILTERED_MA_CROSS = (
    EMA(20) crosses_above EMA(50) AND
    ADX > 25                       # Trend filter
)

# ADX + Bollinger for regime
if ADX < 20:
    use_bollinger_mean_reversion()
else:
    use_trend_following()
```

---

## Implementation

```python
from src.analysis.technical.indicators import Oscillators

osc = Oscillators()

# Calculate ADX/DMI
adx_result = osc.adx(data, period=14)
adx = adx_result['adx']
plus_di = adx_result['plus_di']
minus_di = adx_result['minus_di']
```

---

## Academic Notes

- **Wilder (1978)**: "New Concepts in Technical Trading Systems"
- **Evidence**: ADX useful for regime detection; DI crossovers have mixed results
- **Best Use**: Strategy selection filter rather than primary signal

**Key Insight**: ADX is most valuable for determining WHAT type of strategy to use, not for generating entry signals directly.
