# Candlestick Patterns

## Overview

Japanese candlestick patterns identify potential reversals or continuations based on single or multi-bar price formations. While visually intuitive, academic studies show most patterns have limited predictive power when used alone.

---

## Candlestick Anatomy

```
        High ─┬─
              │  Upper Shadow (Wick)
         ─────┴───── Open (if bearish) / Close (if bullish)
        │         │
        │  Body   │
        │         │
         ─────────── Close (if bearish) / Open (if bullish)
              │  Lower Shadow (Wick)
        Low ──┴──
```

**Key Metrics:**
```python
body = abs(close - open)
upper_shadow = high - max(open, close)
lower_shadow = min(open, close) - low
total_range = high - low
body_pct = body / total_range if total_range > 0 else 0
```

---

## Single Candlestick Patterns

### Doji
**Identification:**
```python
DOJI = abs(open - close) < (high - low) * 0.1
# Body is less than 10% of total range
```

**Meaning**: Indecision; neither bulls nor bears won

**Variations:**
- **Standard Doji**: Cross shape, shadows roughly equal
- **Long-Legged Doji**: Very long shadows
- **Gravestone Doji**: Long upper shadow, no lower
- **Dragonfly Doji**: Long lower shadow, no upper

**Trading:**
```python
REVERSAL_DOJI = DOJI AND (at_resistance OR at_support)
# More meaningful at key levels
```

---

### Hammer / Hanging Man

**Identification:**
```python
HAMMER_SHAPE = (
    lower_shadow >= body * 2 AND     # Long lower shadow
    upper_shadow <= body * 0.3 AND   # Small/no upper shadow
    body > 0                          # Has a body
)

# Context determines meaning
HAMMER = HAMMER_SHAPE AND downtrend   # Bullish reversal
HANGING_MAN = HAMMER_SHAPE AND uptrend  # Bearish warning
```

**Trading:**
```python
HAMMER_SIGNAL = (
    HAMMER AND
    at_support AND
    volume > avg_volume AND
    close > open  # Bullish hammer (green) stronger
)
```

---

### Shooting Star / Inverted Hammer

**Identification:**
```python
SHOOTING_STAR_SHAPE = (
    upper_shadow >= body * 2 AND     # Long upper shadow
    lower_shadow <= body * 0.3 AND   # Small/no lower shadow
    body > 0
)

SHOOTING_STAR = SHOOTING_STAR_SHAPE AND uptrend
INVERTED_HAMMER = SHOOTING_STAR_SHAPE AND downtrend
```

---

### Marubozu

**Identification:**
```python
BULLISH_MARUBOZU = (
    close > open AND                  # Green candle
    upper_shadow < body * 0.05 AND    # No upper shadow
    lower_shadow < body * 0.05        # No lower shadow
)

BEARISH_MARUBOZU = (
    close < open AND
    upper_shadow < body * 0.05 AND
    lower_shadow < body * 0.05
)
```

**Meaning**: Strong conviction in direction

---

## Two-Candlestick Patterns

### Engulfing

**Identification:**
```python
BULLISH_ENGULFING = (
    close[1] < open[1] AND           # Previous red
    close > open AND                  # Current green
    open <= close[1] AND              # Open at/below prev close
    close >= open[1] AND              # Close at/above prev open
    body > body[1]                    # Current body larger
)

BEARISH_ENGULFING = (
    close[1] > open[1] AND           # Previous green
    close < open AND                  # Current red
    open >= close[1] AND
    close <= open[1] AND
    body > body[1]
)
```

**Trading:**
```python
ENGULFING_SIGNAL = (
    BULLISH_ENGULFING AND
    at_support AND
    RSI(14) < 40                      # Oversold confirmation
)
```

---

### Harami

**Identification:**
```python
BULLISH_HARAMI = (
    close[1] < open[1] AND           # Previous red
    close > open AND                  # Current green
    open > close[1] AND              # Body inside previous
    close < open[1]
)

BEARISH_HARAMI = (
    close[1] > open[1] AND
    close < open AND
    open < close[1] AND
    close > open[1]
)
```

**Meaning**: Smaller body inside larger = indecision, potential reversal

---

### Piercing Line / Dark Cloud Cover

**Identification:**
```python
PIERCING_LINE = (
    close[1] < open[1] AND           # Previous red
    open < low[1] AND                 # Gap down open
    close > open AND                  # Current green
    close > (open[1] + close[1]) / 2  # Close above 50% of prev body
)

DARK_CLOUD_COVER = (
    close[1] > open[1] AND           # Previous green
    open > high[1] AND                # Gap up open
    close < open AND                  # Current red
    close < (open[1] + close[1]) / 2  # Close below 50% of prev body
)
```

---

## Three-Candlestick Patterns

### Morning Star / Evening Star

**Identification:**
```python
MORNING_STAR = (
    close[2] < open[2] AND           # First candle: bearish
    body[1] < body[2] * 0.3 AND      # Second: small body (star)
    close > open AND                  # Third: bullish
    close > (open[2] + close[2]) / 2  # Closes above first's midpoint
)

EVENING_STAR = (
    close[2] > open[2] AND
    body[1] < body[2] * 0.3 AND
    close < open AND
    close < (open[2] + close[2]) / 2
)
```

---

### Three White Soldiers / Three Black Crows

**Identification:**
```python
THREE_WHITE_SOLDIERS = (
    close > open AND                  # All three bullish
    close[1] > open[1] AND
    close[2] > open[2] AND
    close > close[1] > close[2] AND   # Progressive higher closes
    open > open[1] > open[2] AND      # Progressive higher opens
    # Bodies roughly same size
    abs(body - body[1]) < body * 0.3
)
```

---

## Pattern Context Rules

### Require Trend Context
```python
def validate_pattern(pattern_type, data, lookback=20):
    # Calculate trend
    trend = "up" if close > SMA(lookback) else "down"

    # Reversal patterns need existing trend
    if pattern_type in ["hammer", "bullish_engulfing", "morning_star"]:
        return trend == "down"  # Bullish reversal needs downtrend

    if pattern_type in ["shooting_star", "bearish_engulfing", "evening_star"]:
        return trend == "up"    # Bearish reversal needs uptrend

    return True
```

### Require Level Context
```python
PATTERN_AT_LEVEL = (
    candlestick_pattern AND
    (at_support OR at_resistance OR at_key_ma)
)
# Patterns at key levels are more significant
```

### Require Volume Confirmation
```python
CONFIRMED_PATTERN = (
    candlestick_pattern AND
    volume > volume.rolling(20).mean()
)
```

---

## Quick Usage (Library)

```python
import pandas as pd
from ordinis.analysis.technical.patterns import CandlestickPatterns

patterns = CandlestickPatterns()

# data must include open/high/low/close columns
detected = patterns.detect(data)

for name, matched in detected.items():
    if matched:
        print(f"Pattern detected: {name}")
```

---

## Implementation

```python
def detect_candlestick_patterns(data):
    patterns = {}

    o, h, l, c = data['open'], data['high'], data['low'], data['close']
    body = abs(c - o)
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l
    total_range = h - l

    # Doji
    patterns['doji'] = body < total_range * 0.1

    # Hammer
    patterns['hammer'] = (
        (lower_shadow >= body * 2) &
        (upper_shadow <= body * 0.3) &
        (c > c.shift(20).rolling(20).mean())  # In downtrend context
    )

    # Bullish Engulfing
    patterns['bullish_engulfing'] = (
        (c.shift(1) < o.shift(1)) &  # Prev bearish
        (c > o) &                      # Current bullish
        (o <= c.shift(1)) &
        (c >= o.shift(1))
    )

    return patterns
```

---

## Academic Notes

- **Caginalp & Laurent (1998)**: Found some predictive value for candlesticks
- **Marshall et al. (2006)**: Limited evidence of profitability on DJIA
- **Morris (2006)**: "Candlestick Charting Explained" - Practitioner reference

**Key Insight**: Use patterns as **confirmation**, not primary signals.
