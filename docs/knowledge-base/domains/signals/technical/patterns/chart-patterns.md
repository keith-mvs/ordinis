# Chart Patterns

## Overview

Chart patterns are price formations that occur over multiple bars/candles, often signaling continuation or reversal. Unlike single candlesticks, these patterns take time to form and typically provide more reliable (though still imperfect) signals.

---

## Pattern Categories

| Type | Patterns | Signal |
|------|----------|--------|
| Reversal | Head & Shoulders, Double Top/Bottom | Trend change |
| Continuation | Triangles, Flags, Pennants | Trend resumption |
| Bilateral | Symmetric Triangle, Rectangle | Could break either way |

---

## Reversal Patterns

### Head and Shoulders

**Formation:**
```
        Head
         /\
    LS  /  \  RS
     /\/    \/\
    /          \
___/____________\___  <- Neckline
```

**Identification:**
```python
HEAD_AND_SHOULDERS = (
    left_shoulder.high < head.high AND
    right_shoulder.high < head.high AND
    abs(left_shoulder.high - right_shoulder.high) < tolerance AND
    neckline.slope <= 0                  # Horizontal or downward
)

# Neckline = line connecting lows between shoulders
NECKLINE_BREAK = close < neckline
```

**Trading:**
```python
# Entry on neckline break
SHORT_ENTRY = close breaks_below neckline AND volume > avg_volume

# Target
TARGET = neckline - (head.high - neckline)  # Measured move

# Stop
STOP = right_shoulder.high
```

**Inverse Head and Shoulders**: Same but inverted (bullish reversal at bottom)

---

### Double Top / Double Bottom

**Double Top:**
```
    /\      /\
   /  \    /  \
  /    \  /    \
 /      \/      \

    First   Second
    Peak    Peak
```

**Identification:**
```python
DOUBLE_TOP = (
    abs(peak1.high - peak2.high) < tolerance AND
    valley_between < both_peaks AND
    peak2.volume < peak1.volume          # Declining volume on second peak
)

DOUBLE_BOTTOM = (
    abs(trough1.low - trough2.low) < tolerance AND
    peak_between > both_troughs AND
    trough2.volume < trough1.volume
)
```

**Trading:**
```python
# Double Top Short Entry
SHORT_ENTRY = close breaks_below valley_low

# Double Bottom Long Entry
LONG_ENTRY = close breaks_above peak_high

# Target = distance from peaks/troughs to middle
TARGET = entry +/- (peak - valley)
```

---

### Triple Top / Triple Bottom

Similar to double, but with three tests of resistance/support.

```python
TRIPLE_TOP = (
    abs(peak1.high - peak2.high) < tolerance AND
    abs(peak2.high - peak3.high) < tolerance AND
    valleys_between < peaks
)
```

---

## Continuation Patterns

### Triangle Patterns

**Ascending Triangle (Bullish):**
```
    _______  <- Flat resistance
   /
  /
 /
```

```python
ASCENDING_TRIANGLE = (
    highs.resistance_line.is_horizontal() AND
    lows.trendline.slope > 0 AND         # Rising lows
    volume.decreasing_trend()
)

BREAKOUT = close > resistance AND volume > avg_volume * 1.5
```

**Descending Triangle (Bearish):**
```
\
 \
  \
   \_______  <- Flat support
```

```python
DESCENDING_TRIANGLE = (
    lows.support_line.is_horizontal() AND
    highs.trendline.slope < 0 AND        # Lower highs
    volume.decreasing_trend()
)

BREAKDOWN = close < support AND volume > avg_volume * 1.5
```

**Symmetric Triangle (Neutral):**
```
\    /
 \  /
  \/
```

```python
SYMMETRIC_TRIANGLE = (
    highs.trendline.slope < 0 AND        # Lower highs
    lows.trendline.slope > 0 AND         # Higher lows
    volume.decreasing_trend()
)

# Break direction determines signal
BREAK_UP = close > upper_trendline
BREAK_DOWN = close < lower_trendline
```

---

### Flags and Pennants

**Bull Flag:**
```
     |
     |
    /|  <- Flagpole (sharp move up)
   / |
  /__|
     /\
    /  \   <- Flag (slight pullback)
   /____\
```

```python
BULL_FLAG = (
    flagpole.return > 10% AND            # Strong prior move
    flag.slope < 0 AND                   # Downward sloping consolidation
    flag.duration < 20 bars AND
    flag.retrace < 0.50 * flagpole       # Shallow pullback
)

BREAKOUT = close > flag.upper_bound AND volume > avg_volume
TARGET = breakout_point + flagpole.height
```

**Pennant:** Similar but triangle-shaped consolidation instead of rectangle.

---

### Wedges

**Rising Wedge (Bearish):**
```
    /
   /|
  / |
 /  /
/  /   <- Both lines rising, but converging
```

```python
RISING_WEDGE = (
    highs.trendline.slope > 0 AND
    lows.trendline.slope > 0 AND
    highs.slope < lows.slope AND         # Converging
    in_uptrend
)

# Bearish - expect break down
BREAKDOWN = close < lower_trendline
```

**Falling Wedge (Bullish):** Opposite - converging down, expect breakout up.

---

### Rectangle / Range

```
_________  <- Resistance


_________  <- Support
```

```python
RECTANGLE = (
    highs cluster near resistance AND
    lows cluster near support AND
    abs(resistance - support) / support < 0.10  # Range bound
)

BREAKOUT = close > resistance OR close < support
```

---

## Pattern Measurement

### Measured Move
```python
# Target = pattern height projected from breakout

def measured_move(pattern_type, pattern_data, breakout_price):
    if pattern_type == "head_shoulders":
        height = pattern_data['head'] - pattern_data['neckline']
        target = breakout_price - height

    elif pattern_type == "triangle":
        height = pattern_data['initial_high'] - pattern_data['initial_low']
        target = breakout_price + height  # or minus for breakdown

    elif pattern_type == "flag":
        flagpole_height = pattern_data['flagpole_top'] - pattern_data['flagpole_bottom']
        target = breakout_price + flagpole_height

    return target
```

---

## Volume Confirmation

```python
# Volume patterns that confirm chart patterns

# Declining volume during pattern formation
VALID_FORMATION = volume.rolling(20).mean().is_declining()

# Volume surge on breakout
CONFIRMED_BREAKOUT = (
    pattern_breakout AND
    volume > volume.rolling(20).mean() * 1.5
)

# Failed breakout (no volume)
FAILED_BREAKOUT = (
    pattern_breakout AND
    volume < volume.rolling(20).mean()
)
```

---

## Common Pitfalls

1. **Subjective Identification**: Patterns in hindsight look obvious
2. **Timeframe Sensitivity**: Same data can show different patterns
3. **Failed Patterns**: Many patterns don't complete or fail
4. **Confirmation Bias**: Seeing patterns that aren't there

---

## Algorithmic Pattern Detection

```python
def detect_head_shoulders(prices, window=50, tolerance=0.02):
    """
    Automated H&S detection using local extrema.
    """
    # Find local maxima and minima
    highs = find_local_maxima(prices, window=5)
    lows = find_local_minima(prices, window=5)

    # Need at least 3 highs and 2 lows
    if len(highs) < 3 or len(lows) < 2:
        return None

    # Check for H&S structure
    for i in range(len(highs) - 2):
        left_shoulder = highs[i]
        head = highs[i + 1]
        right_shoulder = highs[i + 2]

        # Head higher than shoulders
        if head['price'] <= left_shoulder['price']:
            continue
        if head['price'] <= right_shoulder['price']:
            continue

        # Shoulders roughly equal
        shoulder_diff = abs(left_shoulder['price'] - right_shoulder['price'])
        if shoulder_diff / left_shoulder['price'] > tolerance:
            continue

        # Find neckline
        neckline = find_neckline(lows, left_shoulder, head, right_shoulder)

        return {
            'pattern': 'head_shoulders',
            'left_shoulder': left_shoulder,
            'head': head,
            'right_shoulder': right_shoulder,
            'neckline': neckline,
            'target': neckline['price'] - (head['price'] - neckline['price'])
        }

    return None
```

---

## Best Practices

1. **Wait for Confirmation**: Don't anticipate pattern completion
2. **Volume Required**: Breakouts need volume surge
3. **Use Stop Losses**: Pattern failure point defines risk
4. **Multiple Timeframes**: Higher TF patterns more reliable
5. **Don't Force It**: If pattern isn't clear, it's not valid

---

## Academic Notes

- **Bulkowski (2005)**: "Encyclopedia of Chart Patterns"
- **Lo, Mamaysky, Wang (2000)**: Automated pattern recognition study
- **Key Insight**: Patterns work better as confirmation than primary signals

**Best Practice**: Use patterns to define risk (stop at pattern failure point) rather than as high-probability entry signals.
