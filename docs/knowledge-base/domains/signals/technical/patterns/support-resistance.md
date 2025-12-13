# Support and Resistance

## Overview

Support and resistance are price levels where buying or selling pressure historically concentrates. These levels serve as potential reversal or breakout zones and are fundamental to technical analysis.

---

## Definitions

### Support
- Price level where buying pressure exceeds selling pressure
- Price tends to "bounce" upward from support
- Previous support, once broken, often becomes resistance

### Resistance
- Price level where selling pressure exceeds buying pressure
- Price tends to reverse downward from resistance
- Previous resistance, once broken, often becomes support

---

## Types of Support/Resistance

### 1. Horizontal Levels

**Static Price Levels:**
```python
def find_horizontal_levels(prices, lookback=100, touches=3, tolerance=0.02):
    """
    Find price levels touched multiple times.
    """
    levels = []

    for price in prices[-lookback:]:
        # Count touches within tolerance
        touch_count = ((prices >= price * (1 - tolerance)) &
                       (prices <= price * (1 + tolerance))).sum()

        if touch_count >= touches:
            levels.append(price)

    return deduplicate_levels(levels)
```

### 2. Dynamic Levels

**Moving Averages as S/R:**
```python
# Common MA support/resistance levels
MA_20 = SMA(20)   # Short-term
MA_50 = SMA(50)   # Intermediate
MA_200 = SMA(200) # Long-term (institutional)

AT_MA_SUPPORT = (
    close > MA_50 AND
    low <= MA_50 * 1.01 AND
    close > open                         # Bounce candle
)
```

### 3. Trendline Support/Resistance

```python
def draw_trendline(pivot_points, direction='support'):
    """
    Connect swing lows (support) or swing highs (resistance).
    """
    if direction == 'support':
        points = [p for p in pivot_points if p['type'] == 'low']
    else:
        points = [p for p in pivot_points if p['type'] == 'high']

    # Linear regression through points
    slope, intercept = linear_regression(points)

    return slope, intercept
```

### 4. Fibonacci Levels

**Retracement Levels:**
```python
# Common Fibonacci retracement levels
FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]

def fibonacci_retracements(swing_high, swing_low, direction='up'):
    range_size = swing_high - swing_low

    if direction == 'up':  # Pullback in uptrend
        levels = {
            f'fib_{int(level*100)}': swing_high - (range_size * level)
            for level in FIB_LEVELS
        }
    else:  # Rally in downtrend
        levels = {
            f'fib_{int(level*100)}': swing_low + (range_size * level)
            for level in FIB_LEVELS
        }

    return levels
```

### 5. Pivot Points

**Standard Pivot:**
```python
# Daily pivot points
PIVOT = (High[prev] + Low[prev] + Close[prev]) / 3

R1 = (2 × PIVOT) - Low[prev]
R2 = PIVOT + (High[prev] - Low[prev])
R3 = High[prev] + 2 × (PIVOT - Low[prev])

S1 = (2 × PIVOT) - High[prev]
S2 = PIVOT - (High[prev] - Low[prev])
S3 = Low[prev] - 2 × (High[prev] - PIVOT)
```

**Variations:**
- Woodie's Pivot
- Camarilla Pivot
- Fibonacci Pivot

---

## Rule Templates

### Level Identification
```python
# Major support/resistance (3+ touches)
MAJOR_LEVEL = touch_count >= 3

# Minor level (1-2 touches)
MINOR_LEVEL = 1 <= touch_count <= 2

# Recently tested level
FRESH_LEVEL = last_touch < 20 bars
STALE_LEVEL = last_touch > 50 bars
```

### Level Proximity
```python
# Price near level
def near_level(price, level, atr, multiplier=0.5):
    return abs(price - level) < atr * multiplier

AT_SUPPORT = near_level(close, support, ATR(14))
AT_RESISTANCE = near_level(close, resistance, ATR(14))
```

### Level Tests and Breaks
```python
# Support test
SUPPORT_TEST = low <= support AND close > support

# Support break
SUPPORT_BREAK = close < support AND volume > avg_volume

# Failed breakdown (bull trap)
FAILED_BREAKDOWN = (
    close[1] < support AND            # Broke down
    close > support                   # Closed back above
)
```

---

## Trading Strategies

### 1. Bounce Trade
```python
# Long at support
LONG_AT_SUPPORT = (
    near_level(low, support, ATR(14)) AND
    candlestick_reversal_pattern AND   # Hammer, engulfing, etc.
    RSI(14) < 40                       # Oversold confirmation
)

STOP = support - (1.5 * ATR(14))
TARGET = next_resistance OR support + (2 * (entry - stop))
```

### 2. Breakout Trade
```python
# Long on resistance breakout
BREAKOUT_LONG = (
    close > resistance AND
    close[1] <= resistance AND         # Just broke out
    volume > volume.rolling(20).mean() * 1.5
)

STOP = resistance - ATR(14)            # Below old resistance (new support)
TARGET = resistance + (resistance - recent_support)
```

### 3. Retest Entry
```python
# Enter on retest after breakout
RETEST_LONG = (
    close[5:10] > resistance AND       # Recent breakout
    low <= resistance * 1.01 AND       # Retesting old resistance
    close > resistance AND             # Holding above
    volume declining during retest
)
```

### 4. Range Trading
```python
# Trade between S/R in ranging market
RANGE_BOUND = ADX(14) < 20

RANGE_LONG = (
    RANGE_BOUND AND
    near_level(low, support, ATR(14)) AND
    close > open
)

RANGE_SHORT = (
    RANGE_BOUND AND
    near_level(high, resistance, ATR(14)) AND
    close < open
)
```

---

## Role Reversal

```python
# Support becomes resistance (and vice versa)

ROLE_REVERSAL = (
    price broke_below support[past] AND
    now approaching old_support from below AND
    old_support acts as resistance
)

# Trade the reversal
SHORT_AT_OLD_SUPPORT = (
    close[5:10] < old_support AND      # Confirmed break
    high >= old_support * 0.99 AND     # Testing from below
    close < old_support AND            # Rejected
    bearish_candlestick
)
```

---

## Multi-Timeframe S/R

```python
# Weekly S/R more significant than daily
WEEKLY_SUPPORT = calculate_support(weekly_data)
DAILY_SUPPORT = calculate_support(daily_data)

# Confluence zone
STRONG_SUPPORT = (
    near_level(close, WEEKLY_SUPPORT, ATR(14)) AND
    near_level(close, DAILY_SUPPORT, ATR(14))
)

# Priority: Monthly > Weekly > Daily > Intraday
```

---

## Volume at Levels

```python
# High volume at level = strong S/R
def level_strength(level, prices, volumes, lookback=100):
    # Find touches
    at_level = near_level(prices, level, prices.std() * 0.5)

    # Average volume at level
    volume_at_level = volumes[at_level].mean()
    avg_volume = volumes.mean()

    return volume_at_level / avg_volume  # > 1 = strong level
```

---

## Common Pitfalls

1. **Round Numbers Bias**: Levels at 100, 50, etc. are crowded
2. **Moving Levels**: S/R isn't exact; use zones not lines
3. **Broken Levels**: Old S/R loses significance over time
4. **Self-Fulfilling Prophecy**: Popular levels attract orders

---

## Implementation

```python
from ordinis.analysis.technical.patterns import SupportResistanceLocator

locator = SupportResistanceLocator()

# Find recent pivot-based levels (returns SupportResistanceLevels dataclass)
levels = locator.find_levels(
    high=data["high"],
    low=data["low"],
    window=3,
    tolerance=0.003,  # merge nearby pivots into a zone
)

print(levels.support, levels.support_touches)
print(levels.resistance, levels.resistance_touches)
```

---

## Best Practices

1. **Use Zones, Not Lines**: S/R is an area, not exact price
2. **Prioritize Recent Levels**: More touches = stronger
3. **Combine with Volume**: High-volume levels are more significant
4. **Multi-Timeframe**: Higher TF levels take precedence
5. **Wait for Confirmation**: Don't anticipate; react to price action at levels

---

## Academic Notes

- **Osler (2000)**: Stop-loss orders cluster at round numbers
- **Support/Resistance**: Based on order flow and psychology
- **Key Insight**: S/R works because traders collectively act on it

**Best Practice**: Use S/R to define risk (stops beyond levels) and targets, not as entry signals alone.
