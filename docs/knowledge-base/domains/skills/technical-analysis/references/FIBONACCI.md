# Fibonacci Retracement Reference

## Overview

Fibonacci retracement levels use mathematical ratios derived from the Fibonacci sequence to identify potential support and resistance zones during price corrections.

## Key Levels

### Standard Retracement Levels

| Level | Ratio | Calculation | Use Case |
|-------|-------|-------------|----------|
| 0% | - | Swing high/low | Starting point |
| 23.6% | - | Shallow retracement | Minor correction |
| 38.2% | (√5 - 1) / 2 | Moderate retracement | Common correction |
| 50% | - | Midpoint | Psychological level |
| 61.8% | φ (Golden Ratio) | Deep retracement | Critical support/resistance |
| 78.6% | √φ | Very deep | Last support before reversal |
| 100% | - | Full retracement | Complete correction |

### Extension Levels

Used to project targets beyond the original move:
- 127.2%
- 161.8% (Golden Ratio)
- 200%
- 261.8%

## Calculation

### Retracement Levels (Uptrend)

```
Level = High - (High - Low) × Fibonacci Ratio

Example:
High = 100, Low = 50
38.2% Retracement = 100 - (100 - 50) × 0.382 = 80.9
61.8% Retracement = 100 - (100 - 50) × 0.618 = 69.1
```

### Retracement Levels (Downtrend)

```
Level = Low + (High - Low) × Fibonacci Ratio

Example:
High = 100, Low = 50
38.2% Retracement = 50 + (100 - 50) × 0.382 = 69.1
61.8% Retracement = 50 + (100 - 50) × 0.618 = 80.9
```

## Python Implementation

```python
def calculate_fibonacci_levels(high, low, direction='up'):
    """
    Calculate Fibonacci retracement and extension levels.

    Parameters:
    -----------
    high : float
        Swing high price
    low : float
        Swing low price
    direction : str
        'up' for uptrend retracement, 'down' for downtrend

    Returns:
    --------
    dict with all Fibonacci levels
    """
    diff = high - low

    # Retracement levels
    levels = {
        '0.0%': high if direction == 'up' else low,
        '23.6%': None,
        '38.2%': None,
        '50.0%': None,
        '61.8%': None,
        '78.6%': None,
        '100.0%': low if direction == 'up' else high
    }

    # Extensions
    extensions = {
        '127.2%': None,
        '161.8%': None,
        '200.0%': None,
        '261.8%': None
    }

    if direction == 'up':
        # Retracements from high
        levels['23.6%'] = high - diff * 0.236
        levels['38.2%'] = high - diff * 0.382
        levels['50.0%'] = high - diff * 0.500
        levels['61.8%'] = high - diff * 0.618
        levels['78.6%'] = high - diff * 0.786

        # Extensions beyond high
        extensions['127.2%'] = high + diff * 0.272
        extensions['161.8%'] = high + diff * 0.618
        extensions['200.0%'] = high + diff * 1.000
        extensions['261.8%'] = high + diff * 1.618
    else:
        # Retracements from low
        levels['23.6%'] = low + diff * 0.236
        levels['38.2%'] = low + diff * 0.382
        levels['50.0%'] = low + diff * 0.500
        levels['61.8%'] = low + diff * 0.618
        levels['78.6%'] = low + diff * 0.786

        # Extensions beyond low
        extensions['127.2%'] = low - diff * 0.272
        extensions['161.8%'] = low - diff * 0.618
        extensions['200.0%'] = low - diff * 1.000
        extensions['261.8%'] = low - diff * 1.618

    return {'retracements': levels, 'extensions': extensions}


def identify_swing_points(high, low, close, window=20):
    """
    Identify swing high and swing low for Fibonacci calculation.

    Parameters:
    -----------
    high, low, close : pd.Series
        Price data
    window : int
        Lookback window for swing identification

    Returns:
    --------
    tuple of (swing_high, swing_low, swing_high_idx, swing_low_idx)
    """
    import pandas as pd

    # Find highest high and lowest low in window
    swing_high = high.rolling(window).max().iloc[-1]
    swing_low = low.rolling(window).min().iloc[-1]

    # Find indices
    swing_high_idx = high[high == swing_high].index[-1]
    swing_low_idx = low[low == swing_low].index[-1]

    return swing_high, swing_low, swing_high_idx, swing_low_idx
```

## Interpretation

### Retracement Trading

**Bullish Setup** (buying dips):
```
1. Identify uptrend (higher highs, higher lows)
2. Mark recent swing high and low
3. Wait for retracement to key Fibonacci level
4. Enter long at 38.2%, 50%, or 61.8% level
5. Stop below 78.6% or 100% level
6. Target: Previous high or extension levels
```

**Bearish Setup** (selling rallies):
```
1. Identify downtrend (lower highs, lower lows)
2. Mark recent swing high and low
3. Wait for retracement to key Fibonacci level
4. Enter short at 38.2%, 50%, or 61.8% level
5. Stop above 78.6% or 100% level
6. Target: Previous low or extension levels
```

### Level Significance

**23.6% Level**:
- Shallow retracement
- Strong trend continuation
- Often first support/resistance

**38.2% Level**:
- Moderate retracement
- Common in healthy trends
- Good risk/reward entry

**50% Level**:
- Psychological midpoint
- Not pure Fibonacci, but widely watched
- Strong support/resistance

**61.8% Level** (Golden Ratio):
- Deep retracement
- Last chance for trend continuation
- High probability support/resistance
- Most important Fibonacci level

**78.6% Level**:
- Very deep retracement
- Trend may be weakening
- Often coincides with major moving averages

### Confluence with Other Indicators

**Fibonacci + Moving Averages**:
```
Strong confluence when Fibonacci level aligns with:
- 50-day or 200-day MA
- Entry: Price reaches both levels
- Confirmation: Bounce from confluence zone
```

**Fibonacci + Trendlines**:
```
High-probability setup:
- Fibonacci level
- Trendline support/resistance
- Both align at same price
```

**Fibonacci + Volume**:
```
Confirmation:
- Price reaches Fibonacci level
- Volume increases (accumulation/distribution)
- OBV confirms direction
```

## Extension Trading

### Profit Targets

After entry at retracement level, use extensions as profit targets:

```
Entry: 61.8% retracement
Target 1: Previous high (100% extension)
Target 2: 127.2% extension
Target 3: 161.8% extension (Golden Ratio)
```

### Extension Strategy

```python
def calculate_profit_targets(entry, swing_high, swing_low, direction='long'):
    """
    Calculate Fibonacci extension profit targets.

    Parameters:
    -----------
    entry : float
        Entry price
    swing_high : float
        Recent swing high
    swing_low : float
        Recent swing low
    direction : str
        'long' or 'short'

    Returns:
    --------
    dict with profit targets
    """
    diff = swing_high - swing_low

    if direction == 'long':
        targets = {
            'T1': swing_high,
            'T2': swing_high + diff * 0.272,
            'T3': swing_high + diff * 0.618,
            'T4': swing_high + diff * 1.000,
            'T5': swing_high + diff * 1.618
        }
    else:
        targets = {
            'T1': swing_low,
            'T2': swing_low - diff * 0.272,
            'T3': swing_low - diff * 0.618,
            'T4': swing_low - diff * 1.000,
            'T5': swing_low - diff * 1.618
        }

    return targets
```

## Common Patterns

### AB=CD Pattern

Price moves equal distances using Fibonacci ratios:

```
Structure:
A → B: Initial move
B → C: Retracement (typically 38.2% or 61.8%)
C → D: Extension equal to A→B

Trade:
Entry: Point D
Stop: Beyond D
Target: Fibonacci extensions
```

### Three-Drive Pattern

Three successive moves and retracements:

```
Structure:
Drive 1: Initial move
Correction 1: 61.8% retracement
Drive 2: 127.2% extension
Correction 2: 61.8% retracement
Drive 3: 127.2% extension

Trade:
Entry: End of Drive 3
Reversal likely
```

## Automatic Fibonacci Detection

```python
def auto_fibonacci_analysis(data, lookback=100):
    """
    Automatically identify swing points and calculate Fibonacci levels.

    Parameters:
    -----------
    data : pd.DataFrame
        OHLC data
    lookback : int
        Periods to search for swings

    Returns:
    --------
    Analysis with identified levels and current price position
    """
    # Identify swings
    swing_high, swing_low, high_idx, low_idx = identify_swing_points(
        data['High'], data['Low'], data['Close'], lookback
    )

    # Determine direction
    direction = 'up' if high_idx > low_idx else 'down'

    # Calculate levels
    fib = calculate_fibonacci_levels(swing_high, swing_low, direction)

    # Current price
    current_price = data['Close'].iloc[-1]

    # Find nearest level
    nearest_level = None
    min_distance = float('inf')

    for level_name, level_price in fib['retracements'].items():
        if level_price is not None:
            distance = abs(current_price - level_price)
            if distance < min_distance:
                min_distance = distance
                nearest_level = (level_name, level_price)

    return {
        'swing_high': swing_high,
        'swing_low': swing_low,
        'direction': direction,
        'levels': fib,
        'current_price': current_price,
        'nearest_level': nearest_level,
        'distance_to_nearest': min_distance
    }
```

## Best Practices

### Swing Selection

1. **Use significant swings**: Choose major highs/lows, not minor fluctuations
2. **Recent swings**: Most relevant within last 3-6 months
3. **Multiple timeframes**: Validate across daily, weekly, monthly charts
4. **Clear trend**: Works best in established trends, not choppy markets

### Entry Timing

1. **Don't enter at level**: Wait for confirmation
2. **Confirmation signals**:
   - Candlestick reversal pattern
   - Volume increase
   - Momentum divergence
   - Support from other indicators
3. **Stop placement**: Below next Fibonacci level (not at entry level)

### Risk Management

```
Position Sizing:
Risk per trade: 1-2% of capital
Stop distance: Next Fibonacci level or 100% level
Target: Multiple targets at extension levels

Example:
Entry: $50 (61.8% retracement)
Stop: $45 (100% level)
Risk: $5 per share
Target 1: $55 (previous high) - 50% position
Target 2: $58 (127.2% extension) - 30% position
Target 3: $63 (161.8% extension) - 20% position
```

## Limitations

1. **Subjective swing selection**: Different traders identify different swings
2. **Self-fulfilling prophecy**: Works partly because many traders use it
3. **No guarantee**: Levels are areas of interest, not guaranteed support/resistance
4. **Requires confluence**: Best when combined with other analysis
5. **Market conditions**: Less effective in ranging or choppy markets

## Validation

```bash
python scripts/validate_fibonacci.py --test-all
```

Tests:
- Calculation accuracy
- Swing point identification
- Level clustering analysis
- Historical bounce rate at levels
- Confluence detection with other indicators

## Further Reading

- Fibonacci, Leonardo (1202). "Liber Abaci"
- Fischer, Robert (1993). "Fibonacci Applications and Strategies for Traders"
- Brown, Constance (2008). "Fibonacci Analysis"
- CMT Association: Fibonacci Studies curriculum
- Elliott, Ralph Nelson (1938). "The Wave Principle" (Fibonacci-based)
