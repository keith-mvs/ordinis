# Stochastic Oscillator

## Overview

The Stochastic Oscillator, developed by George Lane in the 1950s, measures the current price relative to the high-low range over a specified period. It's based on the observation that prices tend to close near highs in uptrends and near lows in downtrends.

---

## Formula

```
%K = 100 × (Close - Lowest Low) / (Highest High - Lowest Low)
%D = SMA(3) of %K (signal line)

Full Stochastic:
%K = SMA(3) of Raw %K
%D = SMA(3) of %K
```

---

## Types

| Type | %K Smoothing | %D Smoothing | Sensitivity |
|------|--------------|--------------|-------------|
| Fast | None | 3 | Very high |
| Slow | 3 | 3 | Moderate |
| Full | Adjustable | Adjustable | Customizable |

---

## Standard Parameters

| Parameter | Default | Alternatives |
|-----------|---------|--------------|
| %K Period | 14 | 5, 9, 21 |
| %K Smoothing | 3 | 1 (fast), 5 |
| %D Smoothing | 3 | 5 |
| Overbought | 80 | 75 |
| Oversold | 20 | 25 |

---

## Key Concepts

### Reading the Stochastic
- **%K above 80**: Price near top of range (overbought)
- **%K below 20**: Price near bottom of range (oversold)
- **%K crossing %D**: Momentum shift signal

### Lane's Rules
1. Divergence between stochastic and price precedes reversals
2. %K crossing %D is a trading signal
3. Look for extreme readings (>80 or <20) for setups

---

## Rule Templates

### Basic Signals
```python
# Overbought/Oversold
OVERBOUGHT = %K > 80
OVERSOLD = %K < 20

# Extreme readings
EXTREME_OB = %K > 90
EXTREME_OS = %K < 10
```

### Crossover Signals
```python
# Bullish crossover in oversold zone
BUY_SIGNAL = (
    %K crosses_above %D AND
    %K < 25                    # In oversold territory
)

# Bearish crossover in overbought zone
SELL_SIGNAL = (
    %K crosses_below %D AND
    %K > 75                    # In overbought territory
)
```

### With Trend Filter
```python
# Bullish setup with trend
BULLISH_SETUP = (
    close > SMA(50) AND        # Uptrend
    %K < 30 AND                # Oversold
    %K crosses_above %D        # Bullish cross
)

# Bearish setup with trend
BEARISH_SETUP = (
    close < SMA(50) AND        # Downtrend
    %K > 70 AND                # Overbought
    %K crosses_below %D        # Bearish cross
)
```

### Divergence
```python
# Bullish divergence
BULL_DIV = (
    price < price[lookback] AND  # Lower low in price
    %K > %K[lookback] AND        # Higher low in stochastic
    %K < 30                       # In oversold zone
)

# Bearish divergence
BEAR_DIV = (
    price > price[lookback] AND  # Higher high in price
    %K < %K[lookback] AND        # Lower high in stochastic
    %K > 70                       # In overbought zone
)
```

### %K/%D Hook Pattern
```python
# Bullish hook (reversal in oversold)
BULL_HOOK = (
    %K[2] < %D[2] AND            # Was below signal
    %K[1] < %D[1] AND            # Still below
    %K > %D AND                   # Now crossed above
    %K < 25                       # In oversold
)

# Bearish hook (reversal in overbought)
BEAR_HOOK = (
    %K[2] > %D[2] AND            # Was above signal
    %K[1] > %D[1] AND            # Still above
    %K < %D AND                   # Now crossed below
    %K > 75                       # In overbought
)
```

---

## Trading Strategies

### 1. Classic Stochastic Strategy
```python
# Entry rules
LONG = (
    %K crosses_above %D AND
    %K < 25 AND
    ADX(14) < 25               # Low trend strength (range-bound)
)

SHORT = (
    %K crosses_below %D AND
    %K > 75 AND
    ADX(14) < 25
)

STOP = 2 * ATR(14)
TARGET = Middle of range or opposite band
```

### 2. Stochastic + RSI Combo
```python
STRONG_LONG = (
    %K crosses_above %D AND
    %K < 30 AND
    RSI(14) < 35 AND
    RSI(14) > RSI(14)[1]       # RSI turning up
)
```

### 3. Multi-Timeframe Stochastic
```python
# Higher TF direction
DAILY_BULLISH = %K_daily > 50

# Lower TF entry
ENTRY = (
    DAILY_BULLISH AND
    %K_4h < 25 AND
    %K_4h crosses_above %D_4h
)
```

---

## Stochastic POP

A momentum continuation pattern:
```python
# Bullish POP
STOCH_POP = (
    %K > 80 AND                 # Overbought
    %K[1:5].max() < 80 AND      # Recently crossed into OB
    close > close[1]            # Price confirming
)
# Indicates strong momentum, NOT a reversal signal
```

---

## Common Pitfalls

1. **Overbought ≠ Sell**: Strong trends stay overbought
2. **Too Many Signals**: High sensitivity = many false signals
3. **Ignoring Trend**: Works best in ranges, fails in trends
4. **Fast Stochastic Noise**: Use slow stochastic for less noise

---

## Comparison with RSI

| Aspect | Stochastic | RSI |
|--------|------------|-----|
| Sensitivity | Higher | Lower |
| Best for | Ranging markets | Both |
| Signal type | Crossovers + levels | Levels + divergence |
| Whipsaw risk | Higher | Lower |

---

## Implementation

```python
from ordinis.analysis.technical.indicators import Oscillators

osc = Oscillators()

# Slow Stochastic (default)
stoch = osc.stochastic(data, period=14, k_smooth=3, d_smooth=3)
k = stoch['%K']
d = stoch['%D']

# Fast Stochastic
fast_stoch = osc.stochastic(data, period=14, k_smooth=1, d_smooth=3)
```

---

## Academic Notes

- **Lane (1984)**: Original developer's guidelines
- **Pruitt & Hill (2012)**: Found modest predictive value in specific conditions
- **Evidence**: Best combined with trend filters; standalone performance is marginal
