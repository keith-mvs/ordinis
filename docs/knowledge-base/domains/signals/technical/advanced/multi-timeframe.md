# Multi-Timeframe Analysis

## Overview

Multi-timeframe analysis (MTF) examines the same security across multiple timeframes to gain perspective. Higher timeframes reveal the trend, lower timeframes provide entry precision.

---

## Timeframe Hierarchy

### Standard Ratios
```
Position Trading:  Monthly → Weekly → Daily
Swing Trading:     Weekly → Daily → 4H
Day Trading:       Daily → 4H → 1H
Scalping:          4H → 1H → 15M
```

**Rule of Thumb**: Each timeframe should be ~4-6x the next lower

### Timeframe Roles
| Timeframe | Role | Determines |
|-----------|------|------------|
| Higher (3x) | Context | Trend direction, major S/R |
| Current | Decision | Entry/exit signals |
| Lower | Timing | Precise entry, stop placement |

---

## Core Principles

### 1. Trend Alignment
```python
# Trade in direction of higher TF trend
WEEKLY_TREND = weekly_close > weekly_SMA(20)
DAILY_SIGNAL = daily_RSI < 30 AND daily_close > daily_SMA(20)

# Only take daily buy signals when weekly is bullish
VALID_LONG = WEEKLY_TREND AND DAILY_SIGNAL
```

### 2. Nested Structure
```python
# Higher TF sets the range, lower TF trades within

# Weekly defines swing range
WEEKLY_SUPPORT = weekly_low.rolling(10).min()
WEEKLY_RESISTANCE = weekly_high.rolling(10).max()

# Daily trades within that range
DAILY_LONG_AT_SUPPORT = (
    daily_low <= WEEKLY_SUPPORT * 1.01 AND
    daily_bullish_reversal()
)
```

### 3. Confirmation Cascade
```python
# Signal on higher TF, confirm on lower

# Weekly breakout
WEEKLY_BREAKOUT = weekly_close > weekly_resistance

# Wait for daily confirmation
DAILY_CONFIRMATION = (
    WEEKLY_BREAKOUT AND
    daily_close > daily_SMA(20) AND
    daily_RSI > 50
)

# Enter on 4H timing
ENTRY_4H = (
    DAILY_CONFIRMATION AND
    h4_pullback_to_ma() AND
    h4_bullish_candle()
)
```

---

## Implementation Patterns

### Top-Down Analysis
```python
def top_down_analysis(symbol):
    weekly = get_data(symbol, timeframe='weekly')
    daily = get_data(symbol, timeframe='daily')
    hourly = get_data(symbol, timeframe='1h')

    # Step 1: Weekly trend
    weekly_trend = 'up' if weekly['close'] > SMA(weekly['close'], 20) else 'down'

    # Step 2: Daily setup
    if weekly_trend == 'up':
        daily_setup = detect_bullish_setup(daily)
    else:
        daily_setup = detect_bearish_setup(daily)

    # Step 3: Hourly trigger
    if daily_setup['valid']:
        hourly_trigger = find_entry_trigger(hourly, daily_setup['direction'])
        return hourly_trigger

    return None
```

### MTF Indicator Confluence
```python
# Same indicator across timeframes
RSI_WEEKLY = RSI(weekly, 14)
RSI_DAILY = RSI(daily, 14)
RSI_4H = RSI(h4, 14)

# All oversold = strong buy signal
TRIPLE_OVERSOLD = (
    RSI_WEEKLY < 40 AND
    RSI_DAILY < 30 AND
    RSI_4H < 30
)

# Divergence across timeframes
MTF_BULLISH_DIV = (
    RSI_WEEKLY.higher_low() AND
    RSI_DAILY.higher_low() AND
    price.lower_low()
)
```

### MTF Moving Average Filter
```python
# Price relative to MAs across timeframes
def mtf_ma_filter(data_dict):
    signals = {}

    for tf, data in data_dict.items():
        signals[tf] = {
            'above_20': data['close'] > SMA(data['close'], 20),
            'above_50': data['close'] > SMA(data['close'], 50),
            'above_200': data['close'] > SMA(data['close'], 200)
        }

    # Strong uptrend: all TFs above all MAs
    strong_up = all(
        signals[tf]['above_20'] and signals[tf]['above_50']
        for tf in signals
    )

    return {
        'signals': signals,
        'strong_uptrend': strong_up
    }
```

---

## MTF Entry Strategies

### 1. Higher TF Trend + Lower TF Pullback
```python
# Weekly uptrend
WEEKLY_UPTREND = weekly_close > weekly_EMA(21)

# Daily pullback to MA
DAILY_PULLBACK = (
    daily_low <= daily_EMA(21) AND
    daily_close > daily_EMA(21)
)

# 4H entry trigger
ENTRY = (
    WEEKLY_UPTREND AND
    DAILY_PULLBACK AND
    h4_RSI crosses_above 30 AND
    h4_bullish_engulfing
)
```

### 2. Multi-TF Breakout Confirmation
```python
# Daily breaks resistance
DAILY_BREAKOUT = daily_close > daily_resistance

# 4H confirmation
H4_CONFIRM = (
    h4_close > daily_resistance AND
    h4_volume > h4_avg_volume * 1.5
)

# 1H entry on retest
H1_ENTRY = (
    DAILY_BREAKOUT AND
    H4_CONFIRM AND
    h1_low <= daily_resistance * 1.01 AND  # Retesting
    h1_close > daily_resistance             # Holding
)
```

### 3. Divergence Cascade
```python
# Weekly divergence (most powerful)
WEEKLY_DIV = detect_divergence(weekly_RSI, weekly_price)

# Confirmed by daily
DAILY_DIV = detect_divergence(daily_RSI, daily_price)

# Entry on 4H reversal pattern
ENTRY = (
    WEEKLY_DIV AND
    DAILY_DIV AND
    h4_reversal_pattern()
)
```

---

## MTF Stop Placement

```python
def mtf_stop_placement(entry_tf, higher_tf_data, lower_tf_data):
    """
    Use lower TF for tighter stops, higher TF for wider.
    """
    # Tight stop: Below lower TF structure
    tight_stop = lower_tf_data['low'].rolling(5).min()

    # Medium stop: Below entry TF swing low
    medium_stop = find_swing_low(entry_tf_data)

    # Wide stop: Below higher TF support
    wide_stop = higher_tf_data['support']

    return {
        'tight': tight_stop,   # Smaller position, higher R:R
        'medium': medium_stop, # Balanced
        'wide': wide_stop      # Larger position, lower R:R
    }
```

---

## MTF Trend Assessment

```python
def assess_trend_strength(data_dict):
    """
    Score trend strength across timeframes.
    """
    score = 0
    max_score = len(data_dict) * 3  # 3 points per TF

    for tf, data in data_dict.items():
        # +1: Price above short MA
        if data['close'] > SMA(data['close'], 20):
            score += 1
        # +1: Price above long MA
        if data['close'] > SMA(data['close'], 50):
            score += 1
        # +1: Short MA above long MA
        if SMA(data['close'], 20) > SMA(data['close'], 50):
            score += 1

    trend_strength = score / max_score

    return {
        'score': score,
        'max': max_score,
        'strength': trend_strength,
        'rating': 'strong' if trend_strength > 0.8 else
                  'moderate' if trend_strength > 0.5 else 'weak'
    }
```

---

## Common Pitfalls

1. **Analysis Paralysis**: Too many timeframes = indecision
2. **Conflicting Signals**: Different TFs often disagree - use hierarchy
3. **Over-optimization**: Fitting to specific TF combinations
4. **Ignoring Context**: Lower TF signals without higher TF confirmation

---

## Best Practices

1. **Limit Timeframes**: Use 2-3, not more
2. **Clear Hierarchy**: Know which TF has priority
3. **Consistent Rules**: Same indicators/parameters across TFs
4. **Higher TF Wins**: When in conflict, defer to higher
5. **Entry on Lower, Exit on Higher**: Precision in, patience out

---

## Implementation

```python
import pandas as pd
from ordinis.analysis.technical.multi_timeframe import MultiTimeframeAnalyzer

# Price data per timeframe (must include open/high/low/close/volume columns)
data_by_tf = {
    "1h": hourly_df,   # e.g., pd.DataFrame with 1h candles
    "4h": h4_df,
    "1d": daily_df,
}

analyzer = MultiTimeframeAnalyzer()
result = analyzer.analyze(data_by_tf)

print(result.majority_trend)    # "bullish" | "bearish" | "mixed"
print(result.agreement_score)   # 0.0 - 1.0 alignment score
print(result.bias)              # "bullish" | "bearish" | "neutral"

for signal in result.signals:
    print(f"{signal.timeframe}: {signal.trend} (score={signal.score:.2f})")
```

---

## Academic Notes

- **Elder (1993)**: "Triple Screen Trading System"
- **Multiple TF**: Reduces false signals but adds complexity
- **Key Insight**: Higher TF provides context, lower TF provides timing

**Best Practice**: Always check 1 TF higher for context before taking any signal.
