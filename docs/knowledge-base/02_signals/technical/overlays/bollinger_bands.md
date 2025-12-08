# Bollinger Bands

## Overview

Bollinger Bands measure price volatility using standard deviation channels around a moving average. Created by John Bollinger in the 1980s, they adapt to market conditions by widening in high volatility and contracting in low volatility.

---

## Formula

```
Middle Band = SMA(20)
Upper Band = SMA(20) + (2 × StdDev(20))
Lower Band = SMA(20) - (2 × StdDev(20))

%B = (Close - Lower Band) / (Upper Band - Lower Band)
Bandwidth = (Upper Band - Lower Band) / Middle Band
```

---

## Components

| Component | Description | Usage |
|-----------|-------------|-------|
| Middle Band | 20-period SMA | Trend direction, mean |
| Upper Band | +2 std dev | Resistance, overbought |
| Lower Band | -2 std dev | Support, oversold |
| %B | Position within bands | Overbought/oversold |
| Bandwidth | Band width | Volatility state |

---

## Standard Parameters

| Parameter | Default | Alternatives |
|-----------|---------|--------------|
| Period | 20 | 10, 50 |
| Std Dev | 2.0 | 1.5, 2.5, 3.0 |

**Statistical Note**: With 2 standard deviations, ~95% of price action should occur within the bands (assuming normal distribution).

---

## Key Concepts

### The Squeeze

**Definition**: Bandwidth at multi-period lows indicates consolidation before a potential breakout.

```python
# Squeeze detection
BANDWIDTH = (upper - lower) / middle
SQUEEZE = BANDWIDTH < percentile(BANDWIDTH, 20, lookback=126)
TIGHT_SQUEEZE = BANDWIDTH < percentile(BANDWIDTH, 5, lookback=252)
```

**Trading the Squeeze**:
- Wait for price to break outside bands
- Confirm with volume surge
- Direction of breakout determines trade

---

### Walking the Bands

**Definition**: In strong trends, price "walks" along the upper or lower band.

```python
# Walking upper band (strong uptrend)
WALKING_UPPER = close > upper_band AND close[1] > upper_band[1]

# Walking lower band (strong downtrend)
WALKING_LOWER = close < lower_band AND close[1] < lower_band[1]
```

**Warning**: Do NOT fade a walking band - it indicates trend strength.

---

## Rule Templates

### Mean Reversion Signals
```python
# Touch lower band in uptrend
OVERSOLD_BOUNCE = (
    close > SMA(50) AND          # Uptrend filter
    low < lower_band AND         # Touch lower band
    close > lower_band AND       # Close back inside
    RSI(14) < 35                 # Momentum confirmation
)

# Touch upper band in downtrend
OVERBOUGHT_FADE = (
    close < SMA(50) AND          # Downtrend filter
    high > upper_band AND        # Touch upper band
    close < upper_band AND       # Close back inside
    RSI(14) > 65                 # Momentum confirmation
)
```

### %B Signals
```python
# %B overbought/oversold
OVERBOUGHT = percent_b > 1.0    # Above upper band
OVERSOLD = percent_b < 0.0      # Below lower band
NEUTRAL = 0.2 < percent_b < 0.8

# Mean reversion entry
LONG_ENTRY = percent_b < 0 AND percent_b > percent_b[1]  # Below band, turning up
SHORT_ENTRY = percent_b > 1 AND percent_b < percent_b[1]  # Above band, turning down
```

### Squeeze Breakout
```python
# Detect squeeze
SQUEEZE = bandwidth < percentile(bandwidth, 10, 126)
SQUEEZE_DURATION = count consecutive bars where SQUEEZE

# Breakout signals
BULLISH_BREAKOUT = (
    SQUEEZE[1:5].any() AND       # Recent squeeze
    close > upper_band AND       # Break above
    volume > avg_volume * 1.5    # Volume confirmation
)

BEARISH_BREAKOUT = (
    SQUEEZE[1:5].any() AND       # Recent squeeze
    close < lower_band AND       # Break below
    volume > avg_volume * 1.5    # Volume confirmation
)
```

### Double Bottom at Bands
```python
# W-bottom pattern
W_BOTTOM = (
    low[n] < lower_band[n] AND   # First low touches band
    low < lower_band AND         # Second low touches band
    low > low[n] AND             # Higher low
    RSI(14) > RSI(14)[n]         # RSI divergence
)
```

### Bandwidth Expansion/Contraction
```python
# Volatility state
EXPANDING = bandwidth > bandwidth[1] > bandwidth[2]
CONTRACTING = bandwidth < bandwidth[1] < bandwidth[2]

# Volatility regime
HIGH_VOL = bandwidth > percentile(bandwidth, 80, 252)
LOW_VOL = bandwidth < percentile(bandwidth, 20, 252)
```

---

## Trading Strategies

### 1. Mean Reversion (Range Markets)
```python
# Only in low ADX environment
RANGING = ADX(14) < 20

LONG = (
    RANGING AND
    close < lower_band AND
    close > close[1]  # Turning up
)
STOP = lowest(low, 5) - ATR(14)
TARGET = middle_band
```

### 2. Squeeze Breakout (Trend Markets)
```python
# After squeeze, follow breakout
LONG = (
    SQUEEZE[1:10].any() AND
    close > upper_band AND
    ADX(14) > 20 AND
    volume > avg_volume * 1.3
)
STOP = middle_band
TARGET = close + 2 * ATR(14)
```

### 3. Trend Continuation
```python
# Pullback to middle band in trend
UPTREND = close > SMA(50) AND SMA(20) > SMA(50)
PULLBACK = low < middle_band AND close > middle_band

LONG = UPTREND AND PULLBACK
STOP = lower_band
TARGET = upper_band
```

---

## Common Pitfalls

1. **Fading Strong Trends**: Price can walk bands for extended periods
2. **Squeeze False Breakouts**: Not all squeezes lead to big moves
3. **Assuming Normal Distribution**: Markets have fat tails
4. **Ignoring Volume**: Breakouts need volume confirmation
5. **Using Alone**: Best combined with trend filters

---

## Combining with Other Indicators

```python
# Bollinger + RSI
LONG = close < lower_band AND RSI(14) < 30 AND RSI_divergence

# Bollinger + ADX for strategy selection
if ADX(14) > 25:
    use_trend_following()  # Don't fade bands
else:
    use_mean_reversion()   # Fade band touches

# Bollinger + MACD
LONG = close < lower_band AND MACD_histogram > MACD_histogram[1]
```

---

## Implementation

```python
from src.analysis.technical.indicators import VolatilityIndicators

vol = VolatilityIndicators()

# Calculate Bollinger Bands
bb = vol.bollinger_bands(data['close'], period=20, std_dev=2.0)
upper = bb['upper']
middle = bb['middle']
lower = bb['lower']
percent_b = bb['percent_b']
bandwidth = bb['bandwidth']

# Squeeze detection
squeeze = bandwidth < bandwidth.rolling(126).quantile(0.1)
```

---

## Academic Notes

- **Bollinger (2002)**: "Bollinger on Bollinger Bands" - definitive reference
- **Leung & Chong (2003)**: Found positive risk-adjusted returns for BB strategies
- **Lento & Gradojevic (2007)**: BB outperformed buy-and-hold in some markets

**Key Insight**: Effectiveness depends heavily on market regime (trending vs ranging).
