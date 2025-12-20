# Moving Averages

## Overview

Moving averages smooth price data to identify trend direction and provide dynamic support/resistance levels. They are the foundation of many trading systems.

---

## Types of Moving Averages

### Simple Moving Average (SMA)

**Formula:**
```
SMA(n) = Sum(Close, n) / n
```

**Characteristics:**
- Equal weight to all periods
- Most lagging of all MAs
- Smoother, less reactive

**Best For:** Long-term trend identification, major S/R levels

---

### Exponential Moving Average (EMA)

**Formula:**
```
k = 2 / (n + 1)
EMA = Close × k + EMA_prev × (1 - k)
```

**Characteristics:**
- Higher weight to recent prices
- More responsive than SMA
- Industry standard for most strategies

**Best For:** Medium-term trading, crossover systems

---

### Weighted Moving Average (WMA)

**Formula:**
```
WMA = Sum(Price × Weight) / Sum(Weights)
Weights: 1, 2, 3, ..., n (linear)
```

**Characteristics:**
- Linear weighting toward recent
- Between SMA and EMA responsiveness
- Less common in practice

---

### Hull Moving Average (HMA)

**Formula:**
```
HMA = WMA(2 × WMA(n/2) - WMA(n), sqrt(n))
```

**Characteristics:**
- Significantly reduced lag
- Very responsive to price changes
- Can overshoot in choppy markets

**Best For:** Short-term trading, quick trend detection

---

### Kaufman Adaptive Moving Average (KAMA)

**Formula:**
```
ER = Direction / Volatility
SC = (ER × (fast_sc - slow_sc) + slow_sc)^2
KAMA = KAMA_prev + SC × (Price - KAMA_prev)
```

**Characteristics:**
- Self-adjusting to market conditions
- Fast in trends, slow in ranges
- Reduces whipsaws

**Best For:** Adapting to changing market conditions

---

### Volume Weighted Average Price (VWAP)

**Formula:**
```
VWAP = Cumulative(Price × Volume) / Cumulative(Volume)
```

**Characteristics:**
- Resets each trading day
- Institutional execution benchmark
- Shows fair value based on volume

**Best For:** Intraday trading, execution quality

---

## Standard Parameters

| Timeframe | Short | Medium | Long |
|-----------|-------|--------|------|
| Intraday | 9, 10 | 20, 21 | 50 |
| Swing | 20 | 50 | 100, 200 |
| Position | 50 | 100 | 200 |

---

## Rule Templates

### Trend Direction
```python
# Simple trend
UPTREND = close > EMA(50)
DOWNTREND = close < EMA(50)

# Multi-MA trend
STRONG_UPTREND = EMA(20) > EMA(50) > EMA(200) AND close > EMA(20)
STRONG_DOWNTREND = EMA(20) < EMA(50) < EMA(200) AND close < EMA(20)
```

### MA Crossovers
```python
# Classic crossovers
GOLDEN_CROSS = EMA(50) crosses_above EMA(200)
DEATH_CROSS = EMA(50) crosses_below EMA(200)

# Fast crossover for entries
BUY_SIGNAL = EMA(9) crosses_above EMA(21) AND close > EMA(50)
SELL_SIGNAL = EMA(9) crosses_below EMA(21) AND close < EMA(50)
```

### Pullback Entries
```python
# Buy the dip in uptrend
UPTREND = close > EMA(50) AND EMA(20) > EMA(50)
PULLBACK = low < EMA(20) AND close > EMA(20)
PULLBACK_BUY = UPTREND AND PULLBACK AND RSI(14) < 50

# Sell the rally in downtrend
DOWNTREND = close < EMA(50) AND EMA(20) < EMA(50)
RALLY = high > EMA(20) AND close < EMA(20)
RALLY_SHORT = DOWNTREND AND RALLY AND RSI(14) > 50
```

### MA Slope (Trend Strength)
```python
# Rising MAs
STRONG_TREND = EMA(20) > EMA(20)[5] AND EMA(50) > EMA(50)[5]

# Slope calculation
MA_SLOPE = (EMA(20) - EMA(20)[5]) / EMA(20)[5] * 100
STEEP_UPTREND = MA_SLOPE > 2  # 2% rise in 5 bars
```

### Dynamic Support/Resistance
```python
# MA as support in uptrend
SUPPORT_TEST = low < EMA(20) AND close > EMA(20) AND close > EMA(50)
BUY_AT_SUPPORT = SUPPORT_TEST AND bullish_candle

# MA as resistance in downtrend
RESISTANCE_TEST = high > EMA(20) AND close < EMA(20) AND close < EMA(50)
SELL_AT_RESISTANCE = RESISTANCE_TEST AND bearish_candle
```

---

## Common Pitfalls

1. **Whipsaws in Ranges**: MAs give false signals in choppy markets
2. **Lag Problem**: Late entries/exits in fast moves
3. **Parameter Optimization**: Overfitting to historical data
4. **Ignoring Context**: Crossovers fail in mean-reverting regimes

---

## Combining MAs with Other Indicators

```python
# MA + RSI
BUY = EMA(9) crosses_above EMA(21) AND RSI(14) < 70 AND RSI(14) > 30

# MA + Volume
BREAKOUT_BUY = close > EMA(50) AND close[1] < EMA(50) AND volume > avg_volume * 1.5

# MA + ADX
TREND_TRADE = close > EMA(50) AND ADX(14) > 25
```

---

## Implementation

```python
from ordinis.analysis.technical.indicators import MovingAverages

ma = MovingAverages()

# Calculate all MA types
sma_20 = ma.sma(data['close'], period=20)
ema_50 = ma.ema(data['close'], period=50)
wma_20 = ma.wma(data['close'], period=20)
hull_20 = ma.hull_ma(data['close'], period=20)
kama_20 = ma.kama(data['close'], period=20)
vwap = ma.vwap(data)
```

---

## Academic Notes

- **Bessembinder & Chan (1998)**: Mixed evidence on MA profitability after costs
- **Brock, Lakonishok & LeBaron (1992)**: Found positive returns for MA rules on DJIA
- **Sullivan, Timmermann & White (1999)**: Data-snooping concerns with TA rules

**Conclusion**: MAs work better as **filters** (trend confirmation) than primary signals.
