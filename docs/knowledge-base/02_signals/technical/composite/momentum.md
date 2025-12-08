# Momentum Indicators

## Overview

Momentum indicators measure the rate of change in price, identifying acceleration or deceleration in trends. Unlike oscillators with fixed bounds, momentum can theoretically reach any value.

---

## Rate of Change (ROC)

### Formula
```
ROC = ((Close - Close[N]) / Close[N]) × 100

# Percentage change over N periods
```

### Standard Parameters
| Period | Use Case |
|--------|----------|
| 10 | Short-term momentum |
| 14 | Standard |
| 20 | Intermediate |
| 50 | Long-term trends |

### Rule Templates
```python
# Basic momentum
POSITIVE_MOMENTUM = ROC(14) > 0
NEGATIVE_MOMENTUM = ROC(14) < 0

# Momentum strength
STRONG_UP_MOMENTUM = ROC(14) > 5
STRONG_DOWN_MOMENTUM = ROC(14) < -5

# Momentum shift
MOMENTUM_TURNING_UP = ROC(14) > ROC(14)[1] AND ROC(14)[1] < 0
MOMENTUM_TURNING_DOWN = ROC(14) < ROC(14)[1] AND ROC(14)[1] > 0
```

---

## Momentum (MOM)

### Formula
```
MOM = Close - Close[N]

# Price change (not percentage)
```

### Rule Templates
```python
# Zero line crossover
BULLISH_CROSS = MOM(10) crosses_above 0
BEARISH_CROSS = MOM(10) crosses_below 0

# Momentum divergence
BULLISH_DIV = price.lower_low() AND MOM(10).higher_low()
BEARISH_DIV = price.higher_high() AND MOM(10).lower_high()
```

---

## Relative Momentum Index (RMI)

### Formula
```
RMI = RSI calculated on momentum values instead of price
```

### Advantage
Smoother than RSI, reduces whipsaws

---

## Trix

### Formula
```
TRIX = 1-period ROC of triple-smoothed EMA

EMA1 = EMA(Close, N)
EMA2 = EMA(EMA1, N)
EMA3 = EMA(EMA2, N)
TRIX = (EMA3 - EMA3[1]) / EMA3[1] × 100
```

### Parameters
| Period | Use Case |
|--------|----------|
| 12 | Short-term |
| 15 | Standard |
| 18 | Smoother |

### Rule Templates
```python
# Zero line signals
TRIX_BULLISH = TRIX(15) > 0
TRIX_BEARISH = TRIX(15) < 0

# Crossover with signal line
SIGNAL = EMA(TRIX, 9)
BUY = TRIX crosses_above SIGNAL
SELL = TRIX crosses_below SIGNAL
```

---

## Elder's Force Index

### Formula
```
Force Index = (Close - Close[1]) × Volume

EFI = EMA(Force Index, 13)
```

### Interpretation
- Combines price change and volume
- Positive = buying pressure
- Negative = selling pressure

### Rule Templates
```python
# Basic signals
BUYING_PRESSURE = EFI(13) > 0
SELLING_PRESSURE = EFI(13) < 0

# Strength
STRONG_BUYING = EFI(13) > EFI(13).rolling(20).mean() + 2 * EFI(13).rolling(20).std()

# Divergence (powerful signals)
BULLISH_DIV = price.lower_low() AND EFI(13).higher_low()
```

---

## Chande Momentum Oscillator (CMO)

### Formula
```
CMO = ((Sum of Up Days - Sum of Down Days) / (Sum of Up Days + Sum of Down Days)) × 100

Range: -100 to +100
```

### Key Levels
| Level | Interpretation |
|-------|----------------|
| > +50 | Overbought |
| < -50 | Oversold |
| 0 | Neutral |

### Rule Templates
```python
OVERBOUGHT = CMO(14) > 50
OVERSOLD = CMO(14) < -50

# Mean reversion
REVERSAL_LONG = CMO(14) < -50 AND CMO(14) > CMO(14)[1]
REVERSAL_SHORT = CMO(14) > 50 AND CMO(14) < CMO(14)[1]
```

---

## True Strength Index (TSI)

### Formula
```
Double-smoothed momentum:
TSI = 100 × EMA(EMA(PC, r), s) / EMA(EMA(|PC|, r), s)

Where:
- PC = Close - Close[1] (price change)
- r = 25 (first smoothing, standard)
- s = 13 (second smoothing, standard)
```

### Key Levels
| Level | Interpretation |
|-------|----------------|
| > +25 | Bullish |
| < -25 | Bearish |
| 0 | Neutral |

### Rule Templates
```python
# Trend direction
BULLISH = TSI(25, 13) > 0
BEARISH = TSI(25, 13) < 0

# Signal line crossover
SIGNAL = EMA(TSI, 7)
BUY = TSI crosses_above SIGNAL AND TSI < 25
SELL = TSI crosses_below SIGNAL AND TSI > -25
```

---

## Trading Strategies

### 1. Momentum Crossover
```python
LONG_ENTRY = (
    ROC(14) crosses_above 0 AND
    close > SMA(50)                  # Trend filter
)

SHORT_ENTRY = (
    ROC(14) crosses_below 0 AND
    close < SMA(50)
)
```

### 2. Momentum Divergence
```python
# More reliable than oscillator divergence
def detect_momentum_divergence(price, momentum, lookback=20):
    # Price makes lower low
    price_lower_low = price.min() < price[lookback:].min()

    # Momentum makes higher low
    mom_higher_low = momentum.min() > momentum[lookback:].min()

    return price_lower_low AND mom_higher_low
```

### 3. Momentum Ranking
```python
# Rank stocks by momentum for relative strength
def momentum_rank(stocks, period=12):
    returns = {}
    for stock in stocks:
        returns[stock] = ROC(stock, period)

    # Long top quintile, short bottom quintile
    ranked = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    return {
        'long': ranked[:len(ranked)//5],
        'short': ranked[-len(ranked)//5:]
    }
```

### 4. Dual Momentum
```python
# Absolute and relative momentum
ABSOLUTE_MOM = ROC(12) > 0           # Positive return
RELATIVE_MOM = ROC(12) > SPY_ROC(12) # Beat benchmark

# Only invest when both positive
INVEST = ABSOLUTE_MOM AND RELATIVE_MOM

# Otherwise hold cash/bonds
if NOT INVEST:
    hold_cash_or_bonds()
```

---

## Multi-Timeframe Momentum

```python
# Align momentum across timeframes
MOM_DAILY = ROC(14, daily) > 0
MOM_WEEKLY = ROC(14, weekly) > 0
MOM_MONTHLY = ROC(14, monthly) > 0

# Strong confluence
ALL_ALIGNED_BULL = MOM_DAILY AND MOM_WEEKLY AND MOM_MONTHLY
ALL_ALIGNED_BEAR = NOT MOM_DAILY AND NOT MOM_WEEKLY AND NOT MOM_MONTHLY

# Divergence warning
DIVERGENCE = MOM_DAILY AND NOT MOM_WEEKLY
```

---

## Momentum Factor in Quantitative Finance

```python
# Academic momentum factor
# Long winners, short losers based on past 12-1 month returns
# Skip most recent month (reversal effect)

def momentum_factor(prices):
    # 12-month return, skip last month
    mom_12_1 = (prices.shift(21) / prices.shift(252)) - 1

    # Rank stocks
    ranked = mom_12_1.rank(pct=True)

    # Long top 30%, short bottom 30%
    long_stocks = ranked > 0.70
    short_stocks = ranked < 0.30

    return long_stocks, short_stocks
```

---

## Common Pitfalls

1. **Momentum Crashes**: Momentum can reverse violently
2. **Crowded Trade**: Popular momentum strategies become crowded
3. **Transaction Costs**: High turnover in momentum strategies
4. **Mean Reversion**: After extreme momentum, expect pullback

---

## Implementation

```python
from src.analysis.technical.indicators import Oscillators

osc = Oscillators()

# Rate of Change
roc = osc.roc(data, period=14)

# Momentum (price difference)
mom = osc.momentum(data, period=10)

# CMO
cmo = osc.cmo(data, period=14)

# Signals
positive_mom = roc > 0
accelerating = roc > roc.shift(1)
decelerating = roc < roc.shift(1)
```

---

## Academic Notes

- **Jegadeesh & Titman (1993)**: Seminal momentum factor paper
- **Carhart (1997)**: Added momentum to Fama-French factors
- **Key Insight**: Momentum is one of the most robust anomalies in finance

**Best Practice**: Use momentum for stock selection (relative strength) and trend confirmation, not as a standalone timing signal.
