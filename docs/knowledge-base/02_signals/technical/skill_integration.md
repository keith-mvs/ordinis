# Technical Analysis - Skill Integration

**Section**: 02_signals/technical
**Last Updated**: 2025-12-12
**Source Skill**: [technical-analysis](../../../../.claude/skills/technical-analysis/SKILL.md)

---

## Overview

This document provides integration between the technical analysis knowledge base section and the `technical-analysis` Claude skill. The skill provides interactive analysis capabilities while this KB section provides foundational reference material.

---

## Skill Capabilities

The `technical-analysis` skill provides:

- **12 Core Indicators**: ADX, ATR, Bollinger Bands, CCI, Fibonacci, Ichimoku, MA, MACD, OBV, Parabolic SAR, RSI, Stochastic
- **Production Python Code**: Ready-to-use calculator implementations
- **Interpretation Framework**: Signal generation and analysis guidelines
- **Case Studies**: Real-world application examples

---

## Code Library

### Calculator Script

**Location**: `code/technical/calculate_indicators.py`

```bash
# Calculate RSI
python code/technical/calculate_indicators.py --symbol SPY --indicator RSI --period 14

# Calculate MACD
python code/technical/calculate_indicators.py --symbol QQQ --indicator MACD

# Calculate Bollinger Bands
python code/technical/calculate_indicators.py --symbol AAPL --indicator BOLLINGER --period 20 --std-dev 2.0
```

### Available Indicators

| Indicator | Command | Default Period |
|-----------|---------|----------------|
| RSI | `--indicator RSI` | 14 |
| MACD | `--indicator MACD` | 12/26/9 |
| Bollinger Bands | `--indicator BOLLINGER` | 20 |
| ATR | `--indicator ATR` | 14 |
| ADX | `--indicator ADX` | 14 |
| Stochastic | `--indicator STOCHASTIC` | 14/3 |
| CCI | `--indicator CCI` | 20 |
| OBV | `--indicator OBV` | N/A |
| Moving Averages | `--indicator MA` | 20/50/200 |
| Parabolic SAR | `--indicator PSAR` | 0.02/0.20 |
| Fibonacci | `--indicator FIBONACCI` | 14 |

---

## Indicator Class Reference

```python
from calculate_indicators import TechnicalIndicators

calc = TechnicalIndicators()

# RSI
rsi = calc.calculate_rsi(close_prices, period=14)

# MACD
macd_df = calc.calculate_macd(close_prices, fast=12, slow=26, signal=9)
# Returns: DataFrame with 'MACD', 'Signal', 'Histogram' columns

# Bollinger Bands
bb_df = calc.calculate_bollinger_bands(close_prices, period=20, std_dev=2.0)
# Returns: DataFrame with 'Upper', 'Middle', 'Lower', '%B', 'Bandwidth' columns

# ATR
atr = calc.calculate_atr(high, low, close, period=14)

# ADX
adx_df = calc.calculate_adx(high, low, close, period=14)
# Returns: DataFrame with 'ADX', '+DI', '-DI' columns

# Stochastic
stoch_df = calc.calculate_stochastic(high, low, close, k_period=14, d_period=3)
# Returns: DataFrame with '%K', '%D' columns

# CCI
cci = calc.calculate_cci(high, low, close, period=20)

# OBV
obv = calc.calculate_obv(close_prices, volume)

# Moving Averages
ma_df = calc.calculate_moving_averages(close_prices, periods=[20, 50, 200])
# Returns: DataFrame with SMA and EMA for each period

# Parabolic SAR
psar = calc.calculate_parabolic_sar(high, low, af_start=0.02, af_max=0.20)

# Fibonacci Retracement
fib_levels = calc.calculate_fibonacci_retracement(high=150.0, low=100.0, trend_direction='up')
# Returns: Dict with '0.0%', '23.6%', '38.2%', '50.0%', '61.8%', '78.6%', '100.0%' levels
```

---

## Indicator Selection Workflow

### Step 1: Define Analytical Objective

| Objective | Recommended Indicators |
|-----------|----------------------|
| Trend identification | ADX, MA, Parabolic SAR |
| Momentum assessment | RSI, Stochastic, MACD, CCI |
| Volatility measurement | ATR, Bollinger Bands |
| Volume confirmation | OBV |
| Support/resistance | Fibonacci, MA |

### Step 2: Match Time Frame

| Time Frame | Primary Indicators |
|------------|-------------------|
| Intraday | Stochastic, RSI, Parabolic SAR |
| Swing (days-weeks) | MACD, Bollinger Bands, CCI |
| Position (weeks-months) | ADX, MA crossovers |

### Step 3: Multi-Indicator Confirmation

```python
# Strong bullish setup
bullish_signal = (
    macd_df['MACD'].iloc[-1] > macd_df['Signal'].iloc[-1] and  # MACD crossover
    rsi.iloc[-1] > 50 and rsi.iloc[-1] < 70 and                 # RSI bullish but not overbought
    adx_df['ADX'].iloc[-1] > 25 and                             # Strong trend
    adx_df['+DI'].iloc[-1] > adx_df['-DI'].iloc[-1]             # Uptrend confirmed
)

# Strong bearish setup
bearish_signal = (
    macd_df['MACD'].iloc[-1] < macd_df['Signal'].iloc[-1] and
    rsi.iloc[-1] < 50 and rsi.iloc[-1] > 30 and
    adx_df['ADX'].iloc[-1] > 25 and
    adx_df['-DI'].iloc[-1] > adx_df['+DI'].iloc[-1]
)
```

---

## Interpretation Guidelines

### Trend Strength Assessment

```python
adx_value = adx_df['ADX'].iloc[-1]

if adx_value > 25:
    regime = "TRENDING"
    # Use trend-following strategies
    # Trail stops with Parabolic SAR or ATR multiples
elif adx_value < 20:
    regime = "RANGING"
    # Use mean-reversion strategies
    # Apply Bollinger Bands for overbought/oversold
else:
    regime = "TRANSITIONING"
    # Reduce position size
    # Wait for clearer signals
```

### Divergence Detection

```python
def detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 20) -> str:
    """
    Detect bullish or bearish divergence.
    """
    # Price comparison
    price_higher_high = price.iloc[-1] > price.iloc[-lookback:-1].max()
    price_lower_low = price.iloc[-1] < price.iloc[-lookback:-1].min()

    # Indicator comparison
    ind_higher_high = indicator.iloc[-1] > indicator.iloc[-lookback:-1].max()
    ind_lower_low = indicator.iloc[-1] < indicator.iloc[-lookback:-1].min()

    if price_higher_high and not ind_higher_high:
        return "BEARISH_DIVERGENCE"  # Potential reversal down
    elif price_lower_low and not ind_lower_low:
        return "BULLISH_DIVERGENCE"  # Potential reversal up
    else:
        return "NO_DIVERGENCE"
```

### Volatility Regime Adaptation

```python
atr_current = atr.iloc[-1]
atr_average = atr.rolling(20).mean().iloc[-1]

if atr_current > atr_average * 1.5:
    volatility_regime = "HIGH"
    # Widen stop losses
    # Reduce position size
    # Expect larger price swings
elif atr_current < atr_average * 0.5:
    volatility_regime = "LOW"
    # Anticipate volatility expansion
    # Tighten profit targets
    # Consider breakout strategies
else:
    volatility_regime = "NORMAL"
```

---

## Skill Cross-References

### Detailed Reference Materials

For comprehensive methodology and formulas:

- [MOMENTUM_INDICATORS.md](../../../../.claude/skills/technical-analysis/references/MOMENTUM_INDICATORS.md) - RSI, MACD, CCI, Stochastic
- [TREND_INDICATORS.md](../../../../.claude/skills/technical-analysis/references/TREND_INDICATORS.md) - ADX, Ichimoku, MA, Parabolic SAR
- [VOLATILITY_VOLUME.md](../../../../.claude/skills/technical-analysis/references/VOLATILITY_VOLUME.md) - ATR, Bollinger Bands, OBV
- [FIBONACCI.md](../../../../.claude/skills/technical-analysis/references/FIBONACCI.md) - Retracement and extension levels
- [CASE_STUDIES.md](../../../../.claude/skills/technical-analysis/references/CASE_STUDIES.md) - Real-world applications

### KB Section Mapping

| Skill Reference | KB Section |
|-----------------|------------|
| TREND_INDICATORS.md | [trend_indicators/](trend_indicators/) |
| MOMENTUM_INDICATORS.md | [oscillators/](oscillators/), [composite/](composite/) |
| VOLATILITY_VOLUME.md | [volatility/](volatility/), [overlays/bollinger_bands.md](overlays/bollinger_bands.md) |
| FIBONACCI.md | [patterns/support_resistance.md](patterns/support_resistance.md) |

---

## Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
ta>=0.10.0  # Optional, for ta library integration
yfinance>=0.2.0  # For data fetching
```

---

## Academic References

1. **CMT Association**: Chartered Market Technician curriculum
2. **Edwards & Magee**: Technical Analysis of Stock Trends (11th Edition)
3. **Murphy, John J.**: Technical Analysis of the Financial Markets
4. **Pring, Martin J.**: Technical Analysis Explained (5th Edition)
5. **Aronson, D.R.**: Evidence-Based Technical Analysis

---

**Template**: KB Skills Integration v1.0
