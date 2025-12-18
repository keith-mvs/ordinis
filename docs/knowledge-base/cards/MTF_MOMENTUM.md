# Multi-Timeframe Momentum Strategy

---

**Title:** Multi-Timeframe Momentum
**Description:** Combines long-term momentum ranking with short-term stochastic timing
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** momentum, stochastic, multi-timeframe, cross-sectional, timing
**References:** Jegadeesh & Titman (1993), Asness et al. (2013)

---

## Overview

The MTF Momentum strategy combines the well-documented 12-1 month momentum anomaly with intraday stochastic oscillator timing. This dual-timeframe approach captures momentum winners while improving entry prices through pullback detection.

## Mathematical Basis

### 12-1 Month Momentum

Classic momentum factor (skip most recent month to avoid reversal):

$$
\text{Mom}_{12-1} = \frac{P_t}{P_{t-252}} - 1 - \frac{P_t}{P_{t-21}} + 1 = \frac{P_{t-21}}{P_{t-252}} - 1
$$

Simplified:

$$
\text{Mom}_{12-1} = \text{Return}_{t-252 \to t-21}
$$

### Stochastic Oscillator

```
%K = 100 × (Close - Low_n) / (High_n - Low_n)
%D = SMA(%K, 3)
```

Where $n$ = lookback period (default 14).

### Crossover Detection

```
bullish_cross = %K > %D AND %K_prev <= %D_prev
bearish_cross = %K < %D AND %K_prev >= %D_prev
```

## Signal Logic

### Entry Conditions

| Momentum Rank | Stochastic | Signal |
|---------------|------------|--------|
| Top 20% | Bullish cross + %K < 30 | **STRONG LONG** |
| Top 20% | Bullish cross + %K < 50 | **LONG** |
| Bottom 20% | Bearish cross + %K > 70 | **STRONG SHORT** |
| Bottom 20% | Bearish cross + %K > 50 | **SHORT** |
| Middle 60% | Any | **HOLD** |

### Exit Conditions

| Position | Exit Trigger |
|----------|--------------|
| Long | %K > 80 OR stop/TP hit |
| Short | %K < 20 OR stop/TP hit |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `momentum_period` | 252 | Total momentum lookback |
| `skip_period` | 21 | Recent period to skip |
| `stoch_k_period` | 14 | Stochastic %K period |
| `stoch_d_period` | 3 | Stochastic %D smoothing |
| `stoch_oversold` | 30 | Oversold threshold |
| `stoch_overbought` | 70 | Overbought threshold |
| `top_pct` | 0.2 | Top momentum percentile |
| `atr_stop_mult` | 1.5 | ATR stop multiplier |
| `atr_tp_mult` | 2.5 | ATR take-profit multiplier |

## Edge Source

1. **Momentum Anomaly:** 12-1 month returns predict future returns (extensively documented)
2. **Pullback Entry:** Stochastic timing captures mean-reversion within the trend
3. **Reduced Slippage:** Entry on pullbacks improves fill prices
4. **Confirmation:** Two timeframes agreeing reduces false signals

## Universe Ranking

```python
# Cross-sectional momentum ranking
universe_df["momentum"] = universe_df.groupby("date")["return_12_1"].rank(pct=True)

# Long candidates
longs = universe_df[universe_df["momentum"] > 0.8]

# Short candidates
shorts = universe_df[universe_df["momentum"] < 0.2]
```

## Implementation Notes

```python
from ordinis.engines.signalcore.models import MTFMomentumModel
from ordinis.engines.signalcore.core.model import ModelConfig

config = ModelConfig(
    model_id="mtf_momentum",
    model_type="momentum",
    parameters={
        "momentum_period": 252,
        "stoch_oversold": 30,
        "top_pct": 0.2,
    }
)

model = MTFMomentumModel(config)
signal = await model.generate(symbol, df, timestamp)

# For universe ranking
ranks = model.rank_universe(universe_prices_df)
```

## Risk Considerations

- **Momentum Crashes:** Momentum can reverse sharply (2009, 2020)
- **Crowding:** Popular strategy leads to crowded trades
- **Sector Concentration:** Momentum portfolios often sector-concentrated

## Performance Expectations

- **Win Rate:** 50-60%
- **Profit Factor:** 1.4-2.0
- **Best Conditions:** Trending markets with clear winners/losers
- **Worst Conditions:** Choppy markets, momentum reversals

## Recommended Symbols

High-volume, liquid stocks with clear momentum characteristics:

```
AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AMD, COIN, DKNG
```

---

**File:** `src/ordinis/engines/signalcore/models/mtf_momentum.py`
**Status:** ✅ Complete
