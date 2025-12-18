# Moving Average Crossover Strategy

---

**Title:** Golden/Death Cross with Configurable MA Types
**Description:** Classic dual-moving-average crossover system with SMA/EMA selection
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** trend-following, moving-average, crossover, golden-cross, death-cross
**References:** Dow Theory, Technical Analysis of Financial Markets (Murphy)

---

## Overview

The Moving Average Crossover strategy implements the classic "golden cross" (bullish) and "death cross" (bearish) signals using configurable fast and slow moving averages. The strategy supports both Simple Moving Averages (SMA) and Exponential Moving Averages (EMA), with optional trend strength filtering via lookback analysis.

## Mathematical Basis

### Simple Moving Average (SMA)

$$
\text{SMA}_t(n) = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}
$$

### Exponential Moving Average (EMA)

$$
\text{EMA}_t = \alpha \cdot P_t + (1 - \alpha) \cdot \text{EMA}_{t-1}
$$

Where smoothing factor $\alpha = \frac{2}{n + 1}$

### Crossover Detection

$$
\text{CrossUp}_t = (\text{Fast}_{t-1} < \text{Slow}_{t-1}) \land (\text{Fast}_t > \text{Slow}_t)
$$

$$
\text{CrossDown}_t = (\text{Fast}_{t-1} > \text{Slow}_{t-1}) \land (\text{Fast}_t < \text{Slow}_t)
$$

### Signal Strength

Separation between MAs normalized by ATR:

$$
\text{Strength} = \frac{|\text{Fast}_t - \text{Slow}_t|}{\text{ATR}_{14}}
$$

## Signal Logic

| Condition | Action |
|-----------|--------|
| Fast MA crosses above Slow MA | **LONG** (Golden Cross) |
| Fast MA crosses below Slow MA | **SHORT** (Death Cross) |
| Opposite crossover occurs | **EXIT** current position |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_period` | 50 | Fast moving average period |
| `slow_period` | 200 | Slow moving average period |
| `ma_type` | "sma" | Moving average type ("sma" or "ema") |
| `min_bars` | 250 | Minimum data required |
| `require_separation` | False | Require minimum MA separation |
| `min_separation` | 0.01 | Minimum % separation (1%) |

## Edge Source

1. **Trend Persistence:** Markets exhibit momentum; established trends tend to continue
2. **Institutional Lag:** Large players rebalance slowly, creating persistent moves
3. **Behavioral Anchoring:** Round MA numbers attract attention (50, 100, 200)

## Risk Considerations

- **Whipsaw Risk:** Frequent false signals in choppy/sideways markets
- **Lag:** Long MAs react slowly; significant drawdown before exit signal
- **Gap Risk:** Overnight gaps can exceed expected move
- **Crowded Trade:** Popular strategy means compressed alpha

## Implementation Notes

```python
# GPU-accelerated implementation
fast_ma = gpu_engine.compute_ma(close, fast_period, ma_type)
slow_ma = gpu_engine.compute_ma(close, slow_period, ma_type)

# Entry conditions
cross_up = (fast_ma[i-1] < slow_ma[i-1]) and (fast_ma[i] > slow_ma[i])
cross_down = (fast_ma[i-1] > slow_ma[i-1]) and (fast_ma[i] < slow_ma[i])

# Signal strength scaling
separation = abs(fast_ma[i] - slow_ma[i]) / atr[i]
score = min(100, separation * 50)  # Normalize to 0-100
```

## Performance Expectations

- **Win Rate:** 35-45%
- **Profit Factor:** 1.3-2.0
- **Avg Holding Period:** 20-50 bars (trend-dependent)
- **Best Conditions:** Strong trending markets (ADX > 25)
- **Worst Conditions:** Range-bound, low-volatility markets

## Regime Suitability

| Regime | Suitability | Notes |
|--------|-------------|-------|
| BULL | ✅ High | Classic trend-following edge |
| BEAR | ⚠️ Medium | Works but requires short capability |
| SIDEWAYS | ❌ Low | Frequent whipsaws, avoid |
| VOLATILE | ⚠️ Medium | Widen stops, reduce size |
| TRANSITIONAL | ❌ Low | Wait for regime confirmation |

---

**File:** `src/ordinis/application/strategies/moving_average_crossover.py`
**Model:** `MovingAverageCrossoverModel`
**Status:** ✅ Production Ready
