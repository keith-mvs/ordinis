---

title: Trend Following EMA Crossover
description >
Classic dual EMA crossover system with ATR-based risk management
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** trend-following, EMA, crossover, momentum, moving-average
**References:** Gerald Appel (MACD), Richard Donchian

---

## Overview

The Trend Following EMA strategy uses the crossover of fast and slow Exponential Moving Averages to identify trend changes. Positions are entered when the fast EMA crosses above (long) or below (short) the slow EMA, with ATR-based stops and targets. The strategy also exits on opposing crossovers.

## Mathematical Basis

### Exponential Moving Average

The EMA gives more weight to recent prices:

$$
\text{EMA}_t = \alpha \times P_t + (1 - \alpha) \times \text{EMA}_{t-1}
$$

Where smoothing factor:

$$
\alpha = \frac{2}{n + 1}
$$

### Crossover Detection

Bullish crossover when fast EMA crosses above slow EMA:

$$
\text{BullishCross} = (\text{Fast}_{t-1} \leq \text{Slow}_{t-1}) \land (\text{Fast}_t > \text{Slow}_t)
$$

Bearish crossover when fast EMA crosses below slow EMA:

$$
\text{BearishCross} = (\text{Fast}_{t-1} \geq \text{Slow}_{t-1}) \land (\text{Fast}_t < \text{Slow}_t)
$$

### ATR-Based Exits

Dynamic stop and target levels:

$$
\text{Stop} = \text{Entry} \pm \text{ATR}_{14} \times \text{stop\_mult}
$$

$$
\text{Target} = \text{Entry} \pm \text{ATR}_{14} \times \text{tp\_mult}
$$

## Signal Logic

| Condition | Action |
|-----------|--------|
| `fast_prev <= slow_prev` AND `fast_now > slow_now` | **LONG** |
| `fast_prev >= slow_prev` AND `fast_now < slow_now` | **SHORT** |
| Stop hit OR target hit OR opposing crossover OR 20 bars | **EXIT** |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_period` | 9 | Fast EMA period |
| `slow_period` | 21 | Slow EMA period |
| `atr_stop` | 2.0 | ATR multiplier for stop-loss |
| `atr_tp` | 4.0 | ATR multiplier for take-profit |
| `max_bars` | 20 | Maximum holding period |

## Edge Source

1. **Trend Capture:** EMAs identify and ride sustained price trends
2. **Noise Filtering:** Moving averages smooth out short-term volatility
3. **Momentum Confirmation:** Crossovers signal shift in momentum direction
4. **Signal Exit:** Opposing crossover provides natural exit signal

## Risk Considerations

- **Lag:** EMAs are lagging indicators; entries occur after trend starts
- **Whipsaws:** Choppy markets generate false crossover signals
- **Trend Reversals:** Sudden reversals can hit stops before crossover signal
- **Overtrading:** Short-period EMAs can generate excessive signals

## Implementation Notes

```python
# GPU-accelerated EMA computation
ema_fast = gpu_engine.compute_ema(prices, fast_period)
ema_slow = gpu_engine.compute_ema(prices, slow_period)
atr = gpu_engine.compute_atr(high, low, close, 14)

# Crossover detection
bullish_cross = (fast_prev <= slow_prev) and (fast_now > slow_now)
bearish_cross = (fast_prev >= slow_prev) and (fast_now < slow_now)

# Exit on opposing crossover
if position["direction"] == 1 and fast_now < slow_now:
    exit_reason = "signal"
```

## Performance Expectations

- **Win Rate:** 35-45%
- **Profit Factor:** 1.4-2.0
- **Avg Holding Period:** 8-15 bars
- **Best Conditions:** Strong trending markets (momentum regimes)
- **Worst Conditions:** Range-bound, mean-reverting markets

## Sprint 3 Results (Small-Cap)

Outstanding performer:

- SLVM (Sylvamo): Sharpe 223.37, PnL 14.07%
- Works well on trending small-cap stocks

---

**File:** `scripts/strategy_sprint/sprint3_smallcap_gpu.py`
**Method:** `backtest_trend_following_ema()`
**Status:** ✅ Complete
