# MACD Strategy

---

**Title:** MACD Momentum with Histogram Analysis
**Description:** Trades MACD crossovers with histogram-based risk sizing
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** momentum, macd, oscillator, histogram, crossover
**References:** Gerald Appel (1979), Technical Analysis: Power Tools for Active Investors

---

## Overview

The MACD (Moving Average Convergence/Divergence) strategy identifies momentum shifts using the difference between fast and slow EMAs. The signal line crossover provides entry/exit timing, while the histogram magnitude informs position sizing. This implementation includes divergence detection for higher-probability setups.

## Mathematical Basis

### MACD Line

Difference between fast and slow EMAs:

$$
\text{MACD}_t = \text{EMA}(P, n_{\text{fast}}) - \text{EMA}(P, n_{\text{slow}})
$$

### Signal Line

EMA of the MACD line:

$$
\text{Signal}_t = \text{EMA}(\text{MACD}, n_{\text{signal}})
$$

### Histogram

Momentum acceleration/deceleration:

$$
\text{Histogram}_t = \text{MACD}_t - \text{Signal}_t
$$

### Divergence Detection

Bullish divergence: Price makes lower low, MACD makes higher low
$$
\text{BullDiv} = (P_t < P_{t-n}) \land (\text{MACD}_t > \text{MACD}_{t-n})
$$

## Signal Logic

| Condition | Action |
|-----------|--------|
| MACD crosses above Signal Line | **LONG** |
| MACD crosses below Signal Line | **SHORT** |
| Histogram shrinking toward zero | **REDUCE** size |
| Opposite crossover | **EXIT/REVERSE** |

### Position Sizing (Histogram-Based)

$$
\text{Size} = \text{BaseSize} \times \min(1.5, \frac{|\text{Histogram}|}{\text{ATR}_{14}})
$$

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fast_period` | 12 | Fast EMA period |
| `slow_period` | 26 | Slow EMA period |
| `signal_period` | 9 | Signal line EMA period |
| `min_bars` | 50 | Minimum data required |
| `histogram_sizing` | True | Scale size by histogram |
| `use_divergence` | False | Enable divergence signals |

## Edge Source

1. **Momentum Persistence:** Trends accelerate before reversing
2. **Crossover Lag:** Delayed signals filter noise
3. **Histogram Momentum:** Rate of change shows conviction
4. **Divergence Warning:** Price/oscillator divergence precedes reversals

## Risk Considerations

- **Lagging Indicator:** MACD reacts slowly to price changes
- **Whipsaws:** Frequent crossovers in choppy markets
- **Zero-Line Dependence:** Signal strength varies by MACD position
- **False Divergence:** Not all divergences lead to reversals

## Implementation Notes

```python
# GPU-accelerated implementation
fast_ema = gpu_engine.compute_ema(close, fast_period)
slow_ema = gpu_engine.compute_ema(close, slow_period)
macd_line = fast_ema - slow_ema
signal_line = gpu_engine.compute_ema(macd_line, signal_period)
histogram = macd_line - signal_line

# Signal detection
cross_up = (macd_line[i-1] < signal_line[i-1]) and (macd_line[i] > signal_line[i])
cross_down = (macd_line[i-1] > signal_line[i-1]) and (macd_line[i] < signal_line[i])

# Histogram-based sizing
atr = gpu_engine.compute_atr(high, low, close, 14)
size_mult = min(1.5, abs(histogram[i]) / atr[i])
```

## Performance Expectations

- **Win Rate:** 40-50%
- **Profit Factor:** 1.3-1.8
- **Avg Holding Period:** 10-30 bars
- **Best Conditions:** Trending markets with clear momentum
- **Worst Conditions:** Ranging markets, low volatility

## Regime Suitability

| Regime | Suitability | Notes |
|--------|-------------|-------|
| BULL | ✅ High | Strong trend confirmation |
| BEAR | ⚠️ Medium | Useful for downtrend confirmation |
| SIDEWAYS | ❌ Low | Frequent whipsaws |
| VOLATILE | ⚠️ Medium | Histogram sizing helps |
| TRANSITIONAL | ⚠️ Medium | Good for regime detection |

## Advanced: Zero-Line Analysis

MACD position relative to zero provides context:

| MACD Position | Signal | Interpretation |
|---------------|--------|----------------|
| Above zero, rising | Strong LONG | Uptrend accelerating |
| Above zero, falling | Weak LONG | Uptrend decelerating |
| Below zero, falling | Strong SHORT | Downtrend accelerating |
| Below zero, rising | Weak SHORT | Downtrend decelerating |

## Divergence Trading

When enabled, divergence signals take priority:

```python
# Bullish divergence: price lower low, MACD higher low
if (low[i] < low[i-lookback]) and (macd[i] > macd[i-lookback]):
    signal_type = SignalType.ENTRY
    direction = Direction.LONG
    probability *= 1.3  # Higher confidence
```

---

**File:** `src/ordinis/application/strategies/macd.py`
**Model:** `MACDModel`
**Status:** ✅ Production Ready
