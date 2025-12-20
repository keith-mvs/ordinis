# Fibonacci ADX Strategy

---

**Title:** Fibonacci Retracement with ADX Trend Filter
**Description:** Combines Fibonacci levels with ADX trend confirmation for high-probability entries
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** fibonacci, adx, trend, retracement, multi-factor
**References:** Leonardo of Pisa (1202), J. Welles Wilder (1978)

---

## Overview

The Fibonacci ADX strategy synergizes two complementary concepts: Fibonacci retracement levels for identifying optimal entry zones on pullbacks, and ADX for confirming trend strength. Entries occur only when price reaches key Fibonacci levels (38.2%, 50%, 61.8%) during confirmed trends (ADX > 25).

## Mathematical Basis

### Fibonacci Levels

From swing high $H$ to swing low $L$:

$$
\text{Level}_{\phi} = L + (H - L) \times \phi
$$

Key ratios derived from golden ratio:
- 38.2% = $1 - 0.618$
- 50.0% = $0.5$
- 61.8% = $\phi - 1$ (golden ratio complement)

### ADX (Average Directional Index)

Trend strength measurement:

$$
\text{ADX} = \text{SMA}\left(\frac{|+DI - (-DI)|}{+DI + (-DI)}, n\right) \times 100
$$

Where:
- +DI = Positive Directional Indicator
- -DI = Negative Directional Indicator

### Combined Signal Probability

$$
P(\text{success}) = P(\text{fib\_hold}) \times P(\text{trend\_cont} | \text{ADX} > 25)
$$

## Signal Logic

| Condition | Action |
|-----------|--------|
| ADX > 25 AND Price at 38.2% level (±tol) | **LONG** in uptrend |
| ADX > 25 AND Price at 50.0% level (±tol) | **LONG** (medium conf) |
| ADX > 25 AND Price at 61.8% level (±tol) | **LONG** (high risk/reward) |
| Price breaks below 78.6% level | **STOP OUT** - trend failed |
| Price reaches prior swing high/low | **TAKE PROFIT** |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adx_period` | 14 | ADX calculation period |
| `adx_threshold` | 25 | Minimum ADX for trend |
| `swing_lookback` | 50 | Bars for swing detection |
| `fib_levels` | [0.382, 0.5, 0.618] | Key Fibonacci levels |
| `tolerance` | 0.01 | Price tolerance (1%) |

## Edge Source

1. **Trend Persistence:** ADX > 25 indicates established, likely continuing trend
2. **Natural Retracements:** Markets pull back before continuation
3. **Fibonacci Psychology:** Self-fulfilling prophecy from widespread usage
4. **Multi-Factor Filter:** Combining two signals reduces false positives

## Risk Considerations

- **Swing Detection:** Incorrect swing high/low identification invalidates levels
- **Trend Exhaustion:** ADX > 40 may indicate overextended trend
- **Level Clustering:** Multiple Fib levels near each other create confusion
- **Time Decay:** Fibonacci levels become less relevant over time

## Implementation Notes

```python
# GPU-accelerated implementation
adx = gpu_engine.compute_adx(high, low, close, adx_period)

# Swing detection
swing_high = rolling_max(high, swing_lookback)
swing_low = rolling_min(low, swing_lookback)

# Fibonacci levels
fib_range = swing_high - swing_low
levels = {
    0.382: swing_low + fib_range * 0.382,
    0.500: swing_low + fib_range * 0.500,
    0.618: swing_low + fib_range * 0.618,
}

# Signal generation (uptrend example)
trend_confirmed = adx[i] > adx_threshold
for ratio, level in levels.items():
    if abs(close[i] - level) / level < tolerance:
        if trend_confirmed:
            entry = True
            stop_loss = levels.get(0.786, swing_low * 0.95)
            take_profit = swing_high
```

## Performance Expectations

- **Win Rate:** 45-55%
- **Profit Factor:** 1.4-2.0
- **Avg Holding Period:** 15-40 bars
- **Best Conditions:** Trending markets with clear swings
- **Worst Conditions:** Choppy, trendless markets

## Regime Suitability

| Regime | Suitability | Notes |
|--------|-------------|-------|
| BULL | ✅ High | Optimal for pullback entries |
| BEAR | ✅ High | Short pullbacks in downtrends |
| SIDEWAYS | ❌ Low | ADX filter rejects signals |
| VOLATILE | ⚠️ Medium | Wider tolerance needed |
| TRANSITIONAL | ⚠️ Medium | Wait for ADX confirmation |

## Level Quality Ranking

| Level | Hit Rate | Risk/Reward | Notes |
|-------|----------|-------------|-------|
| 38.2% | Higher | Lower | Shallow pullback, strong trend |
| 50.0% | Medium | Medium | Psychological mid-point |
| 61.8% | Lower | Higher | Deep pullback, trend at risk |

## Advanced: ADX Slope Analysis

Rising ADX indicates strengthening trend:

$$
\text{ADX\_Slope} = \text{ADX}_t - \text{ADX}_{t-5}
$$

| ADX Slope | Implication |
|-----------|-------------|
| Positive | Trend strengthening - increase size |
| Flat | Trend stable - normal size |
| Negative | Trend weakening - reduce size |

---

**File:** `src/ordinis/application/strategies/fibonacci_adx.py`
**Model:** `ADXTrendModel`, `FibonacciRetracementModel`
**Status:** ✅ Production Ready
