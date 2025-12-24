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

## High-Profit Enhancements Roadmap

To evolve the strategy from a consistent performer to a high-profit system, the focus must shift from simple mean-reversion targets to capturing explosive trend continuations.

| Enhancement | Description | Goal |
|---|---|---|
| **1. Target Fibonacci Extensions** | Instead of exiting at the previous swing high (0% retracement), hold a partial position for the **1.272 (127.2%)** and **1.618 (161.8%)** extension levels. | Capture the 'second leg' of the trend, significantly increasing average profit per trade. |
| **2. Chandelier Trailing Stops** | Once price breaks the previous swing high, switch from a fixed stop/target to a **Chandelier Exit** (e.g., `High - 3 * ATR`). | Stay in "super-trends" to maximize gains on outlier moves, preventing premature exits. |
| **3. Pyramiding (Adding to Winners)** | If an entry at 50% or 61.8% is successful and price breaks the swing high, add a smaller second position. Roll the stop-loss for the entire position to breakeven. | Compound returns by increasing exposure using unrealized profits ("house money"). |
| **4. Multi-Timeframe Alignment** | Only take LONG signals on the trading timeframe (e.g., 1-hour) if a higher timeframe (e.g., Daily) is also in a confirmed uptrend (e.g., Price > 50-day SMA). | Align with major market flows and filter out short-lived counter-trend rallies, increasing signal probability. |
| **5. Volume Profile Confirmation** | Require pullbacks to Fibonacci levels to occur on **declining volume**. Require the subsequent bounce to occur on **increasing volume**. | Confirm that sellers are exhausted and buyers are returning, adding a layer of Volume Spread Analysis (VSA) to avoid "dead-cat bounces". |

---

## Implementation To-Do List

- [ ] **Task 1 (Risk Management):** Modify `fibonacci_adx.py` to implement **tiered stop-losses**.
  -   *Entry @ 38.2% level* → Stop-loss just below 50.0% level.
  -   *Entry @ 50.0% level* → Stop-loss just below 61.8% level.
  -   *Entry @ 61.8% level* → Stop-loss below swing low (as a last resort).

- [ ] **Task 2 (Profit Taking):** Update signal metadata to include Fibonacci extension levels (1.272, 1.618) as potential `take_profit_2` and `take_profit_3` targets.

- [ ] **Task 3 (ADX Slope):** Integrate ADX slope calculation into the `ADXTrendModel`. Add a `trend_accelerating` boolean to the `adx_signal` metadata. Modify the `FibonacciADXStrategy` to require this condition for entry.

- [ ] **Task 4 (Trailing Stop):** Implement a `ChandelierExit` model in SignalCore. The `PortfolioEngine` should be able to switch to this exit logic after `take_profit_1` (the swing high) is breached.

- [ ] **Task 5 (Swing Detection):** Research and implement a more robust **fractal-based swing detection** method to replace the simple `rolling_max/min` logic. This will make Fibonacci levels more stable and meaningful.

---

**File:** `src/ordinis/application/strategies/fibonacci_adx.py`
**Model:** `ADXTrendModel`, `FibonacciRetracementModel`
**Status:** ✅ Production Ready (Base) | 🚀 Enhancements Pending
