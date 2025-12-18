# Volume-Price Confirmation Strategy

---

**Title:** Volume-Price Confirmation
**Description:** Trades significant price moves confirmed by volume surges above moving average
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** volume, price-action, confirmation, breakout, institutional-flow
**References:** Volume Spread Analysis (Richard Wyckoff), On-Balance Volume

---

## Overview

The Volume-Price Confirmation strategy identifies significant price moves that are validated by abnormally high volume. The premise is that meaningful price changes require institutional participation, which manifests as volume surges. Small price moves on low volume are noise; large price moves on high volume signal conviction.

## Mathematical Basis

### Price Deviation from Mean

Relative price change from 20-period SMA:

$$
\text{PriceChange} = \frac{P_t - \text{SMA}(P, 20)}{\text{SMA}(P, 20)}
$$

### Volume Ratio

Current volume relative to average:

$$
\text{VolRatio} = \frac{V_t}{\text{SMA}(V, 20)}
$$

### Confirmation Signal

Both conditions must be met:

$$
\text{Signal} = (|\text{PriceChange}| > \text{threshold}) \land (\text{VolRatio} > \text{mult})
$$

Default: Price change > 2%, Volume > 2× average

### Direction Determination

Direction follows price deviation:
- PriceChange > threshold → LONG
- PriceChange < -threshold → SHORT

## Signal Logic

| Condition | Action |
|-----------|--------|
| `price_change > 0.02` AND `vol_ratio > 2.0` | **LONG** |
| `price_change < -0.02` AND `vol_ratio > 2.0` | **SHORT** |
| Stop hit OR target hit OR 10 bars elapsed | **EXIT** |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vol_mult` | 2.0 | Volume surge multiplier |
| `price_thresh` | 0.02 | Price deviation threshold (2%) |
| `atr_stop` | 1.5 | ATR multiplier for stop-loss |
| `atr_tp` | 3.0 | ATR multiplier for take-profit |
| `max_bars` | 10 | Maximum holding period |

## Edge Source

1. **Institutional Footprint:** Large players cannot hide their activity in volume
2. **Conviction Filtering:** High-volume moves indicate stronger conviction
3. **Follow-Through:** Volume-confirmed moves tend to continue
4. **Noise Reduction:** Filters out low-volume price noise

## Risk Considerations

- **Volume Spikes:** Earnings, news can cause one-time volume spikes that don't follow through
- **End-of-Day Volume:** Option expiration and rebalancing create artificial volume
- **Small-Cap Illiquidity:** Volume surges on small-caps may be single large orders
- **Mean Reversion:** Extended moves may revert even with volume confirmation

## Implementation Notes

```python
# GPU-accelerated SMA computation
sma_price = gpu_engine.compute_sma(prices, 20)
sma_vol = gpu_engine.compute_sma(volume, 20)
atr = gpu_engine.compute_atr(high, low, close, 14)

# Price and volume conditions
price_change = (current_price - avg_price) / avg_price
vol_ratio = current_vol / avg_vol

# Entry when both conditions met
if price_change > price_thresh and vol_ratio > vol_mult:
    direction = 1  # LONG
elif price_change < -price_thresh and vol_ratio > vol_mult:
    direction = -1  # SHORT
```

## Performance Expectations

- **Win Rate:** 45-55%
- **Profit Factor:** 1.2-1.6
- **Avg Holding Period:** 4-7 bars
- **Best Conditions:** Stocks with clear institutional activity patterns
- **Worst Conditions:** Low-volume stocks, stocks with erratic volume patterns

## Sprint 3 Results (Small-Cap)

Moderate performer across small-cap universe:
- Best on liquid small-caps with consistent volume patterns
- Underperforms on thinly-traded names

---

**File:** `scripts/strategy_sprint/sprint3_smallcap_gpu.py`
**Method:** `backtest_volume_price_confirm()`
**Status:** ✅ Complete
