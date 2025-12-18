# Momentum Breakout Strategy

---

**Title:** Momentum Breakout with Volume Confirmation
**Description:** Trades price breakouts from rolling high/low channels with volume surge confirmation
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** momentum, breakout, volume, channel, small-cap
**References:** Donchian Channels, Volume Spread Analysis

---

## Overview

The Momentum Breakout strategy identifies price breakouts from N-period high/low channels, entering positions only when confirmed by above-average volume. This combination filters out false breakouts that lack institutional participation, improving win rates on small-cap and volatile stocks.

## Mathematical Basis

### Channel Breakout Detection

Rolling channel boundaries over lookback period $n$:

$$
\text{Upper}_t = \max(H_{t-n}, H_{t-n+1}, ..., H_{t-1})
$$

$$
\text{Lower}_t = \min(L_{t-n}, L_{t-n+1}, ..., L_{t-1})
$$

### Volume Confirmation

Volume surge detection using SMA comparison:

$$
\text{VolSurge} = \frac{V_t}{\text{SMA}(V, n)} > \text{threshold}
$$

Default threshold: 1.5× average volume

### ATR-Based Risk Management

Stop-loss and take-profit levels use Average True Range:

$$
\text{Stop} = \text{Entry} \pm \text{ATR}_{14} \times \text{stop\_mult}
$$

$$
\text{Target} = \text{Entry} \pm \text{ATR}_{14} \times \text{tp\_mult}
$$

## Signal Logic

| Condition | Action |
|-----------|--------|
| `price > rolling_high[t-1]` AND `volume > avg_volume × 1.5` | **LONG** |
| `price < rolling_low[t-1]` AND `volume > avg_volume × 1.5` | **SHORT** |
| Stop-loss hit OR target hit OR 15 bars elapsed | **EXIT** |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 20 | Rolling high/low period |
| `breakout_mult` | 1.5 | Volume surge multiplier |
| `atr_stop` | 2.0 | ATR multiplier for stop-loss |
| `atr_tp` | 3.0 | ATR multiplier for take-profit |
| `max_bars` | 15 | Maximum holding period |

## Edge Source

1. **Volume Validation:** Filters out low-conviction breakouts lacking institutional flow
2. **Momentum Persistence:** Genuine breakouts tend to continue in the breakout direction
3. **Channel Psychology:** Round-number and historical high/low levels attract attention

## Risk Considerations

- **False Breakouts:** Even with volume confirmation, some breakouts fail
- **Gap Risk:** Overnight gaps can exceed stop-loss levels
- **Low Liquidity:** Small-cap stocks may have wide spreads on breakouts
- **Time Decay:** Momentum fades after 10-15 bars

## Implementation Notes

```python
# GPU-accelerated implementation
atr = gpu_engine.compute_atr(high, low, close, 14)
sma_vol = gpu_engine.compute_sma(volume, lookback)

# Entry conditions
vol_surge = current_vol > avg_vol * breakout_mult
bullish = current_price > rolling_high[i-1] and vol_surge
bearish = current_price < rolling_low[i-1] and vol_surge
```

## Performance Expectations

- **Win Rate:** 40-50%
- **Profit Factor:** 1.2-1.6
- **Avg Holding Period:** 5-10 bars
- **Best Conditions:** Trending markets with clear breakout levels
- **Worst Conditions:** Choppy, range-bound markets

## Sprint 3 Results (Small-Cap)

Top performers on small-cap universe:
- Strong results on energy stocks (SM, CTRA, OVV)
- Works well with volatile tech names (SMCI, GTLB)

---

**File:** `scripts/strategy_sprint/sprint3_smallcap_gpu.py`
**Method:** `backtest_momentum_breakout()`
**Status:** ✅ Complete
