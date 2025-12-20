# Bollinger Bands Strategy

---

**Title:** Bollinger Bands Mean Reversion with Squeeze Detection
**Description:** Trades band touches with volatility expansion/contraction filtering
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** mean-reversion, volatility, bollinger, squeeze, bands
**References:** John Bollinger (1983), Bollinger on Bollinger Bands

---

## Overview

The Bollinger Bands strategy combines mean reversion at band extremes with volatility-based filtering. It enters positions when price touches outer bands (2 standard deviations) and exits at the moving average centerline. The strategy filters signals using band width to avoid trades during low-volatility "squeeze" conditions where breakouts are more likely.

## Mathematical Basis

### Bollinger Band Construction

Middle band (SMA):
$$
\text{Middle}_t = \text{SMA}(P, n)
$$

Upper and lower bands:
$$
\text{Upper}_t = \text{Middle}_t + k \cdot \sigma_t
$$

$$
\text{Lower}_t = \text{Middle}_t - k \cdot \sigma_t
$$

Where $\sigma_t$ is rolling standard deviation over $n$ periods.

### Band Width (Volatility Measure)

$$
\text{BandWidth}_t = \frac{\text{Upper}_t - \text{Lower}_t}{\text{Middle}_t}
$$

### %B (Position Within Bands)

$$
\%B_t = \frac{P_t - \text{Lower}_t}{\text{Upper}_t - \text{Lower}_t}
$$

- %B > 1: Above upper band
- %B < 0: Below lower band
- %B = 0.5: At centerline

## Signal Logic

| Condition | Action |
|-----------|--------|
| Price ≤ Lower Band AND BandWidth > min | **LONG** - oversold |
| Price ≥ Upper Band AND BandWidth > min | **SHORT** - overbought |
| Price crosses Middle Band | **EXIT** - mean reversion complete |
| Opposite band touch | **REVERSE** position |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `period` | 20 | SMA and std dev period |
| `std_dev` | 2.0 | Standard deviation multiplier |
| `min_band_width` | 0.01 | Minimum band width (1%) |
| `min_bars` | 50 | Minimum data required |
| `use_squeeze_filter` | True | Filter during low volatility |
| `exit_at_middle` | True | Exit at centerline |

## Edge Source

1. **Statistical Mean Reversion:** Price tends to return to moving average
2. **Volatility Clustering:** Low volatility precedes high volatility (Mandelbrot)
3. **Behavioral Extremes:** Traders panic at band edges, creating overshoot

## Risk Considerations

- **Trend Breakouts:** Strong trends can ride the band for extended periods
- **Squeeze Breakouts:** Compression often leads to explosive moves
- **Stop Placement:** Fixed stops may be too tight/wide vs. band width
- **Volatility Regime:** Works best in moderate, stable volatility

## Implementation Notes

```python
# GPU-accelerated implementation
middle = gpu_engine.compute_sma(close, period)
std = gpu_engine.compute_rolling_std(close, period)
upper = middle + std_dev * std
lower = middle - std_dev * std

# Band width filter
band_width = (upper - lower) / middle
is_valid_width = band_width > min_band_width

# Signal generation
touch_lower = (close[i] <= lower[i]) and is_valid_width[i]
touch_upper = (close[i] >= upper[i]) and is_valid_width[i]

# Risk management - bands as targets/stops
if touch_lower:
    stop_loss = lower[i] - (upper[i] - lower[i]) * 0.5
    take_profit = upper[i]  # Opposite band
```

## Performance Expectations

- **Win Rate:** 55-65%
- **Profit Factor:** 1.2-1.5
- **Avg Holding Period:** 5-15 bars
- **Best Conditions:** Stable volatility, mean-reverting markets
- **Worst Conditions:** Trending markets, volatility breakouts

## Regime Suitability

| Regime | Suitability | Notes |
|--------|-------------|-------|
| BULL | ⚠️ Medium | Trade lower band bounces only |
| BEAR | ⚠️ Medium | Trade upper band fades only |
| SIDEWAYS | ✅ High | Optimal regime for BB mean reversion |
| VOLATILE | ⚠️ Medium | Bands widen, signals less reliable |
| TRANSITIONAL | ❌ Low | Watch for squeeze breakouts |

## Squeeze Detection

The "Bollinger Squeeze" indicates low volatility compression:

$$
\text{Squeeze} = \text{BandWidth}_t < \text{BandWidth}_{\text{min}(n)}
$$

**Trading Implications:**
- Avoid mean reversion entries during squeeze
- Prepare for breakout following squeeze release
- Use ATR to confirm volatility expansion

---

**File:** `src/ordinis/application/strategies/bollinger_bands.py`
**Model:** `BollingerBandsModel`
**Status:** ✅ Production Ready
