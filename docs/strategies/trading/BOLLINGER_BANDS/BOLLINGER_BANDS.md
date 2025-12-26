# Bollinger Bands Strategy

Mean-reversion strategy using Bollinger Band extremes, with volatility-compression (“squeeze”) avoidance and trend-risk controls.

---

**Title:** Bollinger Bands Mean Reversion with Squeeze Detection
**Description:** Trades band extremes with volatility and trend gating
**Author:** Ordinis Quantitative Team
**Version:** 1.1.0
**Date:** 2025-12-23
**Status:** review
**Tags:** mean-reversion, volatility, bollinger, squeeze, bands
**References:** John Bollinger (1983), *Bollinger on Bollinger Bands*

---

## Overview

The Bollinger Bands strategy combines mean reversion at band extremes with volatility-based filtering. It targets overshoots at the outer bands and seeks a return toward the middle band (the moving-average “anchor”).

Version 1.1 tightens the specification to reduce false touches, avoid low-volatility traps, and limit fading during strong trends.

### What changed in v1.1

- Entry is defined using **close-based re-entry** (overshoot then re-enter) rather than “touch” ambiguity.
- Squeeze detection is defined using a **rolling quantile** (adaptive per symbol/timeframe).
- Adds an explicit **trend-risk gate** recommendation (e.g., ADX) to avoid “walking the band.”

### Implementation status (important)

The current implemented model(s) may lag this spec. See:

- SignalEngine model (SignalCore implementation): `src/ordinis/engines/signalcore/models/bollinger_bands.py`
- Application strategy wrapper: `src/ordinis/application/strategies/bollinger_bands.py`

Where this document proposes logic not yet present in code, it is labeled **Spec v1.1 (recommended)**.

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

### Spec v1.1 (recommended)

Definitions:

- $C_t$ = close price at time $t$
- $L_t, M_t, U_t$ = lower, middle, upper bands
- $BW_t$ = BandWidth at time $t$

**Long entry (re-entry confirmation):**

- Setup: $C_{t-1} < L_{t-1}$ (closed outside lower band)
- Trigger: $C_t > L_t$ (close re-enters the bands)
- Gate: not in squeeze, and trend filter allows mean reversion

**Short entry (re-entry confirmation):**

- Setup: $C_{t-1} > U_{t-1}$
- Trigger: $C_t < U_t$
- Gate: not in squeeze, and trend filter allows mean reversion

**Exit (mean reversion target):**

- Primary: close crosses the middle band $M_t$ (direction-specific)
- Safety: time stop after $T$ bars if target not reached
- Risk: stop-loss (ATR- or band-range based; see Risk section)

**Reversals:** only allow reversal (flip) in explicitly sideways/mean-reverting regimes; otherwise exit-only.

### Current implementation (as of repo state)

The SignalEngine model (SignalCore implementation) currently:

- Uses “touch/cross” logic near the bands.
- Uses a minimum band width threshold (`min_band_width`) as a low-volatility gate.
- Emits ENTRY on lower-band interactions and EXIT on upper-band interactions.

If you want the runtime to match Spec v1.1, you should update the model logic accordingly.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `period` | 20 | SMA and std dev period |
| `std_dev` | 2.0 | Standard deviation multiplier |
| `min_band_width` | 0.01 | Minimum band width (1%); implemented gate |
| `squeeze_quantile` | 0.15 | Rolling BandWidth quantile for squeeze (Spec v1.1) |
| `squeeze_lookback` | 252 | Lookback window for squeeze quantile (Spec v1.1) |
| `use_squeeze_filter` | True | Avoid trades during volatility compression |
| `trend_filter` | adx | Trend gate type (Spec v1.1) |
| `adx_period` | 14 | ADX window (Spec v1.1) |
| `max_adx` | 25 | Do not fade when ADX exceeds this (Spec v1.1) |
| `time_stop_bars` | 10 | Exit if mean reversion does not occur in time (Spec v1.1) |
| `min_bars` | 50 | Minimum data required |
| `exit_at_middle` | True | Exit at centerline (primary target) |

## Edge Source

1. **Statistical Mean Reversion:** Price tends to return to moving average
2. **Volatility Clustering:** Low volatility precedes high volatility (Mandelbrot)
3. **Behavioral Extremes:** Traders panic at band edges, creating overshoot

## Risk Considerations

- **Trend breakouts (“walk the band”):** Strong trends can ride the band for extended periods; avoid fading when trend strength is high.
- **Squeeze breakouts:** Compression often leads to explosive moves; avoid fading during compression and require expansion confirmation if trading breakouts.
- **Stop Placement:** Fixed stops may be too tight/wide vs. band width
- **Volatility Regime:** Works best in moderate, stable volatility

### Trend-risk gating (Spec v1.1)

Use at least one of:

- ADX filter (recommended): avoid new fades when $ADX > 25$.
- Middle-band slope filter: avoid fading when the midline slope is large in magnitude.

## Implementation Notes

```python
# Reference-style pseudocode (Spec v1.1)
middle = sma(close, period)
std = rolling_std(close, period)
upper = middle + std_dev * std
lower = middle - std_dev * std

band_width = (upper - lower) / middle

# Squeeze filter (adaptive)
q = rolling_quantile(band_width, lookback=squeeze_lookback, quantile=squeeze_quantile)
in_squeeze = band_width < q

# Re-entry confirmation
long_reentry = (close[i - 1] < lower[i - 1]) and (close[i] > lower[i])
short_reentry = (close[i - 1] > upper[i - 1]) and (close[i] < upper[i])

if (not in_squeeze[i]) and long_reentry:
    take_profit = middle[i]
    stop_loss = close[i] - atr_mult * atr[i]
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
| BULL | Medium | Prefer long-only fades near lower band; avoid short fades when trend is strong |
| BEAR | Medium | Prefer short-only fades near upper band; avoid long fades when trend is strong |
| SIDEWAYS | High | Optimal regime for BB mean reversion |
| VOLATILE | Medium | Wider bands; signals can be noisier, use stricter confirmation |
| TRANSITIONAL | Low | High breakout risk; use squeeze awareness and trend filters |

## Squeeze Detection

The “Bollinger Squeeze” indicates low volatility compression. For robustness across symbols/timeframes, v1.1 defines squeeze using a rolling quantile:

$$
    ext{Squeeze}_t = BW_t < Q_q(BW_{t-L:t})
$$

Where $q$ is a low percentile (e.g., 0.15) and $L$ is a sufficiently long lookback window.

**Trading Implications:**
- Avoid mean reversion entries during squeeze
- Prepare for breakout following squeeze release
- Use ATR to confirm volatility expansion

---

**File:** `src/ordinis/application/strategies/bollinger_bands.py`
**Model:** `BollingerBandsModel`
**Status:** review

---

## Document Metadata

```yaml
version: "1.1.0"
created: "2025-12-17"
last_updated: "2025-12-23"
status: "review"
applies_to:
    - "src/ordinis/application/strategies/bollinger_bands.py"
    - "src/ordinis/engines/signalcore/models/bollinger_bands.py"
```

---

**END OF DOCUMENT**
