# RSI Mean Reversion Strategy

---

**Title:** RSI Oversold/Overbought Mean Reversion
**Description:** Trades extreme RSI levels expecting mean reversion to fair value
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** mean-reversion, rsi, oscillator, oversold, overbought
**References:** J. Welles Wilder (1978), New Concepts in Technical Trading Systems

---

## Overview

The RSI Mean Reversion strategy exploits price extremes identified by the Relative Strength Index (RSI). When RSI reaches oversold territory (< 30), the strategy anticipates a bounce; when RSI reaches overbought territory (> 70), it expects a pullback. Position sizing and take-profit levels scale with RSI extremity.

## Mathematical Basis

### Relative Strength Index (RSI)

$$
\text{RSI} = 100 - \frac{100}{1 + \text{RS}}
$$

Where Relative Strength:

$$
\text{RS} = \frac{\text{Avg Gain}_n}{\text{Avg Loss}_n}
$$

### Wilder Smoothing

$$
\text{AvgGain}_t = \frac{(\text{AvgGain}_{t-1} \times (n-1)) + \text{Gain}_t}{n}
$$

### Signal Probability

Based on RSI distance from neutral (50):

$$
P(\text{reversal}) = \frac{|RSI - 50|}{50}
$$

### Dynamic Take-Profit

$$
\text{TP} = \text{Entry} \pm \text{ATR}_{14} \times (1 + \frac{|RSI - 50|}{100})
$$

## Signal Logic

| Condition | Action |
|-----------|--------|
| RSI < 30 (oversold) | **LONG** - expect bounce |
| RSI > 70 (overbought) | **SHORT** - expect pullback |
| RSI returns to 50 ± 5 | **EXIT** - mean achieved |
| Opposite extreme hit | **REVERSE** position |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rsi_period` | 14 | RSI calculation period |
| `oversold` | 30 | Oversold threshold |
| `overbought` | 70 | Overbought threshold |
| `min_bars` | 50 | Minimum data required |
| `extreme_threshold` | 10/90 | Extreme levels for larger size |
| `exit_neutral` | 50 | Exit near neutral RSI |

## Edge Source

1. **Behavioral Overreaction:** Traders overreact to news, pushing prices to extremes
2. **Short-Term Mean Reversion:** Academic evidence supports 1-5 day reversals
3. **Liquidity Dynamics:** Forced selling/buying creates temporary dislocations

## Risk Considerations

- **Trending Markets:** RSI can stay extreme for extended periods during trends
- **"Catching Falling Knife":** Extreme oversold can become more extreme
- **Gap Risk:** Earnings/news gaps ignore RSI levels
- **Regime Dependence:** Works in sideways markets, fails in trends

## Implementation Notes

```python
# GPU-accelerated implementation
rsi = gpu_engine.compute_rsi(close, period=14)
atr = gpu_engine.compute_atr(high, low, close, 14)

# Signal generation
is_oversold = rsi[i] < oversold_threshold
is_overbought = rsi[i] > overbought_threshold

# Probability scales with extremity
prob = abs(rsi[i] - 50) / 50

# Dynamic targets
if is_oversold:
    target = entry + atr[i] * (1 + (50 - rsi[i]) / 100)
    stop = entry - atr[i] * 1.5
```

## Performance Expectations

- **Win Rate:** 55-65%
- **Profit Factor:** 1.1-1.4
- **Avg Holding Period:** 3-10 bars
- **Best Conditions:** Range-bound, mean-reverting markets
- **Worst Conditions:** Strong trending markets (ADX > 30)

## Regime Suitability

| Regime | Suitability | Notes |
|--------|-------------|-------|
| BULL | ⚠️ Medium | Only trade oversold bounces |
| BEAR | ⚠️ Medium | Only trade overbought fades |
| SIDEWAYS | ✅ High | Optimal regime for RSI |
| VOLATILE | ⚠️ Medium | Widen stops, expect larger swings |
| TRANSITIONAL | ❌ Low | False signals common |

## Lopez de Prado Considerations

- Use Purged K-Fold CV for parameter optimization (period, thresholds)
- Apply Deflated Sharpe Ratio to adjust for multiple trials
- Watch for PBO when varying threshold combinations

---

**File:** `src/ordinis/application/strategies/rsi_mean_reversion.py`
**Model:** `RSIMeanReversionModel`
**Status:** ✅ Production Ready
