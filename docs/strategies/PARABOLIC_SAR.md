# Parabolic SAR Strategy

---

**Title:** Parabolic SAR Trend Following with Dynamic Trailing Stop
**Description:** Pure trend-following system using SAR for entries and trailing exits
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** trend-following, sar, trailing-stop, dynamic, wilder
**References:** J. Welles Wilder (1978), New Concepts in Technical Trading Systems

---

## Overview

The Parabolic SAR (Stop and Reverse) strategy implements a pure trend-following system where the SAR indicator serves dual purposes: entry signal when SAR flips position, and dynamic trailing stop that accelerates with the trend. The parabolic curve increasingly tightens the stop as the trend matures, automatically locking in profits.

## Mathematical Basis

### SAR Calculation

For each bar in an uptrend:

$$
\text{SAR}_{t+1} = \text{SAR}_t + \text{AF} \times (\text{EP} - \text{SAR}_t)
$$

Where:
- **AF** = Acceleration Factor (starts at 0.02, max 0.20)
- **EP** = Extreme Point (highest high in uptrend, lowest low in downtrend)

### Acceleration Factor Progression

$$
\text{AF}_{t+1} = \min(\text{AF}_t + \text{increment}, \text{AF}_{\max})
$$

AF increases each time a new EP is reached:
- Initial: 0.02
- Increment: 0.02 per new EP
- Maximum: 0.20

### SAR Flip (Reversal)

$$
\text{Flip}_{\text{Long}} = (\text{Low}_t < \text{SAR}_t) \text{ while in downtrend}
$$

$$
\text{Flip}_{\text{Short}} = (\text{High}_t > \text{SAR}_t) \text{ while in uptrend}
$$

## Signal Logic

| Condition | Action |
|-----------|--------|
| SAR flips from above price to below | **LONG** entry |
| SAR flips from below price to above | **SHORT** entry |
| Price touches SAR while in position | **EXIT** (stop hit) |
| Minimum trend bars not met | **FILTER** (ignore flip) |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `acceleration` | 0.02 | AF increment per new EP |
| `maximum` | 0.20 | Maximum acceleration factor |
| `min_trend_bars` | 3 | Minimum bars before valid entry |
| `min_bars` | 50 | Minimum data required |

## Edge Source

1. **Trend Persistence:** Established trends tend to continue
2. **Dynamic Risk Management:** Stop tightens as trend matures
3. **Automatic Reversal:** Captures both sides of major moves
4. **No Discretion:** Fully systematic, no subjective decisions

## Risk Considerations

- **Choppy Markets:** Frequent whipsaws during consolidation
- **Gap Risk:** Gaps can exceed SAR level significantly
- **Early Entry:** SAR flip may occur before trend confirmation
- **No Target:** Pure trailing stop; may exit prematurely

## Implementation Notes

```python
# GPU-accelerated SAR calculation
sar = np.zeros(len(close))
af = acceleration
ep = high[0] if uptrend else low[0]

for i in range(1, len(close)):
    sar[i] = sar[i-1] + af * (ep - sar[i-1])

    if uptrend:
        if high[i] > ep:
            ep = high[i]
            af = min(af + acceleration, maximum)
        if low[i] < sar[i]:
            # Flip to downtrend
            uptrend = False
            sar[i] = ep
            ep = low[i]
            af = acceleration
    else:
        # Downtrend logic (inverse)
        ...

# Signal generation
flip_long = not uptrend_prev and uptrend_current
flip_short = uptrend_prev and not uptrend_current
```

## Performance Expectations

- **Win Rate:** 35-45%
- **Profit Factor:** 1.5-2.5
- **Avg Holding Period:** Variable (trend-dependent)
- **Best Conditions:** Strong trending markets
- **Worst Conditions:** Range-bound, choppy markets

## Regime Suitability

| Regime | Suitability | Notes |
|--------|-------------|-------|
| BULL | ✅ High | Captures uptrend moves |
| BEAR | ✅ High | Captures downtrend moves |
| SIDEWAYS | ❌ Low | Frequent whipsaws, heavy losses |
| VOLATILE | ⚠️ Medium | Wider initial stop needed |
| TRANSITIONAL | ⚠️ Medium | First flip may be false |

## Acceleration Factor Analysis

| AF Level | Interpretation | Action |
|----------|---------------|--------|
| 0.02-0.06 | Early trend | Wide stop, full size |
| 0.08-0.14 | Maturing trend | Tightening stop, watch closely |
| 0.16-0.20 | Extended trend | Very tight stop, expect reversal |

## Enhancements

### ADX Filter

Add trend strength confirmation:

```python
use_adx_filter = True
adx_threshold = 20

if flip_long and (not use_adx_filter or adx[i] > adx_threshold):
    signal = LONG
```

### Take Profit Override

Add fixed target alongside trailing stop:

```python
take_profit = entry_price * 1.15  # 15% target
if close[i] >= take_profit:
    exit_reason = "target_hit"
```

## Risk Management Integration

The SAR value itself provides the stop-loss:

| Direction | Stop Loss | Take Profit |
|-----------|-----------|-------------|
| LONG | SAR value (below price) | Optional: 1.5× ATR from entry |
| SHORT | SAR value (above price) | Optional: 1.5× ATR from entry |

---

**File:** `src/ordinis/application/strategies/parabolic_sar_trend.py`
**Model:** `ParabolicSARModel`
**Status:** ✅ Production Ready
