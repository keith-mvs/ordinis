# GARCH Volatility Breakout Strategy

---

**Title:** GARCH Volatility Breakout
**Description:** Trades volatility regime changes using GARCH(1,1) forecast vs realized volatility
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** volatility, GARCH, breakout, regime-change, econometric
**References:** Bollerslev (1986), Engle (1982)

---

## Overview

The GARCH Breakout strategy exploits the tendency of GARCH models to lag reality during volatility regime changes. When realized volatility significantly exceeds the GARCH forecast, it signals a regime shift that can be traded directionally.

## Mathematical Basis

### GARCH(1,1) Model

The Generalized Autoregressive Conditional Heteroskedasticity model:

$$
\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

Where:
- $\sigma_t^2$ = conditional variance at time $t$
- $\omega$ = long-run variance weight (constant)
- $\alpha$ = reaction to recent shock (ARCH term)
- $\beta$ = persistence of volatility (GARCH term)
- $\epsilon_{t-1}$ = previous period return shock

**Constraint:** $\alpha + \beta < 1$ for stationarity.

### Breakout Detection

```
breakout_ratio = realized_vol / garch_forecast
signal = breakout_ratio > threshold (default: 2.0)
```

### Realized Volatility

Short-window realized volatility (5-day default):

$$
\text{RV}_t = \sqrt{\frac{252}{n} \sum_{i=0}^{n-1} r_{t-i}^2}
$$

## Signal Logic

| Condition | Action |
|-----------|--------|
| `breakout_ratio > 2.0` AND `recent_return > 0` | **LONG** |
| `breakout_ratio > 2.0` AND `recent_return < 0` | **SHORT** |
| `breakout_ratio < 2.0` | **HOLD** |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `garch_lookback` | 252 | Days for GARCH estimation |
| `realized_vol_window` | 5 | Days for realized vol |
| `breakout_threshold` | 2.0 | Ratio threshold for signal |
| `direction_window` | 5 | Days for direction determination |
| `atr_stop_mult` | 2.0 | ATR multiplier for stop-loss |
| `atr_tp_mult` | 3.0 | ATR multiplier for take-profit |

## Edge Source

1. **Forecast Lag:** GARCH forecasts are backward-looking; during regime changes, they underestimate true volatility
2. **Mean Reversion Delay:** Volatility clusters, so breakouts tend to persist
3. **Directional Bias:** Large volatility expansions often coincide with strong directional moves

## Risk Considerations

- **Whipsaws:** False breakouts in choppy markets
- **Gap Risk:** Overnight gaps can exceed stop-losses
- **Model Risk:** GARCH misspecification during extreme events

## Implementation Notes

```python
from ordinis.engines.signalcore.models import GARCHBreakoutModel
from ordinis.engines.signalcore.core.model import ModelConfig

config = ModelConfig(
    model_id="garch_breakout",
    model_type="volatility",
    parameters={
        "breakout_threshold": 2.0,
        "garch_lookback": 252,
    }
)

model = GARCHBreakoutModel(config)
signal = await model.generate(symbol, df, timestamp)
```

## Dependencies

| Package | Purpose | Fallback |
|---------|---------|----------|
| `arch` | GARCH estimation | EWMA volatility |

## Performance Expectations

- **Win Rate:** 45-55%
- **Profit Factor:** 1.3-1.8
- **Best Conditions:** Trending markets with volatility regime changes
- **Worst Conditions:** Low-volatility sideways markets

---

**File:** `src/ordinis/engines/signalcore/models/garch_breakout.py`
**Status:** ✅ Complete
