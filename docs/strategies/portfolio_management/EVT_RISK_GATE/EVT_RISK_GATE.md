# EVT Risk Gate Strategy

---

**Title:** Extreme Value Theory Risk Gate
**Description:** Overlay strategy using GPD tail estimation to dynamically adjust position sizing
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** risk-management, EVT, tail-risk, VaR, GPD, overlay
**References:** Pickands (1975), McNeil & Frey (2000)

---

## Overview

The EVT Risk Gate is a risk overlay that uses Extreme Value Theory to estimate tail risk more accurately than Gaussian assumptions. When tail risk metrics exceed thresholds, position sizes are reduced proportionally.

## Mathematical Basis

### Generalized Pareto Distribution (GPD)

For exceedances over a high threshold $u$, the tail follows:

$$
F_u(y) = 1 - \left(1 + \frac{\xi y}{\tilde{\sigma}}\right)^{-1/\xi}
$$

Where:
- $y = x - u$ (excess over threshold)
- $\xi$ = shape parameter (tail heaviness)
- $\tilde{\sigma}$ = scale parameter

### Tail Shape Interpretation

| Shape ($\xi$) | Tail Type | Example |
|---------------|-----------|---------|
| $\xi > 0$ | Heavy (Fréchet) | Fat-tailed, extreme events |
| $\xi = 0$ | Exponential (Gumbel) | Moderate tails |
| $\xi < 0$ | Bounded (Weibull) | Light tails with upper bound |

### Value at Risk (VaR)

Using GPD for tail estimation:

$$
\text{VaR}_p = u + \frac{\tilde{\sigma}}{\xi}\left[\left(\frac{n}{N_u}(1-p)\right)^{-\xi} - 1\right]
$$

### Conditional VaR (CVaR / Expected Shortfall)

$$
\text{CVaR}_p = \frac{\text{VaR}_p + \tilde{\sigma} - \xi u}{1 - \xi}
$$

## Signal Logic

The EVT Risk Gate outputs a `position_multiplier` between 0 and 1:

| Condition | Multiplier | Action |
|-----------|------------|--------|
| `VaR < 2%` AND `ξ < 0.2` | 1.0 | Full position |
| `VaR ∈ [2%, 3%]` OR `ξ ∈ [0.2, 0.3]` | 0.5-1.0 | Reduced position |
| `VaR > 3%` OR `ξ > 0.3` | 0.25-0.5 | Minimal position |
| `VaR > 5%` OR `ξ > 0.5` | 0.0 | Exit all positions |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 252 | Days for GPD estimation |
| `threshold_percentile` | 95 | Threshold for tail selection |
| `var_confidence` | 0.99 | VaR confidence level |
| `var_alert_threshold` | 0.03 | VaR level triggering reduction |
| `xi_alert_threshold` | 0.3 | Shape parameter alert level |
| `recalc_frequency` | 5 | Days between recalculation |

## Edge Source

1. **Better Tail Estimation:** GPD captures fat tails that Gaussian VaR misses
2. **Dynamic Risk Scaling:** Automatically reduces exposure before crashes
3. **Shape Awareness:** $\xi$ parameter detects changing tail dynamics

## Integration Pattern

```python
from ordinis.engines.signalcore.models import EVTRiskGate, EVTGatedStrategy

# Create base strategy
base_strategy = ATROptimizedRSIModel(config)

# Wrap with EVT gate
gated_strategy = EVTGatedStrategy(
    base_strategy=base_strategy,
    evt_gate=EVTRiskGate(EVTConfig(var_alert_threshold=0.03))
)

# Signal includes position_multiplier
signal = await gated_strategy.generate(symbol, df, timestamp)
position_size = base_size * signal.metadata["position_multiplier"]
```

## Risk Considerations

- **Estimation Error:** GPD requires sufficient tail observations (~50+ exceedances)
- **Threshold Sensitivity:** Results depend on threshold selection
- **Non-Stationarity:** Tail parameters can change over time

## Diagnostic Outputs

```python
result = evt_gate.fit(returns)

print(f"Shape (ξ): {result.xi:.3f}")
print(f"Scale (σ): {result.sigma:.3f}")
print(f"VaR 99%: {result.var_99:.2%}")
print(f"CVaR 99%: {result.cvar_99:.2%}")
print(f"Multiplier: {result.position_multiplier:.2f}")
```

## Dependencies

| Package | Purpose | Fallback |
|---------|---------|----------|
| `scipy.stats` | GPD fitting | Historical percentile VaR |

## Performance Expectations

- **Purpose:** Risk reduction, not alpha generation
- **Drawdown Reduction:** 20-40% improvement vs unmanaged
- **Return Impact:** Slight reduction during normal periods
- **Best Value:** Crisis periods with fat-tail events

---

**File:** `src/ordinis/engines/signalcore/models/evt_risk_gate.py`
**Status:** ✅ Complete
