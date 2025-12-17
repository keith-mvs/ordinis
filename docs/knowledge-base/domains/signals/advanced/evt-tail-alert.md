# EVT Tail Alert

## Purpose

Flag elevated tail risk using Extreme Value Theory (GPD/GEV) to inform sizing, hedging, or de-risking.

## Inputs

- Returns series (P&L or price returns), sufficient tail samples per asset/book.
- Lookback windows per asset class; ensure point-in-time data.

## Method (outline)

1) Choose threshold u (e.g., 95th/99th percentile of losses).
2) Fit Generalized Pareto (GPD) to exceedances over u.
3) Compute tail metrics: VaR/CVaR at target quantiles; shape parameter ξ as tail heaviness.
4) Signal:
   - tail_alert = 1 if VaR/CVaR breaches limit or ξ > tail_cap.
   - tail_intensity = normalized VaR/CVaR for scaling.

## Outputs

- tail_alert (bool), tail_intensity (0-1 or bps), xi (shape), var_q, cvar_q.

## Usage Notes

- Fit per regime if needed; refresh frequently in volatile periods.
- Ensure enough exceedances; widen window or lower threshold if too few.
- Stress test sensitivity to u and sample size.

## Python Example (GPD Tail)

```python
import numpy as np
import pandas as pd
from scipy.stats import genpareto

def fit_gpd_tail(returns, threshold_q=0.95, alpha=0.99, min_tail=50):
    """Fit GPD to loss tail and compute VaR/CVaR."""
    losses = -returns.dropna()
    u = losses.quantile(threshold_q)
    tail = losses[losses > u] - u
    if len(tail) < min_tail:
        raise ValueError(f"Not enough tail samples ({len(tail)}<{min_tail})")
    xi, loc, scale = genpareto.fit(tail, floc=0)
    var = u + genpareto.ppf(alpha, xi, loc=0, scale=scale)
    # Approx CVaR; for xi near 0, CVaR ~ VaR + scale/(1 - alpha)
    cvar = u + (scale + xi * (var - u)) / max(1 - xi, 1e-6)
    return {"threshold": u, "xi": xi, "var": var, "cvar": cvar, "tail_count": len(tail)}

def rolling_tail_alert(returns, window=252, alpha=0.99):
    records = []
    index = []
    for i in range(window, len(returns) + 1):
        w = returns.iloc[i - window:i]
        index.append(returns.index[i - 1])
        try:
            m = fit_gpd_tail(w, threshold_q=0.95, alpha=alpha)
        except ValueError:
            m = {"threshold": np.nan, "xi": np.nan, "var": np.nan, "cvar": np.nan, "tail_count": np.nan}
        records.append(m)
    return pd.DataFrame(records, index=index)

# Multi-asset helper: compute per column
def tail_alert_matrix(df_returns, window=252, alpha=0.99, var_cap=0.02, xi_cap=0.2):
    alerts = {}
    metrics = {}
    for col in df_returns:
        m = rolling_tail_alert(df_returns[col], window=window, alpha=alpha)
        alert = (m["var"] > var_cap) | (m["xi"] > xi_cap)
        alerts[col] = alert
        metrics[col] = m
    return alerts, metrics

rets = pd.read_csv("returns.csv", parse_dates=["ts"], index_col="ts")["ret"]
metrics = rolling_tail_alert(rets, window=252, alpha=0.99)
tail_alert = (metrics["var"] > 0.02) | (metrics["xi"] > 0.2)  # example thresholds
tail_intensity = metrics["cvar"].fillna(0)

# Example: route to risk gate
risk_gate = tail_alert.reindex(rets.index).fillna(False)
```

## Validation Tips

- Plot var/cvar over time; compare with realized tail losses.
- Sensitivity test threshold_q and window length; ensure stability.
- Monitor tail_count to avoid acting on sparse tails; set min_tail safeguards.
```

## Ensemble Hook

Use tail_intensity to down-weight aggressive signals or increase hedge weight; treat as a risk gate in the ensemble.
