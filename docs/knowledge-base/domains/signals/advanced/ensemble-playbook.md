# Ensemble Playbook

## Purpose

Combine multiple signals into a single decision stream with explicit weighting, gating, and risk-awareness.

## Inputs

- Base signals with aligned timestamps and scopes (e.g., trend, tail risk, orderbook pressure, allocators).
- Quality/confidence per signal (variance, MI, sample size).
- Risk inputs: vol, DD, tail alerts, exposure limits.

## Steps

1) Standardize signals: z-score or min-max per signal; align signs (buy/sell intent).
2) Quality weights: use MI or inverse variance; cap per-signal weight; ensure sum of weights=1.
3) Risk gates: if tail_alert or DD breach, reduce or zero aggressive signals; cap exposure.
4) Aggregate: ensemble = Σ w_i * s_i; optional nonlinear rules (majority vote, thresholding).
5) Decay/refresh: recompute weights on rolling window; decay stale signals.

## Example (pseudo)

```
signals = [trend_slope, tail_intensity, pressure_score, alloc_tilt]
weights = normalize(quality(scores=MI or 1/var))
if tail_alert: weights['trend_slope'] *= 0.5
ensemble = sum(w*s for w, s in zip(weights, signals))
```

## Outputs

- ensemble_score (continuous) and confidence.
- diagnostics: per-signal weight, gating applied.

## Usage Notes

- Keep turnover in check: smooth ensemble over time; apply hysteresis on thresholds.
- Avoid dominance: cap max weight; diversify across signal families.
- Validate out-of-sample; monitor contribution and decay per signal.

## Python Example (simple gated ensemble)

```python
import pandas as pd
import numpy as np

signals = pd.read_csv("signals.csv", parse_dates=["ts"], index_col="ts")
base = signals[["trend_slope", "tail_intensity", "pressure_score", "alloc_tilt"]]
quality = pd.Series([0.4, 0.2, 0.2, 0.2], index=base.columns)  # example weights
weights = quality / quality.sum()

# Standardize signals
z = (base - base.mean()) / base.std(ddof=0)

# Weighted blend
ensemble_raw = (z * weights).sum(axis=1)

# Risk gate: down-weight when tail risk high
tail_alert = signals["tail_intensity"] > 0.02
ensemble = ensemble_raw.where(~tail_alert, ensemble_raw * 0.5)

# Smooth and add hysteresis to reduce churn
ensemble_smoothed = ensemble.ewm(span=5).mean()
long_threshold = 0.5
short_threshold = -0.5

def decision(series):
    pos = 0
    decisions = []
    for val in series:
        if pos == 0:
            if val > long_threshold:
                pos = 1
            elif val < short_threshold:
                pos = -1
        elif pos == 1 and val < 0:
            pos = 0
        elif pos == -1 and val > 0:
            pos = 0
        decisions.append(pos)
    return pd.Series(decisions, index=series.index, name="position")

positions = decision(ensemble_smoothed)

# Diagnostics: contribution per signal
contrib = (z.mul(weights, axis=1))
rolling_contrib = contrib.ewm(span=10).mean()
```

## Additional Patterns

- **Voting**: For discrete signals, take majority vote or require quorum.
- **Vol targeting**: Scale ensemble to target volatility or risk budget.
- **Decay**: Apply exponential decay to stale signals; drop signals with low quality.
- **Cap exposure**: Map ensemble_score to position with max position limits and slope control.
```
