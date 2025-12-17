# Mutual-Information Weighted Ensemble

## Purpose

Combine multiple base signals using mutual information (MI) to weight more informative signals and down-weight redundant/noisy ones.

## Inputs

- Base signals (e.g., trend, value, vol, order book features) aligned in time.
- Target: future return/label used to estimate MI (point-in-time).

## Method (outline)

1) Compute MI between each signal and target; optionally compute pairwise MI to penalize redundancy.
2) Normalize weights: w_i ∝ MI(signal_i, target), adjust for redundancy if desired.
3) Aggregate: ensemble_signal = Σ w_i * signal_i (after z-scoring signals).
4) Optional: decay weights over time based on rolling MI.

## Outputs

- weights per signal, ensemble_signal (continuous), ensemble_confidence (e.g., weighted MI or variance).

## Usage Notes

- Use discretization or k-NN MI estimators suited for continuous variables.
- Recompute MI on rolling windows; watch for sample size adequacy.
- Cap weights to avoid dominance; enforce sum of weights = 1.

## Python Example (MI-weighted blend)

```python
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import numpy as np

signals = pd.read_csv("signals.csv", parse_dates=["ts"], index_col="ts")
target = signals["future_ret"]
X = signals[["s1", "s2", "s3", "s4"]]

# Compute MI weights
mi = mutual_info_regression(X, target, discrete_features=False)
weights = pd.Series(mi, index=X.columns).clip(lower=0)
weights = weights / weights.sum()

# Optional: penalize redundancy (proxy using correlation; replace with MI if available)
pairwise_mi = X.corr().abs()
redundancy_penalty = pairwise_mi.mean(axis=1)
weights = weights / (1 + redundancy_penalty)
weights = weights / weights.sum()

# Standardize signals and blend
z = (X - X.mean()) / X.std(ddof=0)
ensemble = (z * weights).sum(axis=1).rename("ensemble_score")

# Rolling recompute example
window = 252
scores = []
weight_history = []
for i in range(window, len(signals)):
    Xw = X.iloc[i - window:i]
    tw = target.iloc[i - window:i]
    mi_w = mutual_info_regression(Xw, tw)
    w = pd.Series(mi_w, index=Xw.columns).clip(lower=0)
    if w.sum() > 0:
        w = w / w.sum()
    weight_history.append(w)
    latest = ((Xw.iloc[-1:] - Xw.mean()) / Xw.std(ddof=0) * w).sum(axis=1).iloc[0]
    scores.append(latest)
ensemble_rolling = pd.Series(scores, index=signals.index[window:], name="ensemble_score")
weight_history = pd.DataFrame(weight_history, index=signals.index[window:])
```

## Validation Tips

- Check MI stability over time; if weights swing wildly, increase window or regularize.
- Compare ensemble performance to simple averages; ensure MI weighting adds value.
- Guard against data leakage: target must be strictly future returns.
```

## Ensemble Hook

This is the combiner itself; feed base signals plus their quality/confidence scores. Outputs a single ensemble signal and weight diagnostics.
