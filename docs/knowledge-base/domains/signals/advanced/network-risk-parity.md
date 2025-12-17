# Network Risk Parity Signal

## Purpose

Use correlation network structure to tilt weights toward less central assets and away from highly connected (systemic) nodes.

## Inputs

- Returns matrix (assets x time) over a rolling window; consistent calendar/tz.
- Correlation method: Pearson/Spearman; consider shrinkage for stability.

## Method (outline)

1) Compute correlation matrix over window; build graph (edges weighted by |corr|).
2) Compute centrality (e.g., eigenvector, betweenness, degree).
3) Map centrality to risk weight: weight_i ∝ 1 / (centrality_i + ε); normalize to sum=1.
4) Derive signal:
   - weight_tilt = target_weight - current_weight
   - Optional risk cap: enforce min/max weights.

## Outputs

- target_weights (vector), centrality scores, weight_tilt.

## Usage Notes

- Smooth weights over time to reduce churn; add turnover penalty.
- Use robust/shrunk correlations; drop illiquid names if instability is high.
- Recompute on rolling window; stress in regime shifts.

## Python Example (centrality-based weights)

```python
import pandas as pd
import networkx as nx
import numpy as np

rets = pd.read_csv("returns.csv", parse_dates=["ts"], index_col="ts")
window = 252
corr = rets.tail(window).corr()

# Optional: shrink correlation toward identity for stability
shrink = 0.1
corr = (1 - shrink) * corr + shrink * np.eye(len(corr))

G = nx.from_pandas_adjacency(corr.abs())
cent = nx.eigenvector_centrality_numpy(G, max_iter=1000)
cent = pd.Series(cent, name="centrality")
weights = (1 / (cent + 1e-6))
weights = weights / weights.sum()

# Turnover control: blend with prior weights
def smooth_weights(new_w, old_w=None, alpha=0.2):
    if old_w is None:
        return new_w
    blended = alpha * new_w + (1 - alpha) * old_w
    return blended / blended.sum()

old_weights = None
smoothed = smooth_weights(weights, old_weights, alpha=0.2)

# Translate to tilt vs current portfolio
current_w = pd.read_csv("current_weights.csv", index_col=0, squeeze=True)
target_w = smoothed.reindex(current_w.index).fillna(0)
tilt = (target_w - current_w).rename("weight_tilt")

# Enforce min/max and renormalize
min_w, max_w = 0.0, 0.1
target_w = target_w.clip(lower=min_w, upper=max_w)
target_w = target_w / target_w.sum()
```

## Ensemble Hook

Treat target_weights or weight_tilt as a portfolio-level signal; combine with other allocators (e.g., risk parity, vol targeting) via weighted blend.
