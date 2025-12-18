# Network Risk Parity Strategy

---

**Title:** Network Risk Parity
**Description:** Uses correlation network centrality to weight portfolio positions inversely
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** network-theory, risk-parity, centrality, correlation, portfolio-construction
**References:** Billio et al. (2012), Diebold & Yilmaz (2014)

---

## Overview

The Network Risk Parity strategy builds a correlation network from asset returns and calculates centrality measures. Central assets (highly connected, systemically important) receive lower weights, while peripheral assets (more independent) receive higher weights for better diversification.

## Mathematical Basis

### Correlation Network

**Nodes:** Assets

**Edges:** Exists if $|\rho_{ij}| > \text{threshold}$

**Adjacency Matrix:**

$$
A_{ij} = \begin{cases}
1 & \text{if } |\rho_{ij}| \geq \tau \\
0 & \text{otherwise}
\end{cases}
$$

Where $\rho_{ij}$ = correlation between assets $i$ and $j$, and $\tau$ = threshold (default 0.3).

### Eigenvector Centrality

Measures node importance based on connections to other important nodes:

$$
c_i = \frac{1}{\lambda} \sum_j A_{ij} c_j
$$

Or in matrix form: $\mathbf{c}$ is the principal eigenvector of $A$.

### Degree Centrality

Simple count of connections:

$$
c_i^{(deg)} = \frac{\sum_j A_{ij}}{n - 1}
$$

### Inverse Centrality Weighting

Weights inversely proportional to centrality:

$$
w_i = \frac{(c_i + \epsilon)^{-\gamma}}{\sum_j (c_j + \epsilon)^{-\gamma}}
$$

Where $\gamma$ = decay parameter (default 0.5), $\epsilon$ = small constant (0.1).

## Network Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Density | $\frac{2|E|}{n(n-1)}$ | Fraction of possible edges |
| Clustering | Local triangle density | Tendency to cluster |
| Avg Centrality | $\frac{1}{n}\sum c_i$ | Overall connectivity |

## Signal Logic

This is primarily a **portfolio construction** strategy:

| Asset Centrality | Weight | Rationale |
|------------------|--------|-----------|
| High (central) | Low | Systemically risky, reduce |
| Medium | Medium | Average risk contribution |
| Low (peripheral) | High | Diversification benefit |

### Individual Asset Signals

For single-asset signals:

| Weight | Momentum | Signal |
|--------|----------|--------|
| High (> 5%) | Positive | **LONG** |
| Low (< 5%) | - | **REDUCE** |
| Any | Negative | **SELL** |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `corr_lookback` | 60 | Days for correlation |
| `corr_threshold` | 0.3 | Minimum |ρ| for edge |
| `recalc_frequency` | 5 | Days between recalc |
| `centrality_method` | "eigenvector" | Centrality algorithm |
| `weight_decay` | 0.5 | Inverse centrality power |
| `min_weight` | 0.02 | Minimum position weight |
| `max_weight` | 0.30 | Maximum position weight |
| `momentum_lookback` | 20 | Momentum calculation |
| `vol_target` | 0.15 | Annual volatility target |

## Edge Source

1. **Systemic Risk Reduction:** Lower weight to systemically important assets
2. **Diversification:** Higher weight to independent assets
3. **Network Stability:** Monitor network changes for regime shifts
4. **Tail Risk Reduction:** Central assets fall together in crises

## Network Analysis

```python
from ordinis.engines.signalcore.models.network_parity import analyze_correlation_network

analysis = analyze_correlation_network(
    returns_df,
    threshold=0.3
)

print(f"Network density: {analysis['stats'].density:.2%}")
print(f"Average clustering: {analysis['stats'].avg_clustering:.2f}")
print(f"Central assets: {analysis['central_assets']}")
print(f"Peripheral assets: {analysis['peripheral_assets']}")

# Weights
for asset, weight in sorted(analysis['weights'].items(), key=lambda x: -x[1]):
    print(f"{asset}: {weight:.2%}")
```

## Implementation Notes

```python
from ordinis.engines.signalcore.models import NetworkRiskParityModel
from ordinis.engines.signalcore.core.model import ModelConfig

config = ModelConfig(
    model_id="network_parity",
    model_type="portfolio",
    parameters={
        "corr_lookback": 60,
        "corr_threshold": 0.3,
        "centrality_method": "eigenvector",
    }
)

model = NetworkRiskParityModel(config)

# Generate portfolio weights
signals = await model.generate_portfolio_weights(returns_df, timestamp)

for symbol, signal in signals.items():
    print(f"{symbol}: {signal.metadata['target_weight']:.2%} "
          f"(centrality: {signal.metadata['centrality']:.2f})")
```

## Network Visualization

```python
from ordinis.engines.signalcore.models.network_parity import visualize_network

fig = visualize_network(
    returns_df,
    threshold=0.3,
    figsize=(12, 10)
)
fig.savefig("correlation_network.png")
```

**Visual Encoding:**
- Node size: Inverse of centrality (larger = more peripheral)
- Node color: Target weight (darker = higher weight)
- Edges: Correlations above threshold

## Network Stability

```python
# Check for network regime change
is_stable, distance = model.check_network_stability()

if not is_stable:
    print(f"Network changed! Distance: {distance:.3f}")
    # Consider reducing positions or increasing cash
```

## Example Output

For a tech-heavy portfolio:

| Asset | Centrality | Weight | Role |
|-------|------------|--------|------|
| AAPL | 0.85 | 5.2% | Central hub |
| MSFT | 0.82 | 5.8% | Central hub |
| GOOGL | 0.78 | 6.5% | Central |
| NVDA | 0.72 | 7.8% | Semi-central |
| XOM | 0.25 | 15.2% | Peripheral |
| JNJ | 0.22 | 16.5% | Peripheral |
| PG | 0.18 | 18.0% | Most peripheral |

## Dependencies

| Package | Purpose | Fallback |
|---------|---------|----------|
| `networkx` | Visualization | Built-in adjacency |
| `matplotlib` | Plotting | N/A (optional) |

## Risk Considerations

- **Threshold Sensitivity:** Network structure changes with threshold choice
- **Estimation Error:** Correlation estimates noisy with limited data
- **Non-Stationarity:** Correlation structure changes over time
- **Computational Cost:** O(n²) for n assets

## Performance Expectations

- **Return Enhancement:** Marginal (0-2% annual)
- **Risk Reduction:** 10-25% volatility reduction
- **Drawdown Improvement:** 15-30% reduction
- **Sharpe Improvement:** 0.1-0.3
- **Best Use:** Combined with alpha signals

---

**File:** `src/ordinis/engines/signalcore/models/network_parity.py`
**Status:** ✅ Complete
