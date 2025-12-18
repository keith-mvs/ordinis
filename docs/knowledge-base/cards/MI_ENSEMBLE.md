# Mutual Information Ensemble Strategy

---

**Title:** MI-Weighted Ensemble
**Description:** Combines multiple alpha signals weighted by mutual information with forward returns
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** ensemble, mutual-information, information-theory, meta-strategy, signal-combination
**References:** Shannon (1948), Cover & Thomas (2006)

---

## Overview

The MI Ensemble strategy combines multiple trading signals using weights derived from their mutual information with forward returns. Unlike correlation-based weighting, MI captures non-linear dependencies, identifying signals with true predictive power.

## Mathematical Basis

### Mutual Information

Measures shared information between signal $X$ and returns $Y$:

$$
I(X; Y) = H(X) + H(Y) - H(X, Y)
$$

Or equivalently:

$$
I(X; Y) = \sum_{x,y} p(x,y) \log\frac{p(x,y)}{p(x)p(y)}
$$

Where:
- $H(X)$ = entropy of signal
- $H(Y)$ = entropy of returns
- $H(X,Y)$ = joint entropy

### Normalized MI

Scale to [0, 1] range:

$$
\text{NMI}(X; Y) = \frac{2 \cdot I(X; Y)}{H(X) + H(Y)}
$$

### Histogram-Based Estimation

For continuous variables:

```python
# Discretize to bins
x_bins = np.digitize(X, np.percentile(X, np.linspace(0, 100, n_bins)))
y_bins = np.digitize(Y, np.percentile(Y, np.linspace(0, 100, n_bins)))

# Calculate joint and marginal probabilities
joint_prob = histogram2d(x_bins, y_bins) / N
marginal_x = joint_prob.sum(axis=1)
marginal_y = joint_prob.sum(axis=0)

# Compute MI
MI = sum(p_xy * log(p_xy / (p_x * p_y)) for all x, y where p_xy > 0)
```

## Signal Library

Default signals included:

| Signal | Calculation | Captures |
|--------|-------------|----------|
| RSI | 14-period RSI normalized | Mean reversion |
| Momentum | 20-day return z-score | Trend |
| Mean Reversion | -Z-score of price vs MA | Reversion |
| Vol Breakout | Position in rolling range | Breakout |
| Trend Strength | ADX × direction | Trend quality |

### Custom Signals

```python
from ordinis.engines.signalcore.models.mi_ensemble import SignalDefinition

def compute_custom_signal(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change(10).rolling(5).mean()

custom_signal = SignalDefinition(
    name="custom_mom",
    compute=compute_custom_signal
)
```

## Signal Logic

### Ensemble Calculation

$$
\text{Ensemble} = \sum_{i=1}^{n} w_i \cdot \text{signal}_i
$$

Where weights $w_i$ are normalized MI values.

### Entry Conditions

| Ensemble Value | Signal Agreement | Action |
|----------------|------------------|--------|
| `> 0.3` | ≥2 signals positive | **LONG** |
| `< -0.3` | ≥2 signals negative | **SHORT** |
| Otherwise | - | **HOLD** |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mi_lookback` | 252 | Days for MI calculation |
| `mi_bins` | 10 | Bins for discretization |
| `forward_period` | 5 | Forward return period |
| `min_weight` | 0.0 | Minimum signal weight |
| `max_weight` | 0.5 | Maximum signal weight (cap) |
| `recalc_frequency` | 21 | Days between weight recalc |
| `ensemble_threshold` | 0.3 | Threshold for entry |
| `min_signals_agree` | 2 | Required agreeing signals |

## Edge Source

1. **Non-Linear Detection:** MI captures non-linear signal-return relationships
2. **Dynamic Weighting:** Signals contributing predictive power get higher weights
3. **Redundancy Handling:** Correlated signals share weight budget
4. **Adaptive:** Weights recalibrated as market conditions change

## MI Analysis

```python
from ordinis.engines.signalcore.models.mi_ensemble import analyze_signal_mi

# Analyze MI across different forward periods
analysis = analyze_signal_mi(
    df,
    forward_periods=[1, 5, 10, 21]
)

print(analysis.pivot(index="signal", columns="forward_period", values="nmi"))
```

Example output:

| Signal | 1-day | 5-day | 10-day | 21-day |
|--------|-------|-------|--------|--------|
| rsi | 0.02 | 0.05 | 0.08 | 0.06 |
| momentum | 0.01 | 0.03 | 0.07 | 0.12 |
| mean_reversion | 0.03 | 0.06 | 0.05 | 0.03 |
| vol_breakout | 0.02 | 0.04 | 0.06 | 0.08 |

## Implementation Notes

```python
from ordinis.engines.signalcore.models import MIEnsembleModel
from ordinis.engines.signalcore.core.model import ModelConfig

config = ModelConfig(
    model_id="mi_ensemble",
    model_type="ensemble",
    parameters={
        "mi_lookback": 252,
        "forward_period": 5,
        "ensemble_threshold": 0.3,
    }
)

model = MIEnsembleModel(config)
signal = await model.generate(symbol, df, timestamp)

# Check weights
print(f"Signal weights: {signal.metadata['weights']}")
print(f"Signal values: {signal.metadata['signal_values']}")
print(f"Ensemble value: {signal.metadata['ensemble_value']:.3f}")
```

## Risk Considerations

- **Data Mining Bias:** MI optimization can overfit to historical patterns
- **Estimation Error:** MI estimation requires substantial data (252+ observations)
- **Regime Dependence:** Optimal weights change with market regime

## Performance Expectations

- **Win Rate:** 52-58%
- **Profit Factor:** 1.3-1.7
- **Sharpe Ratio:** 0.8-1.5
- **Best Conditions:** Stable market with consistent signal relationships
- **Worst Conditions:** Rapid regime changes

---

**File:** `src/ordinis/engines/signalcore/models/mi_ensemble.py`
**Status:** ✅ Complete
