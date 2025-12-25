# HMM Regime Switching Strategy

---

**Title:** HMM Regime Switching
**Description:** Uses Hidden Markov Model to detect market regimes and adapt strategy accordingly
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** regime-detection, HMM, hidden-markov-model, regime-switching, adaptive
**References:** Baum & Petrie (1966), Hamilton (1989)

---

## Overview

The HMM Regime strategy uses a Hidden Markov Model to identify latent market regimes (Bull, Bear, Neutral). Each regime has distinct return and volatility characteristics. The strategy adapts its behavior—applying momentum in trending regimes and mean reversion in neutral regimes.

## Mathematical Basis

### Hidden Markov Model

**Hidden States:** $S = \{s_1, s_2, ..., s_K\}$ (K regimes)

**Observations:** $O = \{o_1, o_2, ..., o_T\}$ (returns)

**Model Components:**

1. **Initial Distribution:** $\pi_i = P(q_1 = s_i)$

2. **Transition Matrix:** $A_{ij} = P(q_{t+1} = s_j | q_t = s_i)$

3. **Emission Distribution:** $B_i(o) = P(o_t = o | q_t = s_i) \sim N(\mu_i, \sigma_i^2)$

### Viterbi Algorithm

Finds most likely state sequence:

$$
\delta_t(j) = \max_{i} [\delta_{t-1}(i) \cdot A_{ij}] \cdot B_j(o_t)
$$

### Forward Algorithm

Calculates state probabilities:

$$
\alpha_t(j) = \left[\sum_i \alpha_{t-1}(i) \cdot A_{ij}\right] \cdot B_j(o_t)
$$

### Baum-Welch (EM Algorithm)

Estimates parameters from data:

1. **E-step:** Calculate $\gamma_t(i) = P(q_t = s_i | O, \theta)$
2. **M-step:** Update $\mu_i, \sigma_i, A_{ij}, \pi_i$

## Regime Characteristics

| Regime | Return ($\mu$) | Volatility ($\sigma$) | Strategy |
|--------|----------------|----------------------|----------|
| **BULL** (2) | High positive | Moderate | Momentum long |
| **NEUTRAL** (1) | Near zero | Low | Mean reversion |
| **BEAR** (0) | Negative | High | Reduced exposure / Short |

## Signal Logic

### Bull Regime

| Condition | Signal |
|-----------|--------|
| Trend ≥ 0 AND RSI > 50 | **LONG** |
| Otherwise | **HOLD** |

### Bear Regime

| Condition | Signal |
|-----------|--------|
| Trend < 0 AND RSI < 45 | **SHORT** (0.5× size) |
| Otherwise | **HOLD** (reduced exposure) |

### Neutral Regime

| Condition | Signal |
|-----------|--------|
| RSI < 35 | **LONG** (mean reversion) |
| RSI > 65 | **SHORT** (mean reversion) |
| Otherwise | **HOLD** |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_regimes` | 3 | Number of hidden states |
| `lookback` | 252 | Days for HMM training |
| `retrain_frequency` | 21 | Days between retraining |
| `return_period` | 5 | Returns period for observations |
| `rsi_period` | 14 | RSI calculation period |
| `rsi_oversold` | 35 | Oversold threshold |
| `rsi_overbought` | 65 | Overbought threshold |
| `bear_position_mult` | 0.5 | Position reduction in bear |

## Edge Source

1. **Regime Awareness:** Different strategies work in different regimes
2. **Adaptive Behavior:** Automatically switches between momentum and mean reversion
3. **Risk Management:** Reduces exposure in bear markets
4. **Probability-Based:** Uses regime probability for confidence

## Regime Analysis

```python
from ordinis.engines.signalcore.models.hmm_regime import analyze_regimes

result_df = analyze_regimes(df, n_regimes=3)

# Regime statistics
for regime in [0, 1, 2]:
    mask = result_df["regime"] == regime
    returns = result_df.loc[mask, "returns"]
    print(f"Regime {regime}:")
    print(f"  Mean return: {returns.mean():.4f}")
    print(f"  Volatility: {returns.std():.4f}")
    print(f"  Days: {mask.sum()}")
```

## Implementation Notes

```python
from ordinis.engines.signalcore.models import HMMRegimeModel
from ordinis.engines.signalcore.core.model import ModelConfig

config = ModelConfig(
    model_id="hmm_regime",
    model_type="regime",
    parameters={
        "n_regimes": 3,
        "lookback": 252,
        "bear_position_mult": 0.5,
    }
)

model = HMMRegimeModel(config)
signal = await model.generate(symbol, df, timestamp)

# Access regime info
print(f"Current regime: {signal.metadata['regime']}")
print(f"Regime probability: {signal.metadata['regime_probability']:.2%}")
print(f"Regime duration: {signal.metadata['regime_duration']} bars")
print(f"P(transition to bull): {signal.metadata['transition_prob_bull']:.2%}")
```

## Transition Matrix Example

Typical estimated transition matrix:

|  | Bull | Neutral | Bear |
|--|------|---------|------|
| **Bull** | 0.92 | 0.06 | 0.02 |
| **Neutral** | 0.08 | 0.85 | 0.07 |
| **Bear** | 0.05 | 0.15 | 0.80 |

**Interpretation:** Regimes are sticky (high diagonal values), with gradual transitions.

## Dependencies

| Package | Purpose | Fallback |
|---------|---------|----------|
| `hmmlearn` | HMM estimation | `SimpleHMM` (built-in) |

## Risk Considerations

- **Regime Lag:** HMM detects regimes with delay (uses historical data)
- **Misclassification:** Wrong regime leads to wrong strategy
- **Overfitting:** Too many regimes overfit historical data
- **Transition Uncertainty:** Regime changes not predicted, only detected

## Performance Expectations

- **Win Rate:** 50-60%
- **Profit Factor:** 1.4-2.0
- **Drawdown Reduction:** 15-30% vs buy-and-hold
- **Best Conditions:** Markets with distinct regime patterns
- **Worst Conditions:** Rapid regime switching

---

**File:** `src/ordinis/engines/signalcore/models/hmm_regime.py`
**Status:** ✅ Complete
