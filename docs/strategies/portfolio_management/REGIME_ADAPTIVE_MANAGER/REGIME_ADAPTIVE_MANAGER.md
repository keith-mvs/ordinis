# Regime Adaptive Manager

---

**Title:** Dynamic Regime-Based Strategy Orchestration
**Description:** Meta-strategy that detects market regime and allocates across strategy pools
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** regime, adaptive, ensemble, allocation, meta-strategy
**References:** Hamilton (1989) Regime Switching, Ang & Bekaert (2002)

---

## Overview

The Regime Adaptive Manager is a meta-strategy that dynamically allocates capital across three strategy pools (Trend-Following, Mean-Reversion, Volatility) based on detected market regime. It uses Hidden Markov Model-inspired regime detection to classify markets as BULL, BEAR, SIDEWAYS, VOLATILE, or TRANSITIONAL, then applies regime-specific weight allocations.

## Architecture

```
Market Data → RegimeDetector → Regime Classification
                     ↓
              Regime Weights
                     ↓
    ┌────────────────┼────────────────┐
    ↓                ↓                ↓
TrendFollowing   MeanReversion   Volatility
  Ensemble         Ensemble        Ensemble
    ↓                ↓                ↓
    └────────────────┼────────────────┘
                     ↓
           Weighted Signal Aggregation
                     ↓
              Position Sizing
                     ↓
            Final Trading Signal
```

## Mathematical Basis

### Regime Detection

Multi-factor classification using:

$$
\text{Regime} = f(\text{TrendStrength}, \text{Volatility}, \text{Momentum})
$$

**Trend Strength:** ADX-based, smoothed over $n$ periods
**Volatility:** ATR percentile over historical distribution
**Momentum:** RSI deviation from neutral (50)

### Regime Weight Allocation

$$
\text{Signal}_{\text{final}} = \sum_{i} w_i^{(\text{regime})} \cdot \text{Signal}_i
$$

Where $w_i^{(\text{regime})}$ are regime-specific weights.

### Position Size Scaling

$$
\text{Size} = \text{BaseSize} \times \text{Confidence} \times (1 - \text{DrawdownPenalty})
$$

## Default Regime Weights

| Regime | Trend | Mean-Rev | Volatility | Cash |
|--------|-------|----------|------------|------|
| **BULL** | 80% | 10% | 5% | 5% |
| **BEAR** | 20% | 20% | 10% | 50% |
| **SIDEWAYS** | 10% | 70% | 10% | 10% |
| **VOLATILE** | 20% | 20% | 40% | 20% |
| **TRANSITIONAL** | 15% | 15% | 10% | 60% |

## Strategy Pools

### Trend-Following Ensemble
- Moving Average Crossover (50/200)
- Parabolic SAR
- ADX Breakout

### Mean-Reversion Ensemble
- RSI Oversold/Overbought
- Bollinger Bands Fade
- IBS (Internal Bar Strength)

### Volatility Trading Ensemble
- ATR Trailing Strategy
- VIX Mean-Reversion
- Straddle/Strangle Triggers

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `regime_lookback` | 200 | Bars for regime detection |
| `regime_confirm_bars` | 3 | Bars to confirm regime change |
| `use_ensemble` | True | Use ensembles vs single best |
| `base_position_size` | 0.8 | Base allocation when confident |
| `min_position_size` | 0.3 | Minimum during transitions |
| `confidence_scaling` | True | Scale by regime confidence |
| `max_drawdown_threshold` | 0.10 | Risk-off at 10% drawdown |
| `volatility_scaling` | True | Reduce in high volatility |

## Signal Logic

| Condition | Action |
|-----------|--------|
| Regime change detected | Adjust weights smoothly |
| Confidence > 80% | Full weight allocation |
| Confidence 50-80% | Partial allocation |
| Confidence < 50% | High cash, minimal exposure |
| Drawdown > threshold | Force risk-off mode |

## Edge Source

1. **Regime Persistence:** Markets exhibit regime clustering
2. **Strategy Suitability:** Different strategies excel in different conditions
3. **Risk Management:** Automatic de-risking in uncertain regimes
4. **Ensemble Effect:** Diversification across uncorrelated strategies

## Risk Considerations

- **Regime Detection Lag:** Classification happens after regime established
- **Whipsaw Transitions:** Frequent regime changes erode capital
- **Over-Optimization:** Regime weights can be overfit
- **Correlation Spikes:** Strategies may correlate in crisis

## Implementation Notes

```python
# Regime detection
regime_signal = detector.detect(data)
current_regime = regime_signal.regime
confidence = regime_signal.confidence

# Get weights for current regime
weights = regime_weights[current_regime]

# Generate signals from each pool
trend_signal = trend_ensemble.update(data)
reversion_signal = reversion_ensemble.update(data)
volatility_signal = volatility_ensemble.update(data)

# Weighted aggregation
final_signal = (
    weights.trend_following * trend_signal.strength +
    weights.mean_reversion * reversion_signal.strength +
    weights.volatility * volatility_signal.strength
)

# Position sizing with confidence scaling
size = base_size * confidence * (1 - drawdown_penalty)
size = max(min_size, min(1.0, size))
```

## Performance Expectations

- **Win Rate:** 45-55% (regime-dependent)
- **Profit Factor:** 1.2-1.8
- **Max Drawdown:** 15-25% (with risk management)
- **Best Conditions:** Clear, persistent regimes
- **Worst Conditions:** Rapid regime switching

## Regime Detection Metrics

| Metric | BULL | BEAR | SIDEWAYS | VOLATILE |
|--------|------|------|----------|----------|
| ADX | > 25 | > 25 | < 20 | Variable |
| Trend | Up | Down | Flat | Erratic |
| Vol Percentile | < 60 | Any | < 40 | > 70 |
| Momentum (RSI) | > 55 | < 45 | 45-55 | Variable |

## Transition Handling

Smooth transitions prevent whipsaws:

```python
# Exponential smoothing for weights
new_weights = 0.7 * old_weights + 0.3 * target_weights

# Minimum bars in regime before full allocation
if days_in_regime < regime_confirm_bars:
    size_multiplier *= 0.5  # Half size during confirmation
```

## Lopez de Prado Integration

For regime-adaptive testing:
1. **Purged K-Fold CV:** Separate regimes across folds
2. **Regime-Specific Metrics:** Calculate Sharpe per regime
3. **Transition Analysis:** Measure performance during switches
4. **PBO per Regime:** Check overfitting in each regime

---

**File:** `src/ordinis/application/strategies/regime_adaptive/adaptive_manager.py`
**Dependencies:** `regime_detector.py`, `mean_reversion.py`, `trend_following.py`, `volatility_trading.py`
**Status:** ✅ Production Ready
