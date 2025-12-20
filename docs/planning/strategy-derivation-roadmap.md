# Strategy Derivation Roadmap

**Based on:** Ordinis Knowledge Base - Foundations & Signals Domains
**Date:** 2025-12-17
**Status:** Research & Development Queue

---

## Executive Summary

After reviewing the foundations (mathematical methods, stochastic processes, ML, optimization) and signals (technical, quantitative, sentiment, advanced) knowledge base, we have identified **8 implementable strategies** that leverage the existing documentation and infrastructure.

Each strategy is ranked by:
- **Complexity**: Implementation difficulty (1-5)
- **Expected Edge**: Theoretical alpha potential (1-5)
- **Data Requirements**: What data sources are needed
- **Dependencies**: Which existing components can be reused

---

## Strategy 1: Cointegrated Pairs with Ornstein-Uhlenbeck Optimization

### Theoretical Foundation
- [cointegration.md](../foundations/cointegration.md): Engle-Granger and Johansen tests
- [mean-reverting-processes.md](../foundations/mean-reverting-processes.md): OU parameter estimation
- [pairs-trading.md](../signals/quantitative/statistical-arbitrage/pairs-trading.md): Z-score trading

### Concept
Instead of using simple z-score thresholds, fit an Ornstein-Uhlenbeck process to the spread and use the **half-life** to dynamically set entry/exit thresholds and position holding periods.

### Signal Logic
```python
# Estimate OU parameters from spread
params = estimate_ou_parameters(spread)
theta = params['theta']      # Mean reversion speed
half_life = params['half_life_days']

# Dynamic thresholds based on half-life
entry_z = 2.0 * np.sqrt(1 - np.exp(-2*theta))  # ~2 std of stationary distribution
exit_z = 0.5 * entry_z

# Trade if z-score exceeds entry AND half-life < max_holding
if abs(zscore) > entry_z and half_life < 10:
    signal = -np.sign(zscore)  # Mean revert
```

### Key Differentiator
- **Adapts to spread dynamics**: Fast-reverting spreads get tighter thresholds
- **Avoids slow-reverting pairs**: Half-life filter prevents capital lock-up
- **Better risk management**: Use half-life to set stop-loss timing

### Implementation Priority
| Metric | Score |
|--------|-------|
| Complexity | 3/5 |
| Expected Edge | 4/5 |
| Data Requirements | Daily prices for 100+ stocks |
| Dependencies | `regime_detector.py`, `cointegration` tests |

---

## Strategy 2: Kalman Filter Trend + Mean Reversion Hybrid

### Theoretical Foundation
- [signal-processing-for-finance.md](../foundations/signal-processing-for-finance.md): Kalman filter
- [kalman-trend-signal.md](../signals/advanced/kalman-trend-signal.md): Trend extraction
- [atr-optimized-rsi.py](../../src/ordinis/engines/signalcore/models/atr_optimized_rsi.py): Mean reversion

### Concept
Use a Kalman filter to decompose price into **trend** and **residual**. Trade mean reversion on the residual while respecting the trend direction.

### Signal Logic
```python
# Kalman filter outputs
kalman = kalman_trend(prices, q=1e-6, r=1e-3)
trend_slope = kalman['trend_slope']
residual_z = kalman['residual_z']

# Only mean-revert residual when aligned with trend
if trend_slope > 0 and residual_z < -2.0:
    signal = BUY  # Oversold in uptrend
elif trend_slope < 0 and residual_z > 2.0:
    signal = SELL  # Overbought in downtrend
else:
    signal = HOLD
```

### Key Differentiator
- **Avoids counter-trend trades**: Only mean-reverts in trend direction
- **Adaptive noise filtering**: Kalman adapts to volatility regimes
- **Confidence weighting**: Use `1/state_variance` for position sizing

### Implementation Priority
| Metric | Score |
|--------|-------|
| Complexity | 3/5 |
| Expected Edge | 4/5 |
| Data Requirements | Intraday data (existing) |
| Dependencies | `filterpy`, ATR model |

---

## Strategy 3: GARCH Volatility Breakout

### Theoretical Foundation
- [garch-models-volatility-modeling.md](../foundations/garch-models-volatility-modeling.md): GARCH(1,1)
- [atr.md](../signals/technical/volatility/atr.md): Volatility breakouts

### Concept
Use GARCH to forecast next-period volatility. When realized volatility **exceeds** GARCH forecast by >2σ, a volatility breakout is occurring - trade in the direction of the move.

### Signal Logic
```python
from arch import arch_model

# Fit GARCH(1,1) to returns
model = arch_model(returns[-252:], vol='Garch', p=1, q=1)
fitted = model.fit(disp='off')
forecast_vol = np.sqrt(fitted.forecast(horizon=1).variance.iloc[-1, 0])

# Compare to realized
realized_vol = returns[-5:].std() * np.sqrt(252)
vol_ratio = realized_vol / forecast_vol

# Volatility breakout signal
if vol_ratio > 2.0:
    direction = np.sign(returns[-1])  # Trade in direction of move
    signal = direction
else:
    signal = HOLD
```

### Key Differentiator
- **Catches regime changes**: GARCH lags during vol spikes
- **Momentum in vol breakouts**: Volatility expansion often continues
- **Risk management**: Use GARCH forecast for stop distances

### Implementation Priority
| Metric | Score |
|--------|-------|
| Complexity | 2/5 |
| Expected Edge | 3/5 |
| Data Requirements | Daily returns |
| Dependencies | `arch` library |

---

## Strategy 4: Hidden Markov Model Regime Switching

### Theoretical Foundation
- [regime-detection.md](../foundations/regime-detection.md): HMM fitting
- [risk-regimes.md](../signals/sentiment/macro-events/risk-regimes.md): Risk-on/off
- [regime_detector.py](../../src/ordinis/engines/signalcore/regime_detector.py): Existing detector

### Concept
Train a 3-state HMM (Bull, Bear, Neutral) on returns. Switch between strategies based on current regime probability:
- **Bull regime**: Trend-following / momentum
- **Bear regime**: Mean reversion / hedging
- **Neutral regime**: Reduce exposure / pairs

### Signal Logic
```python
from hmmlearn.hmm import GaussianHMM

# Fit 3-state HMM
hmm = GaussianHMM(n_components=3, covariance_type='full')
hmm.fit(returns.reshape(-1, 1))
state_probs = hmm.predict_proba(returns[-1:].reshape(-1, 1))[0]

# Identify regime by mean return
regime_means = hmm.means_.flatten()
bull_idx = np.argmax(regime_means)
bear_idx = np.argmin(regime_means)

# Strategy selection
if state_probs[bull_idx] > 0.6:
    use_strategy = 'momentum'
elif state_probs[bear_idx] > 0.6:
    use_strategy = 'mean_reversion'
else:
    use_strategy = 'reduce_exposure'
```

### Key Differentiator
- **Probabilistic**: Uses probabilities, not hard states
- **Adaptive**: Re-estimates as new data arrives
- **Strategy rotation**: Different strategy per regime

### Implementation Priority
| Metric | Score |
|--------|-------|
| Complexity | 4/5 |
| Expected Edge | 4/5 |
| Data Requirements | Daily returns, multiple assets |
| Dependencies | `hmmlearn`, strategy ensemble |

---

## Strategy 5: Mutual Information Weighted Signal Ensemble

### Theoretical Foundation
- [information-theory.md](../foundations/advanced-mathematics/information-theory.md): Entropy, MI
- [mi-weighted-ensemble.md](../signals/advanced/mi-weighted-ensemble.md): MI weighting

### Concept
Combine multiple signals (RSI, Stochastic, momentum, volatility) by weighting each proportionally to its **mutual information** with future returns. Down-weight redundant signals.

### Signal Logic
```python
from sklearn.feature_selection import mutual_info_regression

# Base signals: RSI, Stochastic, Momentum, Vol
signals = pd.DataFrame({
    'rsi': rsi_values,
    'stoch': stochastic_values,
    'mom': momentum_values,
    'vol': volatility_values
})
target = future_returns

# Compute MI weights (rolling)
mi = mutual_info_regression(signals, target)
weights = mi / mi.sum()

# Penalize redundancy
corr = signals.corr().abs()
redundancy = corr.mean(axis=1)
weights = weights / (1 + redundancy)
weights = weights / weights.sum()

# Ensemble signal
ensemble = (signals.apply(zscore) * weights).sum(axis=1)
```

### Key Differentiator
- **Information-theoretic**: Captures non-linear dependencies
- **Redundancy penalty**: Avoids double-counting correlated signals
- **Adaptive weighting**: Recomputes on rolling basis

### Implementation Priority
| Metric | Score |
|--------|-------|
| Complexity | 3/5 |
| Expected Edge | 4/5 |
| Data Requirements | Multiple signal outputs |
| Dependencies | `sklearn`, base signal models |

---

## Strategy 6: EVT-Gated Risk Management

### Theoretical Foundation
- [extreme-value-theory.md](../foundations/advanced-mathematics/extreme-value-theory.md): GPD, VaR
- [evt-tail-alert.md](../signals/advanced/evt-tail-alert.md): Tail alerts

### Concept
Use Generalized Pareto Distribution to estimate tail risk. When **tail VaR** or **shape parameter ξ** exceeds thresholds, reduce position sizes or activate hedges.

### Signal Logic
```python
from scipy.stats import genpareto

# Fit GPD to loss tail
losses = -returns
u = np.percentile(losses, 95)
tail = losses[losses > u] - u
xi, _, scale = genpareto.fit(tail, floc=0)

# Compute 99% VaR
var_99 = u + genpareto.ppf(0.99, xi, loc=0, scale=scale)

# Tail alert
tail_alert = (var_99 > 0.03) or (xi > 0.3)  # 3% VaR or heavy tail

# Risk gate
if tail_alert:
    position_multiplier = 0.5  # Reduce size by 50%
else:
    position_multiplier = 1.0
```

### Key Differentiator
- **Tail-aware**: Goes beyond normal VaR assumptions
- **Dynamic hedging**: Increases protection when tails fatten
- **Works with any strategy**: Overlay on existing signals

### Implementation Priority
| Metric | Score |
|--------|-------|
| Complexity | 2/5 |
| Expected Edge | 3/5 (risk reduction) |
| Data Requirements | Historical returns (252+ days) |
| Dependencies | `scipy.stats` |

---

## Strategy 7: Network Centrality Risk Parity

### Theoretical Foundation
- [network-theory.md](../foundations/advanced-mathematics/network-theory.md): Correlation networks
- [network-risk-parity.md](../signals/advanced/network-risk-parity.md): Centrality weighting

### Concept
Build a correlation network of portfolio assets. Reduce allocation to **highly central** assets (systemic risk) and increase allocation to **peripheral** assets (diversification).

### Signal Logic
```python
import networkx as nx

# Build correlation network
corr = returns.corr()
G = nx.from_pandas_adjacency(corr.abs())

# Compute eigenvector centrality
centrality = nx.eigenvector_centrality_numpy(G)

# Inverse-centrality weights (less central = more weight)
weights = {k: 1/(v + 0.01) for k, v in centrality.items()}
total = sum(weights.values())
weights = {k: v/total for k, v in weights.items()}

# Compare to current weights for rebalance signal
current = current_portfolio_weights
target = pd.Series(weights)
rebalance_signal = target - current
```

### Key Differentiator
- **Systemic risk aware**: Avoids concentrated exposure
- **Diversification maximizing**: Tilts to uncorrelated assets
- **Regime adaptive**: Network structure changes with market

### Implementation Priority
| Metric | Score |
|--------|-------|
| Complexity | 3/5 |
| Expected Edge | 3/5 |
| Data Requirements | Multi-asset returns |
| Dependencies | `networkx` |

---

## Strategy 8: Cross-Timeframe Momentum with Stochastic Confirmation

### Theoretical Foundation
- [momentum-factor.md](../signals/quantitative/factor-investing/momentum-factor.md): 12-1 momentum
- [stochastic.md](../signals/technical/oscillators/stochastic.md): %K/%D crossovers
- [multi-timeframe.md](../signals/technical/advanced/multi-timeframe.md): Timeframe alignment

### Concept
Combine daily momentum (12-1 month) with intraday stochastic oscillator for entry timing. Only enter momentum trades when stochastic confirms via crossover in the momentum direction.

### Signal Logic
```python
# Daily momentum (skip most recent month)
momentum_12_1 = (price.shift(21) / price.shift(252)) - 1

# Intraday stochastic (14-period, 5-min bars)
stoch_k = stochastic(high, low, close, period=14)
stoch_d = stoch_k.rolling(3).mean()

# Momentum filter
is_winner = momentum_12_1 > momentum_12_1.quantile(0.8)
is_loser = momentum_12_1 < momentum_12_1.quantile(0.2)

# Entry timing
bullish_cross = (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))
bearish_cross = (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))

# Combined signal
if is_winner and bullish_cross and stoch_k < 30:
    signal = BUY
elif is_loser and bearish_cross and stoch_k > 70:
    signal = SELL
```

### Key Differentiator
- **Multi-timeframe confluence**: Daily direction + intraday timing
- **Better entry prices**: Stochastic confirms pullback entries
- **Momentum + mean reversion**: Best of both worlds

### Implementation Priority
| Metric | Score |
|--------|-------|
| Complexity | 2/5 |
| Expected Edge | 4/5 |
| Data Requirements | Daily + intraday |
| Dependencies | Existing ATR model |

---

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1-2)
1. **Strategy 3: GARCH Volatility Breakout** - Simple to implement, good edge
2. **Strategy 6: EVT Risk Gate** - Overlay on existing ATR-RSI strategy
3. **Strategy 8: Multi-Timeframe Momentum** - Leverages existing infrastructure

### Phase 2: Core Development (Week 3-4)
4. **Strategy 2: Kalman Filter Hybrid** - High edge, moderate complexity
5. **Strategy 1: OU Pairs Trading** - Classic quant strategy
6. **Strategy 5: MI Ensemble** - Meta-strategy combining all signals

### Phase 3: Advanced (Week 5+)
7. **Strategy 4: HMM Regime Switching** - Full strategy rotation
8. **Strategy 7: Network Risk Parity** - Portfolio-level optimization

---

## Integration with Existing Infrastructure

### Reusable Components
| Component | Used By Strategies |
|-----------|-------------------|
| `regime_detector.py` | 1, 4, 8 |
| `ATROptimizedRSIModel` | 2, 3, 6 |
| `broker.py` | All |
| `strategy_loader.py` | All |
| `live_trading.py` | All |

### New Dependencies Required
| Library | Used By |
|---------|---------|
| `hmmlearn` | Strategy 4 |
| `arch` | Strategy 3 |
| `filterpy` | Strategy 2 |
| `networkx` | Strategy 7 |

### Config Structure
Each strategy will have its own YAML config under `configs/strategies/`:
```
configs/strategies/
├── atr_optimized_rsi.yaml     # Existing
├── ou_pairs_trading.yaml      # Strategy 1
├── kalman_hybrid.yaml         # Strategy 2
├── garch_breakout.yaml        # Strategy 3
├── hmm_regime.yaml            # Strategy 4
├── mi_ensemble.yaml           # Strategy 5
├── evt_risk_gate.yaml         # Strategy 6
├── network_parity.yaml        # Strategy 7
└── mtf_momentum.yaml          # Strategy 8
```

---

## Success Metrics

Each strategy will be evaluated against:

1. **Walk-Forward Sharpe Ratio** > 0.5
2. **Max Drawdown** < 25%
3. **Win Rate** > 55%
4. **Profit Factor** > 1.5
5. **Robustness**: Profitable in >80% of out-of-sample periods

---

*Document generated by Ordinis Quantitative Research*
