# Ordinis Multi-Model Alternatives Framework

System-level configuration for model selection across all functional components.

**Version**: 1.0.0
**Last Updated**: 2025-12-08
**Status**: Specification

---

## Framework Overview

This specification defines the model architecture for each functional role in the Ordinis trading system. Each component supports a primary model, ML/DL alternatives, and classical statistical substitutes to enable graceful degradation, A/B testing, and resource-aware deployment.

---

## Component Definitions

### 1. Price Forecasting

**Role**: Generate directional price predictions and expected return estimates for signal generation.

**Component ID**: `forecasting.price`

| Tier | Model | Implementation |
|------|-------|----------------|
| **Primary** | LSTM | Sequential pattern recognition with memory cells |
| **ML/DL Alt 1** | Transformer-XL | Long-range dependency modeling with recurrence |
| **ML/DL Alt 2** | Informer | Efficient attention for long sequences |
| **ML/DL Alt 3** | Temporal Fusion Transformer (TFT) | Multi-horizon with interpretable attention |
| **ML/DL Alt 4** | GRU | Lightweight sequential modeling |
| **Classical 1** | ARIMA/SARIMA | Stationary time series with seasonality |
| **Classical 2** | XGBoost/LightGBM | Feature-engineered gradient boosting |
| **Classical 3** | Random Forests | Ensemble decision trees |

**Selection Guidance**:
- Use **LSTM** as default for intraday price patterns
- Switch to **Transformer-XL/Informer** when modeling dependencies >100 timesteps
- Use **TFT** when interpretability of temporal attention is required
- Fall back to **ARIMA** when data is limited (<1000 samples) or stationarity is confirmed
- Use **XGBoost** when feature engineering provides strong predictors

---

### 2. Volatility Estimation

**Role**: Estimate conditional volatility for position sizing, risk metrics, and option pricing.

**Component ID**: `forecasting.volatility`

| Tier | Model | Implementation |
|------|-------|----------------|
| **Primary** | GARCH(1,1) | Symmetric volatility clustering |
| **ML/DL Alt 1** | EGARCH | Asymmetric response to positive/negative shocks |
| **ML/DL Alt 2** | TGARCH (GJR-GARCH) | Threshold-based asymmetry |
| **ML/DL Alt 3** | Neural Volatility Model | Deep learning for nonlinear dynamics |
| **Classical 1** | Stochastic Volatility (SV) | Latent volatility process |
| **Classical 2** | Bayesian SV | Uncertainty-aware volatility estimation |

**Selection Guidance**:
- Use **GARCH(1,1)** as baseline for symmetric volatility patterns
- Switch to **EGARCH/TGARCH** when leverage effect is present (equities exhibit asymmetric vol)
- Use **Neural Volatility** when volatility exhibits regime-dependent nonlinearity
- Use **Bayesian SV** when uncertainty quantification is critical for risk decisions

---

### 3. Sentiment Analysis

**Role**: Extract market sentiment from news, filings, social media, and alternative text data.

**Component ID**: `signals.sentiment`

| Tier | Model | Implementation |
|------|-------|----------------|
| **Primary** | FinBERT | Finance-domain pretrained transformer |
| **ML/DL Alt 1** | BloombergGPT | Large-scale financial LLM |
| **ML/DL Alt 2** | RoBERTa-finance | Robustly optimized BERT for finance |
| **ML/DL Alt 3** | DistilBERT | Lightweight distilled transformer |
| **Classical 1** | Loughran-McDonald Lexicon | Finance-specific word polarity |
| **Classical 2** | Rule-Based Scoring | Pattern matching with domain rules |

**Selection Guidance**:
- Use **FinBERT** as default for earnings calls, SEC filings, news
- Use **BloombergGPT** when available for comprehensive financial understanding
- Use **DistilBERT** for high-throughput, latency-sensitive applications
- Fall back to **Loughran-McDonald** when explainability is required or compute is limited
- Use **Rule-Based** for real-time headline scanning with minimal latency

---

### 4. Trade Execution Optimization

**Role**: Optimize order execution to minimize market impact, slippage, and transaction costs.

**Component ID**: `execution.optimizer`

| Tier | Model | Implementation |
|------|-------|----------------|
| **Primary** | DQN (Deep Q-Network) | Value-based RL for discrete actions |
| **ML/DL Alt 1** | PPO (Proximal Policy Optimization) | Stable policy gradient method |
| **ML/DL Alt 2** | A3C (Asynchronous Advantage Actor-Critic) | Parallel actor-critic training |
| **ML/DL Alt 3** | SAC (Soft Actor-Critic) | Maximum entropy RL for exploration |
| **Classical 1** | VWAP | Volume-weighted average price benchmark |
| **Classical 2** | TWAP | Time-weighted average price benchmark |
| **Classical 3** | Almgren-Chriss | Optimal execution with market impact model |

**Selection Guidance**:
- Use **DQN** for learning adaptive execution in stable market conditions
- Switch to **PPO/SAC** when action space is continuous (price limits, timing)
- Use **A3C** when parallel training infrastructure is available
- Fall back to **VWAP/TWAP** for simple benchmark-tracking execution
- Use **Almgren-Chriss** when market impact is well-characterized and deterministic

---

### 5. Risk Control and Portfolio Management

**Role**: Quantify portfolio risk, allocate capital, and enforce risk constraints.

**Component ID**: `risk.portfolio`

| Tier | Model | Implementation |
|------|-------|----------------|
| **Primary** | Bayesian Risk Model | Uncertainty-aware risk estimation |
| **ML/DL Alt 1** | Neural Risk Model | Deep learning for tail risk |
| **ML/DL Alt 2** | Deep Portfolio Optimization | End-to-end neural allocation |
| **Classical 1** | Monte Carlo Simulation | Scenario-based risk assessment |
| **Classical 2** | CVaR (Conditional VaR) | Expected shortfall optimization |
| **Classical 3** | VaR (Value at Risk) | Threshold-based loss quantile |

**Selection Guidance**:
- Use **Bayesian Risk Model** when parameter uncertainty must inform decisions
- Use **Neural Risk Model** for capturing nonlinear tail dependencies
- Use **Deep Portfolio Optimization** for end-to-end differentiable allocation
- Fall back to **Monte Carlo** for stress testing and scenario analysis
- Use **CVaR** for regulatory compliance and robust optimization
- Use **VaR** for quick risk thresholds (note: underestimates tail risk)

---

## Orchestration Interface

### Model Selection Directive Format

```yaml
component: <component_id>
tier: primary | ml_alt_<n> | classical_<n>
model: <model_name>
reason: <selection_rationale>
fallback_chain: [<ordered_alternatives>]
```

### Example Directives

```yaml
# Default configuration
- component: forecasting.price
  tier: primary
  model: LSTM
  fallback_chain: [GRU, XGBoost, ARIMA]

- component: forecasting.volatility
  tier: ml_alt_1
  model: EGARCH
  reason: Equity markets exhibit leverage effect
  fallback_chain: [GARCH, Bayesian_SV]

- component: signals.sentiment
  tier: primary
  model: FinBERT
  fallback_chain: [DistilBERT, Loughran_McDonald]

- component: execution.optimizer
  tier: classical_1
  model: VWAP
  reason: Low-complexity baseline for paper trading
  fallback_chain: [TWAP, Almgren_Chriss]

- component: risk.portfolio
  tier: primary
  model: Bayesian_Risk_Model
  fallback_chain: [CVaR, Monte_Carlo, VaR]
```

---

## Fallback Policy

### Automatic Degradation Rules

1. **Compute Constraint**: If GPU unavailable, demote ML/DL models to classical alternatives
2. **Data Constraint**: If training data <1000 samples, prefer classical statistical models
3. **Latency Constraint**: If inference latency >100ms required, use lightweight alternatives
4. **Explainability Constraint**: If audit trail required, prefer classical or interpretable models

### Degradation Priority

```
Primary (ML/DL) --> ML/DL Alternative --> Classical --> Rule-Based/Heuristic
```

---

## Resource Requirements

| Component | Primary Model | GPU Required | Min Data | Inference Latency |
|-----------|---------------|--------------|----------|-------------------|
| Price Forecasting | LSTM | Recommended | 5000 bars | ~50ms |
| Volatility Estimation | GARCH | No | 500 bars | ~5ms |
| Sentiment Analysis | FinBERT | Yes | N/A | ~100ms |
| Execution Optimization | DQN | Yes | 10000 episodes | ~20ms |
| Risk/Portfolio | Bayesian | No | 252 days | ~200ms |

---

## Integration Points

### SignalCore Integration

```python
# Model selection at signal generation
signal_config = {
    "forecasting": {"model": "LSTM", "horizon": 5},
    "volatility": {"model": "EGARCH"},
    "sentiment": {"model": "FinBERT", "sources": ["news", "sec_filings"]},
}
```

### RiskGuard Integration

```python
# Risk model selection
risk_config = {
    "portfolio_model": "Bayesian_Risk_Model",
    "stress_test_model": "Monte_Carlo",
    "regulatory_model": "CVaR",
}
```

### FlowRoute Integration

```python
# Execution model selection
execution_config = {
    "optimizer": "VWAP",  # Paper trading default
    "production_optimizer": "DQN",  # Live trading
    "fallback": "TWAP",
}
```

---

## Version Compatibility

| Ordinis Version | Framework Version | Notes |
|-----------------|-------------------|-------|
| 1.x | 1.0.0 | Initial specification |

---

## References

- Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
- Goodfellow, I. et al. (2016). Deep Learning.
- Tsay, R. S. (2010). Analysis of Financial Time Series.
- Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio Transactions.
