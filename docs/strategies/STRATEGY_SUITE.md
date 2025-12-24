# Ordinis Strategy Suite

**Complete Trading Strategy Library**
**Version:** 1.0.0
**Last Updated:** 2025-12-17

---

## Overview

This document serves as the master index for all trading strategies implemented in the Ordinis platform. Strategies are organized by complexity tier and include implementation status, key parameters, and performance characteristics.

---

## Strategy Tiers

### Tier 1: Quick Wins (Low Complexity, Fast Implementation)

| Strategy | File | Status | Expected Edge | Complexity |
|----------|------|--------|---------------|------------|
| **ATR-Optimized RSI** | `atr_optimized_rsi.py` | ✅ Production | +60% (21d) | 2/5 |
| **GARCH Volatility Breakout** | `garch_breakout.py` | ✅ Complete | 3/5 | 2/5 |
| **EVT Risk Gate** | `evt_risk_gate.py` | ✅ Complete | 3/5 (risk) | 2/5 |
| **Multi-Timeframe Momentum** | `mtf_momentum.py` | ✅ Complete | 4/5 | 2/5 |

### Tier 2: Core Quantitative (Medium Complexity)

| Strategy | File | Status | Expected Edge | Complexity |
|----------|------|--------|---------------|------------|
| **Kalman Filter Hybrid** | `kalman_hybrid.py` | ✅ Complete | 4/5 | 3/5 |
| **OU Pairs Trading** | `ou_pairs.py` | ✅ Complete | 4/5 | 3/5 |
| **MI-Weighted Ensemble** | `mi_ensemble.py` | ✅ Complete | 4/5 | 3/5 |

### Tier 3: Advanced (High Complexity)

| Strategy | File | Status | Expected Edge | Complexity |
|----------|------|--------|---------------|------------|
| **HMM Regime Switching** | `hmm_regime.py` | ✅ Complete | 4/5 | 4/5 |
| **Network Risk Parity** | `network_parity.py` | ✅ Complete | 3/5 | 3/5 |

### Tier 4: Sprint 3 Small-Cap Strategies (GPU-Accelerated)

| Strategy | File | Status | Expected Edge | Complexity |
|----------|------|--------|---------------|------------|
| **Momentum Breakout** | `sprint3_smallcap_gpu.py` | ✅ Complete | 3/5 | 2/5 |
| **Mean Reversion RSI** | `sprint3_smallcap_gpu.py` | ✅ Complete | 4/5 | 2/5 |
| **Volatility Squeeze** | `sprint3_smallcap_gpu.py` | ✅ Complete | 4/5 | 2/5 |
| **Trend Following EMA** | `sprint3_smallcap_gpu.py` | ✅ Complete | 4/5 | 2/5 |
| **Volume-Price Confirm** | `sprint3_smallcap_gpu.py` | ✅ Complete | 3/5 | 2/5 |

---

## Strategy Summaries

### 1. ATR-Optimized RSI Mean Reversion
**Location:** `src/ordinis/engines/signalcore/models/atr_optimized_rsi.py`

Mean reversion strategy using RSI<35 entry with ATR-based adaptive stops. Regime-filtered to avoid choppy markets.

**Key Parameters:**
- RSI threshold: 35 (entry), 50 (exit)
- ATR stop multiplier: 1.5×
- ATR take-profit: 1.5-3.0× (per symbol)

**Performance:** +60.1% return, 70-85% win rate, 26.1% max drawdown

---

### 2. GARCH Volatility Breakout
**Location:** `src/ordinis/engines/signalcore/models/garch_breakout.py`

Trades volatility expansions when realized volatility exceeds GARCH(1,1) forecast by >2σ. Direction determined by recent price move.

**Key Parameters:**
- GARCH(1,1) lookback: 252 days
- Breakout threshold: 2.0× forecast
- Realized vol window: 5 days

**Edge:** Captures regime changes as GARCH forecast lags reality.

---

### 3. EVT Risk Gate
**Location:** `src/ordinis/engines/signalcore/models/evt_risk_gate.py`

Overlay strategy using Generalized Pareto Distribution to estimate tail risk. Reduces position sizes when VaR or tail shape exceeds thresholds.

**Key Parameters:**
- GPD threshold: 95th percentile
- VaR confidence: 99%
- Alert triggers: VaR>3% or ξ>0.3

**Edge:** Better tail risk estimation than Gaussian VaR.

---

### 4. Multi-Timeframe Momentum
**Location:** `src/ordinis/engines/signalcore/models/mtf_momentum.py`

Combines daily momentum ranking (12-1 month) with intraday stochastic oscillator for entry timing. Only enters momentum trades when stochastic confirms.

**Key Parameters:**
- Momentum: 12-1 month return
- Stochastic: 14-period %K, 3-period %D
- Entry: Winner + bullish cross + oversold

**Edge:** Better entry prices via stochastic pullback confirmation.

---

### 5. Kalman Filter Hybrid
**Location:** `src/ordinis/engines/signalcore/models/kalman_hybrid.py`

Decomposes price into trend and residual using Kalman filter. Mean-reverts residual only when aligned with trend direction.

**Key Parameters:**
- Process noise (Q): 1e-6
- Observation noise (R): 1e-3
- Residual z-score threshold: 2.0

**Edge:** Avoids counter-trend mean reversion trades.

---

### 6. OU Pairs Trading
**Location:** `src/ordinis/engines/signalcore/models/ou_pairs.py`

Pairs trading with Ornstein-Uhlenbeck parameter estimation. Uses half-life to set dynamic thresholds and maximum holding periods.

**Key Parameters:**
- Cointegration p-value: <0.05
- Half-life filter: <10 days
- Entry z-score: 2.0 (dynamic)

**Edge:** Adapts to spread dynamics; avoids slow-reverting pairs.

---

### 7. MI-Weighted Ensemble
**Location:** `src/ordinis/engines/signalcore/models/mi_ensemble.py`

Meta-strategy combining multiple signals weighted by mutual information with future returns. Penalizes redundant signals.

**Key Parameters:**
- Base signals: RSI, Stochastic, Momentum, Volatility
- MI rolling window: 252 days
- Redundancy penalty: correlation-based

**Edge:** Information-theoretic signal combination; captures non-linear dependencies.

---

### 8. HMM Regime Switching
**Location:** `src/ordinis/engines/signalcore/models/hmm_regime.py`

3-state Hidden Markov Model for regime detection. Switches between strategies based on regime probability.

**Key Parameters:**
- States: Bull, Bear, Neutral
- Strategy mapping: Bull→Momentum, Bear→Mean Reversion
- Transition threshold: 60% probability

**Edge:** Probabilistic regime detection with strategy rotation.

---

### 9. Network Risk Parity
**Location:** `src/ordinis/engines/signalcore/models/network_parity.py`

Portfolio allocation using correlation network centrality. Underweights highly central (systemic) assets; overweights peripheral assets.

**Key Parameters:**
- Centrality: Eigenvector
- Correlation shrinkage: 10%
- Weight smoothing: 20% blend

**Edge:** Systemic risk awareness; diversification maximization.

---

## Directory Structure

```
src/ordinis/engines/signalcore/models/
├── __init__.py
├── atr_optimized_rsi.py      # ✅ Complete
├── garch_breakout.py         # 🔨 Building
├── evt_risk_gate.py          # 🔨 Building
├── mtf_momentum.py           # 🔨 Building
├── kalman_hybrid.py          # 🔨 Building
├── ou_pairs.py               # 🔨 Building
├── mi_ensemble.py            # 🔨 Building
├── hmm_regime.py             # 🔨 Building
└── network_parity.py         # 🔨 Building

configs/strategies/
├── atr_optimized_rsi.yaml    # ✅ Complete
├── garch_breakout.yaml
├── evt_risk_gate.yaml
├── mtf_momentum.yaml
├── kalman_hybrid.yaml
├── ou_pairs.yaml
├── mi_ensemble.yaml
├── hmm_regime.yaml
└── network_parity.yaml
```

---

## Dependencies

| Strategy | Required Libraries |
|----------|-------------------|
| ATR-Optimized RSI | numpy, pandas |
| GARCH Breakout | arch |
| EVT Risk Gate | scipy |
| MTF Momentum | numpy, pandas |
| Kalman Hybrid | filterpy (optional) |
| OU Pairs | statsmodels |
| MI Ensemble | sklearn |
| HMM Regime | hmmlearn |
| Network Parity | networkx |

---

## Quick Start

```python
from ordinis.engines.signalcore.strategy_loader import StrategyLoader

# Load any strategy
loader = StrategyLoader()
loader.load_strategy("configs/strategies/garch_breakout.yaml")

# Get model and generate signals
model = loader.get_model("COIN")
signal = await model.generate("COIN", price_df, timestamp)
```

---

## Detailed Documentation

- [ATR-Optimized RSI Strategy (Official)](./ATR-RSI.md)
- [Strategy Derivation Roadmap](../reference/strategies/strategy-derivation-roadmap.md)

---

*Ordinis Quantitative Research - Strategy Suite v1.0.0*
