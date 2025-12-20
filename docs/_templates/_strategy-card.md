# Strategy Cards Index

---

**Title:** Ordinis Strategy Card Library
**Description:** Quick reference cards for all quantitative trading strategies
**Author:** Ordinis Quantitative Team
**Version:** 1.0.0
**Date:** 2025-12-17
**Tags:** index, strategies, documentation, reference

---

## Overview

Strategy cards are 1-2 page reference documents containing:

- Mathematical basis and formulas
- Signal logic and entry/exit conditions
- Key parameters with defaults
- Edge source explanation
- Implementation examples
- Risk considerations
- Performance expectations

---

## Strategy Cards

### Tier 1: Quick Wins

| Strategy | Card | Focus |
|----------|------|-------|
| GARCH Volatility Breakout | [GARCH_BREAKOUT.md](GARCH_BREAKOUT.md) | Volatility regime changes |
| EVT Risk Gate | [EVT_RISK_GATE.md](EVT_RISK_GATE.md) | Tail risk management |
| Multi-Timeframe Momentum | [MTF_MOMENTUM.md](MTF_MOMENTUM.md) | Momentum + timing |

### Tier 2: Core Quantitative

| Strategy | Card | Focus |
|----------|------|-------|
| Kalman Filter Hybrid | [KALMAN_HYBRID.md](KALMAN_HYBRID.md) | Trend-aligned mean reversion |
| OU Pairs Trading | [OU_PAIRS.md](OU_PAIRS.md) | Statistical arbitrage |
| MI-Weighted Ensemble | [MI_ENSEMBLE.md](MI_ENSEMBLE.md) | Information-theoretic combination |

### Tier 3: Advanced

| Strategy | Card | Focus |
|----------|------|-------|
| HMM Regime Switching | [HMM_REGIME.md](HMM_REGIME.md) | Adaptive regime detection |
| Network Risk Parity | [NETWORK_PARITY.md](NETWORK_PARITY.md) | Correlation-based allocation |

---

## Card Structure

Each card follows this template:

```markdown
# Strategy Name

---
**Title:** ...
**Description:** ...
**Author:** Ordinis Quantitative Team
**Version:** X.Y.Z
**Date:** YYYY-MM-DD
**Tags:** ...
**References:** ...
---

## Overview
## Mathematical Basis
## Signal Logic
## Key Parameters
## Edge Source
## Implementation Notes
## Risk Considerations
## Performance Expectations
```

---

## Related Documentation

- [STRATEGY_SUITE.md](../reference/STRATEGY_SUITE.md) - Master strategy index
- [strategy-derivation-roadmap.md](../reference/strategies/strategy-derivation-roadmap.md) - Theory and derivation
- [atr-optimized-rsi-technical-spec.md](../reference/strategies/atr-optimized-rsi-technical-spec.md) - Production strategy spec

---

## Implementation Files

All strategies located in: `src/ordinis/engines/signalcore/models/`

| File | Class |
|------|-------|
| `garch_breakout.py` | `GARCHBreakoutModel` |
| `evt_risk_gate.py` | `EVTRiskGate`, `EVTGatedStrategy` |
| `mtf_momentum.py` | `MTFMomentumModel` |
| `kalman_hybrid.py` | `KalmanHybridModel` |
| `ou_pairs.py` | `OUPairsModel` |
| `mi_ensemble.py` | `MIEnsembleModel` |
| `hmm_regime.py` | `HMMRegimeModel` |
| `network_parity.py` | `NetworkRiskParityModel` |
