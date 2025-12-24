# Strategy Derivation Roadmap

A lightweight roadmap for turning a trading hypothesis into an Ordinis strategy, including research artifacts, implementation hooks, and verification gates.

---

## Overview

This document is intentionally concise. The key idea is: every strategy should move through the same pipeline:

1. Hypothesis
2. Data & assumptions
3. Backtest harness + baseline
4. Risk controls
5. Walk-forward validation
6. Governance + auditability
7. Paper trading

---

## Required artifacts

- **Hypothesis statement** (what behavior you exploit, why it should persist)
- **Universe + timeframe** (symbols, bar size, market hours)
- **Execution assumptions** (fees, slippage, latency, fill model)
- **Risk constraints** (per-position cap, daily loss, max DD, exposure rules)
- **Validation report** (walk-forward, regime sensitivity, stability across symbols)

---

## Implementation checklist (Ordinis)

- Add a model under `src/ordinis/engines/signalcore/models/`
- Register/load via a YAML under `configs/strategies/`
- Ensure gating:
  - Regime filter (`RegimeDetector`) when appropriate
  - RiskGuard + governance preflight before any execution
- Add tests under `tests/test_engines/test_signalcore/`

---

## Document Metadata

```yaml
version: "0.1.0"
created: "2025-12-23"
last_updated: "2025-12-23"
status: "draft"
```

---

**END OF DOCUMENT**
