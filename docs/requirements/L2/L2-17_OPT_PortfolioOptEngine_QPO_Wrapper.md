# L2-17 — OPT — PortfolioOptEngine (QPO wrapper)

## 1. Identifier
- **ID:** `L2-17`
- **Component:** OPT — PortfolioOptEngine (QPO wrapper)
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Scenario generation and risk-aware optimization (mean-CVaR baseline).
- Constraint compilation and feasibility checks.
- GPU/CPU solver adapter selection (GPU optional).
- Output validation and risk metrics reporting.

### Out of scope
- Trading signal generation (SIG).
- Risk policy enforcement (RISK).

## 3. Responsibilities
- Own optimize() interface and solver adapter abstraction.
- Generate scenarios/returns matrices from inputs or consume precomputed scenarios.
- Enforce constraints and validate output weights.
- Emit optimization artifacts and metrics to ANA/BENCH.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class OptResult:
    weights: Dict[str, float]
    risk_metrics: Dict[str, float]
    solver_used: str
    solve_time_ms: float
    status: str
    diagnostics: Dict[str, Any]

class PortfolioOptEngine:
    def optimize(self, returns: Any, constraints: Dict[str, Any]) -> OptResult: ...
```

### Events
**Consumes**
- ordinis.features.vector.computed (optional for return estimation)
- ordinis.bench.pack.manifest (for scenario windows)

**Emits**
- ordinis.opt.weights
- ordinis.opt.diagnostics

### Schemas
- OptResult schema
- Constraints schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_OPT_INFEASIBLE` | Constraints infeasible | No |
| `E_OPT_SOLVER_FAILED` | Solver failed to converge or crashed | Yes |
| `E_OPT_OUTPUT_INVALID` | Weights violate constraints or sum rules | No |

## 5. Dependencies
### Upstream
- CFG selects solver backend and scenario generator settings
- FEAT/LEARN may provide returns inputs
- PERSIST for storing optimization artifacts

### Downstream
- PORT consumes weights as rebalance targets (optional)
- ANA consumes OPT metrics
- BENCH uses OPT in some runs (optional)

### External services / libraries
- GPU solver (cuOpt) optional; CPU fallback required for DB1 hardware constraints

## 6. Data & State
### Storage
- Persist opt results and diagnostics in `opt_results` table and artifact store.

### Caching
- Cache covariance/scenario computations per window to avoid recompute during BENCH runs.

### Retention
- Optimization artifacts retained 365 days (default) or aligned with run retention.

### Migrations / Versioning
- Constraint schema versioned; changes must be backward compatible.

## 7. Algorithms / Logic
### Mean-CVaR baseline
- Inputs: returns scenarios matrix R (T x N) or historical returns series.
- Objective: maximize expected return - λ * CVaR_α(loss).
- Constraints:
  - Σ w_i = 1
  - bounds: w_min ≤ w_i ≤ w_max
  - optional group constraints (sector caps)

### Feasibility checks
- Before solve: verify bounds and sum constraints are satisfiable.
- Fail fast with `E_OPT_INFEASIBLE` and actionable diagnostics.

### Edge cases
- Missing symbols: drop and renormalize if allowed; otherwise infeasible.
- Degenerate covariance/returns: add ridge regularization per config.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| OPT_SOLVER | string | cpu | P0 | `cpu` in DB1; `gpu` optional. |
| OPT_ALPHA | float | 0.95 | P0 | CVaR confidence. |
| OPT_LAMBDA | float | 1.0 | P0 | Risk aversion. |
| OPT_REG_EPS | float | 1e-6 | P0 | Regularization. |
| OPT_MAX_TIME_MS | int | 2000 | P0 | Solve time budget. |

**Environment variables**
- `ORDINIS_OPT_SOLVER`

## 9. Non-Functional Requirements
- Solve success rate: > 99% on feasible problems in benchmark harness.
- Deterministic outputs given deterministic inputs and fixed solver config.
- CPU fallback required (RTX 2080 Ti insufficient for 49B inference and may be insufficient for heavy GPU optimization).

## 10. Observability
**Metrics**
- `opt_solve_total{status,solver}`
- `opt_solve_time_ms{solver}`
- `opt_infeasible_total`

**Alerts**
- Solver failure spikes
- Infeasible constraint spikes (bad configs)

## 11. Failure Modes & Recovery
- Solver failure: retry with relaxed tolerances or CPU fallback; if still fails, emit diagnostics and return last-known-good weights (optional).
- Infeasible constraints: fail fast; do not emit weights.

## 12. Test Plan
### Unit tests
- Feasibility checks.
- Weight validation (sum=1, bounds, group caps).

### Integration tests
- Run OPT on benchmark returns; ensure stable weights.

### Acceptance criteria
- OPT produces valid weights and risk metrics for baseline scenarios in BENCH.

## 13. Open Questions / Risks
- Exact constraints schema for sector/group constraints in DB1.
- Do we integrate OPT into live trading loop in DB1 or keep as offline tool initially?

## 14. Traceability
### Parent
- DB1 System SRS: PortfolioOptEngine wraps QPO with scenario generation + mean-CVaR optimization.

### Related
- [L2-14 PORT — PortfolioEngine](L2-14_PORT_PortfolioEngine.md)

### Originating system requirements
- OPT-IF-001, OPT-FR-001..003
