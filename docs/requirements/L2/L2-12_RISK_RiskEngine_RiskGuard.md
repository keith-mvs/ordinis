# L2-12 — RISK — RiskEngine (RiskGuard)

## 1. Identifier
- **ID:** `L2-12`
- **Component:** RISK — RiskEngine (RiskGuard)
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Policy-based evaluation of signals and trade intents against portfolio and market state.
- Allow/deny/modify decisions with deterministic reason codes.
- Hard blocks for kill-switch, stale data, drawdown/exposure breaches.
- Instrument-type extension points (equities DB1; futures/options modules P1).

### Out of scope
- Portfolio optimization (OPT).
- Order execution simulation/routing (EXEC).

## 3. Responsibilities
- Own the risk policy pack system and evaluation ordering.
- Produce RiskDecision events with policy_version and detailed reasons.
- Apply adjustments (position sizing caps) without changing direction unless policy requires deny.
- Expose risk snapshots for audit and analytics.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

@dataclass(frozen=True)
class RiskDecision:
    signal_id: str
    allowed: bool
    adjustments: Dict[str, Any]
    reasons: List[str]           # machine-queryable reason codes
    policy_version: str
    risk_snapshot_ref: str

class RiskEngine:
    def evaluate(self, signal: Dict[str, Any], portfolio: Dict[str, Any], market_state: Dict[str, Any]) -> RiskDecision: ...
```

### Events
**Consumes**
- ordinis.signals.generated
- ordinis.ops.data.stale
- ordinis.portfolio.snapshot
- ordinis.market.* (optional for volatility/exposure checks)

**Emits**
- ordinis.risk.decision
- ordinis.risk.reject

### Schemas
- EVT-DATA-014 RiskDecision payload
- Risk snapshot schema (artifact)

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_RISK_POLICY_DENIED` | Policy pack denied the signal | No |
| `E_RISK_POLICY_FAILED` | Policy evaluation failed (exception) | No |
| `E_RISK_SNAPSHOT_FAILED` | Unable to persist risk snapshot | Yes |

## 5. Dependencies
### Upstream
- SIG provides signals
- PORT provides portfolio snapshot
- CFG defines policy pack version and thresholds
- KILL state consulted for hard blocks
- GOV preflight (HARD) for compliance/jurisdiction constraints

### Downstream
- EXEC consumes allowed/adjusted trade intents
- ANA consumes risk decisions for reporting
- LEARN uses risk decisions as labels/features

### External services / libraries
- None required for rule-based DB1; optional math libs for VaR/CVaR estimates

## 6. Data & State
### Storage
- Persist risk snapshots and decisions to SQLite (`risk_decisions`, `risk_snapshots` optional).

### Caching
- Policy pack cache keyed by policy_version.

### Retention
- Risk decisions retained with audit retention; snapshots retained 90-365 days (configurable).

### Migrations / Versioning
- Policy pack versioning; reason code taxonomy is append-only (do not change meaning of existing codes).

## 7. Algorithms / Logic
### Evaluation ordering (deterministic)
1. **Hard blocks** (fail-fast):
   - kill-switch active
   - data stale
   - max drawdown breached
2. **Portfolio constraints**:
   - max position size
   - sector/asset exposure caps
   - leverage caps (if applicable)
3. **Market constraints**:
   - volatility regime blocks
   - liquidity / spread constraints (if available)
4. **Adjustments**:
   - clamp qty to risk budget
   - clamp notional to exposure cap
5. Emit RiskDecision with reasons and policy_version.

### Edge cases
- Multiple signals for same symbol: evaluate independently; downstream PORT may net exposures.
- Missing market_state: treat as degraded; apply conservative defaults or deny based on config.
- Conflicting policies: earliest deny wins; adjustments only applied if overall decision remains allowed.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| RISK_POLICY_VERSION | string | risk_v1 | P0 | Active policy pack. |
| RISK_MAX_DRAWDOWN | float | 0.20 | P0 | Hard stop drawdown threshold. |
| RISK_MAX_POSITION_PCT | float | 0.10 | P0 | Per-symbol exposure cap. |
| RISK_MAX_TURNOVER | float | 2.0 | P0 | Turnover cap per period. |
| RISK_DENY_ON_DATA_STALE | bool | true | P0 | Hard block on stale data. |

**Environment variables**
- `ORDINIS_RISK_POLICY_VERSION`

## 9. Non-Functional Requirements
- Evaluation latency: p95 < 10ms per signal for typical portfolios (<5k positions).
- Deterministic reasons and adjustments for identical inputs.
- Fail-closed posture in prod when critical context missing (configurable).

## 10. Observability
**Metrics**
- `risk_decisions_total{allowed,policy_version}`
- `risk_latency_ms{policy_version}`
- `risk_denied_total{reason_code}`

**Alerts**
- Deny-rate spike (could indicate data issues)
- Drawdown breach triggers kill-switch

## 11. Failure Modes & Recovery
- Policy evaluation exception: deny signal; emit error event; alert.
- Snapshot persistence failure: decision can proceed but must mark `risk_snapshot_ref=missing` and alert (prod may fail closed).
- Context missing (portfolio snapshot): deny by default in prod; dev may warn.

## 12. Test Plan
### Unit tests
- Each policy rule with synthetic portfolios and signals.
- Adjustment math correctness and bounds.

### Integration tests
- SIG→RISK→EXEC paper run on benchmark pack.

### Acceptance criteria
- Risk reasons are machine-queryable and stable across runs.

## 13. Open Questions / Risks
- Reason code taxonomy: finalize list and guarantee backward meaning stability (see L3-05).
- How to handle partial portfolio snapshots (startup) — deny vs conservative sizing.

## 14. Traceability
### Parent
- DB1 System SRS: RiskGuard evaluates and adjusts signals/trades.

### Children / Related
- [L3-05 RISK-POLICY-PACKS — Risk Policy Packs & Reason Codes](../L3/L3-05_RISK-POLICY-PACKS_Risk_Policy_Packs_and_Reason_Codes.md)
- [L2-19 KILL — KillSwitch](L2-19_KILL_KillSwitch.md)

### Originating system requirements
- RISK-IF-001, RISK-FR-001..004
