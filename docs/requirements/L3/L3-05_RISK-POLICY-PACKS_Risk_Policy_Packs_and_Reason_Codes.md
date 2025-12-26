# L3-05 — RISK-POLICY-PACKS — Risk Policy Packs & Reason Codes

## 1. Identifier
- **ID:** `L3-05`
- **Component:** RISK-POLICY-PACKS — Risk Policy Packs & Reason Codes
- **Level:** L3

## 2. Purpose / Scope
### In scope
- Policy pack structure and versioning for risk evaluation.
- Reason code taxonomy (machine queryable, stable meaning).
- Rule ordering and conflict resolution.
- Policy unit test harness (golden fixtures).

### Out of scope
- Portfolio optimization (OPT).
- Execution/broker details (EXEC).

## 3. Responsibilities
- Define the risk policy DSL/representation (Python rules or declarative YAML).
- Define canonical reason codes and ensure stability across versions.
- Provide rule evaluation runtime hooks for RiskEngine.
- Provide fixture-based tests for each rule and for pack-level behavior.

## 4. Interfaces
### Public APIs / SDKs
**Policy pack interface (conceptual)**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

@dataclass(frozen=True)
class PolicyResult:
    allowed: bool
    adjustments: Dict[str, Any]
    reasons: List[str]

class RiskPolicy(Protocol):
    policy_id: str
    priority: int
    def evaluate(self, signal: Dict[str, Any], portfolio: Dict[str, Any], market: Dict[str, Any]) -> PolicyResult: ...

class RiskPolicyPack(Protocol):
    version: str
    policies: List[RiskPolicy]
    def evaluate_all(self, signal, portfolio, market) -> PolicyResult: ...
```

**Reason code format**
- `RISK.<CATEGORY>.<CODE>` (uppercase, dot-separated)
- Example: `RISK.DRAWDOWN.MAX_DRAWDOWN_BREACH`

### Events
**Consumes**
_None._

**Emits**
_None._

### Schemas
- Policy pack manifest schema: {version, policy_ids, hashes}
- Reason code list schema (static export)

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_RISK_POLICY_INVALID` | Policy rule invalid/misconfigured | No |
| `E_RISK_REASON_UNKNOWN` | Policy emitted unknown reason code | No |

## 5. Dependencies
### Upstream
- CFG to select active policy pack version

### Downstream
- RISK engine runtime
- GOV audit references reason codes
- ANA uses reason codes for attribution

### External services / libraries
_None._

## 6. Data & State
### Storage
- Policy packs stored in-repo under `risk_policies/<version>/...` or packaged with the app.
- Optional: persist policy pack manifest hashes to DB for audit.

### Caching
- RiskEngine caches compiled policy pack.

### Retention
- All released policy packs retained indefinitely for audit reproducibility.

### Migrations / Versioning
- Reason codes are append-only; never change semantics of existing reason codes.

## 7. Algorithms / Logic
### Rule ordering
- Policies sorted by ascending `priority`.
- Deny is sticky: once denied, no further policies can re-allow.
- Adjustments merge with deterministic rules (same as HOOK patch rules):
  - For numeric limits: take min (most conservative)
  - For allowlists: intersect
  - For denylists: union

### Baseline rule set (DB1 recommended)
- HARD blocks:
  - kill-switch active → deny `RISK.KILL.ACTIVE`
  - stale data → deny `RISK.DATA.STALE`
  - max drawdown breach → deny `RISK.DRAWDOWN.MAX_DRAWDOWN_BREACH`
- Exposure caps:
  - per-symbol notional cap → adjust `RISK.EXPOSURE.SYMBOL_CAP`
  - sector cap (if sector metadata available) → adjust `RISK.EXPOSURE.SECTOR_CAP`
- Liquidity/spread (optional DB1):
  - deny if spread > threshold → `RISK.LIQUIDITY.SPREAD_TOO_WIDE`

### Edge cases
- Missing metadata (sector): either skip sector rules or deny based on strictness config.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| RISK_POLICY_VERSION | string | risk_v1 | P0 | Active pack version. |
| RISK_STRICT_METADATA | bool | false | P0 | Deny when required metadata missing. |

## 9. Non-Functional Requirements
- Policy evaluation deterministic and side-effect free.
- Every deny/adjust must include at least one reason code.
- Unknown reason codes are hard errors (contract enforcement).

## 10. Observability
Risk decisions and reason codes are emitted as part of RiskDecision events and tracked by RISK metrics. See [L2-12 RISK](../L2/L2-12_RISK_RiskEngine_RiskGuard.md).

## 11. Failure Modes & Recovery
- Policy pack missing: deny all signals and alert (fail safe).
- Policy emitted unknown reason: treat as policy invalid; deny; alert.

## 12. Test Plan
### Unit tests
- Rule fixtures: each rule has pass/fail cases with expected reasons/adjustments.
- Pack-level golden tests: composite scenarios produce expected deny/adjust set.

### Acceptance criteria
- Reason code list is complete and stable across releases.

## 13. Open Questions / Risks
- DSL choice: pure Python rules (flexible) vs declarative YAML (audit-friendly).
- Standardize reason codes across GOV and RISK (shared taxonomy).

## 14. Traceability
### Parent
- [L2-12 RISK — RiskEngine](../L2/L2-12_RISK_RiskEngine_RiskGuard.md)

### Originating system requirements
- RISK-FR-001..004
