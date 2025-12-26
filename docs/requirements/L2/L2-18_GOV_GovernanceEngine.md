# L2-18 — GOV — GovernanceEngine

## 1. Identifier
- **ID:** `L2-18`
- **Component:** GOV — GovernanceEngine
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Cross-cutting policy checks (allow/deny/warn/modify) at every engine boundary.
- Audit log as system of record (append-only) with standardized audit records.
- Compliance mapping hooks (jurisdiction tags, data handling tags) for DB1 baseline.
- HumanGate integration for CodeGen patch approval and privileged operations.

### Out of scope
- Full regulatory reporting automation (P2).
- External compliance integrations (case management) (P2).

## 3. Responsibilities
- Own policy pack versioning and evaluation runtime.
- Own audit schema, hash chaining, and audit completeness checks.
- Provide a uniform preflight(ctx) interface and audit(event) sink.
- Enforce model allowlists and data handling policies for AI plane.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

Decision = Literal["allow","deny","warn","modify"]

@dataclass(frozen=True)
class GovernanceDecision:
    decision: Decision
    reasons: List[str]
    patch: Dict[str, Any]
    policy_version: str

class GovernanceEngine:
    def preflight(self, ctx: Dict[str, Any]) -> GovernanceDecision: ...
    def audit(self, record: Dict[str, Any]) -> str: ...  # audit_ref
    def load_policy_pack(self, version: str) -> None: ...
```

### Events
**Consumes**
- ordinis.* (audit inputs from all components)
- ordinis.cfg.change.requested
- ordinis.cg.patch.proposed

**Emits**
- ordinis.gov.preflight.decision
- ordinis.gov.audit.appended
- ordinis.gov.change.approved|denied

### Schemas
- EVT-DATA-020 AuditRecord
- GovernanceDecision schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_GOV_DENIED` | Governance denied action | No |
| `E_GOV_POLICY_MISSING` | Policy pack not loaded/unknown | No |
| `E_GOV_AUDIT_WRITE_FAILED` | Audit append failed | Yes |

## 5. Dependencies
### Upstream
- CFG for policy_version and thresholds
- SEC for redaction and actor identity
- PERSIST for audit storage

### Downstream
- HOOK pipeline uses GOV as a HARD preflight/audit hook
- All engines depend on GOV decisions for gating privileged actions

### External services / libraries
- Cryptographic hash function for audit chaining

## 6. Data & State
### Storage
- SQLite `audit_log` append-only table with hash chain fields.
- Optional `policy_packs` table storing loaded pack metadata and hashes.

### Caching
- Policy pack cache keyed by policy_version and hash.

### Retention
- Audit retention per PERSIST policy; default aligned with DB retention policy.

### Migrations / Versioning
- Audit schema is append-only; backward compatible extensions only.

## 7. Algorithms / Logic
### Preflight model
- Inputs: context containing component_id, action, actor, instrument_type, jurisdiction, data_tags, and payload summary.
- Policies evaluate deterministically and return allow/deny/warn/modify with reason codes.
- Modify decision may patch:
  - max order size
  - model selection (force fallback model)
  - disable certain sources (Synapse)

### Audit model
- Every preflight and every side-effect action emits an AuditRecord.
- AuditRecord includes policy_version, model_used (if any), and refs to inputs/outputs.
- Hash chaining enforced; chain verification runs periodically and on startup.

### Edge cases
- GOV unavailable: prod fail closed for EXEC and config changes; dev may fail open for non-side-effect actions.
- High-volume auditing: store references to payloads rather than full payloads to control size.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| GOV_POLICY_VERSION | string | gov_v1 | P0 | Active policy pack. |
| GOV_FAIL_CLOSED_EXEC | bool | true | P0 | Block execution if GOV unavailable. |
| GOV_AUDIT_HASH_ENABLE | bool | true | P0 | Hash chain enabled. |
| GOV_AUDIT_SAMPLE_RATE | float | 1.0 | P0 | Must be 1.0 for side effects. |

**Environment variables**
- `ORDINIS_GOV_POLICY_VERSION`

## 9. Non-Functional Requirements
- Preflight latency: p95 < 5ms (in-memory policy eval).
- Audit append must be durable; failure triggers immediate alert and optionally kill-switch.
- Policies are deterministic and versioned; no dynamic network calls in policy evaluation (DB1).

## 10. Observability
**Metrics**
- `gov_preflight_total{decision,policy_version}`
- `gov_preflight_latency_ms{policy_version}`
- `gov_audit_append_total{result}`
- `gov_audit_chain_verify_total{result}`

**Alerts**
- Audit append failures
- Policy pack missing

## 11. Failure Modes & Recovery
- Audit write failure: fail closed for side effects; raise ops alert.
- Policy pack load failure: deny all privileged actions until resolved.
- Hash chain broken: trigger incident; potentially kill-switch.

## 12. Test Plan
### Unit tests
- Policy evaluation determinism and reason codes.
- Audit record hash chaining and verification.

### Integration tests
- Hook pipeline enforces GOV decisions at SIG/RISK/EXEC boundaries.
- CodeGen patch requires HumanGate approval.

### Acceptance criteria
- 100% of EXEC submits have corresponding audit records with policy_version.

## 13. Open Questions / Risks
- Compliance mapping scope for DB1 (SEC/FINRA/MiFID references) vs deferring to Phase 2.
- Audit payload size strategy: what fields become refs vs inline?

## 14. Traceability
### Parent
- DB1 System SRS: Governance-first approach with comprehensive audit trails.

### Related
- [L2-02 HOOK — Hook Framework](L2-02_HOOK_Hook_and_Middleware_Framework.md)
- [L2-05 PERSIST — Persistence Layer](L2-05_PERSIST_Persistence_Layer.md)

### Originating system requirements
- GOV-IF-001, GOV-FR-001..003
