# L2-19 — KILL — KillSwitch

## 1. Identifier
- **ID:** `L2-19`
- **Component:** KILL — KillSwitch
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Global kill-switch state for halting trading side effects.
- Enforcement hooks integrated into ORCH/RISK/EXEC.
- Optional cancel-all behavior on activation (adapter-dependent).

### Out of scope
- External hardware kill switches (P2).
- Two-man rule reset in DB1 (P1).

## 3. Responsibilities
- Maintain kill-switch state and reason codes.
- Expose query and set APIs with governance enforcement.
- Emit kill-switch events for audit and alerting.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal

KillState = Literal["ACTIVE","INACTIVE"]

@dataclass(frozen=True)
class KillSwitchStatus:
    state: KillState
    reason: Optional[str]
    activated_at: Optional[str]
    actor: Optional[str]

class KillSwitch:
    def get(self) -> KillSwitchStatus: ...
    def activate(self, reason: str, actor: str) -> KillSwitchStatus: ...   # requires GOV approval in prod
    def deactivate(self, actor: str) -> KillSwitchStatus: ...              # requires GOV approval
```

### Events
**Consumes**
- ordinis.ops.kill.activate.requested
- ordinis.ops.kill.deactivate.requested

**Emits**
- ordinis.ops.kill.activated
- ordinis.ops.kill.deactivated
- ordinis.ops.kill.block

### Schemas
- KillSwitchStatus schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_KILL_ALREADY_ACTIVE` | Kill-switch already active | No |
| `E_KILL_ALREADY_INACTIVE` | Kill-switch already inactive | No |
| `E_KILL_GOV_DENIED` | Governance denied state change | No |

## 5. Dependencies
### Upstream
- GOV for approval and audit
- PERSIST for durable state (optional; DB1 recommends persist)

### Downstream
- ORCH gates pipeline stages
- RISK hard-blocks signals
- EXEC blocks order submission

### External services / libraries
_None._

## 6. Data & State
### Storage
- SQLite `kill_switch` single-row table storing current state and metadata.

### Caching
- In-memory cached state with TTL; invalidated on state change event.

### Retention
- Kill events retained per audit retention; state table holds latest only.

### Migrations / Versioning
- KillSwitchStatus schema versioned; state table migration unlikely.

## 7. Algorithms / Logic
### Enforcement
- ORCH checks kill state before SIG.
- RISK checks kill state as hard block.
- EXEC checks kill state before submit (defense-in-depth).

### Activation triggers
- Manual operator activation (preferred).
- Automatic activation on:
  - ledger invariant breach
  - audit write failure (prod)
  - broker circuit open for prolonged duration (configurable)

### Edge cases
- Deactivation requires explicit GOV approval and audit record.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| KILL_PERSIST_ENABLE | bool | true | P0 | Persist kill state to DB. |
| KILL_AUTO_ON_LEDGER_BREACH | bool | true | P0 | Auto-activate. |
| KILL_AUTO_ON_AUDIT_FAIL | bool | true | P0 | Auto-activate. |
| KILL_CANCEL_ALL_ON_ACTIVATE | bool | false | P0 | Requires broker cancel support. |

## 9. Non-Functional Requirements
- Activation latency: p95 < 10ms to block new submits (single process).
- Fail-safe: default posture is ACTIVE if state cannot be read at startup (prod).

## 10. Observability
**Metrics**
- `kill_state{state=active|inactive}`
- `kill_activations_total{reason}`

**Alerts**
- Kill-switch activation always alerts.

## 11. Failure Modes & Recovery
- State read failure: assume ACTIVE in prod; INACTIVE allowed in dev (configurable).
- Persistence failure: keep in-memory state, emit alert; do not allow deactivation without durable write in prod.

## 12. Test Plan
### Unit tests
- Activate/deactivate transitions and idempotency.
- Enforcement checks in ORCH/RISK/EXEC wrappers.

### Integration tests
- Activate kill while orders pending; ensure new submits blocked.

### Acceptance criteria
- Under kill-switch active, no new OrderSubmitted events occur.

## 13. Open Questions / Risks
- Do we require two-man rule in DB1 for prod deactivation? (recommended P1).
- How aggressively do we auto-activate on transient broker errors vs persistent failures?

## 14. Traceability
### Parent
- DB1 System SRS: Kill-switch gating prevents catastrophic loss.

### Related
- [L2-18 GOV — GovernanceEngine](L2-18_GOV_GovernanceEngine.md)
- [L2-13 EXEC — ExecutionEngine](L2-13_EXEC_ExecutionEngine_FlowRoute.md)

### Originating system requirements
- KILL-FR-001, KILL-FR-002
