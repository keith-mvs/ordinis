# L3-03 — HOOK-PIPELINE — Hook Pipeline & Context

## 1. Identifier
- **ID:** `L3-03`
- **Component:** HOOK-PIPELINE — Hook Pipeline & Context
- **Level:** L3

## 2. Purpose / Scope
### In scope
- HookStage enumeration and canonical stage ordering.
- HookContext schema and redaction requirements.
- Decision merge rules and patch semantics.
- Timeout and budgeting rules per hook/stage.

### Out of scope
- Engine-specific hook implementations (owned by respective engines).

## 3. Responsibilities
- Define the contract for HookContext fields and lifecycle.
- Define deterministic merge semantics for modify patches.
- Define error codes, retry posture, and observability hooks for pipeline.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Protocol

HookStage = Literal["preflight","validate_input","execute","validate_output","audit","emit_metrics","postflight","on_error"]

@dataclass
class HookContext:
    schema_version: str
    component_id: str
    action: str
    correlation_id: str
    trace_id: str
    span_id: str
    ts_start: str
    actor: Optional[Dict[str, Any]]
    input_summary: Dict[str, Any]      # MUST be redacted
    output_summary: Optional[Dict[str, Any]] = None
    error_summary: Optional[Dict[str, Any]] = None
    tags: Dict[str, str] = None

@dataclass
class HookDecision:
    decision: Literal["allow","deny","modify"]
    reasons: List[str]
    patch: Dict[str, Any] = None
```

### Events
**Consumes**
_None._

**Emits**
- ordinis.ops.hook.decision
- ordinis.ops.hook.error

### Schemas
- HookContext schema v1
- HookDecision schema v1

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_HOOK_TIMEOUT` | Hook exceeded budget | Yes |
| `E_HOOK_PATCH_INVALID` | Modify patch failed schema/merge rules | No |
| `E_HOOK_DENIED` | HARD hook denied | No |

## 5. Dependencies
### Upstream
- SEC redactor (mandatory before logging/audit)
- CFG for budgets and enablement

### Downstream
- HOOK manager and all engine wrappers
- GOV audit sink (HookContext stored as attachment refs)

### External services / libraries
_None._

## 6. Data & State
### Storage
- HookContext not persisted directly; stored as redacted attachments in audit trail if enabled.

### Caching
- N/A

### Retention
- Hook decision events retained with ops logs; audit attachments follow audit retention.

### Migrations / Versioning
- HookContext schema versioned; only additive changes allowed without major bump.

## 7. Algorithms / Logic
### Canonical stage ordering
1. preflight (GOV/HARD)
2. validate_input
3. execute
4. validate_output
5. audit
6. emit_metrics
7. postflight
8. on_error

### Patch merge semantics (deterministic)
- JSON Merge Patch semantics for dicts (RFC 7396-like) with restrictions:
  - cannot delete required keys
  - cannot widen action scope (e.g., increase order size beyond original)
  - all patches validated against a component-specific patch schema
- Lists:
  - default: replace (to avoid non-deterministic append ordering)
  - append allowed only for explicitly appendable fields

### Deny precedence
- Any HARD hook deny => overall deny.
- SOFT hook deny => recorded as warning; does not block unless configured.

### Timeouts
- HOOK_STAGE_BUDGET_MS applies per-hook per-stage.
- Exceeding budget triggers timeout decision:
  - HARD: deny
  - SOFT: allow + warning

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| HOOK_STAGE_BUDGET_MS | int | 50 | P0 | Budget per hook stage. |
| HOOK_PATCH_SCHEMA_STRICT | bool | true | P0 | Validate modify patches strictly. |
| HOOK_LIST_MERGE_MODE | string | replace | P0 | replace|append (append only for allowlisted fields). |

## 9. Non-Functional Requirements
- Determinism: identical hook chain yields identical merged output.
- Safety: patches cannot expand risk exposure; component patch schema must enforce monotonic safety.

## 10. Observability
**Metrics**
- `hook_stage_latency_ms{component,stage}`
- `hook_patch_reject_total{component}`
- `hook_timeout_total{component,hook_id}`

## 11. Failure Modes & Recovery
- Patch invalid: treat as deny for HARD hooks; warning for SOFT hooks depending on config.
- Redaction missing: treat as hard error in prod; do not emit ctx.

## 12. Test Plan
### Unit tests
- Merge semantics for dicts/lists.
- Patch schema validation rejects unsafe modifications.
- Timeout behavior for HARD vs SOFT.

### Acceptance criteria
- Hook pipeline contract tests pass for all engine wrappers.

## 13. Open Questions / Risks
- Whether list merge default should be replace (safer) or append (more ergonomic).
- Define per-component patch schemas (EXEC and RISK are highest priority).

## 14. Traceability
### Parent
- [L2-02 HOOK — Hook Framework](../L2/L2-02_HOOK_Hook_and_Middleware_Framework.md)

### Originating system requirements
- HOOK-FR-001..003
