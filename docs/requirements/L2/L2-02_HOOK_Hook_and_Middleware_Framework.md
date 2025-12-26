# L2-02 — HOOK — Hook & Middleware Framework

## 1. Identifier
- **ID:** `L2-02`
- **Component:** HOOK — Hook & Middleware Framework
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Standard hook stages and execution order for every engine boundary.
- Hook registration, prioritization, and decision aggregation (allow/deny/modify).
- Hard vs soft hook semantics and isolation guarantees.
- Shared HookContext contract (IDs, actor, input/output refs, timing).

### Out of scope
- Business logic of individual engines (signals, risk rules, broker routing).
- UI or operator workflows (owned by GOV/ops).

## 3. Responsibilities
- Provide a reusable middleware framework for engine entrypoints (decorators/wrappers).
- Guarantee deterministic hook invocation order and consistent error handling.
- Emit hook decision telemetry and hook-failure events without leaking secrets.
- Provide test harness utilities to assert hook execution and policy coverage.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Literal, List

HookStage = Literal[
    "preflight",
    "validate_input",
    "execute",
    "validate_output",
    "audit",
    "emit_metrics",
    "postflight",
    "on_error",
]

@dataclass
class HookContext:
    component_id: str
    action: str
    correlation_id: str
    trace_id: str
    span_id: str
    ts_start: str
    input_ref: Optional[str] = None
    output_ref: Optional[str] = None
    tags: Dict[str, str] = None
    data: Dict[str, Any] = None   # request/response fragments (redacted)
    error: Optional[str] = None

@dataclass
class HookDecision:
    decision: Literal["allow","deny","modify"]
    reasons: List[str]
    patch: Dict[str, Any] = None

class Hook(Protocol):
    hook_id: str
    priority: int          # lower runs earlier
    mode: Literal["HARD","SOFT"]
    def handle(self, stage: HookStage, ctx: HookContext) -> HookDecision: ...

class HookManager:
    def register(self, hook: Hook) -> None: ...
    def run(self, stage: HookStage, ctx: HookContext) -> HookDecision: ...
```

### Events
**Consumes**
- ordinis.ops.hook.* (optional; for hook debugging/replay)

**Emits**
- ordinis.ops.hook.error
- ordinis.ops.hook.decision

### Schemas
- EVT-DATA-001 EventEnvelope
- HOOK context schema (internal; stored as audit attachments)

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_HOOK_DENIED` | A HARD hook denied the action | No |
| `E_HOOK_FAILED` | Hook raised exception; HARD hooks deny, SOFT hooks log | No |
| `E_HOOK_TIMEOUT` | Hook exceeded time budget | Yes |

## 5. Dependencies
### Upstream
- CFG (hook enable/disable, budgets)
- SEC (redaction policy)

### Downstream
- All engines: ORCH, SIG, RISK, EXEC, PORT, ANA, OPT, HEL, SYN, CTX, CG, LEARN, BENCH
- GOV preflight/audit hooks (HARD)

### External services / libraries
- OpenTelemetry (span creation around stages)

## 6. Data & State
### Storage
- No persistent state required in DB1.
- Optional: persist hook decisions as part of GOV audit trail (PERSIST).

### Caching
- In-process cache of resolved hook chain per component+action (invalidate on config change).

### Retention
- Hook decision events retained per audit retention policy (see PERSIST).

### Migrations / Versioning
- HookContext schema versioned; changes require backward compatible expansion.

## 7. Algorithms / Logic
### Decision aggregation
- Hooks execute in ascending `priority`.
- If any HARD hook returns **deny**, the stage fails immediately with `E_HOOK_DENIED`.
- If a hook returns **modify**, its `patch` is merged into `ctx.data` using deterministic merge rules:
  - Scalars overwrite
  - Dicts deep-merge
  - Lists append (unless key is declared replace-only)
- SOFT hook exceptions are swallowed and emitted to `ordinis.ops.hook.error`.
- HARD hook exceptions are treated as deny.

### Time budgeting
- Each stage has a configurable budget (ms).
- If a hook exceeds budget: HARD => deny; SOFT => warning + continue.

### Redaction
- `ctx.data` must be redacted using SEC policies before log/audit emission.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| HOOK_ENABLE | bool | true | P0 | Master enable. |
| HOOK_STAGE_BUDGET_MS | int | 50 | P0 | Per-hook per-stage time budget. |
| HOOK_FAIL_OPEN_SOFT | bool | true | P0 | SOFT hook failure does not block. |
| HOOK_FAIL_CLOSED_HARD | bool | true | P0 | HARD hook failure blocks. |

**Environment variables**
- `HOOK_ENABLE`
- `HOOK_STAGE_BUDGET_MS`

## 9. Non-Functional Requirements
- Hook overhead: p95 < 2ms per engine boundary for default hook chain.
- Deterministic behavior: same inputs yield same merged patches and decisions.
- No secret leakage via ctx.data/logs (SEC redaction required).

## 10. Observability
**Metrics**
- `hook_invocations_total{component,stage,hook_id,result}`
- `hook_latency_ms{component,stage,hook_id}`
- `hook_denied_total{component,stage,hook_id}`

**Logs**
- `hook_id`, `component_id`, `stage`, `decision`, `reasons`, `trace_id`.

**Traces**
- One span per stage + child span per hook: `hook.<stage>.<hook_id>`.

**Alerts**
- High deny rate spikes (could indicate bad deployment/config).

## 11. Failure Modes & Recovery
- Hook exception (SOFT): emit `ops.hook.error`, continue.
- Hook exception (HARD): deny and short-circuit.
- Hook timeout: treat as failure per mode.
- Hook chain misconfiguration: fail startup if duplicate hook_id or invalid priorities.

## 12. Test Plan
### Unit tests
- Hook ordering by priority.
- Decision aggregation rules (deny beats modify; deterministic merge).
- Timeout behavior for HARD vs SOFT.
- Redaction invocation for ctx.data before log/audit.

### Integration tests
- Instrument a dummy engine entrypoint and assert stage order.
- Simulate GOV hook as HARD and ensure deny blocks downstream call.

### Acceptance criteria
- Contract test suite passes for all engine wrappers.

## 13. Open Questions / Risks
- Exact merge semantics for list fields (append vs replace by key).
- Do we allow hooks to mutate ctx directly (discourage) vs patch-only (prefer patch-only).

## 14. Traceability
### Parent
- DB1 System SRS: Governance-first hooks across all engines.

### Children / Related
- [L3-03 HOOK-PIPELINE — Hook Pipeline & Context](../L3/L3-03_HOOK-PIPELINE_Hook_Pipeline_and_Context.md)
- [L2-18 GOV — GovernanceEngine](L2-18_GOV_GovernanceEngine.md)

### Originating system requirements
- HOOK-FR-001, HOOK-FR-002, HOOK-FR-003, HOOK-IF-001
