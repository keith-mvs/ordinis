# L2-08 — ORCH — OrchestrationEngine

## 1. Identifier
- **ID:** `L2-08`
- **Component:** ORCH — OrchestrationEngine
- **Level:** L2

## 2. Purpose / Scope
### In scope
- End-to-end pipeline coordination (data → feature → signal → risk → execution → portfolio → analytics).
- Cycle correlation IDs, trace roots, and run tracking.
- Mode selection: live cycle, paper cycle, offline backtest.
- Stage-level retries for idempotent stages and failure containment policies.

### Out of scope
- Distributed orchestration across multiple hosts (single-host DB1).
- UI scheduling (cron/ops tooling).

## 3. Responsibilities
- Define the canonical stage graph and ordering for DB1 execution plane.
- Create correlation_id and root trace for each cycle/backtest run.
- Invoke HOOK/GOV gates at stage boundaries.
- Persist run metadata and results references to PERSIST.
- Trigger kill-switch on catastrophic invariants violations per policy.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

@dataclass(frozen=True)
class CycleResult:
    correlation_id: str
    status: Literal["ok","degraded","failed","blocked"]
    stage_results: Dict[str, Any]
    started_at: str
    finished_at: str

class OrchestrationEngine:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def status(self) -> Dict[str, Any]: ...
    def run_cycle(self, trigger_event: Dict[str, Any]) -> CycleResult: ...
    def run_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]: ...
```

### Events
**Consumes**
- ordinis.market.* (cycle triggers may be timer-based or market-based)
- ordinis.ops.run.backtest.requested

**Emits**
- ordinis.ops.cycle.started
- ordinis.ops.cycle.completed
- ordinis.ops.stage.failed
- ordinis.ops.kill.block

### Schemas
- CycleResult schema (stored as artifact)
- EVT canonical event envelope

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_ORCH_STAGE_FAILED` | A pipeline stage failed | No |
| `E_ORCH_BLOCKED` | Pipeline blocked by GOV/KILL | No |
| `E_ORCH_TIMEOUT` | Cycle exceeded time budget | Yes |

## 5. Dependencies
### Upstream
- CFG for stage enablement, time budgets, retry policy
- OBS for tracing and metrics
- SB for event triggers (optional) and stage events
- GOV/HOOK for gating

### Downstream
- ING, FEAT, SIG, RISK, EXEC, PORT, ANA, OPT, LEARN (optional), CTX/SYN/HEL (dev-plane)
- PERSIST for run metadata and artifacts

### External services / libraries
- Scheduler (timer) for periodic cycles (DB1: simple loop or cron wrapper)

## 6. Data & State
### Storage
- SQLite `runs` table: run_id/correlation_id, start/end timestamps, status.
- Artifacts stored in `artifacts` table or artifact store (reports, metrics JSON).

### Caching
- In-memory state: last cycle status, stage health cache.

### Retention
- Run metadata retained per DB retention policy (default 365 days dev/paper).

### Migrations / Versioning
- Run schema versioned; artifacts include schema_version and code commit hash.

## 7. Algorithms / Logic
### Canonical stage ordering (DB1)
1. **ING**: ingest + normalize market data (or replay data in backtest)
2. **FEAT**: compute feature vectors
3. **SIG**: generate candidate signals
4. **RISK**: evaluate and adjust/deny signals
5. **EXEC**: create/submit orders; capture fills
6. **PORT/LEDG**: apply executions to portfolio/ledger; update state
7. **ANA**: compute KPIs + performance snapshot
8. **GOV audit**: ensure audit completeness (HARD)

### Failure containment
- If **ANA** fails: mark cycle degraded; trading can continue (configurable).
- If **EXEC** fails: mark cycle failed; depending on policy, retry only if idempotent_key supported.
- If **GOV audit** fails: fail closed in prod; fail open in dev (configurable but not recommended).

### Retries
- Retries permitted only for stages annotated `idempotent=True`.
- Backoff: exponential with jitter.

### Edge cases
- Kill-switch active: short-circuit before SIG.
- Data stale: short-circuit before SIG and emit block reason.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| ORCH_CYCLE_BUDGET_MS | int | 5000 | P0 | End-to-end cycle budget. |
| ORCH_STAGE_BUDGET_MS | map | see defaults | P0 | Per-stage budgets. |
| ORCH_RETRY_MAX | int | 1 | P0 | Retries per stage (idempotent only). |
| ORCH_FAIL_CLOSED_ON_AUDIT | bool | true | P0 | In prod must be true. |
| ORCH_DEGRADED_ALLOW_EXEC | bool | true | P0 | Continue trading if analytics fails. |

**Environment variables**
- `ORDINIS_ORCH_CYCLE_BUDGET_MS`

## 9. Non-Functional Requirements
- Deterministic stage ordering and decision traceability per correlation_id.
- Cycle orchestration overhead: p95 < 10ms excluding downstream engine execution time.
- No duplicate side effects: idempotency_key enforced for order submission pathways.

## 10. Observability
**Metrics**
- `orch_cycle_total{status}`
- `orch_cycle_latency_ms`
- `orch_stage_latency_ms{stage}`
- `orch_stage_fail_total{stage,error_code}`

**Traces**
- One root span per cycle: `orch.cycle`.
- Child spans per stage: `orch.stage.<name>`.

**Alerts**
- Consecutive failed cycles
- Audit failures
- Kill-switch triggers

## 11. Failure Modes & Recovery
- Downstream engine failure: mark stage failed; apply retry policy; emit stage.failed.
- Time budget exceeded: stop remaining non-critical stages; emit cycle timeout.
- Persistence unavailable: fail closed if cannot write run metadata (configurable).

## 12. Test Plan
### Unit tests
- Stage ordering and gating logic.
- Retry policy only for idempotent stages.
- Degraded mode behavior (analytics failure does not block exec when allowed).

### Integration tests
- Full pipeline with in-memory bus and paper broker.
- Kill-switch engaged prevents order submission.

### Acceptance criteria
- A complete cycle produces: signals, risk decisions, (optional) orders, portfolio updates, analytics report, audit records.

## 13. Open Questions / Risks
- Cycle trigger model: timer-based vs event-based (ticks) in DB1 (recommend timer + buffered bars).
- How to represent partial completion in CycleResult for UI/ops consumption.

## 14. Traceability
### Parent
- DB1 System SRS: Orchestration coordinates the pipeline with retries/tracing.

### Related
- [L2-07 SB — StreamingBus](L2-07_SB_StreamingBus.md)
- [L2-18 GOV — GovernanceEngine](L2-18_GOV_GovernanceEngine.md)
- [L2-19 KILL — KillSwitch](L2-19_KILL_KillSwitch.md)

### Originating system requirements
- ORCH-IF-001, ORCH-FR-001..004
