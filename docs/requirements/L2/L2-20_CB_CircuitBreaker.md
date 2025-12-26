# L2-20 — CB — CircuitBreaker

## 1. Identifier
- **ID:** `L2-20`
- **Component:** CB — CircuitBreaker
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Standard circuit breaker implementation for external dependencies (broker, data vendor, model endpoints).
- State machine CLOSED/OPEN/HALF_OPEN with configurable thresholds.
- Fast-fail behavior in OPEN state.

### Out of scope
- Distributed CB state across hosts (single-host DB1).
- Adaptive ML-based CB tuning (P2).

## 3. Responsibilities
- Provide reusable CB library with deterministic behavior.
- Expose metrics and events when CB state changes.
- Integrate with EXEC/HEL/ING adapters.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Literal, Any, Dict

CBState = Literal["CLOSED","OPEN","HALF_OPEN"]

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int
    success_threshold: int
    open_timeout_ms: int

class CircuitBreaker:
    def __init__(self, name: str, cfg: CircuitBreakerConfig): ...
    def state(self) -> CBState: ...
    def call(self, fn: Callable[[], Any]) -> Any: ...
    def record_success(self) -> None: ...
    def record_failure(self, err: Exception) -> None: ...
```

### Events
**Consumes**
_None._

**Emits**
- ordinis.ops.cb.state_changed
- ordinis.ops.cb.fast_fail

### Schemas
- CB state change event schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_CB_OPEN` | Circuit breaker is OPEN; call blocked | Yes |
| `E_CB_HALF_OPEN_REJECT` | Circuit breaker HALF_OPEN test call failed; re-opened | Yes |

## 5. Dependencies
### Upstream
- CFG for thresholds and timeouts
- OBS for metrics

### Downstream
- EXEC broker adapter calls
- HEL backend adapter calls
- ING vendor adapter calls

### External services / libraries
_None._

## 6. Data & State
### Storage
- DB1: in-memory CB state; optional persistence on shutdown/startup (not required).

### Caching
- N/A (CB is already stateful).

### Retention
- CB events retained per ops log retention.

### Migrations / Versioning
- CB event schema versioned; backward compatible additions only.

## 7. Algorithms / Logic
### State machine
- CLOSED: all calls pass; failures increment counter.
- OPEN: calls fast-fail (E_CB_OPEN) until open_timeout expires.
- HALF_OPEN: allow limited test calls:
  - if success_count >= success_threshold => CLOSED
  - if any failure => OPEN

### Edge cases
- Time source monotonic clock for timeouts.
- Thread safety: CB must be safe for concurrent calls (lock or atomic counters).

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| CB_FAILURE_THRESHOLD | int | 5 | P0 | Failures before OPEN. |
| CB_SUCCESS_THRESHOLD | int | 2 | P0 | Successes to close from HALF_OPEN. |
| CB_OPEN_TIMEOUT_MS | int | 30000 | P0 | Open window. |

## 9. Non-Functional Requirements
- Fast-fail latency in OPEN: p95 < 1ms.
- Thread-safe under concurrent calls.

## 10. Observability
**Metrics**
- `cb_state{breaker,state}`
- `cb_failures_total{breaker}`
- `cb_fast_fail_total{breaker}`

**Alerts**
- CB stuck OPEN beyond threshold (dependency outage).

## 11. Failure Modes & Recovery
- CB misconfiguration (threshold=0): fail startup validation.
- Clock anomalies: use monotonic clock to avoid time warp.

## 12. Test Plan
### Unit tests
- Transition CLOSED→OPEN on failures.
- OPEN→HALF_OPEN after timeout.
- HALF_OPEN→CLOSED on successes.
- Fast fail behavior timing.

### Integration tests
- EXEC broker calls wrapped with CB; simulate failure to open CB.

### Acceptance criteria
- CB protects EXEC and HEL from cascading failures.

## 13. Open Questions / Risks
- Do we persist CB state across restarts (probably unnecessary for DB1)?

## 14. Traceability
### Parent
- DB1 System SRS: Circuit breakers for external dependencies.

### Related
- [L2-13 EXEC — ExecutionEngine](L2-13_EXEC_ExecutionEngine_FlowRoute.md)
- [L2-21 HEL — Helix Provider](L2-21_HEL_Helix_LLM_Provider.md)

### Originating system requirements
- CB-FR-001, CB-FR-002
