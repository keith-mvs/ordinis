# L2-13 — EXEC — ExecutionEngine (FlowRoute)

## 1. Identifier
- **ID:** `L2-13`
- **Component:** EXEC — ExecutionEngine (FlowRoute)
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Order intent to broker submission (paper/live adapters) with idempotency.
- Fill simulation model for backtests.
- Order lifecycle events and reconciliation hooks.
- Circuit breaker and throttling around external broker calls.

### Out of scope
- High-frequency co-located execution optimization (HFT).
- Complex smart order routing across multiple venues (P1+).

## 3. Responsibilities
- Own OrderIntent → Order submission workflow and idempotency guarantees.
- Provide BrokerAdapter interface and DB1 paper broker implementation.
- Emit ExecutionReport and order lifecycle events.
- Ensure side-effect safety: persist before submit, dedupe on retry.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Protocol

@dataclass(frozen=True)
class ExecutionReport:
    order_id: str
    status: Literal["submitted","partial_fill","filled","rejected","canceled"]
    fills: List[Dict[str, Any]]
    avg_price: float
    fees: float
    slippage: float
    raw_broker_ref: Optional[str]

class BrokerAdapter(Protocol):
    adapter_id: str
    def submit(self, order: Dict[str, Any]) -> Dict[str, Any]: ...   # returns broker receipt
    def cancel(self, broker_order_id: str) -> Dict[str, Any]: ...
    def fetch_order(self, broker_order_id: str) -> Dict[str, Any]: ...
    def fetch_positions(self) -> List[Dict[str, Any]]: ...
    def fetch_account(self) -> Dict[str, Any]: ...

class ExecutionEngine:
    def execute(self, order_intent: Dict[str, Any], market_state: Dict[str, Any]) -> ExecutionReport: ...
    def cancel(self, order_id: str) -> Dict[str, Any]: ...
    def sync_broker_state(self) -> Dict[str, Any]: ...
```

### Events
**Consumes**
- ordinis.risk.decision (allowed intents)
- ordinis.orders.intent (internal)
- ordinis.market.* (for fill models in sim)

**Emits**
- ordinis.orders.submitted
- ordinis.executions.report
- ordinis.orders.rejected
- ordinis.ops.broker.circuit_open

### Schemas
- EVT-DATA-015 OrderIntent
- EVT-DATA-016 OrderSubmitted
- EVT-DATA-017 ExecutionReport

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_EXEC_IDEMPOTENCY_CONFLICT` | Duplicate idempotency_key maps to different order intent | No |
| `E_EXEC_BROKER_REJECTED` | Broker rejected order | No |
| `E_EXEC_BROKER_TIMEOUT` | Broker call timed out | Yes |
| `E_EXEC_CIRCUIT_OPEN` | Circuit breaker open; broker calls blocked | Yes |

## 5. Dependencies
### Upstream
- RISK provides allowed/adjusted intents
- CFG selects broker adapter and throttles
- PERSIST stores order intents and lifecycle
- CB provides circuit breaker around broker calls
- GOV enforces venue/instrument constraints

### Downstream
- PORT/LEDG consume execution reports to update positions/cash
- ANA consumes executions for slippage and performance attribution

### External services / libraries
- Broker APIs/SDKs (paper broker in DB1; live broker later)

## 6. Data & State
### Storage
- SQLite tables: `orders`, `order_events`, `fills` (PERSIST).
- Optional raw broker receipts stored as artifacts (redacted).

### Caching
- Idempotency cache by idempotency_key (also persisted).
- Throttling counters and CB state in memory (persist CB state optional).

### Retention
- Orders/fills retained per retention policy (default 365 days).
- Raw broker receipts retention shorter (default 30 days).

### Migrations / Versioning
- Broker adapter interface versioned; order schema changes require EVT schema bump.

## 7. Algorithms / Logic
### Execute flow (live/paper)
1. Receive OrderIntent (post-risk) with idempotency_key.
2. Persist intent (tx) with unique index on idempotency_key.
3. Check CB state; if OPEN => return E_EXEC_CIRCUIT_OPEN.
4. Submit to BrokerAdapter with timeout + retry policy (retry only if idempotency supported).
5. Persist submission receipt and emit `orders.submitted`.
6. Track fills:
   - Paper/live: subscribe to broker updates or poll.
   - Backtest: simulate fills using FillModel.
7. Emit ExecutionReport including fills, fees, slippage.

### Edge cases
- Crash after submit but before persistence: on restart, reconcile by querying broker using client_order_id.
- Partial fills: emit incremental execution reports; PORT applies idempotently.
- Rejects: include broker reason in redacted form.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| EXEC_BROKER_ADAPTER | string | paper | P0 | `paper` in DB1. |
| EXEC_TIMEOUT_MS | int | 2000 | P0 | Broker call timeout. |
| EXEC_MAX_RETRIES | int | 0 | P0 | Default 0 for live; sim can retry. |
| EXEC_MAX_ORDERS_PER_MIN | int | 120 | P0 | Throttle. |
| EXEC_SLIPPAGE_MODEL | string | fixed_bps | P0 | Backtest slippage model id. |

**Environment variables**
- `ORDINIS_EXEC_BROKER_ADAPTER`

## 9. Non-Functional Requirements
- Idempotency: duplicate order submission prevented under retries/restarts.
- Broker call timeout enforced; no unbounded blocking.
- Paper/backtest fill model deterministic for benchmark packs.

## 10. Observability
**Metrics**
- `exec_orders_total{status,adapter}`
- `exec_latency_ms{adapter}`
- `exec_reject_total{reason}`
- `exec_slippage_bps`
- `exec_circuit_state{adapter}`

**Alerts**
- Circuit breaker open
- Reject spike
- Order submit latency spike

## 11. Failure Modes & Recovery
- Broker API down: CB opens; orders blocked; trigger kill-switch if persistent.
- DB persistence failure: fail closed before side effects (do not submit).
- Fill tracking failure: degrade to polling; emit alert.

## 12. Test Plan
### Unit tests
- Idempotency_key uniqueness enforcement.
- Slippage and fee computation correctness (fill model).
- CB transitions and fast-fail behavior.

### Integration tests
- RISK→EXEC→PORT apply execution report updates positions correctly.
- Crash/restart simulation around submit path.

### Acceptance criteria
- No duplicate orders are created under retry/restart scenarios.

## 13. Open Questions / Risks
- Live broker target for Phase 1/DB1 (paper only vs real adapter).
- Order reconciliation strategy granularity: poll interval and rate limits.

## 14. Traceability
### Parent
- DB1 System SRS: FlowRoute routes and executes orders with fill models.

### Children / Related
- [L3-04 EXEC-BROKER-ADAPTER — Broker Adapter Contract](../L3/L3-04_EXEC-BROKER-ADAPTER_Broker_Adapter_Contract.md)
- [L2-20 CB — CircuitBreaker](L2-20_CB_CircuitBreaker.md)

### Originating system requirements
- EXEC-IF-001, EXEC-FR-001..005
