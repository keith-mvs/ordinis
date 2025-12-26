# L3-04 — EXEC-BROKER-ADAPTER — Broker Adapter Contract

## 1. Identifier
- **ID:** `L3-04`
- **Component:** EXEC-BROKER-ADAPTER — Broker Adapter Contract
- **Level:** L3

## 2. Purpose / Scope
### In scope
- BrokerAdapter interface and required semantics (submit/cancel/fetch).
- Idempotency requirements (client_order_id/idempotency_key).
- Paper broker reference implementation contract.
- Error mapping and retry posture.

### Out of scope
- Smart routing across venues (P1).
- Co-location/HFT execution tuning (P2).

## 3. Responsibilities
- Define the adapter contract and canonical request/response shapes.
- Define error code mapping from broker-native errors to platform errors.
- Define reconciliation hooks (fetch_positions/account) for state sync.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from typing import Protocol, Dict, Any, List, Optional, Literal

class BrokerAdapter(Protocol):
    adapter_id: str
    def submit(self, order: Dict[str, Any]) -> Dict[str, Any]: ...
    def cancel(self, broker_order_id: str) -> Dict[str, Any]: ...
    def fetch_order(self, broker_order_id: str) -> Dict[str, Any]: ...
    def fetch_positions(self) -> List[Dict[str, Any]]: ...
    def fetch_account(self) -> Dict[str, Any]: ...
```

**Canonical submit request (OrderRequest)**
```json
{
  "client_order_id": "uuid",
  "idempotency_key": "uuid-or-hash",
  "symbol": "AAPL",
  "side": "buy|sell",
  "qty": 10,
  "order_type": "market|limit",
  "limit_price": 189.12,
  "time_in_force": "day|gtc",
  "tags": {"strategy_id":"...","risk_policy":"..."}
}
```

### Events
**Consumes**
_None._

**Emits**
- ordinis.ops.broker.error
- ordinis.ops.broker.reconcile

### Schemas
- OrderRequest schema
- BrokerReceipt schema (normalized)
- BrokerOrderStatus schema (normalized)

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_BROKER_RATE_LIMIT` | Broker rate limited request | Yes |
| `E_BROKER_TIMEOUT` | Broker request timed out | Yes |
| `E_BROKER_REJECT` | Broker rejected order (non-retryable) | No |
| `E_BROKER_AUTH` | Authentication/authorization failed | No |

## 5. Dependencies
### Upstream
- SEC for credentials
- CB for circuit breaking calls
- OBS for tracing

### Downstream
- EXEC uses adapter

### External services / libraries
- Broker SDK/HTTP client

## 6. Data & State
### Storage
- Adapter should not own persistence; EXEC/PERSIST do.

### Caching
- Adapter may cache auth tokens with TTL (securely) but must not log them.

### Retention
- Adapter must not persist secrets; receipts may be stored by EXEC as redacted artifacts.

### Migrations / Versioning
- Adapter interface version bump requires coordinated EXEC update and tests.

## 7. Algorithms / Logic
### Idempotency contract
- submit() must accept a stable `client_order_id` and `idempotency_key`.
- If broker supports idempotency: adapter passes through and uses broker idempotency features.
- If broker does not: adapter must detect duplicates by querying open orders by client_order_id and return the existing broker_order_id.

### Error mapping
- Map broker-native errors into standardized error codes and include:
  - broker_error_code
  - broker_message (redacted)
  - retry_after_ms (if provided)

### Reconciliation
- fetch_positions/account must return normalized shapes to allow EXEC/PORT reconcile.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| BROKER_BASE_URL | string | "" | P1 | If using HTTP broker. |
| BROKER_PAPER_STARTING_CASH | float | 100000.0 | P0 | Paper broker only. |
| BROKER_RATE_LIMIT_QPS | int | 5 | P0 | Default conservative. |

## 9. Non-Functional Requirements
- submit() must respect EXEC timeout budget and CB policy.
- fetch_positions/account must complete < 2s in DB1 for typical portfolios.

## 10. Observability
**Metrics**
- `broker_submit_total{adapter,result}`
- `broker_submit_latency_ms{adapter}`
- `broker_reject_total{adapter,reason}`

## 11. Failure Modes & Recovery
- Auth failure: fail fast; do not retry; alert.
- Rate limit: retry with backoff; respect broker Retry-After header if present.
- Partial outage: CB opens; EXEC blocks new orders; reconcile continues if possible.

## 12. Test Plan
### Unit tests
- Error mapping to standard codes.
- Idempotency: duplicate submit returns same broker_order_id.

### Integration tests
- Paper broker adapter handles submit/cancel/fetch and emits expected receipts.

### Acceptance criteria
- EXEC can swap adapters without changing upstream engine code.

## 13. Open Questions / Risks
- Which live broker is targeted first (Alpaca, IBKR, Tradier, etc.)?
- How to normalize broker order statuses across vendors (state machine mapping).

## 14. Traceability
### Parent
- [L2-13 EXEC — ExecutionEngine](../L2/L2-13_EXEC_ExecutionEngine_FlowRoute.md)

### Originating system requirements
- EXEC-IF-001, EXEC-FR-001..005
