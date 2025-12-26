# L2-07 — SB — StreamingBus

## 1. Identifier
- **ID:** `L2-07`
- **Component:** SB — StreamingBus
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Unified event bus for ingesting and fan-out of all platform events.
- Publish/subscribe API with schema validation gates.
- Backpressure controls and dead-letter queue routing.
- In-memory adapter for DB1; pluggable adapters (Redis Streams/Kafka) for later phases.

### Out of scope
- Cross-datacenter replication and geo-failover (P2).
- Guaranteed exactly-once semantics (DB1 uses at-least-once).

## 3. Responsibilities
- Transport events between engines with trace/correlation propagation intact.
- Enforce EVT schema validation at publish and subscribe boundaries.
- Provide subscriber lifecycle management and bounded concurrency.
- Provide DLQ for rejected or repeatedly failing events.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Protocol, Literal

@dataclass(frozen=True)
class PublishAck:
    ok: bool
    event_id: str
    ts_published: str

@dataclass(frozen=True)
class SubscriptionFilter:
    topic: str              # supports wildcards e.g. ordinis.market.*
    partition_key: Optional[str] = None

Handler = Callable[[Dict[str, Any]], None]  # receives EventEnvelope dict

class StreamingBus(Protocol):
    def publish(self, event: Dict[str, Any]) -> PublishAck: ...
    def subscribe(self, flt: SubscriptionFilter, handler: Handler) -> str: ...
    def unsubscribe(self, subscription_id: str) -> None: ...
    def health(self) -> Dict[str, Any]: ...
```

### Events
**Consumes**
- All producers publish DB1 events via `publish`

**Emits**
- ordinis.ops.dlq.appended
- ordinis.ops.schema.reject
- ordinis.ops.bus.backpressure

### Schemas
- EVT-DATA-001 EventEnvelope (required for all publishes)
- DLQ record schema (envelope + failure metadata)

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_SB_SCHEMA_REJECT` | Event rejected due to schema validation failure | No |
| `E_SB_BACKPRESSURE` | Publish/dispatch rejected due to queue limits | Yes |
| `E_SB_HANDLER_FAILED` | Subscriber handler failed and exceeded retry policy | No |
| `E_SB_UNAVAILABLE` | Bus adapter unavailable | Yes |

## 5. Dependencies
### Upstream
- EVT for validation rules
- CFG for adapter selection and limits
- OBS for metrics/tracing

### Downstream
- ORCH (cycle triggers and orchestration events)
- All engines consuming streams (ING/FEAT/SIG/RISK/EXEC/PORT/ANA/LEARN/BENCH)

### External services / libraries
- DB1: in-process queue implementation (asyncio/threads)
- P1: Redis Streams client
- P2: Kafka client

## 6. Data & State
### Storage
- DB1: in-memory queues; optional persistence of DLQ to SQLite (`dlq` table).

### Caching
- Subscription routing tables cached in-memory (topic wildcard resolution).

### Retention
- DLQ retention per PERSIST policies; in-memory events are ephemeral.

### Migrations / Versioning
- DLQ schema versioned; backward compatible additions only.

## 7. Algorithms / Logic
### Publish flow
1. Validate event via EVT (envelope + payload).
2. Route to matching subscriptions using topic wildcard matching.
3. Enforce backpressure: if queue depth > max, reject publish or spill to DLQ depending on policy.
4. Dispatch to handlers with bounded concurrency.

### Delivery semantics (DB1)
- At-least-once per subscriber.
- Handler failures trigger retry with exponential backoff up to `SB_MAX_RETRIES`.
- After retries exceeded, event is appended to DLQ with failure metadata.

### Edge cases
- Poison messages: deterministic DLQ routing with error_code and stack hash.
- Handler slowness: per-subscription concurrency cap; slow handler cannot starve others.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| SB_BACKEND | string | inmemory | P0 | `inmemory` in DB1. |
| SB_MAX_QUEUE_DEPTH | int | 10000 | P0 | Per-topic queue cap. |
| SB_MAX_CONCURRENCY | int | 64 | P0 | Max concurrent handlers. |
| SB_MAX_RETRIES | int | 3 | P0 | Handler retry attempts. |
| SB_RETRY_BACKOFF_MS | int | 50 | P0 | Base backoff. |
| SB_DLQ_ENABLE | bool | true | P0 | DLQ routing on failure. |
| SB_VALIDATE_ON_PUBLISH | bool | true | P0 | EVT gate. |

**Environment variables**
- `ORDINIS_SB_BACKEND`

## 9. Non-Functional Requirements
- Dispatch latency (in-memory): p95 < 5ms under 1k events/sec steady-state.
- No unbounded memory growth: queue depth caps enforced.
- Schema rejection rate and DLQ rate observable and alertable.

## 10. Observability
**Metrics**
- `sb_publish_total{result,event_type}`
- `sb_publish_latency_ms`
- `sb_dispatch_latency_ms{subscription}`
- `sb_queue_depth{topic}`
- `sb_dlq_total{reason}`

**Logs**
- DLQ appends include event_id, event_type, error_code, trace_id.

**Traces**
- `sb.publish`, `sb.dispatch` spans with event_type attribute.

## 11. Failure Modes & Recovery
- Adapter failure: publish fails fast with E_SB_UNAVAILABLE; ORCH may degrade or halt depending on profile.
- Queue overflow: reject publish with E_SB_BACKPRESSURE or route to DLQ.
- Handler failure: retry; if exhausted, DLQ.

## 12. Test Plan
### Unit tests
- Topic routing with wildcards.
- Backpressure enforcement.
- Retry + DLQ routing for handler failures.
- Schema reject path.

### Integration tests
- End-to-end publish from ING and consume in FEAT/SIG.

### Acceptance criteria
- Under synthetic load, bus remains stable and queue caps prevent memory blowout.

## 13. Open Questions / Risks
- Should DLQ always persist to SQLite in DB1 or keep in-memory with periodic flush?
- Wildcard topic matching semantics: glob vs regex vs exact prefix matching.

## 14. Traceability
### Parent
- DB1 System SRS: Event-driven architecture foundation.

### Children / Related
- [L3-02 SB-ADAPTERS — StreamingBus Adapters](../L3/L3-02_SB-ADAPTERS_StreamingBus_Adapters.md)
- [L2-01 EVT — Event Model](L2-01_EVT_Event_Model_and_Signal_Set.md)

### Originating system requirements
- SB-IF-001, SB-FR-001..005, SB-NFR-001
