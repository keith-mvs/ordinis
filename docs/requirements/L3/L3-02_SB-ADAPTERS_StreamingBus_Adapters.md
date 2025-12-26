# L3-02 — SB-ADAPTERS — StreamingBus Adapters

## 1. Identifier
- **ID:** `L3-02`
- **Component:** SB-ADAPTERS — StreamingBus Adapters
- **Level:** L3

## 2. Purpose / Scope
### In scope
- Adapter interface for StreamingBus backends.
- DB1 InMemory adapter implementation details.
- Planned Redis Streams adapter contract (P1) and Kafka adapter contract (P2).
- DLQ routing contract across adapters.

### Out of scope
- Business schemas (EVT) and engine logic.

## 3. Responsibilities
- Provide a stable Adapter interface that satisfies StreamingBus contract.
- Define adapter capabilities (ordering, persistence, partitions).
- Define DLQ append contract and record schema.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, Protocol

@dataclass(frozen=True)
class AdapterCaps:
    persistent: bool
    ordered_by_partition: bool
    max_payload_bytes: int
    supports_dlq: bool

class BusAdapter(Protocol):
    adapter_id: str
    def caps(self) -> AdapterCaps: ...
    def publish(self, event: Dict[str, Any]) -> None: ...
    def subscribe(self, topic_pattern: str, handler: Callable[[Dict[str, Any]], None]) -> str: ...
    def unsubscribe(self, sub_id: str) -> None: ...
    def dlq_append(self, record: Dict[str, Any]) -> None: ...
    def health(self) -> Dict[str, Any]: ...
```

### Events
**Consumes**
- All EVT events passed from StreamingBus facade

**Emits**
- ordinis.ops.dlq.appended
- ordinis.ops.bus.adapter.unhealthy

### Schemas
- DLQ record schema: {event, error_code, error_detail_hash, first_seen_ts, attempts, last_error_ts}

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_SB_ADAPTER_UNAVAILABLE` | Adapter backend unavailable | Yes |
| `E_SB_ADAPTER_PAYLOAD_TOO_LARGE` | Payload exceeds adapter max size | No |
| `E_SB_ADAPTER_DLQ_FAILED` | DLQ append failed | Yes |

## 5. Dependencies
### Upstream
- CFG selects adapter and parameters
- OBS for metrics

### Downstream
- SB facade used by all engines

### External services / libraries
- DB1 InMemory: Python threading/asyncio
- P1 Redis Streams: redis client
- P2 Kafka: kafka client

## 6. Data & State
### Storage
- InMemory adapter has no storage.
- Redis/Kafka adapters persist via their backends.
- DLQ persistence can be via SQLite `dlq` table (recommended DB1).

### Caching
- Subscription routing caches in SB facade; adapters may have connection pooling caches.

### Retention
- DLQ retention per PERSIST; backend retention policies apply for Redis/Kafka.

### Migrations / Versioning
- DLQ schema versioned; adapter-specific metadata fields must be namespaced under `adapter_meta`.

## 7. Algorithms / Logic
### InMemory adapter (DB1)
- One in-process dispatcher thread/task.
- Topic routing via prefix-glob match.
- Bounded queues per topic to enforce backpressure.
- Handler execution in a bounded worker pool.

### Redis Streams adapter (P1 contract)
- Use one stream per topic domain, or one per symbol partition (configurable).
- Consumer groups per subscriber.
- At-least-once delivery; ack after handler success.

### Kafka adapter (P2 contract)
- Topics per domain, partition key = `partition_key` in envelope.
- Consumer group per engine.
- Exactly-once not required; idempotency at consumer required.

### DLQ contract
- Record includes:
  - `event` (original envelope, potentially truncated for size)
  - `error_code`, `error_detail_hash`
  - `attempts`, `first_seen_ts`, `last_seen_ts`
  - `adapter_id`, `topic`

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| SB_BACKEND | string | inmemory | P0 | Adapter id. |
| SB_MAX_PAYLOAD_BYTES | int | 1048576 | P0 | 1MB default. |
| SB_REDIS_URL | string | "" | P1 | Only used if backend=redis. |
| SB_KAFKA_BOOTSTRAP | string | "" | P2 | Only used if backend=kafka. |

## 9. Non-Functional Requirements
- Adapters must preserve partition ordering when caps.ordered_by_partition = true.
- InMemory adapter must enforce bounded queues.

## 10. Observability
**Metrics**
- `sb_adapter_publish_total{adapter,result}`
- `sb_adapter_latency_ms{adapter,op}`
- `sb_adapter_unhealthy_total{adapter}`

## 11. Failure Modes & Recovery
- Backend down: adapter health returns unhealthy; SB facade returns E_SB_UNAVAILABLE; ORCH may halt or degrade.
- DLQ append failure: log + alert; in prod treat as critical (auditability).

## 12. Test Plan
### Unit tests
- InMemory queue caps and backpressure.
- Topic routing correctness.

### Integration tests
- Publish/subscribe across multiple topics and partitions; verify ordering per partition.

### Acceptance criteria
- InMemory adapter handles load without unbounded memory growth.

## 13. Open Questions / Risks
- Partition strategy for Redis/Kafka (topic-per-symbol vs partition key).
- DLQ storage: SQLite always-on vs adapter-native DLQ (recommended SQLite for DB1).

## 14. Traceability
### Parent
- [L2-07 SB — StreamingBus](../L2/L2-07_SB_StreamingBus.md)

### Related
- [L2-05 PERSIST — Persistence Layer](../L2/L2-05_PERSIST_Persistence_Layer.md)

### Originating system requirements
- SB-FR-001..005
