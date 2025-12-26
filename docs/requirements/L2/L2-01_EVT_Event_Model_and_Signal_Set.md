# L2-01 — EVT — Event Model & Signal Set

## 1. Identifier
- **ID:** `L2-01`
- **Component:** EVT — Event Model & Signal Set
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Canonical event envelope, topic taxonomy, and naming rules for all engine-to-engine communication.
- Schema definitions for all DB1 event payloads (market, features, signals, risk, orders, executions, portfolio, analytics, governance, AI, learning, ops).
- Schema versioning policy (semver) and backward compatibility contract.
- Idempotency and correlation semantics (event_id, correlation_id, idempotency_key).
- Validation libraries/utilities for producers and consumers (runtime + CI).

### Out of scope
- Transport selection and delivery semantics (owned by StreamingBus).
- Vendor-specific raw message parsing (owned by ING).
- Strategy/model logic (owned by SIG).

## 3. Responsibilities
- Own the authoritative definition of the DB1 event ontology and payload schemas.
- Provide a validation API used at every publish/consume boundary.
- Provide schema-compatibility tests for CI (producer/consumer regression).
- Define shared enumerations (instrument_type, order_type, reason_code format, etc.).
- Define canonical error codes for schema violations and incompatible versions.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

EventType = str
SchemaVersion = str

@dataclass(frozen=True)
class EventEnvelope:
    event_id: str
    event_type: EventType
    schema_version: SchemaVersion
    ts_event: str
    ts_ingest: str
    source: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    correlation_id: str
    partition_key: str
    payload: Dict[str, Any]
    tags: Dict[str, str]
    idempotency_key: Optional[str] = None
    sig: Optional[str] = None  # HMAC or signature (optional in DB1)

def validate_envelope(evt: EventEnvelope) -> None: ...
def validate_payload(event_type: EventType, schema_version: SchemaVersion, payload: Dict[str, Any]) -> None: ...
def validate_event(evt: EventEnvelope) -> None: ...
def normalize_event_type(event_type: str) -> str: ...
def schema_hash(event_type: EventType, schema_version: SchemaVersion) -> str: ...
```

### Events
**Consumes**
_None._

**Emits**
_None._

### Schemas
- EVT-DATA-001 Canonical envelope (EventEnvelope)
- EVT-DATA-010..021 DB1 payload schemas (MarketTick, Bar, FeatureVector, Signal, RiskDecision, OrderIntent, OrderSubmitted, ExecutionReport, PositionUpdate, AnalyticsReport, AuditRecord, AIRequest/AIResponse)

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_EVT_SCHEMA_INVALID` | Envelope or payload failed schema validation | No |
| `E_EVT_SCHEMA_UNKNOWN` | Unknown event_type or schema_version not registered | No |
| `E_EVT_SCHEMA_INCOMPATIBLE` | Consumer cannot parse producer schema (compat break) | No |

## 5. Dependencies
### Upstream
_None._

### Downstream
- All engines/services publishing or consuming events
- StreamingBus schema gate
- GovernanceEngine audit records

### External services / libraries
- Schema validation runtime (JSON Schema or Pydantic models)
- Time library with nanos support

## 6. Data & State
### Storage
- Schema registry stored in-repo (e.g., `schemas/ordinis/.../*.json`) for DB1.
- Optional persisted schema snapshots tied to run_config_snapshot (PERSIST).

### Caching
- Compiled schema cache in-process keyed by (event_type, schema_version, schema_hash).

### Retention
- Schemas are immutable once released; retained indefinitely.

### Migrations / Versioning
- Schema changes follow semver; breaking changes require major version bump.
- CI enforces backward compatibility for minor/patch bumps.

## 7. Algorithms / Logic
### Core flow
1. Producer constructs `EventEnvelope` with required fields (IDs, timestamps, trace/correlation).
2. Producer calls `validate_event(evt)` prior to publish.
3. StreamingBus re-validates envelope/payload and rejects invalid events to DLQ.
4. Consumers validate again at handler boundary (defense-in-depth).

### Compatibility rules
- **Minor/Patch bump:** consumer must accept producer payload with unknown optional fields ignored.
- **Major bump:** consumer must explicitly opt-in to new major version.

### Edge cases
- Clock skew: `ts_ingest` is authoritative for ordering within ingestion systems; `ts_event` preserved for market truth.
- Duplicate delivery: side-effect-capable events must include `idempotency_key`.
- Partitioning: `partition_key` must be stable (e.g., symbol) to preserve per-asset ordering.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| EVT_SCHEMA_STRICT | bool | true | P0 | Reject unknown fields when true. |
| EVT_ALLOW_UNKNOWN_FIELDS | bool | false | P0 | If true, unknown fields are ignored with warning. |
| EVT_VALIDATE_ON_PUBLISH | bool | true | P0 | Producer-side validation. |
| EVT_VALIDATE_ON_CONSUME | bool | true | P0 | Consumer-side validation. |

**Environment variables**
- `EVT_SCHEMA_STRICT`
- `EVT_VALIDATE_ON_PUBLISH`

## 9. Non-Functional Requirements
- Validation overhead: p95 < 1ms per event for typical payload sizes (< 4KB).
- Zero ambiguous schemas: all required/optional fields explicitly declared.
- Backwards compatibility enforced in CI for non-major bumps.

## 10. Observability
**Metrics**
- `evt_validate_total{result=pass|fail,event_type}`
- `evt_validate_latency_ms{event_type}`
- `evt_schema_unknown_total{event_type,version}`

**Logs**
- On reject: include `event_id`, `event_type`, `schema_version`, `trace_id`, and first failing path.

**Traces**
- Envelope validation span: `evt.validate`.

## 11. Failure Modes & Recovery
- Schema reject at publish (producer error): fail-fast, emit `ops.schema.reject`.
- Schema reject at consume (consumer behind): route to DLQ; optionally pause consumer.
- Incompatible schema deployment: blocked by CI; if in prod, trigger kill-switch + rollback.

## 12. Test Plan
### Unit tests
- Validate envelope required fields (missing/invalid types).
- Validate each payload schema with boundary values and missing required fields.
- Validate event_type naming normalization and namespace enforcement.

### Integration tests
- StreamingBus publish/subscribe with schema gate enabled.
- Backward compatibility tests: old consumer parsing new producer (minor bump).

### Acceptance criteria
- 100% of published events in DB1 pass validation.
- Invalid events are rejected and appear in DLQ with a deterministic failure reason.

## 13. Open Questions / Risks
- Final decision: JSON Schema vs Pydantic-first schemas (or both with codegen).
- Do we sign events (HMAC) in DB1 or defer to DB2?
- Do we standardize numeric precision (float vs Decimal) for prices/fees?

## 14. Traceability
### Parent
- DB1 System SRS: Event-driven architecture with canonical schema.

### Children / Related
- [L3-01 EVT-SIGNALS — Event Ontology & Schemas](../L3/L3-01_EVT-SIGNALS_Event_Ontology_and_Schemas.md)
- [L2-07 SB — StreamingBus](L2-07_SB_StreamingBus.md)

### Originating system requirements
- EVT-DATA-001, EVT-DATA-002, EVT-DATA-003, EVT-DATA-004
- EVT-DATA-010..021
