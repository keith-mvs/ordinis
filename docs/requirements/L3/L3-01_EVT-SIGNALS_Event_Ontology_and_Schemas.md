# L3-01 — EVT-SIGNALS — Event Ontology & Schemas

## 1. Identifier
- **ID:** `L3-01`
- **Component:** EVT-SIGNALS — Event Ontology & Schemas
- **Level:** L3

## 2. Purpose / Scope
### In scope
- Canonical topic namespaces and event_type strings for DB1.
- Payload schema definitions for core trading loop events.
- Schema validation rules and compatibility tests.

### Out of scope
- Bus adapter delivery semantics (SB).
- Business logic producing the events (owned by engines).

## 3. Responsibilities
- Define and maintain the authoritative list of DB1 event types.
- Define JSON schema (or equivalent) for each event payload and envelope.
- Define required tags fields (instrument_type, venue, jurisdiction, env).
- Provide CI contract tests for schema compatibility.

## 4. Interfaces
### Public APIs / SDKs
The EVT layer exposes schema artifacts and validators. Public APIs are defined in [L2-01 EVT](../L2/L2-01_EVT_Event_Model_and_Signal_Set.md).

        **Schema artifact layout (recommended)**
        - `schemas/ordinis/envelope/v1.json`
        - `schemas/ordinis/market/bar/v1.json`
        - `schemas/ordinis/features/vector/v1.json`
        - `schemas/ordinis/signals/signal/v1.json`
        - `schemas/ordinis/risk/decision/v1.json`
        - `schemas/ordinis/orders/intent/v1.json`
        - `schemas/ordinis/execution/report/v1.json`
        - `schemas/ordinis/portfolio/snapshot/v1.json`
        - `schemas/ordinis/analytics/report/v1.json`
        - `schemas/ordinis/gov/audit/v1.json`

        **Canonical event envelope (excerpt)**
        ```json
{
  "event_id": "uuid",
  "event_type": "ordinis.market.bar.closed",
  "schema_version": "1.0.0",
  "ts_event": "RFC3339",
  "ts_ingest": "RFC3339",
  "source": "ing.file_replay",
  "trace_id": "otel-trace-id",
  "span_id": "otel-span-id",
  "correlation_id": "uuid",
  "partition_key": "AAPL",
  "payload": { },
  "tags": {
    "instrument_type": "equity",
    "venue": "paper",
    "jurisdiction": "US"
  }
}
```

### Events
**Consumes**
_None._

**Emits**
_None._

### Schemas
- `ordinis.market.bar.closed` (Bar v1)
- `ordinis.features.vector.computed` (FeatureVector v1)
- `ordinis.signals.generated` (Signal v1)
- `ordinis.risk.decision` (RiskDecision v1)
- `ordinis.orders.intent` (OrderIntent v1)
- `ordinis.orders.submitted` (OrderSubmitted v1)
- `ordinis.executions.report` (ExecutionReport v1)
- `ordinis.portfolio.snapshot` (PortfolioSnapshot v1)
- `ordinis.analytics.report` (AnalyticsReport v1)
- `ordinis.gov.audit.appended` (AuditRecord v1)

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_EVT_SCHEMA_INVALID` | Event payload/envelope invalid | No |
| `E_EVT_TOPIC_UNKNOWN` | event_type not in registry | No |
| `E_EVT_TAGS_MISSING` | Required tags missing (instrument_type/jurisdiction/env) | No |

## 5. Dependencies
### Upstream
- None (authoritative definitions)

### Downstream
- SB schema gate
- All producers/consumers
- GOV audit consumers

### External services / libraries
- JSON schema validator (implementation choice)

## 6. Data & State
### Storage
- Schema files stored in-repo under `schemas/` and optionally embedded into package distribution.

### Caching
- Compiled schema cache in-process.

### Retention
- Schema artifacts retained indefinitely; never delete released versions.

### Migrations / Versioning
- Schema versioning: semver; breaking changes require major bump + dual-publish period.

## 7. Algorithms / Logic
### Topic namespace rules
- Prefix: `ordinis.<domain>.<entity>.<verb>`
- Domains: `market`, `features`, `signals`, `risk`, `orders`, `executions`, `portfolio`, `analytics`, `gov`, `ai`, `learn`, `ops`
- Examples:
  - `ordinis.market.bar.closed`
  - `ordinis.signals.generated`

### Required tags (DB1 baseline)
- `instrument_type`: equity|etf|future|fx|crypto (expandable)
- `venue`: paper|sim|<broker_id>
- `jurisdiction`: US|EU|... (string)
- `env`: dev|paper|prod

### Compatibility checks (CI)
- For each schema:
  - validate example fixtures
  - validate backward compatibility for minor/patch bumps:
    - old schema validator must accept new payload with additional optional fields

## 8. Configuration
EVT schema behavior is configured by EVT settings (strictness, unknown fields). See [L2-01 EVT](../L2/L2-01_EVT_Event_Model_and_Signal_Set.md).

## 9. Non-Functional Requirements
- Schema coverage: 100% of event_type strings used in code exist in schema registry.
- Schema fixtures: each schema has at least one valid and one invalid fixture in CI.

## 10. Observability
EVT schema validation telemetry is emitted by producers and by SB. See [L2-01 EVT](../L2/L2-01_EVT_Event_Model_and_Signal_Set.md) and [L2-07 SB](../L2/L2-07_SB_StreamingBus.md).

## 11. Failure Modes & Recovery
- Unknown event_type: reject at SB publish and route to DLQ.
- Missing required tags: reject; this is a hard auditability failure.

## 12. Test Plan
### Unit/contract tests
- `schemas/` compile and validate with validator.
- Fixtures validate expected pass/fail.
- Compatibility tests between prior and current versions.

### Acceptance criteria
- No schema regressions; CI fails on any breaking change without major bump.

## 13. Open Questions / Risks
- Decide canonical precision types for money (float vs decimal); affects schema numeric types.
- Do we encode timestamps with nanos precision (string) or integer epoch nanos?

## 14. Traceability
### Parent
- [L2-01 EVT — Event Model & Signal Set](../L2/L2-01_EVT_Event_Model_and_Signal_Set.md)

### Related
- [L2-07 SB — StreamingBus](../L2/L2-07_SB_StreamingBus.md)

### Originating system requirements
- EVT-DATA-001..004, EVT-DATA-010..021
