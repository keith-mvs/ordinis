# L2-09 — ING — Ingestion & Normalization

## 1. Identifier
- **ID:** `L2-09`
- **Component:** ING — Ingestion & Normalization
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Market/alt-data adapter interface and DB1 reference adapters (file/CSV/Parquet replay; optional live vendor adapter).
- Normalization into canonical EVT payloads (MarketTick/Bar/News if enabled).
- Staleness detection and data-quality tagging.
- Publishing normalized events to StreamingBus.

### Out of scope
- Commercial vendor contracting and licensing.
- Long-term raw data archival pipelines (BENCH/PERSIST handle benchmark storage).

## 3. Responsibilities
- Own the provider adapter abstraction for raw inputs.
- Own normalization to canonical schemas and schema validation gate before publish.
- Emit data-quality and staleness signals used by RISK/GOV.
- Support deterministic replay for backtests.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from typing import Protocol, Iterator, Any, Dict, Optional
from dataclasses import dataclass

class DataSourceAdapter(Protocol):
    adapter_id: str
    def connect(self) -> None: ...
    def read(self) -> Iterator[Dict[str, Any]]: ...      # provider-native messages
    def close(self) -> None: ...

@dataclass(frozen=True)
class NormalizedBatch:
    events: list[Dict[str, Any]]  # EventEnvelope dicts
    source_lag_ms: float

class IngestionEngine:
    def ingest_once(self) -> NormalizedBatch: ...
    def ingest_stream(self) -> None: ...
```

### Events
**Consumes**
- Provider-native (non-EVT) messages from adapters

**Emits**
- ordinis.market.tick.received
- ordinis.market.bar.closed
- ordinis.ops.data.stale
- ordinis.ops.data.quality

### Schemas
- EVT-DATA-010 MarketTick
- EVT-DATA-011 Bar
- EVT-DATA-001 EventEnvelope

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_ING_PROVIDER_DOWN` | Data provider adapter unavailable | Yes |
| `E_ING_NORMALIZE_FAILED` | Provider message could not be normalized | No |
| `E_ING_DATA_STALE` | Data lag exceeded threshold | Yes |

## 5. Dependencies
### Upstream
- CFG for provider selection and thresholds
- SEC for provider API keys
- EVT for schema validation

### Downstream
- SB for publish
- FEAT and SIG (consume market events)
- RISK (data stale can block trading)

### External services / libraries
- File readers (CSV/Parquet) for replay
- Optional vendor SDKs (Polygon/IEX/Finnhub) for live mode

## 6. Data & State
### Storage
- Optional: persist raw provider messages for replay (DB1: off by default).
- Benchmark raw data stored via BENCH pack storage.

### Caching
- Connection/session objects cached per adapter; reconnect logic on failure.

### Retention
- Raw storage retention governed by BENCH/PERSIST policies if enabled.

### Migrations / Versioning
- Normalization mapping versioned per provider adapter version.

## 7. Algorithms / Logic
### Normalize → publish
1. Read provider-native message.
2. Map to canonical MarketTick/Bar payload.
3. Wrap in EventEnvelope with trace/correlation (or attach to upstream correlation for replay).
4. Validate via EVT and publish to SB.

### Staleness detection
- Compute `lag = now_utc - ts_event`.
- If lag > `ING_MAX_STALENESS_MS`, emit `ops.data.stale` and mark all downstream stages `blocked_by_data`.

### Edge cases
- Out-of-order ticks: preserve ts_event ordering per symbol where feasible; downstream can re-order within window.
- Missing fields: treat as normalize failure; route to DLQ with reason and provider message snippet (redacted).

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| ING_PROVIDER | string | file_replay | P0 | `file_replay` or vendor adapter id. |
| ING_MAX_STALENESS_MS | int | 2000 | P0 | Threshold for stale market data. |
| ING_SYMBOL_UNIVERSE | list[str] | [] | P0 | Symbols to ingest. |
| ING_TIMEFRAME | string | 1m | P0 | Bar aggregation timeframe for replay. |
| ING_PUBLISH_TICKS | bool | false | P0 | Enable tick stream (high volume). |

**Environment variables**
- `ORDINIS_ING_PROVIDER`

## 9. Non-Functional Requirements
- Normalization correctness: 0 schema-invalid events emitted in steady state.
- Throughput: sustain bar-based ingestion for 5k symbols at 1m bars on a single host.
- Backtest replay determinism: same input files yield identical emitted events.

## 10. Observability
**Metrics**
- `ing_messages_total{provider,result}`
- `ing_normalize_latency_ms{provider}`
- `ing_lag_ms{provider}`
- `ing_stale_total{provider}`

**Alerts**
- Provider down
- Persistent data staleness

## 11. Failure Modes & Recovery
- Provider disconnect: reconnect with exponential backoff; if exceeds threshold, trigger kill-switch in prod.
- Normalization failures: emit to DLQ and continue; if rate exceeds threshold, block trading.
- Stale data: emit `ops.data.stale` and block SIG/EXEC per ORCH policy.

## 12. Test Plan
### Unit tests
- Provider message to canonical payload mapping (golden fixtures).
- Staleness computation.

### Integration tests
- Replay a known dataset file and assert produced events match expected hashes.

### Acceptance criteria
- In replay mode, ING produces deterministic bar events for the entire universe and window.

## 13. Open Questions / Risks
- Tick ingestion in DB1: default off to avoid TB/month growth; confirm strategy frequency targets.
- Do we support news ingestion in DB1 or defer to later phase?

## 14. Traceability
### Parent
- DB1 System SRS: Ingest and normalize data into the bus.

### Related
- [L2-07 SB — StreamingBus](L2-07_SB_StreamingBus.md)
- [L2-01 EVT — Event Model](L2-01_EVT_Event_Model_and_Signal_Set.md)

### Originating system requirements
- ING-FR-001..003
