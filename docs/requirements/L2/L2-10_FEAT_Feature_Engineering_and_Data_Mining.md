# L2-10 — FEAT — Feature Engineering & Data Mining

## 1. Identifier
- **ID:** `L2-10`
- **Component:** FEAT — Feature Engineering & Data Mining
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Streaming feature computation from canonical market events (bars/ticks).
- Feature versioning and publication of FeatureVector events.
- Deterministic enrichment jobs (consume → compute → republish).
- Optional persistence of computed features for training/backtesting.

### Out of scope
- Large-scale distributed feature store (P2).
- Non-deterministic feature generation (must be deterministic for DB1).

## 3. Responsibilities
- Own feature pipeline definitions and feature_version management.
- Compute technical indicators and engineered features needed by SIG/OPT.
- Guarantee deterministic outputs given identical input streams.
- Publish FeatureVector events with provenance (inputs window_ref).

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from typing import Dict, Any, Optional, Protocol
from dataclasses import dataclass

@dataclass(frozen=True)
class FeatureVector:
    symbol: str
    tf: str
    feature_version: str
    features: Dict[str, float]
    window_ref: str

class FeaturePipeline(Protocol):
    pipeline_id: str
    feature_version: str
    def on_bar(self, bar: Dict[str, Any]) -> Optional[FeatureVector]: ...
    def warmup(self, history_bars: list[Dict[str, Any]]) -> None: ...
    def reset(self) -> None: ...

class FeatureEngine:
    def start(self) -> None: ...
    def stop(self) -> None: ...
```

### Events
**Consumes**
- ordinis.market.bar.closed
- ordinis.market.tick.received (optional)

**Emits**
- ordinis.features.vector.computed
- ordinis.ops.feature.reject

### Schemas
- EVT-DATA-012 FeatureVector
- EVT-DATA-011 Bar

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_FEAT_INVALID_INPUT` | Input bar/tick missing required fields | No |
| `E_FEAT_COMPUTE_FAILED` | Feature computation exception | No |
| `E_FEAT_VERSION_UNKNOWN` | Requested feature pipeline version not registered | No |

## 5. Dependencies
### Upstream
- SB for subscription to market streams
- EVT for schema validation
- CFG for enabling pipelines and persistence

### Downstream
- SIG consumes FeatureVector
- OPT consumes FeatureVector or derived returns/covariances
- LEARN consumes features for dataset build

### External services / libraries
- Numeric computing libs (numpy/pandas) for indicators

## 6. Data & State
### Storage
- Optional feature persistence: SQLite `features` table or Parquet partitioned by (symbol, tf, date).

### Caching
- Rolling window caches per symbol/tf for indicator computation.

### Retention
- Feature persistence retention per training/backtest needs; default 365 days for DB1 local store.

### Migrations / Versioning
- Feature schema adds must be backward compatible; breaking changes require feature_version bump.

## 7. Algorithms / Logic
### Compute model
- FEAT subscribes to bar events for each symbol/tf.
- Maintains rolling windows (e.g., last N bars) per symbol.
- On each bar close:
  1. Update rolling window.
  2. Compute indicators (returns, vol, momentum, RSI, MACD, etc.).
  3. Emit FeatureVector with `feature_version` and `window_ref`.

### Determinism
- All computations use fixed window definitions and deterministic numeric operations.
- No randomness without explicit seed stored in config snapshot.

### Edge cases
- Insufficient warmup: emit no FeatureVector until minimum bars met; emit `ops.feature.warmup`.
- NaNs: reject vector if any feature is NaN/inf; emit reject event to DLQ.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| FEAT_PIPELINES | list[str] | ["tech_v1"] | P0 | Enabled pipelines by id. |
| FEAT_WARMUP_BARS | int | 200 | P0 | Minimum bars before compute. |
| FEAT_PERSIST_ENABLE | bool | false | P0 | Persist feature vectors for training. |
| FEAT_REJECT_ON_NAN | bool | true | P0 | Reject NaN/inf vectors. |

**Environment variables**
- `ORDINIS_FEAT_PERSIST_ENABLE`

## 9. Non-Functional Requirements
- Compute latency: p95 < 20ms per bar per symbol for typical feature sets (<200 features).
- No schema-invalid FeatureVector events published.
- Feature computation must not allocate unbounded memory (window caps).

## 10. Observability
**Metrics**
- `feat_vectors_total{pipeline,result}`
- `feat_compute_latency_ms{pipeline}`
- `feat_nan_reject_total{pipeline}`
- `feat_window_size{pipeline,symbol}`

**Alerts**
- High reject rate (NaNs) indicates upstream data corruption.

## 11. Failure Modes & Recovery
- Compute exception: reject vector, emit to DLQ; do not crash pipeline.
- Memory pressure: shed load by disabling optional pipelines (config) or reducing universe.
- Persistence failure: continue compute but emit `ops.persist.error` (depending on strictness).

## 12. Test Plan
### Unit tests
- Deterministic outputs for known bar sequences (golden vectors).
- NaN handling and reject logic.

### Integration tests
- Replay ingestion → feature computation → signal consumption path.

### Acceptance criteria
- Feature vectors produced match expected hashes for benchmark datasets.

## 13. Open Questions / Risks
- Feature set definition for DB1 baseline (which indicators are mandatory vs optional).
- Do we standardize feature scaling/normalization here or inside SIG?

## 14. Traceability
### Parent
- DB1 System SRS: Mining/enrichment jobs read from bus and republish enriched events.

### Related
- [L2-11 SIG — SignalEngine](L2-11_SIG_SignalEngine_SignalCore.md)
- [L2-25 LEARN — LearningEngine](L2-25_LEARN_LearningEngine.md)

### Originating system requirements
- FEAT-FR-001..003
