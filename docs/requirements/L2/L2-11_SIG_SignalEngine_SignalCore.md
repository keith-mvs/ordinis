# L2-11 — SIG — SignalEngine (SignalCore)

## 1. Identifier
- **ID:** `L2-11`
- **Component:** SIG — SignalEngine (SignalCore)
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Generation of trading signals from FeatureVector/market frames.
- StrategyPack plugin API and model inference wrappers (non-LLM decisioning).
- Signal sanity checks and schema-conformant publication.
- Provenance metadata (strategy_id/model_id/feature_version).

### Out of scope
- Risk policy enforcement (RISK).
- Order routing and execution (EXEC).
- LLM-driven signals (explicitly disallowed for executable signals in DB1).

## 3. Responsibilities
- Own the strategy/plugin architecture for signal generation.
- Execute model inference deterministically and within latency budgets.
- Emit Signal events with TTL/horizon and confidence bounds enforced.
- Provide an optional explanation reference (not narrative) for audit/debug.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Protocol

Direction = Literal["LONG","SHORT","FLAT"]

@dataclass(frozen=True)
class Signal:
    signal_id: str
    symbol: str
    direction: Direction
    confidence: float
    horizon: str
    ttl_ms: int
    model_id: str
    strategy_id: str
    feature_version: str
    metadata: Dict[str, Any]

class StrategyPack(Protocol):
    strategy_id: str
    def on_frame(self, frame: Dict[str, Any], portfolio: Dict[str, Any]) -> List[Signal]: ...
    def warmup(self, history: List[Dict[str, Any]]) -> None: ...
    def reset(self) -> None: ...

class SignalEngine:
    def generate_signals(self, frame: Dict[str, Any], portfolio: Dict[str, Any]) -> List[Signal]: ...
```

### Events
**Consumes**
- ordinis.features.vector.computed
- ordinis.market.bar.closed (optional direct strategies)
- ordinis.portfolio.snapshot (optional for context)

**Emits**
- ordinis.signals.generated
- ordinis.signals.reject

### Schemas
- EVT-DATA-013 Signal payload
- EVT-DATA-012 FeatureVector

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_SIG_STRATEGY_FAILED` | StrategyPack threw exception | No |
| `E_SIG_INVALID_OUTPUT` | Signal failed sanity/schema checks | No |
| `E_SIG_WARMUP_INSUFFICIENT` | Warmup not met; no signals produced | No |

## 5. Dependencies
### Upstream
- FEAT provides feature vectors
- CFG defines enabled strategies/models and thresholds
- EVT schema validation
- HOOK/GOV preflight validation on output (HARD in prod)

### Downstream
- RISK consumes signals
- ANA consumes signals for analysis/attribution
- LEARN consumes signal/outcome pairs

### External services / libraries
- ML inference libs (xgboost/onnxruntime/torch) depending on strategy models

## 6. Data & State
### Storage
- Optional: persist raw signals to SQLite (`signals` table) for training/analysis.

### Caching
- In-memory loaded model artifacts keyed by model_id.
- Warmup buffers per symbol and strategy.

### Retention
- Signal persistence retention default 90 days (configurable).

### Migrations / Versioning
- Signal schema changes require EVT schema version bump; strategy/model versions tracked separately.

## 7. Algorithms / Logic
### Primary flow
1. Consume FeatureVector.
2. For each enabled StrategyPack:
   - If warmup satisfied, compute candidate signals.
3. Apply sanity checks:
   - confidence ∈ [0,1]
   - ttl_ms > 0
   - direction ∈ {LONG,SHORT,FLAT}
   - required provenance fields present
4. Publish `ordinis.signals.generated`.

### AI boundary (non-authority rule)
- Cortex may generate *analysis artifacts* and *code changes*, but cannot publish executable Signal events.
- Any LLM-derived suggestions must be converted into explicit StrategyPack logic and pass tests/governance.

### Edge cases
- Conflicting signals across strategies: publish all with provenance; downstream RISK/PORT may net/resolve.
- Missing feature keys: strategy decides; engine ensures outputs still valid.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| SIG_STRATEGIES | list[str] | ["baseline_momo_v1"] | P0 | Enabled strategies. |
| SIG_MAX_SIGNALS_PER_SYMBOL | int | 5 | P0 | Cap for safety. |
| SIG_DEFAULT_TTL_MS | int | 60000 | P0 | TTL for signals missing explicit TTL. |
| SIG_REJECT_ON_INVALID | bool | true | P0 | Reject invalid outputs rather than clamp silently. |

**Environment variables**
- `ORDINIS_SIG_STRATEGIES`

## 9. Non-Functional Requirements
- Signal generation latency: p95 < 50ms per symbol per cycle (bar-based).
- No invalid signals published; reject rate alertable.
- Deterministic output for deterministic inputs (no randomness unless seeded).

## 10. Observability
**Metrics**
- `sig_signals_total{strategy,result}`
- `sig_latency_ms{strategy}`
- `sig_reject_total{strategy,reason}`

**Logs**
- Strategy failures include strategy_id, symbol, feature_version, trace_id.

## 11. Failure Modes & Recovery
- Strategy exception: isolate to strategy; continue others; emit reject + alert on rate.
- Model artifact missing: disable strategy and emit `ops.strategy.disabled` (configurable).
- Warmup insufficient: emit no signal, track warmup state metrics.

## 12. Test Plan
### Unit tests
- Sanity checks for confidence bounds and TTL.
- StrategyPack contract tests (warmup/reset determinism).

### Integration tests
- FEAT→SIG→RISK pipeline on benchmark pack.

### Acceptance criteria
- At least one baseline strategy produces deterministic signals on benchmark windows.

## 13. Open Questions / Risks
- Baseline strategy set for DB1 profitability exploration (momentum/mean-reversion/vol breakout).
- Conflict resolution approach: netting in RISK vs PORT.

## 14. Traceability
### Parent
- DB1 System SRS: Signal generation from features/market data.

### Children / Related
- [L3-06 SIG-STRATEGY-PACKS — Strategy Pack API & Model Runner](../L3/L3-06_SIG-STRATEGY-PACKS_Strategy_Pack_API_and_Model_Runner.md)
- [L2-12 RISK — RiskEngine](L2-12_RISK_RiskEngine_RiskGuard.md)

### Originating system requirements
- SIG-IF-001, SIG-FR-001..004
