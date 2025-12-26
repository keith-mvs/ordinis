# L3-06 — SIG-STRATEGY-PACKS — Strategy Pack API & Model Runner

## 1. Identifier
- **ID:** `L3-06`
- **Component:** SIG-STRATEGY-PACKS — Strategy Pack API & Model Runner
- **Level:** L3

## 2. Purpose / Scope
### In scope
- StrategyPack plugin interface and lifecycle.
- Model artifact loading and inference runner contract (ONNX/torch/xgboost).
- Determinism rules and seeding requirements.
- Strategy pack packaging/versioning and registry.

### Out of scope
- Risk enforcement (RISK).
- Order routing (EXEC).
- LLM-based strategy decisions (disallowed for executable signals in DB1).

## 3. Responsibilities
- Define the StrategyPack contract used by SignalEngine.
- Provide a ModelRunner abstraction with standard preprocessing/postprocessing.
- Define strategy packaging (manifest + dependencies + model refs).
- Provide contract tests to verify strategy determinism and output sanity.

## 4. Interfaces
### Public APIs / SDKs
**StrategyPack interface**

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Literal

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
    version: str
    def warmup(self, history: List[Dict[str, Any]]) -> None: ...
    def on_frame(self, frame: Dict[str, Any], portfolio: Dict[str, Any]) -> List[Signal]: ...
    def reset(self) -> None: ...
```

**Strategy pack manifest (recommended)**
```json
{
  "strategy_id": "baseline_momo",
  "version": "1.0.0",
  "feature_version": "tech_v1",
  "model_artifacts": [{"artifact_id":"model_abc","role":"primary"}],
  "params": {"lookback": 60, "threshold": 0.7},
  "deterministic": true
}
```

### Events
**Consumes**
_None._

**Emits**
_None._

### Schemas
- Strategy pack manifest schema
- Signal payload schema (EVT-DATA-013)

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_STRAT_WARMUP_NOT_MET` | Warmup requirements not met | No |
| `E_STRAT_MODEL_MISSING` | Referenced model artifact missing | No |
| `E_STRAT_OUTPUT_INVALID` | Generated signal invalid | No |

## 5. Dependencies
### Upstream
- CFG for enabled strategies and parameters
- PERSIST/LEARN for model artifact registry

### Downstream
- SIG engine uses strategy packs
- BENCH uses strategy packs in pack runs

### External services / libraries
- Inference runtime(s) depending on model type

## 6. Data & State
### Storage
- Strategy pack manifests stored in repo or in `./data/strategies/` with checksums.

### Caching
- Loaded model artifacts cached per process with eviction based on memory budget.

### Retention
- Strategy packs retained indefinitely for reproducibility; mark deprecated but do not delete.

### Migrations / Versioning
- Strategy version bump required when feature_version changes or output semantics change.

## 7. Algorithms / Logic
### Lifecycle
1. Load StrategyPack + manifest.
2. Warmup with history bars/features until minimum lookback satisfied.
3. For each frame:
   - preprocess features
   - run inference (if model-based)
   - postprocess into 0..N Signal objects
4. Reset on run boundary.

### Determinism rules (DB1 hard requirements)
- No RNG unless seeded from config snapshot and recorded in manifest.
- Any stochastic model must run with fixed seeds and deterministic backend flags where supported.
- ModelRunner must record:
  - model_id/artifact_id
  - preprocessing version hash
  - inference runtime version

### Edge cases
- Strategy returns too many signals: engine will cap; strategy should respect `SIG_MAX_SIGNALS_PER_SYMBOL`.
- Strategy missing required features: may output none; must not crash.

## 8. Configuration
Strategy packs are configured in SIG config:
- `SIG_STRATEGIES` list
- per-strategy params in `SIG_STRATEGY_PARAMS.<strategy_id>`

## 9. Non-Functional Requirements
- StrategyPack must not exceed latency budget per frame (SIG enforces).
- Strategy outputs must validate against EVT schema and sanity constraints.

## 10. Observability
SIG metrics break down by `strategy_id`. Strategy packs should add:
- `strategy_custom_metric_*` only if stable and documented in manifest.

## 11. Failure Modes & Recovery
- Model artifact not found: strategy disabled; emit alert and skip.
- Inference error: treat as strategy failure; skip and alert.

## 12. Test Plan
### Unit/contract tests
- Warmup behavior determinism.
- Output schema validation on golden frames.
- Latency budget check using synthetic load.

### Acceptance criteria
- A strategy pack passes contract tests before being allowed in SIG_STRATEGIES for prod.

## 13. Open Questions / Risks
- ModelRunner standard: ONNX runtime for portability vs native framework per model type.
- Where to store strategy manifests: in repo (preferred) vs DB artifact store.

## 14. Traceability
### Parent
- [L2-11 SIG — SignalEngine](../L2/L2-11_SIG_SignalEngine_SignalCore.md)

### Originating system requirements
- SIG-FR-001..004
