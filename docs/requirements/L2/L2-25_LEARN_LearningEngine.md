# L2-25 — LEARN — LearningEngine

## 1. Identifier
- **ID:** `L2-25`
- **Component:** LEARN — LearningEngine
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Outcome capture and labeling for signals/trades/strategies.
- Dataset building for SIG models and RAG eval sets.
- Training/evaluation harness integration and model artifact registry.
- Governed rollout gates, shadow mode, rollback and drift monitoring.

### Out of scope
- Full-scale MLOps platform (Kubeflow etc.) (P2).
- Online learning in production (DB1 offline only).

## 3. Responsibilities
- Record events/outcomes into training datasets with provenance.
- Train new model artifacts and generate evaluation reports.
- Gate rollouts using benchmark packs and acceptance thresholds.
- Monitor drift for deployed models (signals + LLM) and raise alerts.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class ModelArtifact:
    artifact_id: str
    model_type: str        # signal_model, embed_model, prompt_pack, etc.
    version: str
    path_ref: str
    metrics: Dict[str, float]
    created_at: str
    data_snapshot_ref: str

class LearningEngine:
    def record_event(self, event: Dict[str, Any]) -> None: ...
    def build_dataset(self, dataset_id: str, window: Dict[str, Any]) -> str: ...
    def train(self, train_config: Dict[str, Any]) -> ModelArtifact: ...
    def evaluate(self, artifact_id: str, eval_pack_id: str) -> Dict[str, Any]: ...
    def rollout(self, artifact_id: str, target: Dict[str, Any]) -> Dict[str, Any]: ...
```

### Events
**Consumes**
- ordinis.signals.generated
- ordinis.risk.decision
- ordinis.executions.report
- ordinis.portfolio.snapshot
- ordinis.analytics.report
- ordinis.ai.response (for LLM drift)

**Emits**
- ordinis.learn.dataset.built
- ordinis.learn.model.trained
- ordinis.learn.eval.completed
- ordinis.learn.rollout.completed
- ordinis.learn.drift.alert

### Schemas
- ModelArtifact schema
- Dataset manifest schema
- Eval report schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_LEARN_DATA_MISSING` | Not enough labeled data for training | No |
| `E_LEARN_TRAIN_FAILED` | Training job failed | Yes |
| `E_LEARN_EVAL_FAILED` | Evaluation failed | Yes |
| `E_LEARN_ROLLOUT_DENIED` | Governance gate denied rollout | No |

## 5. Dependencies
### Upstream
- PERSIST for storing datasets and artifacts
- CFG for training parameters and thresholds
- GOV for rollout approval and audit
- BENCH for evaluation packs

### Downstream
- SIG consumes trained signal model artifacts (via model registry path refs)
- SYN consumes updated embeddings/index (optional)
- HEL/CTX consume prompt packs/safety profiles (optional)

### External services / libraries
- ML training libs (xgboost/torch) depending on model types

## 6. Data & State
### Storage
- Artifacts stored as files (e.g., `./data/models/...`) with references stored in SQLite `model_registry` table.
- Datasets stored as Parquet in `./data/datasets/...` with manifest in DB.

### Caching
- Cache recent dataset manifests and model artifacts metadata.

### Retention
- Datasets retained 90-365 days (configurable).
- Model artifacts retained indefinitely or until replaced by policy (keep last N).

### Migrations / Versioning
- Model registry schema versioned; artifacts include code commit and training config hash.

## 7. Algorithms / Logic
### Data capture and labeling
- For each signal_id:
  - capture features + signal + risk decision
  - when execution/portfolio outcome realized, label profit/loss and risk-adjusted outcome
- Store labels with horizon (e.g., 1d/5d/20d) to support multiple objectives.

### Evaluation gates (baseline)
- Signal models must beat baseline on:
  - Sharpe improvement threshold
  - max drawdown not worse than baseline beyond tolerance
- LLM prompt packs must beat baseline on:
  - schema-valid output rate
  - hallucination proxy (citation coverage)

### Drift monitoring (baseline)
- Signal model drift: feature distribution shift, prediction distribution shift, performance decay.
- LLM drift: refusal rate drift, latency drift, schema-fail drift, embedding shift.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| LEARN_ENABLE | bool | true | P0 | Data capture always-on. |
| LEARN_TRAIN_SCHEDULE | string | manual | P0 | `manual` in DB1. |
| LEARN_MIN_SAMPLES | int | 1000 | P0 | Minimum training samples. |
| LEARN_EVAL_PACKS | list[str] | ["bench_core_v1"] | P0 | Required eval packs. |
| LEARN_ROLLOUT_MODE | string | shadow | P0 | `shadow` then `active`. |

## 9. Non-Functional Requirements
- Reproducibility: training jobs record data snapshot refs and config hashes.
- No ungoverned rollouts: promotion to active requires GOV approval.
- Drift detection runs at least daily in DB1 (or per backtest batch).

## 10. Observability
**Metrics**
- `learn_events_total{type}`
- `learn_train_total{model_type,status}`
- `learn_eval_total{pack,status}`
- `learn_drift_score{model,metric}`

**Alerts**
- Drift threshold breach
- Repeated train failures

## 11. Failure Modes & Recovery
- Training failure: retry on transient; keep last known good artifact active.
- Data missing: no-op with explicit error; alert only if expected schedule.
- Rollout denied: record decision and keep current artifact.

## 12. Test Plan
### Unit tests
- Dataset builder output schema and labeling correctness.
- Model registry CRUD and versioning.

### Integration tests
- Train a toy model on a small dataset and evaluate against a benchmark pack.

### Acceptance criteria
- Any rollout produces an audit record including eval metrics and approvals.

## 13. Open Questions / Risks
- What is the minimum viable labeled dataset structure for DB1 strategies?
- How to store and evaluate prompt packs for Cortex (prompt versioning).

## 14. Traceability
### Parent
- DB1 System SRS: Continuous learning with eval and governed rollout.

### Children / Related
- [L3-12 LEARN-ROLL-OUT — Model Rollout & Drift Monitoring](../L3/L3-12_LEARN-ROLL-OUT_Model_Rollout_and_Drift_Monitoring.md)
- [L2-26 BENCH — Benchmark Packs](L2-26_BENCH_Benchmark_Packs_and_Backtest_Harness.md)

### Originating system requirements
- LEARN-IF-001, LEARN-FR-001..004
