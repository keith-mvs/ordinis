# L3-12 — LEARN-ROLL-OUT — Model Rollout & Drift Monitoring

## 1. Identifier
- **ID:** `L3-12`
- **Component:** LEARN-ROLL-OUT — Model Rollout & Drift Monitoring
- **Level:** L3

## 2. Purpose / Scope
### In scope
- Model registry states (candidate → shadow → active → retired).
- Rollout gating rules using BENCH scorecards and acceptance thresholds.
- Rollback procedure and last-known-good selection.
- Drift monitoring metrics and alert thresholds for signal and LLM models.

### Out of scope
- Online learning and automated retraining in production (DB1 offline).
- Enterprise MLOps pipeline orchestrators (P2).

## 3. Responsibilities
- Define rollout state machine and required artifacts per promotion.
- Define acceptance criteria thresholds for promotion to active.
- Define drift detection algorithms and response playbooks.

## 4. Interfaces
### Public APIs / SDKs
### Model rollout state machine
- `CANDIDATE` (trained, evaluated)  
- `SHADOW` (used for logging only; not used to make trading decisions)  
- `ACTIVE` (used by SIG/HEL where allowed)  
- `RETIRED` (kept for reproducibility but not selectable)

### Required artifacts for promotion
- Training config + hash
- Data snapshot reference + hash
- Evaluation scorecards across required BENCH packs
- Governance approval record

### Events
**Consumes**
- ordinis.learn.model.trained
- ordinis.learn.eval.completed
- ordinis.bench.scorecard.generated

**Emits**
- ordinis.learn.rollout.completed
- ordinis.learn.rollback.completed
- ordinis.learn.drift.alert

### Schemas
- ModelRegistryRecord schema
- DriftAlert schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_LEARN_GATE_FAILED` | Evaluation gate failed; cannot promote | No |
| `E_LEARN_APPROVAL_MISSING` | Governance approval missing | No |
| `E_LEARN_DRIFT_BREACH` | Drift metric breached threshold | No |

## 5. Dependencies
### Upstream
- BENCH scorecards
- GOV approvals
- OBS for monitoring telemetry

### Downstream
- SIG model selection uses registry ACTIVE entries
- HEL prompt packs/model configs may use registry

### External services / libraries
_None._

## 6. Data & State
### Storage
- Model registry stored in SQLite `model_registry` with state + metadata + artifact refs.
- Drift metrics stored in `model_drift` table or time-series metrics via OBS.

### Caching
- ACTIVE model lookup cached in-memory for fast selection.

### Retention
- Keep last N retired models per type (default 10) for reproducibility and rollback.

### Migrations / Versioning
- Registry schema versioned; states are append-only (new states allowed).

## 7. Algorithms / Logic
### Acceptance gates (baseline DB1)
Signal models:
- Must improve risk-adjusted return vs baseline:
  - ΔSharpe >= +0.10 (configurable)
  - MaxDD not worse by more than +2% absolute (configurable)
- Must not increase system failure rates (DLQ/error rate) in benchmark runs.

LLM prompt packs / workflow schemas:
- Schema-valid output rate >= 99% on eval set
- Citation coverage >= threshold for grounded tasks
- Refusal rate within expected band

### Shadow mode
- Run candidate in parallel and log:
  - predictions
  - confidence distribution
  - latency
- Do not allow candidate to affect execution decisions.

### Drift detection (baseline)
- Feature drift: PSI or KS test on key features
- Prediction drift: divergence vs baseline distribution
- Performance drift: rolling Sharpe/return decay
- LLM drift: refusal/schema-fail/latency drift, embedding shift (if available)

### Response playbook
- On drift breach:
  - alert + create incident
  - optionally roll back to last-known-good ACTIVE model (configurable)

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| LEARN_GATE_DELTA_SHARPE | float | 0.10 | P0 | Promotion threshold. |
| LEARN_GATE_MAXDD_TOL | float | 0.02 | P0 | MaxDD tolerance. |
| LEARN_SHADOW_DURATION_DAYS | int | 7 | P0 | Shadow period before active. |
| LEARN_DRIFT_PSI_THRESH | float | 0.2 | P0 | PSI threshold. |

## 9. Non-Functional Requirements
- No model can become ACTIVE without passing gates and approvals (hard).
- Rollback must be O(1): switch pointer to last-known-good in registry.

## 10. Observability
**Metrics**
- `learn_model_state_total{model_type,state}`
- `learn_gate_pass_total{model_type,pack}`
- `learn_drift_alert_total{model_type,metric}`

**Alerts**
- Any drift breach
- Any gate failure for scheduled promotions

## 11. Failure Modes & Recovery
- Missing scorecards: treat as gate failure; do not promote.
- Approval service down: do not promote; retry later.

## 12. Test Plan
### Unit tests
- State machine transitions enforce prerequisites.
- Gate evaluation logic on synthetic scorecards.

### Integration tests
- Train toy model, run BENCH pack eval, promote to shadow, then active with GOV approval.

### Acceptance criteria
- Any ACTIVE model has an auditable lineage to training data and eval results.

## 13. Open Questions / Risks
- Exact drift tests for feature sets (which features are monitored).
- Whether rollbacks are automatic in prod or require explicit human confirmation.

## 14. Traceability
### Parent
- [L2-25 LEARN — LearningEngine](../L2/L2-25_LEARN_LearningEngine.md)

### Related
- [L2-26 BENCH — Benchmark Packs](../L2/L2-26_BENCH_Benchmark_Packs_and_Backtest_Harness.md)

### Originating system requirements
- LEARN-FR-001..004
