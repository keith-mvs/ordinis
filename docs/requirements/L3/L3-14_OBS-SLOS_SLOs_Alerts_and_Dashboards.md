# L3-14 — OBS-SLOS — SLOs, Alerts & Dashboards

## 1. Identifier
- **ID:** `L3-14`
- **Component:** OBS-SLOS — SLOs, Alerts & Dashboards
- **Level:** L3

## 2. Purpose / Scope
### In scope
- Service-level objectives (SLOs) for trading loop and AI dev-plane.
- Alert rules and thresholds for key failure modes.
- Dashboard metric sets (trading KPIs, system KPIs, AI KPIs).
- Runbook links and incident response triggers.

### Out of scope
- Vendor-specific dashboard tooling; uses Prometheus-style metric names.
- On-call schedules and paging integrations (external).

## 3. Responsibilities
- Define DB1 SLOs and how they are measured.
- Define alert thresholds and severities.
- Define minimal dashboard panels and their metric queries.

## 4. Interfaces
### Public APIs / SDKs
This spec defines observability standards consumed by [L2-06 OBS](../L2/L2-06_OBS_Observability.md).

### Core SLOs (DB1 baseline)
1. **Cycle success rate**
   - SLI: `orch_cycle_total{status="ok"}/orch_cycle_total`
   - Target: ≥ 99% (paper), ≥ 99.9% (prod)
2. **Execution submission success**
   - SLI: `exec_orders_total{status="submitted"}/exec_orders_total`
   - Target: ≥ 99% (paper), ≥ 99.5% (prod)
3. **Audit completeness**
   - SLI: `% exec submits with gov audit record`
   - Target: 100% (prod)
4. **Data freshness**
   - SLI: `ing_lag_ms p95`
   - Target: ≤ 2s (bar-based)

### Events
**Consumes**
_None._

**Emits**
- ordinis.ops.alert.raised
- ordinis.ops.slo.breached

### Schemas
- Alert event schema
- SLO breach event schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_SLO_BREACH` | SLO breached | No |

## 5. Dependencies
### Upstream
- OBS metrics and logs

### Downstream
- ORCH, EXEC, PERSIST, GOV, ING, SB consumers for metric emission

### External services / libraries
_None._

## 6. Data & State
### Storage
- SLO definitions stored in config files under `observability/slo.yaml` and alert rules under `observability/alerts.yaml`.

### Caching
- N/A

### Retention
- Metrics retention is deployment-defined; minimum recommended 30 days for DB1.

### Migrations / Versioning
- SLO definitions versioned; changes require change log and GOV approval in prod.

## 7. Algorithms / Logic
### Alert severities (recommended)
- SEV1: Kill-switch activated, audit write failure, ledger invariant breach, broker circuit open > threshold.
- SEV2: Data staleness persistent, high DLQ rate, high reject rate.
- SEV3: Analytics missing, model drift warnings, elevated latency.

### Dashboard panels (minimum)
Trading
- equity curve, drawdown
- slippage bps, fees
- turnover, exposure

System
- cycle latency p50/p95
- stage latencies
- error rates, DLQ counts
- DB lock contention

AI
- hel latency, refusal rate
- ctx schema failure rate
- syn retrieval empty rate

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| OBS_ALERT_ENABLE | bool | true | P0 | Emit alert events. |
| OBS_SLO_ENABLE | bool | true | P0 | Compute SLO SLIs. |
| OBS_SLO_WINDOW_MIN | int | 60 | P0 | Rolling window. |

## 9. Non-Functional Requirements
- SLO computation must not materially impact runtime (<1% CPU).
- Alerting must be reliable; if alert emission fails, log prominently.

## 10. Observability
This spec itself is the observability definition. See [L2-06 OBS](../L2/L2-06_OBS_Observability.md) for primitives.

## 11. Failure Modes & Recovery
- Missing metrics: SLO cannot be computed; treat as degraded and alert in prod.
- Alert storm: apply rate limits and dedupe by alert_key within a window.

## 12. Test Plan
### Unit tests
- Alert rule expressions compile and reference existing metric names.
- SLO definitions reference existing SLIs.

### Integration tests
- Simulate failures (audit write fail, data staleness) and verify alerts fire.

### Acceptance criteria
- A baseline dashboard shows system health and trading KPIs for any run.

## 13. Open Questions / Risks
- Finalize SLO targets for prod vs paper based on strategy frequency and broker reliability.
- Which alerts should auto-activate kill-switch (recommended: audit fail, ledger breach, broker outage).

## 14. Traceability
### Parent
- [L2-06 OBS — Observability](../L2/L2-06_OBS_Observability.md)

### Originating system requirements
- OBS-FR-001..003
