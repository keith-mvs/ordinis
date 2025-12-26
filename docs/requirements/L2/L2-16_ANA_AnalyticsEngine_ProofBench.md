# L2-16 — ANA — AnalyticsEngine (ProofBench)

## 1. Identifier
- **ID:** `L2-16`
- **Component:** ANA — AnalyticsEngine (ProofBench)
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Performance analytics for backtests and live/paper runs.
- Trading KPIs: Sharpe/Sortino/CAGR/MaxDD/VaR/CVaR/turnover/slippage etc.
- System KPIs: cycle latency, error rates, DLQ rates, audit coverage.
- Optional narrative generation via Cortex (non-blocking).

### Out of scope
- GUI dashboards (visualization layer).
- Regulatory reporting formats (MiFID II/SEC filings) (P2).

## 3. Responsibilities
- Compute KPI scorecards for each run and benchmark pack.
- Emit AnalyticsReport events and persist metrics artifacts.
- Provide benchmarking comparisons to reference benchmarks (SPY etc.).
- Optionally call Cortex for narrative summaries (never required for core metrics).

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class AnalyticsReport:
    run_id: str
    period: str
    metrics: Dict[str, float]
    narrative_ref: Optional[str]
    artifacts: Dict[str, str]  # refs to curves/attribution

class AnalyticsEngine:
    def analyze(self, run_id: str) -> AnalyticsReport: ...
    def analyze_simulation(self, sim_results: Dict[str, Any]) -> AnalyticsReport: ...
```

### Events
**Consumes**
- ordinis.executions.report
- ordinis.portfolio.snapshot
- ordinis.signals.generated
- ordinis.risk.decision
- ordinis.ops.* (system metrics aggregation)

**Emits**
- ordinis.analytics.report
- ordinis.analytics.narrative.generated (optional)

### Schemas
- EVT-DATA-019 AnalyticsReport
- KPI schema (metrics map keys standardized)

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_ANA_INPUT_MISSING` | Required input series not available | No |
| `E_ANA_COMPUTE_FAILED` | Metrics computation failed | No |
| `E_ANA_NARRATIVE_FAILED` | Narrative generation failed (non-blocking) | No |

## 5. Dependencies
### Upstream
- PERSIST for reading trades/positions/ledger series
- CFG for KPI selection and benchmarks
- OBS for system KPI ingestion
- CTX/HEL/SYN optional for narrative

### Downstream
- LEARN uses analytics outcomes for labeling
- BENCH uses analytics reports for scorecards
- GOV may reference analytics for monitoring thresholds

### External services / libraries
- Math/stats libraries (numpy/scipy) for metrics

## 6. Data & State
### Storage
- Persist analytics metrics to `analytics_reports` table and artifact store (equity curve, drawdown series).

### Caching
- Cache intermediate time series for repeated KPI calculations within a run.

### Retention
- Reports retained 365 days (default) or aligned with audit retention in prod.

### Migrations / Versioning
- Metric key namespace is append-only; do not change meaning of existing keys without version bump.

## 7. Algorithms / Logic
### KPI computation (trading)
- Returns series from portfolio equity snapshots.
- Compute:
  - CAGR, volatility, Sharpe, Sortino
  - Max drawdown, Calmar
  - Win rate, profit factor
  - Turnover, holding period
  - Slippage and fees attribution
  - VaR/CVaR (historical; scenario-based optional via OPT)

### KPI computation (system)
- Cycle latency p50/p95, stage latencies
- Error rate, retry counts, DLQ volume
- Audit coverage: % actions with audit records

### Edge cases
- Short runs: Sharpe undefined; return NaN? DB1 rule: output 0 and include `metric_quality` flags.
- Missing benchmarks: omit benchmark metrics but preserve core metrics.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| ANA_BENCHMARK_SYMBOL | string | SPY | P0 | Benchmark for equity strategies. |
| ANA_METRICS_SET | list[str] | default | P0 | Which metrics to compute. |
| ANA_NARRATIVE_ENABLE | bool | false | P0 | Optional Cortex narrative. |
| ANA_VAR_CONFIDENCE | float | 0.95 | P0 | VaR/CVaR confidence. |

**Environment variables**
- `ORDINIS_ANA_NARRATIVE_ENABLE`

## 9. Non-Functional Requirements
- Analytics compute time: p95 < 2s for 1-year daily backtest with <10k trades.
- Metrics must be reproducible given identical inputs.
- Narrative generation must not block metrics production (hard requirement).

## 10. Observability
**Metrics**
- `ana_compute_latency_ms`
- `ana_reports_total{result}`
- `ana_narrative_fail_total`

**Alerts**
- Missing analytics for completed runs
- KPI anomalies (e.g., NaN rate)

## 11. Failure Modes & Recovery
- Missing inputs: emit report with `metric_quality=degraded` and explicit missing inputs list.
- Computation failure: fail analytics stage; ORCH may allow trading continuation depending on policy.
- Narrative failure: ignore; emit warning.

## 12. Test Plan
### Unit tests
- Metrics formulas against known fixtures.
- Handling of short series and missing data.

### Integration tests
- Run BENCH pack and ensure full KPI set produced.

### Acceptance criteria
- KPI set includes both trading and system KPIs for every completed run.

## 13. Open Questions / Risks
- Exact KPI key taxonomy (names) to standardize for dashboards and gating.
- Do we compute CVaR from realized returns only or from scenario generator (OPT integration)?

## 14. Traceability
### Parent
- DB1 System SRS: ProofBench provides backtesting/performance analysis and narrative.

### Related
- [L2-26 BENCH — Benchmark Packs](L2-26_BENCH_Benchmark_Packs_and_Backtest_Harness.md)
- [L2-23 CTX — Cortex](L2-23_CTX_Cortex_LLM_Engine.md)

### Originating system requirements
- ANA-IF-001, ANA-FR-001..003
