# L2-26 — BENCH — Benchmark Packs & Backtest Harness

## 1. Identifier
- **ID:** `L2-26`
- **Component:** BENCH — Benchmark Packs & Backtest Harness
- **Level:** L2

## 2. Purpose / Scope
### In scope
- Historical benchmark packs: 3/6/9/12-month windows sampled across 10–15 years.
- Regime tagging (bull/bear/vol spike) and stress windows.
- Replay harness that runs packs through ORCH pipeline deterministically.
- Scorecard generation with trading + system KPIs.

### Out of scope
- External dataset acquisition/licensing.
- Massive distributed backtesting clusters (single-host DB1).

## 3. Responsibilities
- Define pack manifest schema and sampling rules (reproducible seeds).
- Produce replay streams identical to live ingestion events (EVT compliant).
- Compute scorecards using ANA outputs; gate models via LEARN.
- Persist pack artifacts and reports for reproducibility.

## 4. Interfaces
### Public APIs / SDKs
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class BenchPackManifest:
    pack_id: str
    universe: List[str]
    start: str
    end: str
    timeframe: str
    regimes: List[str]
    seed: int
    data_refs: Dict[str, str]

class BenchmarkHarness:
    def generate_pack(self, spec: Dict[str, Any]) -> BenchPackManifest: ...
    def run_pack(self, pack_id: str, orch_config: Dict[str, Any]) -> Dict[str, Any]: ...
    def score_pack(self, run_id: str) -> Dict[str, Any]: ...
```

### Events
**Consumes**
- Benchmark pack generation requests (ops tooling)

**Emits**
- ordinis.bench.pack.generated
- ordinis.bench.pack.run.completed
- ordinis.bench.scorecard.generated

### Schemas
- BenchPackManifest schema
- Scorecard schema (trading + system KPIs)

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_BENCH_DATA_MISSING` | Historical data missing for requested window/universe | No |
| `E_BENCH_REPLAY_FAILED` | Replay harness failed | Yes |
| `E_BENCH_SCORE_FAILED` | Scorecard computation failed | No |

## 5. Dependencies
### Upstream
- PERSIST for storing packs and results
- ING for normalization mapping (replay uses same schemas)
- ANA for KPI computation
- ORCH for pipeline execution

### Downstream
- LEARN uses packs for evaluation and gating
- GOV may use scorecards for monitoring thresholds

### External services / libraries
- Parquet readers/writers for historical data packs

## 6. Data & State
### Storage
- Pack data stored as Parquet under `./data/bench/<pack_id>/...` with manifest in SQLite `bench_packs` table.
- Run results stored via ORCH/ANA/PERSIST.

### Caching
- Cache pack manifests and metadata; avoid re-reading Parquet repeatedly.

### Retention
- Packs retained per storage budget; default keep last N packs per universe + all stress packs.

### Migrations / Versioning
- Manifest schema versioned; packs include schema_version and data hash.

## 7. Algorithms / Logic
### Pack generation
- Select random windows of length {3,6,9,12} months across last 10–15 years.
- Ensure coverage across:
  - sectors (if equity universe is sector-stratified)
  - regimes (tag via benchmark index drawdown/volatility)
  - stress windows (explicit include list: 2008, 2020, rate shocks)
- Persist:
  - raw historical bars
  - derived features (optional)
  - manifest with hashes and seed

### Replay harness
- Replays bar events in chronological order with controlled wall-clock speed.
- Uses the same EVT schemas as live ingestion.
- Produces a run_id and feeds into ANA for scorecards.

### Scorecard
- Trading KPIs: Sharpe, Sortino, CAGR, MaxDD, Calmar, hit rate, profit factor, turnover, slippage/fees, CVaR.
- System KPIs: cycle latency p50/p95, errors, DLQ events, audit coverage, uptime for the run.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| BENCH_WINDOWS_MONTHS | list[int] | [3,6,9,12] | P0 | Window sizes. |
| BENCH_YEARS_LOOKBACK | int | 15 | P0 | Lookback range. |
| BENCH_PACKS_PER_UNIVERSE | int | 50 | P0 | Sample count. |
| BENCH_INCLUDE_STRESS | bool | true | P0 | Include stress windows. |
| BENCH_REPLAY_SPEED | float | 0.0 | P0 | 0.0 = as-fast-as-possible. |

## 9. Non-Functional Requirements
- Determinism: replay of the same pack yields identical events and metrics (within numeric tolerance).
- Throughput: run 1-year daily backtest in < 60s on DB1 hardware target (ballpark).
- Storage: packs must be budgeted; avoid TB/month growth (tick default off).

## 10. Observability
**Metrics**
- `bench_packs_total{status}`
- `bench_run_latency_ms`
- `bench_scorecards_total{status}`

**Alerts**
- Pack generation failures
- Scorecard missing for completed pack run

## 11. Failure Modes & Recovery
- Data missing: fail pack generation; emit actionable diagnostics listing missing symbols/dates.
- Replay failure: retry once; if still fails, mark pack run failed and attach logs.
- Scorecard compute failure: mark run degraded; metrics missing is a hard failure for gating.

## 12. Test Plan
### Unit tests
- Manifest schema and hashing.
- Regime tagging logic for known windows.

### Integration tests
- Generate a small pack and run through ORCH+ANA pipeline.

### Acceptance criteria
- A DB1 baseline pack run produces a complete KPI scorecard and audit trail.

## 13. Open Questions / Risks
- Define canonical universes (equity sectors, ETFs, futures) and storage budget for DB1.
- Stress window list finalization (what periods to include by default).

## 14. Traceability
### Parent
- DB1 System SRS: Benchmark packs enable pre-deployment validation and gating.

### Children / Related
- [L3-13 BENCH-PACKS — Pack Generator & Regime Tagging](../L3/L3-13_BENCH-PACKS_Benchmark_Pack_Generator_and_Regime_Tagging.md)
- [L2-16 ANA — AnalyticsEngine](L2-16_ANA_AnalyticsEngine_ProofBench.md)

### Originating system requirements
- BENCH-FR-001..004
