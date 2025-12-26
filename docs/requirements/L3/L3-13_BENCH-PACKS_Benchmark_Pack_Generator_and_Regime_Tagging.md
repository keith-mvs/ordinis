# L3-13 — BENCH-PACKS — Benchmark Pack Generator & Regime Tagging

## 1. Identifier
- **ID:** `L3-13`
- **Component:** BENCH-PACKS — Benchmark Pack Generator & Regime Tagging
- **Level:** L3

## 2. Purpose / Scope
### In scope
- Pack sampling algorithm across 10–15 years with 3/6/9/12 month windows.
- Regime tagging rules (bull/bear/sideways/vol spike).
- Stress window inclusion list and validation rules.
- Manifest hashing and reproducibility guarantees.

### Out of scope
- Analytics metrics computation (ANA).
- Training and gating (LEARN) beyond providing packs/labels.

## 3. Responsibilities
- Define how packs are sampled and how seeds map to windows.
- Define regime tagging and threshold selection.
- Define stress window catalog and update procedure.

## 4. Interfaces
### Public APIs / SDKs
BenchmarkHarness APIs are in [L2-26 BENCH](../L2/L2-26_BENCH_Benchmark_Packs_and_Backtest_Harness.md). This spec defines generator internals.

### Regime tagging (baseline)
- Compute benchmark index returns + realized volatility over the window.
- Tags:
  - `BULL`: cumulative return > +10% and max drawdown < 10%
  - `BEAR`: cumulative return < -10% or max drawdown > 20%
  - `SIDEWAYS`: |cum return| < 5% and vol below median
  - `VOL_SPIKE`: realized vol > 90th percentile of lookback distribution

### Events
**Consumes**
_None._

**Emits**
_None._

### Schemas
- BenchPackManifest schema
- RegimeTag schema

### Error codes
| Code | Meaning | Retryable |
|---|---|---|
| `E_BENCH_REGIME_TAG_FAILED` | Unable to compute regime tags | No |
| `E_BENCH_SEED_INVALID` | Seed or sampling spec invalid | No |

## 5. Dependencies
### Upstream
- Historical price data access (Parquet) for benchmark index and universe
- CFG for thresholds and stress list

### Downstream
- BENCH harness uses generator
- LEARN uses regime tags for eval stratification

### External services / libraries
_None._

## 6. Data & State
### Storage
- Stress window catalog stored as versioned JSON in repo `bench/stress_windows.json`.
- Manifests stored in SQLite and file-based path.

### Caching
- Cache benchmark index series and volatility percentiles to speed pack generation.

### Retention
- Stress window catalog retained indefinitely; packs retained per storage policy.

### Migrations / Versioning
- Regime tagging threshold changes require pack generator version bump; packs record generator_version.

## 7. Algorithms / Logic
### Sampling algorithm (DB1 baseline)
1. Inputs:
   - universe list
   - windows_months list (3/6/9/12)
   - years_lookback (10–15)
   - packs_per_universe N
   - seed S
2. Derive deterministic RNG from seed S.
3. For each pack i in 1..N:
   - choose window length L from windows_months
   - choose start date uniformly over available history such that start+L fits
   - choose timeframe (daily or 1m) per spec
   - attach benchmark index series for the same window
4. Compute regime tags and attach to manifest.
5. Hash manifest:
   - `manifest_hash = H(json_canonical(manifest_without_hash))`

### Stress windows
- Always include known stress periods if `BENCH_INCLUDE_STRESS=true`.
- Stress windows are explicit date ranges in catalog.

### Edge cases
- Missing data for a symbol: either drop symbol (and record) or fail pack depending on strictness config.

## 8. Configuration
| Key | Type | Default | Required | Notes |
|---|---:|---:|---:|---|
| BENCH_INCLUDE_STRESS | bool | true | P0 | Include stress windows. |
| BENCH_REGIME_THRESH_BULL | float | 0.10 | P0 | Bull threshold. |
| BENCH_REGIME_THRESH_BEAR | float | -0.10 | P0 | Bear threshold. |
| BENCH_REGIME_VOL_PCTL | float | 0.90 | P0 | Vol spike percentile. |
| BENCH_STRICT_DATA | bool | true | P0 | Fail pack if data missing. |

## 9. Non-Functional Requirements
- Determinism: sampling must be fully reproducible from seed and data hashes.
- Coverage: packs must span multiple regimes; generator should enforce minimum counts per regime (optional).

## 10. Observability
**Metrics**
- `bench_pack_gen_total{status}`
- `bench_regime_counts{regime}`
- `bench_missing_data_total{symbol}`

**Alerts**
- Generator failures
- Regime distribution skew (if enforced)

## 11. Failure Modes & Recovery
- Regime tag computation fails: mark pack as degraded; require fix before using for gating.
- Missing data: if strict, fail pack and list missing; if not strict, drop and record.

## 12. Test Plan
### Unit tests
- Deterministic sampling given seed.
- Regime tagging classification on known synthetic series.

### Integration tests
- Generate packs from a small historical dataset and ensure manifests stable across runs.

### Acceptance criteria
- Pack manifests include generator_version and manifest_hash and are stable.

## 13. Open Questions / Risks
- Regime thresholds calibration (10%/20% DD are placeholders; calibrate per asset class).
- Whether to enforce minimum number of packs per regime in DB1 baseline.

## 14. Traceability
### Parent
- [L2-26 BENCH — Benchmark Packs](../L2/L2-26_BENCH_Benchmark_Packs_and_Backtest_Harness.md)

### Originating system requirements
- BENCH-FR-001..004
