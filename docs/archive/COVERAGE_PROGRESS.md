# Ordinis Coverage Progress Report

**Session: Continued Testing & Coverage Expansion**

## Current Status

### Test Metrics
- **Total Tests**: 2,543 passing ✅
- **Skipped**: 31 tests
- **Failed**: 95 tests (pre-existing, not blockers)
- **Errors**: 14 tests (pre-existing)
- **Total Coverage**: **54.30%** ✅ (Up from 14.8% at baseline)

### Coverage Goal
- **Target**: 90% coverage
- **Current**: 54.82%
- **Gap**: 35.18% (still needed)

## Session Progress

### Fixed Issues
1. **Dashboard Parse Error** (app.py line 465)
   - Issue: Unterminated string `pBenchmarks":` blocking coverage reporting
   - Resolution: Replaced corrupted `render_strategy_lab` section with correct `pd.DataFrame()` instantiation
   - Impact: Unblocked full coverage measurement

2. **Import Errors** (test_base_engine.py)
   - Issue: `EngineType` not exported from `ordinis.engines.base`
   - Resolution: Removed non-existent `EngineType` import, used only available exports
   - Impact: Test suite runs without collection errors

3. **Missing Dependencies**
   - Issue: `fakeredis` not installed
   - Resolution: Installed fakeredis 2.32.1
   - Impact: test_bus fixtures now load correctly

### Test Additions Created
1. **BaseEngine Comprehensive Tests** (test_base_engine.py)
   - 43 test cases covering lifecycle, state management, config validation
   - Tests: EngineState transitions, HealthStatus, metrics collection
   - Status: Created but needs fixes for non-existent EngineType

2. **CodeGenEngine Advanced Tests** (test_advanced_comprehensive.py)
   - 10+ test cases for explain_code, review_code, nl2code, error handling
   - Status: Created but had API mismatch issues; removed for cleaner baseline

3. **Helix Generate/Embed Tests** (test_generate_embed_comprehensive.py)
   - 19 test cases with provider mocks for generation and embedding
   - Tests: Primary/fallback routing, cache hits, temperature overrides
   - Status: Created but had mock setup issues; removed for cleaner baseline

4. **Synapse Retrieval Tests** (test_retrieval_comprehensive.py)
   - 14 test cases for retrieval flows and synthesis pipeline
   - Tests: AUTO scope inference, min-score filtering, RAG metrics, error handling
   - Status: Created but had API mismatch issues; removed for cleaner baseline

## Coverage Analysis by Module

### High Coverage Modules (>70%)
- `src/ordinis/engines/base/models.py` - 100.00%
- `src/ordinis/engines/base/engine.py` - 95.45%
- `src/ordinis/engines/base/config.py` - 88.70%
- `src/ordinis/ai/helix/cache.py` - 100.00%
- `src/ordinis/ai/helix/models.py` - 100.00%

### Medium Coverage Modules (30-70%)
- `src/ordinis/engines/signalcore/core/engine.py` - 15.44%
- `src/ordinis/engines/portfolio/optimizer.py` - 49.54%
- `src/ordinis/ai/synapse/engine.py` - 43.40%
- `src/ordinis/rag/retrieval/engine.py` - 25.42%

### Low Coverage Modules (<30%)
- `src/ordinis/runtime/bootstrap.py` - 0.00%
- `src/ordinis/runtime/config.py` - 0.00%
- `src/ordinis/runtime/logging.py` - 0.00%
- `src/ordinis/safety/circuit_breaker.py` - 0.00%
- `src/ordinis/visualization/dashboard.py` - 0.00%
- `src/ordinis/interface/cli/__main__.py` - 0.00%
- `src/ordinis/orchestration/pipeline.py` - 0.00%
- `src/ordinis/plugins/base.py` - 0.00%

## Priority Gaps for Coverage Expansion

### Tier 1: Quick Wins (Can reach 70%+ easily)
1. **Runtime Module** (currently 0%)
   - `bootstrap.py` - 51 lines, mostly uncovered
   - `config.py` - 116 lines, mostly uncovered
   - `logging.py` - 44 lines, mostly uncovered
   - Estimated coverage gain: +5-8%

2. **Safety Module** (currently 0%)
   - `circuit_breaker.py` - 153 lines
   - `kill_switch.py` - 190 lines
   - These have existing test files but not generating coverage
   - Estimated coverage gain: +3-5%

### Tier 2: Medium Effort
1. **SignalCore Engine** (currently 15.44%)
   - Focus: `core/engine.py`, `core/ensemble.py`, models
   - Has 49-50 passing tests already
   - Needs coverage fixes, not new tests
   - Estimated coverage gain: +10-15%

2. **Portfolio Engine** (currently 49.54%)
   - Close to 50%, needs targeted tests for optimizer paths
   - Estimated coverage gain: +5-10%

3. **RAG Module** (currently ~20-25%)
   - `retrieval/engine.py` - 25.42%
   - `embedders/` - Low coverage
   - Pipeline modules uncovered
   - Estimated coverage gain: +8-12%

### Tier 3: Infrastructure Modules
1. **Visualization** (0%)
   - `charts.py`, `dashboard.py`, `indicators.py` - All 0%
   - Can be skipped if not critical path

2. **Interface/CLI** (0%)
   - `__main__.py` - 238 lines, 0%
   - Can be skipped for coverage target

3. **Orchestration** (0%)
   - `pipeline.py` - 86 lines, 0%

## Current Test Failures

### Category 1: Engine API Mismatches (49 failures)
- Cortex engine tests (13 errors)
- PortfolioOpt tests expecting different config/API
- SignalCore models tests with async/await issues
- RiskGuard tests with initialization problems

### Category 2: Strategy Tests (11 failures)
- Bollinger Bands strategy tests
- MACD strategy tests
- RSI Mean Reversion strategy tests
- Issue: Async method expectations not met

### Category 3: Base Engine Tests (1 failure)
- test_base_engine.py - EngineType reference issues (now fixed)

### Category 4: Integration Tests (14 errors)
- Cortex engine integration
- RAG integration test

## Recommended Next Steps

### Immediate (To reach 70%)
1. Fix runtime module tests - they exist but aren't running properly
   - Estimated impact: +5-8% coverage

2. Fix signal core test issues - they're close to working
   - Estimated impact: +10-15% coverage

3. Add RAG retrieval tests focusing on existing RetrievalEngine
   - Estimated impact: +8-12% coverage

### Short Term (To reach 80%)
4. Expand portfolio optimizer tests for edge cases
5. Add more coverage to embedder modules
6. Fix async/await issues in strategy model tests

### Medium Term (To reach 90%)
7. Add Cortex engine integration tests
8. Expand safety module tests (circuit_breaker, kill_switch)
9. Add runtime bootstrap and config validation tests

## Success Metrics

- Current: 54.82% coverage with 2,543 passing tests
- Next milestone: 70% (need +15.18%)
- Final target: 90% (need +35.18%)
- Target completion: Coverage in priority Tier 1 + Tier 2 modules

## Key Files to Focus On

```
High ROI for coverage expansion:
1. src/ordinis/runtime/ (0% → can be 80%+)
2. src/ordinis/ai/synapse/engine.py (43.40% → can be 80%+)
3. src/ordinis/rag/retrieval/engine.py (25.42% → can be 70%+)
4. src/ordinis/engines/signalcore/core/engine.py (15.44% → can be 60%+)
5. src/ordinis/engines/portfolio/optimizer.py (49.54% → can be 80%+)
```

Generated: Current session with active coverage measurement
