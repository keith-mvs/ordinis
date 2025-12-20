# Consolidated Gap Analysis Report

**Version:** 1.0  
**Date:** 2025-01-21  
**Status:** Comprehensive Analysis - Updated  
**Methodology:** Code archaeology, grep analysis, Git history review, prior report synthesis

---

## Executive Summary

This report consolidates findings from **5 prior gap-analysis reports** and re-runs the same analysis methodology to identify current status, newly resolved items, and remaining gaps. Special attention is given to **Synapse-space changes** following commit `47e6a35` ("feat(rag): implement SYNAPSE_RAG_DATABASE_REVIEW recommendations R1-R8").

### Reports Consolidated

| Report | Date | Scope | Current Status |
|--------|------|-------|----------------|
| [SYNAPSE_RAG_DATABASE_REVIEW.md](../../SYNAPSE_RAG_DATABASE_REVIEW.md) | Dec 19, 2025 | RAG/Chroma/SQLite integration | **R1-R8 IMPLEMENTED** ✅ |
| [LEARNING_ENGINE_REVIEW.md](LEARNING_ENGINE_REVIEW.md) | Jan 20, 2025 | LearningEngine architecture | Gaps remain 🔴 |
| [ML_ENHANCEMENTS_REVIEW.md](ML_ENHANCEMENTS_REVIEW.md) | Jan 21, 2025 | SignalCore ML systems | Gaps remain 🔴 |
| [PORTFOLIO_ARCHITECTURE_REVIEW.md](../../PORTFOLIO_ARCHITECTURE_REVIEW.md) | Dec 2024 | Portfolio/PortfolioOpt engines | Mostly implemented ✅ |
| [SENSITIVITY_ANALYSIS_REPORT.md](../analysis/SENSITIVITY_ANALYSIS_REPORT.md) | — | Position sizing sensitivity | Informational |

### Overall System Posture

| Domain | Critical Gaps | High Gaps | Medium Gaps | Status |
|--------|--------------|-----------|-------------|--------|
| **Synapse/RAG** | 0 | 0 | 2 | 🟢 Healthy |
| **LearningEngine** | 2 | 3 | 2 | 🔴 At Risk |
| **ML/SignalCore** | 2 | 3 | 2 | 🔴 At Risk |
| **Portfolio** | 0 | 1 | 3 | 🟡 Stable |
| **Observability** | 0 | 1 | 1 | 🟡 Stable |
| **TOTAL** | **4** | **8** | **10** | |

---

## 1. What Changed Since Last Baseline

### 1.1 Synapse-Space Changes (Major)

**Commit:** `47e6a35` feat(rag): implement SYNAPSE_RAG_DATABASE_REVIEW recommendations R1-R8

The following Synapse review recommendations have been **fully implemented**:

| ID | Recommendation | Implementation | Evidence |
|----|----------------|----------------|----------|
| **R1** | Add `session_id` to SQLite trades/orders | ✅ Implemented | Schema migration applied |
| **R2** | Create `TradeVectorIngester` pipeline | ✅ Implemented | [trade_ingester.py](../../src/ordinis/rag/pipeline/trade_ingester.py) (524 lines) |
| **R3** | Implement `DualWriteManager` with saga pattern | ✅ Implemented | [dual_write.py](../../src/ordinis/core/dual_write.py) (643 lines) |
| **R4** | Add embedding version to collection metadata | ✅ Implemented | Versioned metadata in create_collection() |
| **R5** | Deterministic ID generation | ✅ Implemented | [id_generator.py](../../src/ordinis/rag/vectordb/id_generator.py) (367 lines) |
| **R6** | Add `indexed_at`, `checksum` to metadata | ✅ Implemented | Metadata fields added |
| **R7** | Create SQLite `sessions` table | ✅ Implemented | FK relationships established |
| **R8** | Add FTS5 virtual table on trades.metadata | ✅ Implemented | FTS5 enabled |

**Key new modules:**

- `src/ordinis/core/dual_write.py` - Saga pattern for cross-store transactions
- `src/ordinis/rag/pipeline/trade_ingester.py` - SQLite → ChromaDB sync
- `src/ordinis/rag/vectordb/id_generator.py` - Content-hash deterministic IDs
- `src/ordinis/rag/context/assembler.py` - Tiered context assembly (session-aware)

### 1.2 Prometheus/Observability Changes

**Status:** Partially implemented

| Component | Status | Evidence |
|-----------|--------|----------|
| Prometheus metrics exporter | ✅ Implemented | [metrics_exporter.py](../../src/ordinis/monitoring/metrics_exporter.py) |
| Prometheus collectors | ✅ Implemented | [collectors.py](../../src/ordinis/monitoring/collectors.py) |
| Cortex engine metrics | ✅ Implemented | Counter, Histogram in [engine.py](../../src/ordinis/engines/cortex/core/engine.py) |
| LearningEngine metrics | ❌ Not implemented | No instrumentation |
| SignalCore metrics | ❌ Not implemented | No latency tracking |

### 1.3 Portfolio Changes

**Status:** Mostly complete per PORTFOLIO_ARCHITECTURE_REVIEW.md

All 7 gap-fix modules were created:

- PortfolioOptAdapter, TransactionCostModel, ExecutionFeedbackCollector
- RegimeSizingHook, Enhanced Governance rules
- InstrumentTypes, RiskAttributionEngine

**Remaining integration work:** Wire adapters into live workflows (marked in task list).

---

## 2. Consolidated Gap Inventory

### 2.1 Critical Gaps (P0) — Action Required Immediately

| ID | Gap | Domain | Evidence | Impact | Fix Effort |
|----|-----|--------|----------|--------|------------|
| **G-ML-1** | No model serialization/persistence | SignalCore | `ModelRegistry._models` is in-memory dict; no `save()`/`load()` | Models lost on restart; no reproducibility | Medium |
| **G-ML-2** | Training/serving skew in LSTM | SignalCore | [lstm_model.py#L160](../../src/ordinis/engines/signalcore/models/lstm_model.py) - scaler not persisted | Wrong predictions in production | Low |
| **G-LE-1** | No ModelEvaluator/evaluation gates | LearningEngine | Grep: 0 matches for `ModelEvaluator\|EvaluationGate` | Bad models can be promoted | Medium |
| **G-LE-2** | Auto-retraining disabled by default | LearningEngine | `auto_retrain_enabled=False` in config | Models degrade over time | Low |

### 2.2 High Gaps (P1) — Address Within 30 Days

| ID | Gap | Domain | Evidence | Impact | Fix Effort |
|----|-----|--------|----------|--------|------------|
| **G-ML-3** | No ONNX Runtime for inference | SignalCore | Grep: 0 matches for `onnxruntime\|torch.onnx` | 2-5x latency opportunity missed | Medium |
| **G-ML-4** | No feature store implementation | SignalCore | No `FeatureStore` class in src/ | Feature drift, inconsistent preprocessing | Medium |
| **G-ML-5** | No inference latency benchmarking | SignalCore | No `@timer` decorators on model.generate() | Cannot verify p95 ≤ 200ms SLA | Low |
| **G-LE-3** | Drift detection not wired to actions | LearningEngine | `check_drift()` appends to list, no trigger | No auto-remediation | Low |
| **G-LE-4** | No MLflow/artifact persistence | LearningEngine | Models stored in-memory only | No experiment tracking | Medium |
| **G-PF-1** | Portfolio adapters not wired to live workflows | Portfolio | Task checklist incomplete | Features not usable | Medium |
| **G-OB-1** | LearningEngine missing Prometheus metrics | Observability | No instrumentation in learning/ | No visibility into ML pipeline | Low |

### 2.3 Medium Gaps (P2) — Address Within 60 Days

| ID | Gap | Domain | Evidence | Impact | Fix Effort |
|----|-----|--------|----------|--------|------------|
| **G-ML-6** | No mixed-precision/quantization | SignalCore | Grep: 0 matches for `cuda.amp\|autocast` | Suboptimal GPU utilization | Low |
| **G-LE-5** | Training job execution is stub-only | LearningEngine | `submit_training_job()` doesn't execute | Manual training required | Medium |
| **G-LE-6** | Backtest/live parity validation missing | LearningEngine | No parity testing framework | Overfitted models may deploy | Medium |
| **G-SY-1** | No BM25/FTS hybrid retrieval in RAG | Synapse | Vector-only retrieval | Keyword queries underperform | Low |
| **G-SY-2** | No embedding cache (re-embeds on index) | Synapse | Every index run calls embedding API | Slow indexing, API costs | Low |
| **G-PF-2** | Factor data loading not implemented | Portfolio | RiskAttributionEngine lacks live data | Attribution incomplete | Medium |
| **G-PF-3** | Multi-asset integration incomplete | Portfolio | InstrumentRegistry exists but not wired | Cannot trade derivatives | Medium |
| **G-PF-4** | State management not transactional | Portfolio | No `begin_transaction()`/`rollback()` | Concurrent state issues | Medium |
| **G-OB-2** | No distributed tracing spans | Observability | No OpenTelemetry in learning/signal engines | Hard to debug latency | Medium |

---

## 3. Risk & Priority Matrix

### 3.1 Risk-Impact Quadrant

```
                    HIGH IMPACT
                         │
         G-ML-1 ●──────┼────────● G-LE-1
         G-ML-2 ●      │        ● G-LE-2
                       │
    LOW EFFORT ────────┼──────── HIGH EFFORT
                       │
         G-LE-3 ●      │        ● G-ML-3
         G-ML-5 ●      │        ● G-LE-4
         G-OB-1 ●──────┼────────● G-ML-4
                       │
                    LOW IMPACT
```

### 3.2 Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FOUNDATIONAL (Fix First)                         │
├─────────────────────────────────────────────────────────────────────┤
│  G-ML-2 (Training/Serving Skew) ─┐                                  │
│                                  ├──▶ G-ML-1 (Model Persistence)    │
│  G-ML-4 (Feature Store) ─────────┘                                  │
│                                                                     │
│  G-LE-2 (Enable Auto-Retrain) ──▶ G-LE-3 (Wire Drift → Retrain)    │
│                                                                     │
│  G-OB-1 (Add Metrics) ──────────▶ G-LE-1 (Evaluation Gates)         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ENHANCEMENT (Fix Second)                         │
├─────────────────────────────────────────────────────────────────────┤
│  G-ML-1 ──▶ G-LE-4 (MLflow Integration)                            │
│                                                                     │
│  G-ML-3 (ONNX Runtime) ──▶ G-ML-5 (Latency Benchmarks)             │
│                                                                     │
│  G-ML-6 (Mixed Precision)                                          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION (Fix Last)                          │
├─────────────────────────────────────────────────────────────────────┤
│  G-SY-1 (Hybrid Retrieval), G-SY-2 (Embedding Cache)               │
│  G-PF-2 (Factor Data), G-PF-3 (Multi-Asset)                        │
│  G-LE-5 (Training Executor), G-LE-6 (Parity Testing)               │
│  G-OB-2 (Distributed Tracing)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Domain-Specific Findings

### 4.1 Synapse/RAG Domain — 🟢 Healthy

**Summary:** All 8 critical recommendations from SYNAPSE_RAG_DATABASE_REVIEW.md have been implemented.

**Remaining (P2):**

- G-SY-1: Hybrid BM25+vector retrieval not yet active (FTS5 table exists, but retrieval logic uses vector-only)
- G-SY-2: No embedding cache; re-embeds documents on every index run

**Evidence of fixes:**

```
src/ordinis/core/dual_write.py           # 643 lines - saga pattern
src/ordinis/rag/pipeline/trade_ingester.py    # 524 lines - SQLite→Chroma sync
src/ordinis/rag/vectordb/id_generator.py      # 367 lines - deterministic IDs
```

### 4.2 LearningEngine Domain — 🔴 At Risk

**Summary:** Core infrastructure exists but the closed-loop is not operational.

**Critical issues:**

1. **G-LE-1**: No `ModelEvaluator` class — models can be promoted without validation
2. **G-LE-2**: `auto_retrain_enabled=False` — no automatic model refresh

**Key findings from re-analysis:**

```python
# config.py - still disabled
auto_retrain_enabled: bool = False

# engine.py - drift alerts not actionable
def check_drift(...) -> list[DriftAlert]:
    # ...
    self._drift_alerts.append(alert)  # Only appends, no action taken
```

### 4.3 ML/SignalCore Domain — 🔴 At Risk

**Summary:** 33+ models exist but production-readiness gaps persist.

**Critical issues:**

1. **G-ML-1**: Models stored in-memory only (`ModelRegistry._models = {}`)
2. **G-ML-2**: LSTM normalizer not persisted with model state

**Key findings from re-analysis:**

```python
# train.py:142 - only saves state_dict, not scaler
torch.save(model.model.state_dict(), Path(args.output_dir) / "model.pt")

# lstm_model.py:160 - comment acknowledges the bug
# "wrong for inference but ok for this demo"
```

**Missing infrastructure (grep confirmed 0 matches):**

- ONNX Runtime: `torch.onnx|onnxruntime`
- Feature Store: `FeatureStore|feature_store`
- MLflow: `MLflow|mlflow`
- Mixed Precision: `cuda.amp|autocast|GradScaler`

### 4.4 Portfolio Domain — 🟡 Stable

**Summary:** All 7 new modules created per review; integration pending.

**Completed:**

- PortfolioOptAdapter, TransactionCostModel, ExecutionFeedbackCollector
- RegimeSizingHook, Enhanced Governance rules
- InstrumentTypes, RiskAttributionEngine

**Pending (from task checklist):**

- G-PF-1: Wire adapters into PortfolioEngine's rebalance workflow
- G-PF-2: Load factor data from market data adapters
- G-PF-3: Update PortfolioEngine to use InstrumentRegistry
- G-PF-4: Implement transactional state updates

### 4.5 Observability Domain — 🟡 Stable

**Summary:** Prometheus infrastructure exists but ML-specific metrics missing.

**Implemented:**

- `src/ordinis/monitoring/metrics_exporter.py` — HTTP endpoint for Prometheus
- `src/ordinis/monitoring/collectors.py` — Counter, Gauge, Histogram definitions
- Cortex engine has Counter and Histogram instrumentation

**Gaps:**

- G-OB-1: LearningEngine has no Prometheus metrics
- G-OB-2: No OpenTelemetry distributed tracing

---

## 5. Recommended Fix Sequence

### Sprint 1 (This Week) — Critical Fixes

| Priority | Gap ID | Task | Effort | Owner |
|----------|--------|------|--------|-------|
| P0 | G-ML-2 | Persist scaler with LSTM model state | 2h | — |
| P0 | G-LE-2 | Set `auto_retrain_enabled=True` with safeguards | 1h | — |
| P0 | G-LE-3 | Wire `check_drift()` → `_trigger_retraining()` | 2h | — |
| P0 | G-ML-1 | Add `save()`/`load()` to Model ABC + LSTMModel | 4h | — |
| P0 | G-LE-1 | Create `ModelEvaluator` with basic threshold gates | 4h | — |

### 30-Day Plan — High Priority

| Week | Focus | Gap IDs | Deliverables |
|------|-------|---------|--------------|
| Week 1 | Persistence & Evaluation | G-ML-1, G-ML-2, G-LE-1 | Model checkpointing, evaluator class |
| Week 2 | MLflow Integration | G-LE-4 | Experiment tracking, artifact store |
| Week 3 | ONNX Runtime | G-ML-3 | Export LSTM to ONNX, inference wrapper |
| Week 4 | Observability | G-OB-1, G-ML-5 | Prometheus metrics for ML pipeline |

### 60-Day Plan — Medium Priority

| Focus | Gap IDs | Deliverables |
|-------|---------|--------------|
| Feature Store | G-ML-4 | Versioned feature serving, point-in-time retrieval |
| Training Executor | G-LE-5 | Async job execution, GPU management |
| Portfolio Integration | G-PF-1, G-PF-2 | Wire adapters, load factor data |
| Mixed Precision | G-ML-6 | AMP training, INT8 quantization |

### 90-Day Plan — Optimization

| Focus | Gap IDs | Deliverables |
|-------|---------|--------------|
| Hybrid Retrieval | G-SY-1 | BM25 + vector fusion scoring |
| Embedding Cache | G-SY-2 | Local cache with TTL |
| Parity Testing | G-LE-6 | Backtest/live comparison framework |
| Distributed Tracing | G-OB-2 | OpenTelemetry spans across engines |
| Multi-Asset | G-PF-3 | InstrumentRegistry integration |

---

## 6. Traceability Appendix

### 6.1 Gap → Source Report Mapping

| Gap ID | Source Report | Original ID | Section |
|--------|---------------|-------------|---------|
| G-ML-1 | ML_ENHANCEMENTS_REVIEW.md | GAP-1 | §3.1 |
| G-ML-2 | ML_ENHANCEMENTS_REVIEW.md | GAP-2 | §3.2 |
| G-ML-3 | ML_ENHANCEMENTS_REVIEW.md | I1 | §4.1 |
| G-ML-4 | ML_ENHANCEMENTS_REVIEW.md | GAP-3 | §3.3 |
| G-ML-5 | ML_ENHANCEMENTS_REVIEW.md | GAP-4 | §3.4 |
| G-ML-6 | ML_ENHANCEMENTS_REVIEW.md | GAP-5 | §3.5 |
| G-LE-1 | LEARNING_ENGINE_REVIEW.md | GAP-1 | §3 |
| G-LE-2 | LEARNING_ENGINE_REVIEW.md | GAP-2 | §3 |
| G-LE-3 | LEARNING_ENGINE_REVIEW.md | GAP-3 | §3 |
| G-LE-4 | LEARNING_ENGINE_REVIEW.md | GAP-4 | §3 |
| G-LE-5 | LEARNING_ENGINE_REVIEW.md | GAP-7 | §3 |
| G-LE-6 | LEARNING_ENGINE_REVIEW.md | GAP-6 | §3 |
| G-SY-1 | SYNAPSE_RAG_DATABASE_REVIEW.md | §5.4 | Retrieval Quality |
| G-SY-2 | SYNAPSE_RAG_DATABASE_REVIEW.md | §5.5 | Embedding Pipeline |
| G-PF-1 | PORTFOLIO_ARCHITECTURE_REVIEW.md | §6 Task List | Phase 1 |
| G-PF-2 | PORTFOLIO_ARCHITECTURE_REVIEW.md | §6 Task List | Phase 5 |
| G-PF-3 | PORTFOLIO_ARCHITECTURE_REVIEW.md | Gap 4 | §2 |
| G-PF-4 | PORTFOLIO_ARCHITECTURE_REVIEW.md | Gap 2 | §2 |
| G-OB-1 | LEARNING_ENGINE_REVIEW.md | GAP-5 | §3 |
| G-OB-2 | LEARNING_ENGINE_REVIEW.md | GAP-5 | §3 |

### 6.2 Git Baseline Commits

| Report | Baseline Commit | Analysis Date |
|--------|-----------------|---------------|
| SYNAPSE_RAG_DATABASE_REVIEW.md | Pre-47e6a35 | Dec 19, 2025 |
| LEARNING_ENGINE_REVIEW.md | ~1f73cee | Jan 20, 2025 |
| ML_ENHANCEMENTS_REVIEW.md | 47e6a35 | Jan 21, 2025 |
| PORTFOLIO_ARCHITECTURE_REVIEW.md | ~c48b8e1 | Dec 2024 |
| **This Report** | **47e6a35 (HEAD)** | **Jan 21, 2025** |

### 6.3 Evidence Files by Gap

| Gap ID | Key Files | Line Ranges |
|--------|-----------|-------------|
| G-ML-1 | src/ordinis/engines/signalcore/core/model.py | L100-200 |
| G-ML-2 | src/ordinis/engines/signalcore/models/lstm_model.py | L65-67, L160 |
| G-ML-2 | src/ordinis/engines/signalcore/train.py | L142 |
| G-LE-1 | src/ordinis/engines/learning/core/engine.py | L481-507 |
| G-LE-2 | src/ordinis/engines/learning/core/config.py | L55 |
| G-LE-3 | src/ordinis/engines/learning/core/engine.py | L563-598 |
| G-OB-1 | src/ordinis/monitoring/collectors.py | Full file |

---

## 7. Limitations & Assumptions

1. **Git history parsing:** Some commits may have been squashed; individual file changes within `47e6a35` were not fully enumerated.

2. **Test coverage not verified:** This analysis focused on source code gaps, not test coverage metrics.

3. **Live trading validation:** Gaps were identified via static analysis; runtime behavior in production not tested.

4. **External dependencies:** ONNX Runtime, MLflow, and OpenTelemetry are proposed additions; compatibility with existing dependencies not verified.

5. **Performance benchmarks:** Latency targets (p95 ≤ 200ms) are from README; no baseline measurements exist.

---

## 8. Conclusion

The Ordinis trading system has made significant progress, particularly in the **Synapse/RAG domain** where all 8 critical recommendations have been implemented. However, the **LearningEngine** and **ML/SignalCore** domains have critical gaps that prevent closed-loop ML operations:

- Models cannot be persisted or reproduced
- Evaluation gates do not exist
- Auto-retraining is disabled
- Training/serving skew causes incorrect predictions

**Immediate action items:**

1. Fix LSTM scaler persistence (2h)
2. Enable auto-retraining with safeguards (1h)
3. Create ModelEvaluator class (4h)
4. Implement model save/load (4h)

With these fixes, the system will be positioned for reliable ML operations.

---

**Report generated by:** AI Architecture Review  
**Methodology:** Same as prior reports (grep_search, file_search, read_file, git log analysis)  
**Next review:** After Sprint 1 completion
