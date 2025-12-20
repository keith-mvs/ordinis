# LearningEngine Architecture Review

**Version:** 1.0  
**Date:** 2025-01-20  
**Author:** AI Architecture Review  
**Status:** Comprehensive Analysis Complete

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Map](#2-current-architecture-map)
3. [Key Gaps & Risks](#3-key-gaps--risks)
4. [Prioritized Recommendations](#4-prioritized-recommendations)
5. [Interface & Contract Improvements](#5-interface--contract-improvements)
6. [Performance & Latency Optimizations](#6-performance--latency-optimizations)
7. [Feedback Loop & MLOps Enhancements](#7-feedback-loop--mlops-enhancements)
8. [Observability, Monitoring, and Drift Detection](#8-observability-monitoring-and-drift-detection)
9. [Testing, Backtest/Live Parity, and Validation Gates](#9-testing-backtestlive-parity-and-validation-gates)
10. [Implementation Plan](#10-implementation-plan)
11. [Appendix: Repository References](#appendix-repository-references)

---

## 1. Executive Summary

### Assessment Overview

The **LearningEngine** provides foundational continuous learning infrastructure with event collection, training job management, model registry, drift detection, and governance integration. However, critical gaps exist in the feedback loop that prevent full closed-loop ML operations.

### Current State

| Component | Status | Maturity |
|-----------|--------|----------|
| Event Collection | ✅ Implemented | Production-ready |
| Training Job Management | ✅ Implemented | Basic (manual trigger) |
| Model Registry | ✅ Implemented | Basic (in-memory) |
| Drift Detection | ⚠️ Partial | Exists but not wired to actions |
| Evaluation Gates | ❌ Missing | No ModelEvaluator class |
| Auto-Retraining | ❌ Disabled | `auto_retrain_enabled=False` |
| A/B Testing | ✅ Implemented | Framework exists |
| Circuit Breakers | ✅ Implemented | Production-ready |
| Observability | ⚠️ Partial | No Prometheus/OTEL integration |

### Critical Findings

1. **No Evaluation Gate Pattern** - Model promotion lacks automated quality gates beyond basic threshold checks
2. **Auto-Retraining Disabled** - `LearningEngineConfig.auto_retrain_enabled` defaults to `False`
3. **Drift Detection Not Actionable** - Alerts generated but not wired to retraining triggers
4. **Missing Model Store** - No MLflow/artifact persistence; models stored in-memory only
5. **Backtest/Live Parity Gap** - No systematic validation that backtest models match live behavior

### ROI Impact

| Enhancement | Effort | Impact | Priority |
|-------------|--------|--------|----------|
| Evaluation Gates | Medium | High (prevents bad model deploys) | P0 |
| Wire Drift → Retrain | Low | High (closes feedback loop) | P0 |
| Enable Auto-Retraining | Low | Medium (automation) | P1 |
| MLflow Integration | Medium | High (reproducibility) | P1 |
| Prometheus Metrics | Medium | High (observability) | P1 |

---

## 2. Current Architecture Map

### Component Topology

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              LearningEngine Ecosystem                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────┐     ┌────────────────────┐     ┌──────────────────────┐   │
│  │ OrchestrationEng │────▶│   LearningEngine   │◀────│   FeedbackCollector  │   │
│  │ (via Protocol)   │     │                    │     │   (collectors/)      │   │
│  └──────────────────┘     └────────────────────┘     └──────────────────────┘   │
│           │                        │                           │                 │
│           │                        ▼                           │                 │
│           │               ┌────────────────────┐               │                 │
│           │               │   Model Registry   │               │                 │
│           │               │   (in-memory)      │               │                 │
│           │               └────────────────────┘               │                 │
│           │                        │                           │                 │
│           │                        ▼                           │                 │
│           │               ┌────────────────────┐               │                 │
│           │               │   TrainingJobs     │               │                 │
│           │               │   (queue-based)    │               │                 │
│           │               └────────────────────┘               │                 │
│           │                        │                           │                 │
│           ▼                        ▼                           ▼                 │
│  ┌──────────────────┐     ┌────────────────────┐     ┌──────────────────────┐   │
│  │  SignalEngine    │     │   DriftDetection   │     │  CircuitBreaker      │   │
│  │  RiskEngine      │     │   (check_drift)    │     │  Monitor             │   │
│  │  ExecutionEngine │     │                    │     │                      │   │
│  │  PortfolioEngine │     └────────────────────┘     └──────────────────────┘   │
│  └──────────────────┘                                                           │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                      ClosedLoop Feedback (feedback/)                      │   │
│  │  ┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐   │   │
│  │  │FeedbackCollector│  │ ABTestingFramework  │  │ContinuousImprovement│   │   │
│  │  │ (closed_loop)   │  │                     │  │    Pipeline         │   │   │
│  │  └─────────────────┘  └─────────────────────┘  └─────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Files

| Module | Path | Lines | Purpose |
|--------|------|-------|---------|
| **LearningEngine** | `src/ordinis/engines/learning/core/engine.py` | 623 | Core engine with event collection, training, registry |
| **LearningEngineConfig** | `src/ordinis/engines/learning/core/config.py` | ~80 | Configuration dataclass |
| **Models** | `src/ordinis/engines/learning/core/models.py` | 220 | EventType, TrainingJob, ModelVersion, DriftAlert |
| **FeedbackCollector** | `src/ordinis/engines/learning/collectors/feedback.py` | 1933 | Production feedback with circuit breakers |
| **ClosedLoop** | `src/ordinis/engines/learning/feedback/closed_loop.py` | 790 | A/B testing, continuous improvement |
| **Governance** | `src/ordinis/engines/learning/hooks/governance.py` | ~100 | Audit and policy hooks |

### Data Flow

```
Market Events                 Trade Outcomes               Model Predictions
     │                              │                            │
     ▼                              ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       FeedbackCollector.record_*()                       │
│   - record_trade_outcome()                                               │
│   - record_signal_accuracy()                                             │
│   - record_execution_failure()                                           │
│   - record_regime_change()                                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LearningEngine.record_event()                         │
│   - Buffers events by type (EventType enum)                              │
│   - Triggers flush when buffer full                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
           ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
           │   SQLite    │  │  ChromaDB   │  │   Memory    │
           │   Storage   │  │  (vectors)  │  │   Buffer    │
           └─────────────┘  └─────────────┘  └─────────────┘
```

### Protocol Integration

From [src/ordinis/engines/orchestration/core/engine.py](../../../src/ordinis/engines/orchestration/core/engine.py#L91):

```python
class LearningEngineProtocol(Protocol):
    """Protocol for learning engine interface."""

    async def update(self, results: list[Any]) -> None:
        """Update models based on results."""
        ...
```

The protocol defines minimal interface - only `update()` method is required. This is **too thin** and doesn't expose evaluation, drift, or retraining capabilities.

---

## 3. Key Gaps & Risks

### GAP-1: No ModelEvaluator or Evaluation Gate (P0 - Critical)

**Finding:** Grep search for `ModelEvaluator|EvaluationGate|model_eval|benchmark_pass` returned **0 matches**.

**Current State:**

- `evaluate_model()` exists but only compares metrics to thresholds
- No standardized evaluation harness
- No benchmark dataset management
- No statistical significance testing for model comparisons

**Risk:** Bad models can be promoted to production without rigorous validation.

**Evidence:**

```python
# From engine.py:481-507
async def evaluate_model(
    self,
    version_id: str,
    benchmark_name: str,
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> EvaluationResult:
    # Only checks if metrics >= thresholds * config factor
    passed = all(
        metrics.get(k, 0) >= v * self.config.eval_benchmark_threshold
        for k, v in thresholds.items()
    )
```

### GAP-2: Auto-Retraining Disabled by Default (P0 - Critical)

**Finding:** `auto_retrain_enabled=False` in [config.py](../../../src/ordinis/engines/learning/core/config.py)

**Current State:**

- `RetrainingTrigger` class exists in `closed_loop.py`
- Trigger logic exists but is never called automatically
- Manual intervention required for all retraining

**Risk:** Models degrade over time without intervention.

**Evidence:**

```python
# From config.py
@dataclass
class LearningEngineConfig(BaseEngineConfig):
    auto_retrain_enabled: bool = False  # <-- DISABLED
```

### GAP-3: Drift Detection Not Wired to Actions (P1 - High)

**Finding:** `check_drift()` generates alerts but doesn't trigger retraining.

**Current State:**

- Drift alerts are appended to `_drift_alerts` list
- Health check shows "degraded" when many alerts exist
- No automatic response (retrain, notify, rollback)

**Evidence:**

```python
# From engine.py:563-598
def check_drift(...) -> list[DriftAlert]:
    # ...
    if change_pct > threshold:
        alert = DriftAlert(...)
        alerts.append(alert)
        self._drift_alerts.append(alert)  # Only appends, no action
        _logger.warning(...)
    return alerts
```

### GAP-4: No MLflow/Model Store Integration (P1 - High)

**Finding:** Models stored only in-memory; no artifact persistence.

**Current State:**

- `_model_versions` is a dict in memory
- `_production_models` is a dict in memory
- No serialization to disk
- No experiment tracking

**Risk:**

- Model lineage is lost on restart
- No reproducibility
- Cannot compare experiments

**Evidence:**

```python
# From engine.py - all in-memory
self._model_versions: dict[str, list[ModelVersion]] = {}
self._production_models: dict[str, ModelVersion] = {}
```

### GAP-5: Missing Prometheus/OpenTelemetry Integration (P1 - High)

**Finding:** No metrics export infrastructure.

**Current State:**

- Custom `get_stats()` method returns dict
- No Prometheus counters/gauges
- No distributed tracing spans
- No alerting integration

**Evidence:** Grep for `prometheus|metrics\.|opentelemetry|tracing|span` showed no LearningEngine-specific instrumentation.

### GAP-6: Backtest/Live Parity Validation Missing (P2 - Medium)

**Finding:** No systematic check that backtest performance matches live.

**Current State:**

- Backtest runs separately from live
- No parity testing framework
- No confidence intervals on backtest results

**Risk:** Overfitted backtest results don't transfer to live trading.

### GAP-7: Training Job Execution is Stub-Only (P2 - Medium)

**Finding:** `submit_training_job()` stores job metadata but doesn't execute training.

**Current State:**

- Jobs added to `_training_jobs` dict
- No actual training loop
- No GPU/distributed training support

**Evidence:**

```python
# From engine.py - job is created but never executed
async def submit_training_job(self, ...) -> TrainingJob:
    job = TrainingJob(...)
    self._training_jobs[job.job_id] = job
    return job  # No execution!
```

---

## 4. Prioritized Recommendations

### P0: Critical (This Sprint)

#### REC-1: Implement ModelEvaluator Class

Create a dedicated evaluation harness with:

- Benchmark dataset registry
- Statistical significance testing (A/B, bootstrap)
- Holdout validation
- Cross-validation support
- Threshold configuration per model type

**Proposed Interface:**

```python
class ModelEvaluator(Protocol):
    async def evaluate(
        self,
        model: ModelVersion,
        benchmark: BenchmarkDataset,
        holdout: pd.DataFrame | None = None,
    ) -> EvaluationResult:
        """Evaluate model against benchmark with statistical tests."""
        ...
    
    async def compare_models(
        self,
        model_a: ModelVersion,
        model_b: ModelVersion,
        benchmark: BenchmarkDataset,
    ) -> ComparisonResult:
        """Compare two models with significance testing."""
        ...
```

**Location:** `src/ordinis/engines/learning/evaluation/evaluator.py`

#### REC-2: Enable and Wire Auto-Retraining

1. Change default: `auto_retrain_enabled=True`
2. Wire drift alerts to `RetrainingTrigger`
3. Add configurable triggers:
   - Performance degradation > X%
   - N consecutive losing trades
   - Drift alert count > threshold
   - Time-based (weekly/monthly)

**Implementation:**

```python
# In LearningEngine._handle_drift_alert()
async def _handle_drift_alert(self, alert: DriftAlert) -> None:
    if alert.severity == "critical" and self.config.auto_retrain_enabled:
        trigger = RetrainingTrigger(
            trigger_type=TriggerType.DRIFT,
            model_name=alert.model_name,
            severity=alert.severity,
        )
        await self._trigger_retraining(trigger)
```

#### REC-3: Wire Circuit Breaker to Signal Flow

Ensure all engines check `FeedbackCollector.should_allow_signals()` before generating signals.

**Integration Point:** `SignalCore._generate_signals()` should call:

```python
allowed, reason = self._feedback_collector.should_allow_signals()
if not allowed:
    raise SignalBlockedError(reason)
```

### P1: High (30 Days)

#### REC-4: Add MLflow Integration

```python
# Proposed: src/ordinis/engines/learning/store/mlflow_store.py
class MLflowModelStore:
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
    
    async def log_model(
        self,
        model: Any,
        metrics: dict[str, float],
        params: dict[str, Any],
        artifacts: list[Path],
    ) -> str:
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            return mlflow.active_run().info.run_id
```

#### REC-5: Add Prometheus Metrics

```python
# Proposed metrics
learning_events_total = Counter(
    "learning_events_total",
    "Total events recorded",
    ["event_type", "source_engine"]
)

training_jobs_active = Gauge(
    "training_jobs_active",
    "Currently active training jobs"
)

drift_alerts_total = Counter(
    "drift_alerts_total",
    "Total drift alerts generated",
    ["model_name", "severity"]
)

model_evaluation_duration_seconds = Histogram(
    "model_evaluation_duration_seconds",
    "Time to evaluate a model"
)
```

#### REC-6: Implement Training Job Executor

Create async training executor with:

- Job queue (asyncio.Queue or Redis)
- GPU resource management
- Timeout handling
- Progress reporting

```python
class TrainingExecutor:
    async def execute_job(self, job: TrainingJob) -> TrainingResult:
        async with self._resource_lock:
            job.status = TrainingStatus.RUNNING
            try:
                result = await self._run_training(job)
                job.status = TrainingStatus.COMPLETED
                return result
            except Exception as e:
                job.status = TrainingStatus.FAILED
                job.error_message = str(e)
                raise
```

### P2: Medium (60-90 Days)

#### REC-7: Backtest/Live Parity Framework

```python
class ParityValidator:
    async def validate_parity(
        self,
        model: ModelVersion,
        backtest_results: BacktestResult,
        live_results: LiveTradingResult,
        tolerance_pct: float = 0.1,
    ) -> ParityResult:
        """Validate that live performance matches backtest within tolerance."""
        ...
```

#### REC-8: Shadow Mode for New Models

Add shadow rollout that runs new models in parallel without affecting live trading:

```python
class ShadowRunner:
    async def run_shadow(
        self,
        production_model: ModelVersion,
        candidate_model: ModelVersion,
        duration: timedelta,
    ) -> ShadowResult:
        """Run candidate in shadow mode, compare to production."""
        ...
```

#### REC-9: Feature Store Integration

Add feature versioning and serving:

```python
class FeatureStore:
    async def get_features(
        self,
        entity_id: str,
        feature_names: list[str],
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """Retrieve point-in-time correct features."""
        ...
```

---

## 5. Interface & Contract Improvements

### Current Protocol (Too Thin)

```python
# From orchestration/core/engine.py:91
class LearningEngineProtocol(Protocol):
    async def update(self, results: list[Any]) -> None:
        """Update models based on results."""
        ...
```

### Proposed Enhanced Protocol

```python
from typing import Protocol, Any
from datetime import datetime

class LearningEngineProtocol(Protocol):
    """Enhanced protocol for learning engine interface."""

    # Event Collection
    def record_event(self, event: LearningEvent) -> str:
        """Record a learning event, returns event_id."""
        ...

    async def flush_events(self) -> int:
        """Flush buffered events to storage, returns count."""
        ...

    # Training
    async def submit_training_job(
        self,
        model_name: str,
        training_data: Any,
        config: dict[str, Any],
    ) -> TrainingJob:
        """Submit a training job."""
        ...

    def get_training_job(self, job_id: str) -> TrainingJob | None:
        """Get training job status."""
        ...

    # Model Registry
    async def register_model_version(
        self,
        model_name: str,
        version: str,
        artifact_path: str,
        metrics: dict[str, float],
    ) -> ModelVersion:
        """Register a new model version."""
        ...

    async def promote_model(
        self,
        model_name: str,
        version_id: str,
        target_stage: ModelStage,
        rollout_strategy: RolloutStrategy,
    ) -> ModelVersion | None:
        """Promote a model to a new stage."""
        ...

    def get_production_model(self, model_name: str) -> ModelVersion | None:
        """Get current production model."""
        ...

    # Evaluation
    async def evaluate_model(
        self,
        version_id: str,
        benchmark_name: str,
        metrics: dict[str, float],
        thresholds: dict[str, float],
    ) -> EvaluationResult:
        """Evaluate a model version."""
        ...

    # Drift Detection
    def check_drift(
        self,
        model_name: str,
        current_metrics: dict[str, float],
    ) -> list[DriftAlert]:
        """Check for drift against baseline."""
        ...

    # Circuit Breaker (New)
    def should_allow_signals(self) -> tuple[bool, str]:
        """Check if signal generation should proceed."""
        ...

    # Health
    async def health_check(self) -> HealthStatus:
        """Get engine health status."""
        ...
```

### Type Definitions to Export

Add to `src/ordinis/engines/learning/__init__.py`:

```python
from ordinis.engines.learning.core.engine import LearningEngine
from ordinis.engines.learning.core.config import LearningEngineConfig
from ordinis.engines.learning.core.models import (
    EventType,
    LearningEvent,
    TrainingJob,
    TrainingStatus,
    ModelVersion,
    ModelStage,
    RolloutStrategy,
    EvaluationResult,
    DriftAlert,
)

__all__ = [
    "LearningEngine",
    "LearningEngineConfig",
    "EventType",
    "LearningEvent",
    "TrainingJob",
    "TrainingStatus",
    "ModelVersion",
    "ModelStage",
    "RolloutStrategy",
    "EvaluationResult",
    "DriftAlert",
]
```

---

## 6. Performance & Latency Optimizations

### Current Bottlenecks

| Operation | Current | Target | Optimization |
|-----------|---------|--------|--------------|
| Event buffering | Unbounded list | 100ms flush | Ring buffer with async flush |
| SQLite writes | Sync per record | Batch writes | Batch every 100 records or 1s |
| ChromaDB embedding | Per-record | Batch embedding | Batch every 50 records |
| Drift check | Linear scan | O(1) lookup | Precompute baseline diffs |

### Proposed Optimizations

#### 1. Async Event Buffer with Ring Buffer

```python
from collections import deque
import asyncio

class AsyncEventBuffer:
    def __init__(self, max_size: int = 10_000, flush_interval: float = 1.0):
        self._buffer: deque[LearningEvent] = deque(maxlen=max_size)
        self._flush_interval = flush_interval
        self._flush_task: asyncio.Task | None = None
    
    def append(self, event: LearningEvent) -> None:
        self._buffer.append(event)
        if len(self._buffer) >= self._buffer.maxlen * 0.9:
            asyncio.create_task(self._trigger_flush())
    
    async def start_periodic_flush(self) -> None:
        self._flush_task = asyncio.create_task(self._periodic_flush())
    
    async def _periodic_flush(self) -> None:
        while True:
            await asyncio.sleep(self._flush_interval)
            await self._flush()
```

#### 2. Batch Database Writes

```python
async def _batch_store_sqlite(self, records: list[FeedbackRecord]) -> None:
    if not records or not self.db_manager:
        return
    
    async with self.db_manager._lock:
        await self.db_manager._connection.executemany(
            INSERT_SQL,
            [record.to_tuple() for record in records]
        )
        await self.db_manager._connection.commit()
```

#### 3. Precomputed Drift Thresholds

```python
def set_baseline(self, model_name: str, metrics: dict[str, float]) -> None:
    self._baseline_metrics[model_name] = metrics
    # Precompute thresholds for fast comparison
    self._drift_thresholds[model_name] = {
        k: (v * (1 - self.config.drift_threshold_pct),
            v * (1 + self.config.drift_threshold_pct))
        for k, v in metrics.items()
    }

def check_drift_fast(self, model_name: str, current_metrics: dict[str, float]) -> bool:
    thresholds = self._drift_thresholds.get(model_name)
    if not thresholds:
        return False
    for k, v in current_metrics.items():
        if k in thresholds:
            low, high = thresholds[k]
            if v < low or v > high:
                return True
    return False
```

---

## 7. Feedback Loop & MLOps Enhancements

### Current Feedback Loop (Incomplete)

```
Market Data → Signals → Trades → Outcomes → FeedbackCollector → SQLite (end)
                                                      ↓
                                              LearningEngine.record_event()
                                                      ↓
                                              _events buffer (end - no action)
```

### Target Feedback Loop (Complete)

```
Market Data → Signals → Trades → Outcomes → FeedbackCollector
                ↑                                   ↓
                │                           LearningEngine
                │                                   ↓
                │                ┌──────────────────┴──────────────────┐
                │                ↓                                     ↓
                │         DriftDetector                        EventAggregator
                │                ↓                                     ↓
                │         RetrainingTrigger ◀─────────────────── TrainingExecutor
                │                ↓                                     ↓
                │         ModelEvaluator                         ModelRegistry
                │                ↓                                     │
                │         EvaluationGate ─────────────────────────────┤
                │                ↓                                     │
                │         RolloutController ◀─────────────────────────┘
                │                ↓
                └─────── ProductionModels (updated)
```

### Closed-Loop Implementation Checklist

| Component | Status | Action Required |
|-----------|--------|-----------------|
| FeedbackCollector → LearningEngine | ✅ | None |
| LearningEngine → DriftDetector | ✅ | None |
| DriftDetector → RetrainingTrigger | ❌ | Wire `_handle_drift_alert()` |
| RetrainingTrigger → TrainingExecutor | ❌ | Implement executor |
| TrainingExecutor → ModelRegistry | ❌ | Auto-register trained models |
| ModelRegistry → ModelEvaluator | ❌ | Create evaluator class |
| ModelEvaluator → EvaluationGate | ❌ | Add pass/fail gates |
| EvaluationGate → RolloutController | ⚠️ | Wire to rollout strategies |
| RolloutController → SignalEngine | ⚠️ | Update model references |

### MLOps Pipeline Stages

```yaml
# Proposed: configs/mlops/pipeline.yaml
stages:
  - name: data_collection
    trigger: continuous
    output: event_buffer
    
  - name: data_validation
    trigger: hourly
    input: event_buffer
    validation:
      - schema_check
      - null_rate < 0.01
      - feature_drift < 0.1
    
  - name: training
    trigger: on_demand | scheduled | drift_detected
    input: validated_data
    output: model_artifact
    
  - name: evaluation
    trigger: after_training
    input: model_artifact
    benchmarks:
      - pack_01_bull
      - pack_04_bear
      - pack_05_crypto
    thresholds:
      sharpe: 1.5
      max_drawdown: 0.20
      win_rate: 0.55
    
  - name: staging
    trigger: evaluation_passed
    rollout: shadow
    duration: 24h
    
  - name: production
    trigger: staging_passed
    rollout: canary
    initial_traffic: 10%
    ramp_schedule:
      - 10%: 1h
      - 25%: 4h
      - 50%: 12h
      - 100%: 24h
```

---

## 8. Observability, Monitoring, and Drift Detection

### Proposed Metrics Taxonomy

#### Event Metrics

```python
# Counter: Total events by type
learning_events_total{event_type, source_engine, status}

# Gauge: Current buffer size
learning_event_buffer_size{engine}

# Histogram: Event processing latency
learning_event_processing_duration_seconds{event_type}
```

#### Training Metrics

```python
# Counter: Training jobs by status
training_jobs_total{model_name, status}

# Gauge: Active training jobs
training_jobs_active{model_name}

# Histogram: Training duration
training_job_duration_seconds{model_name, job_type}

# Gauge: GPU utilization during training
training_gpu_utilization{device}
```

#### Model Registry Metrics

```python
# Gauge: Models by stage
model_versions_total{model_name, stage}

# Counter: Promotions
model_promotions_total{model_name, from_stage, to_stage, rollout_strategy}

# Counter: Rollbacks
model_rollbacks_total{model_name, reason}
```

#### Drift Metrics

```python
# Gauge: Current drift from baseline (per metric)
model_drift_ratio{model_name, metric_name}

# Counter: Drift alerts
drift_alerts_total{model_name, severity, drift_type}

# Gauge: Time since last baseline update
model_baseline_age_seconds{model_name}
```

#### Circuit Breaker Metrics

```python
# Gauge: Circuit breaker state (0=closed, 1=open, 2=half-open)
circuit_breaker_state{engine}

# Counter: Circuit breaker trips
circuit_breaker_trips_total{engine, error_type}

# Histogram: Time in open state
circuit_breaker_open_duration_seconds{engine}
```

### Dashboard Panels (Grafana)

#### Overview Dashboard

| Panel | Query | Type |
|-------|-------|------|
| Events/Minute | `rate(learning_events_total[1m])` | Graph |
| Active Training | `training_jobs_active` | Stat |
| Production Models | `model_versions_total{stage="production"}` | Table |
| Drift Alerts (24h) | `increase(drift_alerts_total[24h])` | Stat |
| Circuit Breaker Status | `circuit_breaker_state` | State Timeline |

#### Training Dashboard

| Panel | Query | Type |
|-------|-------|------|
| Job Queue Depth | `training_jobs_active` | Gauge |
| Training Duration | `histogram_quantile(0.95, training_job_duration_seconds)` | Graph |
| GPU Utilization | `training_gpu_utilization` | Graph |
| Success Rate | `rate(training_jobs_total{status="completed"}[1h]) / rate(training_jobs_total[1h])` | Stat |

### Alerting Rules

```yaml
# Proposed: configs/alerts/learning.yaml
groups:
  - name: learning_engine_alerts
    rules:
      - alert: HighDriftRate
        expr: increase(drift_alerts_total[1h]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High drift alert rate detected"
          
      - alert: CircuitBreakerOpen
        expr: circuit_breaker_state == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker is OPEN for {{ $labels.engine }}"
          
      - alert: TrainingJobStuck
        expr: training_jobs_active > 0 and 
              time() - training_job_start_time > 3600
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Training job running longer than 1 hour"
          
      - alert: NoEventsReceived
        expr: increase(learning_events_total[5m]) == 0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "No learning events received in 10 minutes"
```

---

## 9. Testing, Backtest/Live Parity, and Validation Gates

### Current Test Coverage

| Module | Tests | Lines Covered | Notes |
|--------|-------|---------------|-------|
| `core/engine.py` | 921 lines of tests | ~80% | Lifecycle, events, registry |
| `core/config.py` | Basic | ~90% | Simple validation |
| `core/models.py` | Basic | ~85% | Enum coverage |
| `collectors/feedback.py` | Partial | ~40% | Circuit breaker untested |
| `feedback/closed_loop.py` | Partial | ~30% | A/B framework untested |

### Missing Test Categories

1. **Integration Tests**
   - LearningEngine ↔ OrchestrationEngine integration
   - FeedbackCollector ↔ SQLite ↔ ChromaDB flow
   - Drift detection → Retraining trigger

2. **Load Tests**
   - 10,000 events/second throughput
   - Concurrent training job handling
   - Large model registry (1000+ versions)

3. **Chaos Tests**
   - Database connection failure during flush
   - Training job timeout
   - Circuit breaker recovery

### Proposed Test Structure

```
tests/test_engines/test_learning/
├── __init__.py
├── conftest.py              # Fixtures
├── test_engine.py           # Core engine (existing)
├── test_config.py           # Configuration (existing)
├── test_models.py           # Data models (existing)
├── test_feedback.py         # Feedback collector (partial)
├── integration/
│   ├── __init__.py
│   ├── test_orchestration_integration.py   # NEW
│   ├── test_storage_integration.py         # NEW
│   └── test_end_to_end.py                  # NEW
├── load/
│   ├── __init__.py
│   ├── test_event_throughput.py            # NEW
│   └── test_concurrent_training.py         # NEW
└── chaos/
    ├── __init__.py
    ├── test_database_failure.py            # NEW
    ├── test_circuit_breaker_recovery.py    # NEW
    └── test_training_timeout.py            # NEW
```

### Backtest/Live Parity Testing

```python
# Proposed: tests/test_engines/test_learning/parity/test_parity.py

class TestBacktestLiveParity:
    """Validate backtest results match live trading within tolerance."""

    @pytest.fixture
    def backtest_results(self) -> BacktestResult:
        """Load cached backtest results."""
        return load_backtest_results("pack_01_bull_2024")

    @pytest.fixture
    def live_results(self) -> LiveTradingResult:
        """Load live trading results for comparison period."""
        return load_live_results("2024-01-01", "2024-03-31")

    def test_sharpe_parity(
        self,
        backtest_results: BacktestResult,
        live_results: LiveTradingResult,
    ) -> None:
        """Sharpe ratio should be within 20% of backtest."""
        bt_sharpe = backtest_results.metrics["sharpe"]
        live_sharpe = live_results.metrics["sharpe"]
        
        parity_ratio = live_sharpe / bt_sharpe
        assert 0.8 <= parity_ratio <= 1.2, (
            f"Sharpe parity failed: backtest={bt_sharpe:.2f}, "
            f"live={live_sharpe:.2f}, ratio={parity_ratio:.2f}"
        )

    def test_win_rate_parity(
        self,
        backtest_results: BacktestResult,
        live_results: LiveTradingResult,
    ) -> None:
        """Win rate should be within 10% of backtest."""
        bt_wr = backtest_results.metrics["win_rate"]
        live_wr = live_results.metrics["win_rate"]
        
        assert abs(live_wr - bt_wr) <= 0.10, (
            f"Win rate parity failed: backtest={bt_wr:.2%}, "
            f"live={live_wr:.2%}"
        )

    def test_drawdown_parity(
        self,
        backtest_results: BacktestResult,
        live_results: LiveTradingResult,
    ) -> None:
        """Max drawdown should not exceed backtest by more than 50%."""
        bt_dd = abs(backtest_results.metrics["max_drawdown"])
        live_dd = abs(live_results.metrics["max_drawdown"])
        
        assert live_dd <= bt_dd * 1.5, (
            f"Drawdown parity failed: backtest={bt_dd:.2%}, "
            f"live={live_dd:.2%}"
        )
```

### Evaluation Gates

```python
# Proposed: src/ordinis/engines/learning/evaluation/gates.py

from enum import Enum
from dataclasses import dataclass
from typing import Any

class GateType(str, Enum):
    PERFORMANCE = "performance"
    STATISTICAL = "statistical"
    SAFETY = "safety"
    PARITY = "parity"

@dataclass
class EvaluationGate:
    name: str
    gate_type: GateType
    threshold: float
    comparison: str  # "gte", "lte", "eq"
    required: bool = True

    def evaluate(self, value: float) -> bool:
        if self.comparison == "gte":
            return value >= self.threshold
        elif self.comparison == "lte":
            return value <= self.threshold
        elif self.comparison == "eq":
            return abs(value - self.threshold) < 1e-6
        return False

# Standard gates
PRODUCTION_GATES = [
    EvaluationGate("sharpe", GateType.PERFORMANCE, 1.5, "gte"),
    EvaluationGate("max_drawdown", GateType.PERFORMANCE, 0.20, "lte"),
    EvaluationGate("win_rate", GateType.PERFORMANCE, 0.55, "gte"),
    EvaluationGate("profit_factor", GateType.PERFORMANCE, 1.8, "gte"),
    EvaluationGate("p_value", GateType.STATISTICAL, 0.05, "lte"),
    EvaluationGate("parity_ratio", GateType.PARITY, 0.8, "gte"),
]
```

---

## 10. Implementation Plan

### Sprint 1 (Current Sprint - This Week)

| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Create `ModelEvaluator` class with basic thresholds | P0 | 4h | - |
| Wire `check_drift()` → `_handle_drift_alert()` → `_trigger_retraining()` | P0 | 2h | - |
| Add `should_allow_signals()` check to SignalCore | P0 | 2h | - |
| Enable `auto_retrain_enabled=True` with safeguards | P0 | 1h | - |
| Add unit tests for new components | P0 | 3h | - |

### 30-Day Plan

| Week | Focus | Deliverables |
|------|-------|--------------|
| Week 1 | Evaluation Gates | `ModelEvaluator`, `EvaluationGate`, basic benchmarks |
| Week 2 | MLflow Integration | Model persistence, experiment tracking, artifact store |
| Week 3 | Training Executor | Async job execution, GPU resource management |
| Week 4 | Prometheus Metrics | Instrumentation, Grafana dashboards, alerting rules |

### 60-Day Plan

| Focus Area | Deliverables |
|------------|--------------|
| Shadow Mode | `ShadowRunner` class, parallel execution, comparison reports |
| Canary Rollout | Traffic splitting, progressive rollout, auto-rollback |
| Parity Testing | Backtest/live comparison framework, statistical validation |
| Load Testing | 10K events/sec benchmark, stress test suite |

### 90-Day Plan

| Focus Area | Deliverables |
|------------|--------------|
| Feature Store | Feature versioning, point-in-time retrieval, online serving |
| Distributed Training | Multi-GPU support, Ray/Dask integration |
| AutoML Integration | Hyperparameter optimization, NAS for signal models |
| Full CI/CD Pipeline | Model CI, automated benchmarks, deployment gates |

---

## Appendix: Repository References

### Core Files

| File | Path | Purpose |
|------|------|---------|
| LearningEngine | [src/ordinis/engines/learning/core/engine.py](../../../src/ordinis/engines/learning/core/engine.py) | Main engine class |
| Config | [src/ordinis/engines/learning/core/config.py](../../../src/ordinis/engines/learning/core/config.py) | Configuration |
| Models | [src/ordinis/engines/learning/core/models.py](../../../src/ordinis/engines/learning/core/models.py) | Data classes |
| FeedbackCollector | [src/ordinis/engines/learning/collectors/feedback.py](../../../src/ordinis/engines/learning/collectors/feedback.py) | Production feedback |
| ClosedLoop | [src/ordinis/engines/learning/feedback/closed_loop.py](../../../src/ordinis/engines/learning/feedback/closed_loop.py) | A/B testing |
| Governance | [src/ordinis/engines/learning/hooks/governance.py](../../../src/ordinis/engines/learning/hooks/governance.py) | Audit hooks |

### Related Engines

| Engine | Path | Integration Point |
|--------|------|-------------------|
| Orchestration | [src/ordinis/engines/orchestration/core/engine.py](../../../src/ordinis/engines/orchestration/core/engine.py) | `LearningEngineProtocol` |
| SignalCore | [src/ordinis/engines/signalcore/core/engine.py](../../../src/ordinis/engines/signalcore/core/engine.py) | Model consumption |
| RiskGuard | [src/ordinis/engines/riskguard/core/engine.py](../../../src/ordinis/engines/riskguard/core/engine.py) | Circuit breaker |

### Tests

| Test File | Path | Coverage |
|-----------|------|----------|
| Engine Tests | [tests/test_engines/test_learning/test_engine.py](../../../tests/test_engines/test_learning/test_engine.py) | 921 lines |
| Config Tests | [tests/test_engines/test_learning/test_config.py](../../../tests/test_engines/test_learning/test_config.py) | Basic |
| Model Tests | [tests/test_engines/test_learning/test_models.py](../../../tests/test_engines/test_learning/test_models.py) | Basic |
| Feedback Tests | [tests/test_engines/test_learning/test_feedback.py](../../../tests/test_engines/test_learning/test_feedback.py) | Partial |

### Documentation

| Document | Path | Content |
|----------|------|---------|
| Integration Guide | [docs/archive/LEARNING_ENGINE_INTEGRATION.md](../archive/LEARNING_ENGINE_INTEGRATION.md) | Phase 1 integration |
| System Architecture | [docs/architecture/LAYERED_SYSTEM_ARCHITECTURE.md](../architecture/LAYERED_SYSTEM_ARCHITECTURE.md) | Overall architecture |
| README | [README.md](../../../README.md) | LearningEngine overview |

---

**End of Review**

*This document should be updated as implementations progress. Track completion in the project board.*
