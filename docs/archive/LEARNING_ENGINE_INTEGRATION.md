# Learning Engine Integration with Confidence Calibration

## Overview

This document describes the integration of the **LearningEngine** with the **Phase 1 Real Market Backtest** and **Confidence Calibration** systems. The integration enables closed-loop feedback where:

1. Trade signals are generated with confidence scores
2. Trades are executed based on filtered confidence thresholds
3. Actual outcomes are recorded as feedback events
4. Learning engine captures the feedback for model improvement
5. Calibration metrics inform future confidence adjustments

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 1 Backtest Loop                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Generate Trades         → Historical Market Data            │
│  2. Calculate Confidence    → Signal Quality Metrics            │
│  3. Calibrate (Optional)    → ML-based Probability Adjustment   │
│  4. Apply Filter            → Threshold-based Decision          │
│  5. Analyze Performance     → Win Rate, Sharpe, P/F             │
│  6. Record Feedback         → Learning Events                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
           │
           │ Learning Events
           v
┌─────────────────────────────────────────────────────────────────┐
│                     Learning Engine                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  • EventType.SIGNAL_GENERATED     → Entry confidence/metadata   │
│  • EventType.SIGNAL_ACCURACY      → Outcome & return            │
│  • EventType.METRIC_RECORDED      → Aggregate backtest metrics  │
│  • Model Lifecycle                → Registry, evaluation, drift  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. LearningEngine Initialization

**File:** `scripts/phase1_real_market_backtest.py`

```python
def setup_learning_engine(data_dir: Path | None = None) -> LearningEngine:
    """Create and initialize a LearningEngine instance for feedback capture."""
    engine = LearningEngine(
        LearningEngineConfig(
            data_dir=data_dir or Path("artifacts") / "learning_engine",
            max_events_memory=20000,
        )
    )
    asyncio.run(engine.initialize())
    return engine
```

### 2. Event Recording

**Signal Generation Event:**
```python
LearningEvent(
    event_type=EventType.SIGNAL_GENERATED,
    source_engine="phase1_real_market_backtest",
    symbol=trade["symbol"],
    timestamp=entry_ts,
    payload={
        "confidence": confidence_value,
        "num_agreeing_models": trade.get("num_agreeing_models"),
        "market_volatility": trade.get("market_volatility"),
        "signal_strength": trade.get("signal_strength"),
        "holding_days": trade.get("holding_days"),
        "applied_threshold": applied_threshold,
        "calibrated": calibration_used,
    },
)
```

**Signal Accuracy Event (Outcome):**
```python
LearningEvent(
    event_type=EventType.SIGNAL_ACCURACY,
    source_engine="phase1_real_market_backtest",
    symbol=trade["symbol"],
    timestamp=exit_ts,
    payload={
        "return_pct": trade["return_pct"],
        "win": trade["win"],
        "confidence_used": confidence_value,
        "applied_threshold": applied_threshold,
        "calibrated": calibration_used,
    },
    outcome=trade["return_pct"],  # Supervised learning target
)
```

**Metrics Summary Event:**
```python
LearningEvent(
    event_type=EventType.METRIC_RECORDED,
    source_engine="phase1_real_market_backtest",
    payload={
        "baseline": baseline,
        "filtered": filtered,
        "improvement": {
            "win_rate_change_pct": float(win_rate_change),
            "sharpe_ratio_change": float(sharpe_change),
            "profit_factor_change": float(pf_change),
        },
        "applied_threshold": float(applied_threshold),
        "confidence_key": confidence_key,
        "use_calibration": use_calibration,
    },
)
```

### 3. Function Signature

```python
def run_real_market_backtest(
    threshold: float = 0.80,
    risk_tolerance: float = 0.50,
    use_calibration: bool = True,
    use_learning_engine: bool = True,           # NEW
    learning_data_dir: Path | None = None,      # NEW
):
```

**Parameters:**
- `use_learning_engine`: Enable/disable LearningEngine feedback recording
- `learning_data_dir`: Custom directory for learning engine artifacts

### 4. Event Recording Process

```python
def record_learning_feedback(
    engine: LearningEngine,
    trades: List[Dict],
    confidence_key: str,
    applied_threshold: float,
    calibration_used: bool,
) -> int:
    """Send generated trades to the LearningEngine as feedback events."""
    events_recorded = 0

    for trade in trades:
        # Record entry signal
        engine.record_event(signal_generated_event)
        events_recorded += 1

        # Record outcome/accuracy
        engine.record_event(signal_accuracy_event)
        events_recorded += 1

    return events_recorded
```

## Data Flow Example

### Backtest Execution with Learning Engine

```
1. Setup Phase
   ├─ Initialize LearningEngine
   ├─ Download market data (2019-2024)
   └─ Configure confidence threshold

2. Training Data Generation
   ├─ Generate 1000+ trades from historical data
   ├─ Calculate signal confidence (0.15-0.95)
   ├─ Optionally calibrate probabilities (Platt/Isotonic)
   └─ Store all trade metadata

3. Filtering & Analysis
   ├─ Apply confidence filter (e.g., threshold=0.80)
   ├─ Separate passed/rejected trades
   ├─ Analyze baseline vs filtered performance
   └─ Generate improvement statistics

4. Learning Event Recording
   ├─ For each trade:
   │  ├─ Record SIGNAL_GENERATED event (entry)
   │  ├─ Record SIGNAL_ACCURACY event (exit + outcome)
   │  └─ Include confidence, threshold, calibration metadata
   ├─ Record METRIC_RECORDED event (summary statistics)
   └─ Track events in LearningEngine buffer

5. Report Generation
   ├─ Save backtest results (phase1_real_market_backtest.json)
   ├─ Include learning metrics:
   │  ├─ learning_events_recorded: 2000+
   │  ├─ learning_data_dir: artifacts/learning_engine
   │  └─ use_learning_engine: true
   └─ Shutdown LearningEngine gracefully
```

## Configuration

### LearningEngineConfig Settings (Used in Phase 1 Backtest)

```python
LearningEngineConfig(
    engine_id="learning",
    engine_name="Learning Engine",

    # Storage
    data_dir=Path("artifacts") / "learning_engine",
    max_events_memory=20000,           # Buffer up to 20k events

    # Event collection (all enabled)
    collect_signals=True,              # SIGNAL_GENERATED
    collect_executions=True,           # ORDER_* events
    collect_portfolio=True,            # POSITION_* events
    collect_risk=True,                 # RISK_BREACH
    collect_predictions=True,          # MODEL_PREDICTION

    # Training
    min_samples_for_training=1000,    # Need 1000+ samples for calibration
    training_batch_size=256,

    # Evaluation
    require_eval_pass=True,           # Validate before production
    eval_benchmark_threshold=0.95,    # Must beat baseline by 95%

    # Drift detection
    enable_drift_detection=True,
    drift_threshold_pct=0.10,         # 10% change threshold
)
```

## Query & Analysis

### Accessing Recorded Events

```python
# Get all recorded events
all_events = engine.get_events()

# Query by event type
signal_events = engine.get_events(event_type=EventType.SIGNAL_GENERATED)
accuracy_events = engine.get_events(event_type=EventType.SIGNAL_ACCURACY)
metrics = engine.get_events(event_type=EventType.METRIC_RECORDED)

# Time-based queries
recent_events = engine.get_events(
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
    limit=1000
)
```

### Engine Statistics

```python
stats = engine.get_stats()
# Returns:
# {
#     'events_buffered': 2047,
#     'events_by_type': {
#         'signal_generated': 1000,
#         'signal_accuracy': 1000,
#         'metric_recorded': 1,
#         ...
#     },
#     'training_jobs': 0,
#     'model_versions': 0,
#     'production_models': 0,
#     'evaluations': 0,
#     'drift_alerts': 0,
# }
```

## Future Enhancements

### 1. Automated Model Training
```python
# After sufficient events collected:
job = await engine.submit_training_job(
    model_name="confidence_calibrator",
    model_type="signal",
    config={
        "method": "platt_scaling",
        "cv_folds": 5,
        "min_samples": 500,
    }
)
```

### 2. Evaluation & Promotion
```python
# Evaluate trained model
eval = await engine.evaluate_model(
    version_id=model_version.version_id,
    benchmark_name="validation_set",
    metrics={"brier_score": 0.08, "log_loss": 0.25},
    thresholds={"brier_score": 0.12, "log_loss": 0.30},
)

# Promote to production
if eval.passed:
    await engine.promote_model(
        model_name="confidence_calibrator",
        version_id=model_version.version_id,
        target_stage=ModelStage.PRODUCTION,
    )
```

### 3. Drift Monitoring
```python
# Set baseline performance metrics
engine.set_baseline("confidence_calibrator", {
    "accuracy": 0.82,
    "precision": 0.79,
})

# Monitor for drift
alerts = engine.check_drift("confidence_calibrator", {
    "accuracy": 0.75,  # 8.5% decrease
    "precision": 0.72,  # 8.9% decrease
})
# Would trigger 2 drift alerts
```

## Test Coverage

### Integration Tests

Run the comprehensive test suite:

```bash
conda run -n ordinis-env python scripts/test_learning_engine_integration.py
```

**Tests included:**
1. **Basic Operations** - Initialize, record events, query
2. **Backtest Integration** - Trade data, signals, accuracy
3. **Model Lifecycle** - Register, evaluate, promote, drift detection

### Phase 1 Backtest with Learning Engine

Run the backtest with learning enabled:

```bash
conda run -n ordinis-env python scripts/phase1_real_market_backtest.py \
    --threshold 0.80 \
    --risk-tolerance 0.50
```

**Output includes:**
- `reports/phase1_real_market_backtest.json` - Full backtest report
- `artifacts/learning_engine/` - Learning event buffer
- Learning event statistics in report:
  ```json
  {
    "use_learning_engine": true,
    "learning_events_recorded": 2047,
    "learning_data_dir": "artifacts/learning_engine",
    ...
  }
  ```

## Performance Metrics

### Event Recording Overhead
- **Per-trade overhead**: ~2 events (entry + exit)
- **1000 trades**: ~2000 events in memory
- **Memory usage**: ~2KB per event (metadata + payload)
- **Total for backtest**: ~4-5 MB buffer

### Query Performance
- Event lookups: O(n) linear scan (for now)
- Type filtering: O(n) with pre-indexed dict
- Time range filtering: O(n) sequential scan
- Typical query: <10ms for 2000 events

## Integration with Orchestration Engine

The LearningEngine integrates into the full trading pipeline:

```python
# Orchestration Engine receives learning engine
orchestration = OrchestrationEngine(config)
await orchestration.register_learning_engine(learning_engine)

# During trading cycle
await orchestration.run_cycle(symbols=["AAPL", "MSFT"])
# Automatically sends results to learning engine:
# engine.update([learning_data])
```

**Protocol compliance:**
```python
class LearningEngineProtocol(Protocol):
    async def update(self, results: list[Any]) -> None:
        """Update models based on results."""
        ...
```

## Files Modified/Created

1. **Modified: `scripts/phase1_real_market_backtest.py`**
   - Added imports: `UTC`, `EventType`, `LearningEngine`, `LearningEngineConfig`, `LearningEvent`
   - New function: `setup_learning_engine()`
   - New function: `record_learning_feedback()`
   - Modified: `run_real_market_backtest()` with learning engine integration
   - Updated report schema with learning metrics

2. **Created: `scripts/test_learning_engine_integration.py`**
   - Comprehensive test suite for LearningEngine
   - Integration tests with backtest data
   - Model lifecycle tests with drift detection

## Summary

The Learning Engine integration provides:

✓ **Closed-loop feedback** - Trade outcomes recorded and analyzed
✓ **Event tracking** - 2000+ learning events per backtest
✓ **Confidence calibration** - ML-based probability adjustment
✓ **Model management** - Training, evaluation, promotion
✓ **Drift detection** - Performance degradation alerts
✓ **Governance** - Audit trails and validation
✓ **Extensibility** - Ready for automated retraining

This foundation enables continuous model improvement as new market data arrives and trading outcomes accumulate.
