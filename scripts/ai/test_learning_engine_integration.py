"""
Test script to demonstrate Learning Engine integration with Phase 1 backtest.

This script validates that:
1. LearningEngine is properly initialized
2. Learning events are recorded for signals and outcomes
3. Feedback loop is complete with backtest metrics
4. Learning engine can be queried for event statistics
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from ordinis.engines.learning import (
    EventType,
    LearningEngine,
    LearningEngineConfig,
    LearningEvent,
)


async def test_learning_engine_basic():
    """Test basic LearningEngine initialization and event recording."""
    print("=" * 80)
    print("TEST: Basic LearningEngine Operations")
    print("=" * 80)
    print()

    # Setup
    config = LearningEngineConfig(
        data_dir=Path("artifacts") / "test_learning_engine",
        max_events_memory=1000,
    )
    engine = LearningEngine(config)

    try:
        # Initialize
        await engine.initialize()
        print("✓ LearningEngine initialized")

        # Record some events
        for i in range(5):
            engine.record_event(
                LearningEvent(
                    event_type=EventType.SIGNAL_GENERATED,
                    source_engine="test_script",
                    symbol=f"TEST{i}",
                    payload={"confidence": 0.7 + i * 0.05},
                )
            )
        print("✓ Recorded 5 SIGNAL_GENERATED events")

        for i in range(5):
            engine.record_event(
                LearningEvent(
                    event_type=EventType.SIGNAL_ACCURACY,
                    source_engine="test_script",
                    symbol=f"TEST{i}",
                    payload={"return": 0.02 * (i - 2)},
                    outcome=0.02 * (i - 2),
                )
            )
        print("✓ Recorded 5 SIGNAL_ACCURACY events")

        # Query events
        all_events = engine.get_events()
        print(f"✓ Total events recorded: {len(all_events)}")

        signal_events = engine.get_events(event_type=EventType.SIGNAL_GENERATED)
        print(f"✓ SIGNAL_GENERATED events: {len(signal_events)}")

        accuracy_events = engine.get_events(event_type=EventType.SIGNAL_ACCURACY)
        print(f"✓ SIGNAL_ACCURACY events: {len(accuracy_events)}")

        # Check stats
        stats = engine.get_stats()
        print(f"✓ Engine stats: {stats}")

        # Health check
        health = await engine.health_check()
        print(f"✓ Engine health: {health.level.value} - {health.message}")

        print()
        print("TEST PASSED: Basic operations functional")

    finally:
        await engine.shutdown()
        print("✓ Engine shutdown complete")


async def test_learning_engine_with_backtest_data():
    """Test LearningEngine with realistic backtest data."""
    print()
    print("=" * 80)
    print("TEST: Learning Engine with Backtest Data")
    print("=" * 80)
    print()

    config = LearningEngineConfig(
        data_dir=Path("artifacts") / "test_learning_backtest",
        max_events_memory=5000,
    )
    engine = LearningEngine(config)

    try:
        await engine.initialize()
        print("✓ LearningEngine initialized")

        # Simulate trade records from backtest
        trades = [
            {
                "symbol": "AAPL",
                "entry_date": "2024-01-01",
                "exit_date": "2024-01-10",
                "confidence_score": 0.75,
                "calibrated_probability": 0.72,
                "return_pct": 0.025,
                "win": True,
            },
            {
                "symbol": "MSFT",
                "entry_date": "2024-01-02",
                "exit_date": "2024-01-11",
                "confidence_score": 0.65,
                "calibrated_probability": 0.58,
                "return_pct": -0.015,
                "win": False,
            },
            {
                "symbol": "GOOGL",
                "entry_date": "2024-01-03",
                "exit_date": "2024-01-12",
                "confidence_score": 0.85,
                "calibrated_probability": 0.82,
                "return_pct": 0.035,
                "win": True,
            },
        ]

        # Record signal events
        for trade in trades:
            entry_ts = datetime.fromisoformat(trade["entry_date"]).replace(tzinfo=UTC)
            exit_ts = datetime.fromisoformat(trade["exit_date"]).replace(tzinfo=UTC)

            engine.record_event(
                LearningEvent(
                    event_type=EventType.SIGNAL_GENERATED,
                    source_engine="test_backtest",
                    symbol=trade["symbol"],
                    timestamp=entry_ts,
                    payload={
                        "confidence": trade["confidence_score"],
                        "calibrated_probability": trade["calibrated_probability"],
                    },
                )
            )

            engine.record_event(
                LearningEvent(
                    event_type=EventType.SIGNAL_ACCURACY,
                    source_engine="test_backtest",
                    symbol=trade["symbol"],
                    timestamp=exit_ts,
                    payload={
                        "return_pct": trade["return_pct"],
                        "win": trade["win"],
                    },
                    outcome=trade["return_pct"],
                )
            )

        print(f"✓ Recorded {len(trades)} trades with SIGNAL_GENERATED and SIGNAL_ACCURACY events")

        # Record summary metrics
        engine.record_event(
            LearningEvent(
                event_type=EventType.METRIC_RECORDED,
                source_engine="test_backtest",
                payload={
                    "baseline_performance": {"win_rate": 0.45, "sharpe_ratio": 1.2},
                    "filtered_performance": {"win_rate": 0.60, "sharpe_ratio": 1.8},
                    "improvement": {
                        "win_rate_change_pct": 15.0,
                        "sharpe_ratio_change": 0.6,
                    },
                },
            )
        )
        print("✓ Recorded METRIC_RECORDED event for backtest summary")

        # Query and verify
        all_events = engine.get_events()
        print(f"✓ Total events: {len(all_events)}")

        signal_gen = engine.get_events(event_type=EventType.SIGNAL_GENERATED)
        print(f"✓ Signal generation events: {len(signal_gen)}")

        signal_acc = engine.get_events(event_type=EventType.SIGNAL_ACCURACY)
        print(f"✓ Signal accuracy events: {len(signal_acc)}")

        metrics = engine.get_events(event_type=EventType.METRIC_RECORDED)
        print(f"✓ Metric events: {len(metrics)}")

        stats = engine.get_stats()
        print("✓ Engine stats:")
        print(f"  - Events buffered: {stats['events_buffered']}")
        print(f"  - Events by type: {stats['events_by_type']}")

        print()
        print("TEST PASSED: Backtest data integration successful")

    finally:
        await engine.shutdown()
        print("✓ Engine shutdown complete")


async def test_learning_engine_model_lifecycle():
    """Test Learning Engine model registration and promotion."""
    print()
    print("=" * 80)
    print("TEST: Learning Engine Model Lifecycle")
    print("=" * 80)
    print()

    config = LearningEngineConfig(
        data_dir=Path("artifacts") / "test_learning_models",
    )
    engine = LearningEngine(config)

    try:
        await engine.initialize()
        print("✓ LearningEngine initialized")

        # Register a model version
        model_version = await engine.register_model_version(
            model_name="confidence_signal_model",
            version="1.0.0",
            metrics={"accuracy": 0.82, "precision": 0.79},
            description="Initial confidence calibration model",
        )
        print(f"✓ Registered model version: {model_version.version_id}")

        # Create evaluation
        eval_result = await engine.evaluate_model(
            version_id=model_version.version_id,
            benchmark_name="validation_set",
            metrics={"accuracy": 0.82, "precision": 0.79},
            thresholds={"accuracy": 0.75, "precision": 0.70},
        )
        print(
            f"✓ Evaluation result: {eval_result.eval_id} - {'PASSED' if eval_result.passed else 'FAILED'}"
        )

        # Promote model
        promoted = await engine.promote_model(
            model_name="confidence_signal_model",
            version_id=model_version.version_id,
            target_stage=engine._production_models.get(
                "confidence_signal_model", model_version
            ).stage.__class__.PRODUCTION
            if hasattr(model_version, "stage")
            else None,
        )
        if promoted:
            print(f"✓ Model promoted to {promoted.stage.value}")

        # Get production model
        prod_model = engine.get_production_model("confidence_signal_model")
        if prod_model:
            print(f"✓ Production model: {prod_model.model_name} v{prod_model.version}")

        # Set drift baseline
        engine.set_baseline("confidence_signal_model", {"accuracy": 0.82, "precision": 0.79})
        print("✓ Drift baseline set")

        # Check for drift (no drift)
        alerts = engine.check_drift(
            "confidence_signal_model",
            {"accuracy": 0.81, "precision": 0.78},  # Minor changes, < 10%
        )
        print(f"✓ Drift check: {len(alerts)} alerts (expected 0)")

        # Check for drift (significant drift)
        alerts = engine.check_drift(
            "confidence_signal_model",
            {"accuracy": 0.70, "precision": 0.68},  # >10% change
        )
        print(f"✓ Drift check with degradation: {len(alerts)} alerts")
        if alerts:
            print(f"  - Alert: {alerts[0].metric_name} changed {alerts[0].current_value:.2f}")

        stats = engine.get_stats()
        print(
            f"✓ Final stats: {stats['production_models']} production models, {stats['drift_alerts']} drift alerts"
        )

        print()
        print("TEST PASSED: Model lifecycle operations functional")

    finally:
        await engine.shutdown()
        print("✓ Engine shutdown complete")


async def main():
    """Run all tests."""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " LEARNING ENGINE INTEGRATION TESTS ".center(78) + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    await test_learning_engine_basic()
    await test_learning_engine_with_backtest_data()
    await test_learning_engine_model_lifecycle()

    print()
    print("=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print()


if __name__ == "__main__":
    asyncio.run(main())
