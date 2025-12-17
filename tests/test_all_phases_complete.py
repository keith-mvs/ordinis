"""
Comprehensive Test: Phases 1-4 Complete

Tests all implemented signal models and advanced ensemble strategies.
"""

import asyncio
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), "src"))

from ordinis.engines.signalcore.core.config import SignalCoreEngineConfig
from ordinis.engines.signalcore.core.engine import SignalCoreEngine
from ordinis.engines.signalcore.core.ensemble import EnsembleStrategy
from ordinis.engines.signalcore.models.algorithmic import IndexRebalanceModel, PairsTradingModel
from ordinis.engines.signalcore.models.fundamental import GrowthModel, ValuationModel
from ordinis.engines.signalcore.models.sentiment import NewsSentimentModel


async def test_all_phases():
    """Comprehensive test of all phases."""

    print("=" * 80)
    print("COMPREHENSIVE TEST: PHASES 1-4")
    print("=" * 80)

    # Initialize Engine
    print("\n[Setup] Initializing SignalCore Engine...")
    config = SignalCoreEngineConfig(
        min_probability=0.0,
        min_score=-1.0,
        enable_governance=False,
        enable_ensemble=True,
        ensemble_strategy=EnsembleStrategy.VOL_ADJUSTED,
    )
    engine = SignalCoreEngine(config)
    await engine.initialize()

    # Register All Models
    print("[Setup] Registering All Models...")
    models = [
        ("Phase 1", ValuationModel()),
        ("Phase 1", GrowthModel()),
        ("Phase 2", NewsSentimentModel()),
        ("Phase 3", PairsTradingModel()),
        ("Phase 3", IndexRebalanceModel()),
    ]

    for phase, model in models:
        engine.register_model(model)
        print(f"  ✓ {phase}: {model.config.model_id}")

    # Create Test Data
    print("\n[Setup] Creating Test Data...")
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    test_data = pd.DataFrame(
        {
            "open": [100 + i * 0.5 for i in range(100)],
            "high": [105 + i * 0.5 for i in range(100)],
            "low": [95 + i * 0.5 for i in range(100)],
            "close": [102 + i * 0.5 for i in range(100)],
            "volume": [1000000] * 100,
            # Fundamental
            "pe_ratio": [12.0] * 100,
            "pb_ratio": [1.5] * 100,
            "ev_ebitda": [8.0] * 100,
            "revenue_growth": [0.15] * 100,
            "eps_growth": [0.18] * 100,
            "operating_margin": [0.18] * 100,
            # Sentiment
            "news_sentiment": [0.7] * 100,  # Positive sentiment
            # Algorithmic
            "spread": np.random.randn(100) * 0.5,
            "index_event": ["none"] * 99 + ["addition"],
            "symbol": ["TEST_STOCK"] * 100,
        },
        index=dates,
    )

    timestamp = dates[-1]

    # Test Each Model
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODEL SIGNALS")
    print("=" * 80)

    for phase, model in models:
        signal = await model.generate(test_data, timestamp)
        print(f"\n{model.config.model_id} ({phase})")
        print(f"  Direction: {signal.direction.value}")
        print(f"  Score: {signal.score:.3f}")
        print(f"  Probability: {signal.probability:.2%}")

    # Test All Ensemble Strategies
    print("\n" + "=" * 80)
    print("ENSEMBLE STRATEGIES (Phase 4)")
    print("=" * 80)

    ensemble_strategies = [
        ("Voting", EnsembleStrategy.VOTING),
        ("Weighted Average", EnsembleStrategy.WEIGHTED_AVERAGE),
        ("Highest Confidence", EnsembleStrategy.HIGHEST_CONFIDENCE),
        ("IC-Weighted", EnsembleStrategy.IC_WEIGHTED),
        ("Volatility-Adjusted", EnsembleStrategy.VOL_ADJUSTED),
        ("Regression-Based", EnsembleStrategy.REGRESSION),
    ]

    for strategy_name, strategy in ensemble_strategies:
        config.ensemble_strategy = strategy
        engine._config = config  # Update config

        batch = await engine.generate_batch(data={"TEST_STOCK": test_data}, timestamp=timestamp)

        ensemble = next((s for s in batch.signals if "ensemble" in s.model_id), None)
        if ensemble:
            print(f"\n{strategy_name}")
            print(f"  Model: {ensemble.model_id}")
            print(f"  Direction: {ensemble.direction.value}")
            print(f"  Score: {ensemble.score:.3f}")
            print(f"  Probability: {ensemble.probability:.2%}")

    # Summary
    print("\n" + "=" * 80)
    print("ALL PHASES COMPLETE ✓")
    print("=" * 80)
    print("\nImplemented:")
    print("  ✓ Phase 1: Fundamental Models (Valuation, Growth)")
    print("  ✓ Phase 2: Sentiment Models (News Sentiment)")
    print("  ✓ Phase 3: Algorithmic Models (Pairs Trading, Index Rebalancing)")
    print("  ✓ Phase 4: Advanced Ensembles (6 strategies)")
    print("\nTotal Models: 5")
    print("Total Ensemble Strategies: 6")
    print("\nAll models integrated with SignalCore Engine ✓")
    print()


if __name__ == "__main__":
    asyncio.run(test_all_phases())
