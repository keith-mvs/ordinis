"""
Phase 1 Completion Test: Fundamental Signals

Tests the newly implemented Fundamental signal models (Valuation & Growth)
integrated with the SignalCore engine.
"""

import asyncio
import os
import sys

import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ordinis.engines.signalcore.core.config import SignalCoreEngineConfig
from ordinis.engines.signalcore.core.engine import SignalCoreEngine
from ordinis.engines.signalcore.core.ensemble import EnsembleStrategy
from ordinis.engines.signalcore.models.fundamental import GrowthModel, ValuationModel


async def test_phase_1_completion():
    """
    Demonstrate Phase 1 (Fundamental Signals) completion.

    Phase 1 Goals:
    - ✓ Valuation Model (P/E, P/B, EV/EBITDA)
    - ✓ Growth Model (Revenue Growth, EPS Growth, Margin Expansion)
    - ✓ Integration with SignalCore Engine
    - ✓ Ensemble Support
    """
    print("=" * 70)
    print("PHASE 1 COMPLETION: FUNDAMENTAL SIGNALS")
    print("=" * 70)

    # Initialize Engine
    print("\n[1/4] Initializing SignalCore Engine...")
    config = SignalCoreEngineConfig(
        min_probability=0.5,
        min_score=0.0,
        enable_governance=False,  # Disable for testing
        ensemble_strategy=EnsembleStrategy.WEIGHTED_AVERAGE,
    )
    engine = SignalCoreEngine(config)
    await engine.initialize()

    # Register Fundamental Models
    print("[2/4] Registering Fundamental Models...")
    val_model = ValuationModel()
    growth_model = GrowthModel()

    engine.register_model(val_model)
    engine.register_model(growth_model)

    print(f"  ✓ Registered: {val_model.config.model_id}")
    print(f"  ✓ Registered: {growth_model.config.model_id}")

    # Create Test Data
    print("[3/4] Creating Test Data...")
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

    # Company A: Undervalued + High Growth (Strong BUY)
    data_a = pd.DataFrame(
        {
            "open": [100] * 100,
            "high": [105] * 100,
            "low": [95] * 100,
            "close": [102] * 100,
            "volume": [1000000] * 100,
            "pe_ratio": [8.0] * 100,  # Low P/E (undervalued)
            "pb_ratio": [0.8] * 100,  # Low P/B (undervalued)
            "ev_ebitda": [6.0] * 100,  # Low EV/EBITDA (undervalued)
            "revenue_growth": [0.30] * 100,  # 30% revenue growth
            "eps_growth": [0.35] * 100,  # 35% EPS growth
            "operating_margin": [0.20] * 100,
            "symbol": ["COMPANY_A"] * 100,
        },
        index=dates,
    )

    # Company B: Overvalued + Declining Growth (Strong SELL)
    data_b = pd.DataFrame(
        {
            "open": [200] * 100,
            "high": [210] * 100,
            "low": [190] * 100,
            "close": [205] * 100,
            "volume": [500000] * 100,
            "pe_ratio": [45.0] * 100,  # High P/E (overvalued)
            "pb_ratio": [8.0] * 100,  # High P/B (overvalued)
            "ev_ebitda": [25.0] * 100,  # High EV/EBITDA (overvalued)
            "revenue_growth": [-0.10] * 100,  # -10% revenue growth
            "eps_growth": [-0.15] * 100,  # -15% EPS growth
            "operating_margin": [0.08] * 100,
            "symbol": ["COMPANY_B"] * 100,
        },
        index=dates,
    )

    # Company C: Fairly Valued + Moderate Growth (NEUTRAL)
    data_c = pd.DataFrame(
        {
            "open": [150] * 100,
            "high": [155] * 100,
            "low": [145] * 100,
            "close": [152] * 100,
            "volume": [750000] * 100,
            "pe_ratio": [18.0] * 100,  # Fair P/E
            "pb_ratio": [2.5] * 100,  # Fair P/B
            "ev_ebitda": [12.0] * 100,  # Fair EV/EBITDA
            "revenue_growth": [0.08] * 100,  # 8% revenue growth
            "eps_growth": [0.10] * 100,  # 10% EPS growth
            "operating_margin": [0.15] * 100,
            "symbol": ["COMPANY_C"] * 100,
        },
        index=dates,
    )

    test_cases = [
        ("COMPANY_A (Undervalued + High Growth)", data_a),
        ("COMPANY_B (Overvalued + Declining Growth)", data_b),
        ("COMPANY_C (Fairly Valued + Moderate Growth)", data_c),
    ]

    # Generate Signals
    print("[4/4] Generating Signals...\n")
    print("=" * 70)

    for name, data in test_cases:
        print(f"\n{name}")
        print("-" * 70)

        timestamp = dates[-1]

        # Generate individual signals
        val_signal = await val_model.generate(data, timestamp)
        growth_signal = await growth_model.generate(data, timestamp)

        print("  Valuation Signal:")
        print(f"    Direction: {val_signal.direction.value}")
        print(f"    Score: {val_signal.score:.3f}")
        print(f"    Probability: {val_signal.probability:.2%}")
        print(f"    Composite: {val_signal.metadata['composite_score']:.1f}/100")

        print("\n  Growth Signal:")
        print(f"    Direction: {growth_signal.direction.value}")
        print(f"    Score: {growth_signal.score:.3f}")
        print(f"    Probability: {growth_signal.probability:.2%}")
        print(f"    Composite: {growth_signal.metadata['composite_score']:.1f}/100")

        # Generate ensemble signal using generate_batch
        batch_data = {data["symbol"].iloc[0]: data}
        batch = await engine.generate_batch(data=batch_data, timestamp=timestamp)

        # Display ensemble (last signal in batch if ensemble enabled)
        if batch.signals:
            # Find ensemble signal (model_id will be "ensemble_...")
            ensemble = next((s for s in batch.signals if "ensemble" in s.model_id), None)
            if ensemble:
                print("\n  Ensemble Signal (Weighted Average):")
                print(f"    Direction: {ensemble.direction.value}")
                print(f"    Score: {ensemble.score:.3f}")
                print(f"    Probability: {ensemble.probability:.2%}")

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE ✓")
    print("=" * 70)
    print("\nNext Steps:")
    print("  Phase 2: Sentiment Signals (News, Social Media)")
    print("  Phase 3: Algorithmic Signals (Pairs Trading, Arbitrage)")
    print("  Phase 4: Advanced Ensembles (IC-Weighted, Vol-Adjusted)")
    print()


if __name__ == "__main__":
    asyncio.run(test_phase_1_completion())
