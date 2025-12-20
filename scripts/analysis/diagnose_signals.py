"""
Diagnostic script to identify signal generation bottlenecks.

This script checks:
1. Why signals aren't being generated
2. Which models are being loaded
3. What signal thresholds are too strict
4. How to tune models for live trading
"""

import asyncio

import numpy as np
import pandas as pd

from ordinis.engines.signalcore.core.engine import SignalCoreEngine
from ordinis.engines.signalcore.core.signal import SignalBatch


async def diagnose_signal_generation():
    """Diagnose signal generation issues."""

    print("=" * 70)
    print("SIGNAL GENERATION DIAGNOSTIC")
    print("=" * 70)

    # 1. Initialize engine
    print("\n[1] Initializing SignalCore Engine...")
    engine = SignalCoreEngine()
    print("✓ Engine initialized")

    # 2. Check registered models
    print("\n[2] Checking registered models...")
    registry = engine.model_registry if hasattr(engine, "model_registry") else {}
    models = engine.models if hasattr(engine, "models") else []

    print("   Registered models in engine:")
    if hasattr(engine, "models"):
        for i, model in enumerate(engine.models, 1):
            print(f"   {i}. {model.__class__.__name__}")
    else:
        print("   (Models attribute not found)")

    # 3. Generate sample data
    print("\n[3] Generating sample OHLCV data...")
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    np.random.seed(42)
    price = 100
    prices = []
    for _ in range(100):
        price = price * (1 + np.random.normal(0, 0.02))
        prices.append(price)

    prices = np.array(prices)

    sample_data = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.005, 100)),
            "high": prices * (1 + abs(np.random.normal(0.01, 0.01, 100))),
            "low": prices * (1 - abs(np.random.normal(0.01, 0.01, 100))),
            "close": prices,
            "volume": np.random.normal(1e6, 1e5, 100).astype(int),
        },
        index=dates,
    )

    # Ensure OHLC relationships
    sample_data["high"] = sample_data[["open", "close", "high"]].max(axis=1) * 1.001
    sample_data["low"] = sample_data[["open", "close", "low"]].min(axis=1) * 0.999

    print(f"✓ Generated {len(sample_data)} bars")
    print("\n   Sample data statistics:")
    print(
        f"   Close price range: ${sample_data['close'].min():.2f} - ${sample_data['close'].max():.2f}"
    )
    print(f"   Volatility (std): {sample_data['close'].pct_change().std() * 100:.2f}%")
    print(f"   Volume (avg): {sample_data['volume'].mean() / 1e6:.1f}M")

    # 4. Try to generate signals
    print("\n[4] Attempting signal generation...")
    try:
        signals = await engine.generate_signals(symbol="AAPL", data=sample_data, lookback=20)

        if isinstance(signals, SignalBatch):
            print("✓ Generated signal batch:")
            print(f"   Total signals in batch: {len(signals.signals)}")

            if signals.signals:
                for i, signal in enumerate(signals.signals[:5]):  # Show first 5
                    print(f"\n   Signal {i+1}:")
                    print(f"     - Score: {signal.score:.3f}")
                    print(f"     - Probability: {signal.probability:.3f}")
                    print(f"     - Direction: {signal.direction}")
                    print(f"     - Model: {signal.metadata.get('model_name', 'unknown')}")
            else:
                print("   ✗ No signals generated!")
                print("\n   DIAGNOSIS: Signal thresholds may be too strict")
                print("   Action: Lower confidence thresholds in models")
        else:
            print(f"✓ Generated signals: {signals}")

    except Exception as e:
        print(f"✗ Error generating signals: {e}")
        import traceback

        traceback.print_exc()

    # 5. Analyze model behavior
    print("\n[5] Checking individual model behavior...")

    if hasattr(engine, "models"):
        for model in engine.models[:3]:  # Check first 3
            print(f"\n   Model: {model.__class__.__name__}")
            try:
                signal = await model.generate(symbol="AAPL", data=sample_data, lookback=20)

                if signal:
                    print("     ✓ Generated signal")
                    print(f"       Score: {signal.score:.3f}")
                    print(f"       Probability: {signal.probability:.3f}")
                else:
                    print("     ✗ No signal (returned None)")

            except Exception as e:
                print(f"     ✗ Error: {e}")

    # 6. Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    print("""
✓ If NO signals are being generated:
  1. Check model threshold configuration
  2. Lower confidence/probability thresholds
  3. Enable ensemble voting to accept signals from fewer models
  4. Add debug logging to model.generate() methods

✓ If SOME signals are being generated:
  1. Extract IC scores from backtest
  2. Use high-IC models in live trading
  3. Disable or down-weight low-IC models
  4. Combine with position sizing based on confidence

✓ For live deployment:
  1. Run backtests on real data (2023, 2024, 2025)
  2. Extract IC scores for each model
  3. Create production config with IC-weighted ensemble
  4. Start paper trading to validate
  5. Gradually scale to real capital

✓ Key metrics to monitor:
  - Information Coefficient (IC): Should be > 0.05 for edge
  - Hit Rate: Target > 50%
  - Sharpe Ratio: Target > 1.0
  - Max Drawdown: Keep < 20%
  - Profit Factor: Target > 1.5
    """)


async def main():
    await diagnose_signal_generation()


if __name__ == "__main__":
    asyncio.run(main())
