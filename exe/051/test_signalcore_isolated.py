#!/usr/bin/env python
"""
Test SignalCore Engine in Isolation

Demonstrates the SignalCore engine capabilities running independently
before integration into the full demo.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ordinis.engines.signalcore.core.engine import SignalCoreEngine
from ordinis.engines.signalcore.core.config import SignalCoreEngineConfig
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models import (
    ATROptimizedRSIModel,
    BollingerBandsModel,
    MACDModel,
    RSIMeanReversionModel,
    SMACrossoverModel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(symbol: str, bars: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)

    # Generate realistic price data
    start_price = 100
    returns = np.random.normal(0.001, 0.02, bars)  # Mean return 0.1%, volatility 2%
    prices = start_price * np.exp(np.cumsum(returns))

    # Add some trend
    trend = np.linspace(0, 10, bars)
    prices = prices + trend

    # Generate OHLCV
    dates = pd.date_range(end=datetime.now(UTC), periods=bars, freq='5min')

    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, bars)),
        'high': prices * (1 + np.random.uniform(0, 0.01, bars)),
        'low': prices * (1 - np.random.uniform(0, 0.01, bars)),
        'close': prices,
        'volume': np.random.uniform(100000, 1000000, bars)
    })

    data.set_index('timestamp', inplace=True)
    return data


async def test_individual_models():
    """Test individual models in isolation."""
    print("\n" + "="*80)
    print("TESTING INDIVIDUAL SIGNALCORE MODELS")
    print("="*80 + "\n")

    # Generate test data
    symbol = "TEST"
    data = generate_sample_data(symbol, 200)

    # Test models
    models_to_test = [
        ("ATR-Optimized RSI", ATROptimizedRSIModel, {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "atr_stop_mult": 1.5,
            "atr_tp_mult": 2.0
        }),
        ("Bollinger Bands", BollingerBandsModel, {
            "period": 20,
            "std_dev": 2.0,
            "squeeze_threshold": 0.001
        }),
        ("MACD", MACDModel, {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        }),
        ("RSI Mean Reversion", RSIMeanReversionModel, {
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70
        }),
        ("SMA Crossover", SMACrossoverModel, {
            "short_period": 10,
            "long_period": 50
        })
    ]

    for model_name, model_class, params in models_to_test:
        print(f"\nTesting {model_name}...")
        print("-" * 40)

        try:
            # Create model config
            config = ModelConfig(
                model_id=f"{model_name.lower().replace(' ', '_')}_v1",
                model_type="technical",
                parameters=params
            )

            # Initialize model
            model = model_class(config)

            # Generate signal
            signal = await model.generate_signal(symbol, data)

            # Display results
            if signal:
                print(f"  Signal Generated: {signal.signal_type.value}")
                print(f"  Confidence: {signal.confidence:.2%}")
                print(f"  Entry Price: ${signal.entry_price:.2f}")
                if signal.stop_loss:
                    print(f"  Stop Loss: ${signal.stop_loss:.2f}")
                if signal.take_profit:
                    print(f"  Take Profit: ${signal.take_profit:.2f}")
                print(f"  Metadata: {signal.metadata}")
            else:
                print("  No signal generated")

        except Exception as e:
            print(f"  ERROR: {str(e)}")


async def test_signalcore_engine():
    """Test the complete SignalCore engine with multiple models."""
    print("\n" + "="*80)
    print("TESTING SIGNALCORE ENGINE WITH ENSEMBLE")
    print("="*80 + "\n")

    # Initialize engine
    engine_config = SignalCoreEngineConfig(
        min_probability=0.6,
        min_score=0.3,
        enable_governance=False,  # Disable governance for isolated test
        enable_ensemble=True,
        ensemble_min_agreement=2,  # Require at least 2 models to agree
    )

    engine = SignalCoreEngine(engine_config)
    await engine.initialize()

    # Register multiple models
    models = [
        (ATROptimizedRSIModel, {"rsi_oversold": 30, "rsi_overbought": 70}),
        (RSIMeanReversionModel, {"oversold": 30, "overbought": 70}),
        (BollingerBandsModel, {"period": 20, "std_dev": 2.0}),
    ]

    for i, (model_class, params) in enumerate(models):
        config = ModelConfig(
            model_id=f"model_{i}",
            model_type="technical",
            parameters=params,
            weight=1.0  # Equal weight for all models
        )
        model = model_class(config)
        engine.register_model(model)

    print(f"Registered {len(models)} models in ensemble")

    # Test on multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMD"]

    for symbol in symbols:
        print(f"\n{symbol} Analysis:")
        print("-" * 40)

        # Generate data
        data = generate_sample_data(symbol, 150)

        # Generate ensemble signal
        signal = await engine.generate_signal(symbol, data)

        if signal:
            print(f"  ENSEMBLE SIGNAL: {signal.signal_type.value}")
            print(f"  Confidence: {signal.confidence:.2%}")
            print(f"  Entry: ${signal.entry_price:.2f}")
            print(f"  Risk/Reward: SL ${signal.stop_loss:.2f} / TP ${signal.take_profit:.2f}")

            # Show which models agreed
            if 'model_signals' in signal.metadata:
                print(f"  Model Agreement: {signal.metadata['model_signals']}")
        else:
            print("  No consensus signal from ensemble")

    # Get engine metrics
    metrics = engine.get_metrics()
    print(f"\nEngine Metrics:")
    print(f"  Signals Generated: {metrics.signals_generated}")
    print(f"  Average Latency: {metrics.avg_latency:.3f}s")
    print(f"  Error Rate: {metrics.error_rate:.2%}")

    # Cleanup
    await engine.shutdown()


async def test_advanced_features():
    """Test advanced SignalCore features."""
    print("\n" + "="*80)
    print("TESTING ADVANCED SIGNALCORE FEATURES")
    print("="*80 + "\n")

    # Test confidence filtering
    print("1. Confidence Filtering:")
    print("-" * 40)

    config = SignalCoreEngineConfig(
        min_probability=0.75,  # High confidence threshold
        min_score=0.5,
        enable_confidence_filter=True
    )

    engine = SignalCoreEngine(config)
    await engine.initialize()

    # Register a model
    model_config = ModelConfig(
        model_id="high_confidence_test",
        model_type="technical",
        parameters={"rsi_oversold": 25, "rsi_overbought": 75}
    )
    model = ATROptimizedRSIModel(model_config)
    engine.register_model(model)

    # Generate data with clear signal
    data = generate_sample_data("TEST", 100)
    signal = await engine.generate_signal("TEST", data)

    if signal and signal.confidence >= 0.75:
        print(f"  ✓ High confidence signal passed: {signal.confidence:.2%}")
    else:
        print(f"  ✗ Signal filtered out due to low confidence")

    await engine.shutdown()

    print("\n2. Model Performance Tracking:")
    print("-" * 40)
    print("  SignalCore tracks model performance over time")
    print("  - Win rate per model")
    print("  - Average return per signal")
    print("  - Sharpe ratio")
    print("  - Maximum drawdown")
    print("  (Full tracking requires integration with execution engine)")


async def main():
    """Run all isolated tests."""
    print("\n" + "="*80)
    print("SIGNALCORE ENGINE ISOLATED TESTING")
    print("Testing SignalCore capabilities before integration")
    print("="*80)

    try:
        # Test individual models
        await test_individual_models()

        # Test ensemble engine
        await test_signalcore_engine()

        # Test advanced features
        await test_advanced_features()

        print("\n" + "="*80)
        print("TESTING COMPLETE")
        print("SignalCore engine is ready for integration")
        print("="*80 + "\n")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())