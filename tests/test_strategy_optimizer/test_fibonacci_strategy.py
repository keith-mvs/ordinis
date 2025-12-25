import asyncio
from datetime import datetime

import numpy as np
import pandas as pd

from ordinis.application.strategies.fibonacci_adx import FibonacciADXStrategy


def test_fibonacci_strategy_instantiation_and_signal():
    # Create synthetic OHLCV data (300 days)
    idx = pd.date_range(end=pd.Timestamp.now(), periods=300, freq="D")
    prices = np.linspace(100.0, 150.0, num=300)

    df = pd.DataFrame(
        {
            "open": prices * 0.99,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(100, 1000, size=300),
        },
        index=idx,
    )

    strategy = FibonacciADXStrategy(
        name="test_fib",
        adx_period=14,
        adx_threshold=10,
        swing_lookback=50,
        fib_levels=[0.382, 0.5, 0.618],
        tolerance=0.03,
    )

    # Ensure model attributes exist
    assert hasattr(strategy, "adx_model")
    assert hasattr(strategy, "fib_model")

    # Run signal generation (should not raise)
    sig = asyncio.run(strategy.generate_signal(df, df.index[-1]))

    # Accept either a Signal object or None
    assert sig is None or hasattr(sig, "signal_type")

    # Check description and required bars
    desc = strategy.get_description()
    assert isinstance(desc, str) and len(desc) > 0
    assert isinstance(strategy.required_bars, int) and strategy.required_bars > 0
