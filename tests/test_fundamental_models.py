import asyncio
import os
import sys

import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ordinis.engines.signalcore.models.fundamental.growth import GrowthModel
from ordinis.engines.signalcore.models.fundamental.valuation import ValuationModel


async def test_fundamental_models():
    print("Testing Fundamental Models...")

    # Create dummy data
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    data = pd.DataFrame(
        {
            "open": [100] * 10,
            "high": [105] * 10,
            "low": [95] * 10,
            "close": [102] * 10,
            "volume": [1000] * 10,
            "pe_ratio": [
                10,
                12,
                14,
                16,
                18,
                20,
                25,
                30,
                35,
                40,
            ],  # Increasing PE (getting expensive)
            "pb_ratio": [1.0] * 10,
            "ev_ebitda": [8.0] * 10,
            "revenue_growth": [
                0.25,
                0.24,
                0.23,
                0.22,
                0.21,
                0.20,
                0.15,
                0.10,
                0.05,
                0.0,
            ],  # Declining growth
            "eps_growth": [0.25] * 10,
            "operating_margin": [0.15] * 10,
            "symbol": ["TEST"] * 10,
        },
        index=dates,
    )

    timestamp = dates[-1]

    # Test Valuation Model
    print("\n--- Valuation Model ---")
    val_model = ValuationModel()
    val_signal = await val_model.generate(data, timestamp)
    print(f"Signal: {val_signal.direction} (Score: {val_signal.score:.2f})")
    print(f"Metadata: {val_signal.metadata}")

    # Test Growth Model
    print("\n--- Growth Model ---")
    growth_model = GrowthModel()
    growth_signal = await growth_model.generate(data, timestamp)
    print(f"Signal: {growth_signal.direction} (Score: {growth_signal.score:.2f})")
    print(f"Metadata: {growth_signal.metadata}")


if __name__ == "__main__":
    asyncio.run(test_fundamental_models())
