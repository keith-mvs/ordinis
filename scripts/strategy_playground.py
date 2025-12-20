import asyncio
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType
from ordinis.runtime import initialize, load_config

# --- 1. Define Your Custom Strategy Here ---


class MyCustomStrategy(Model):
    """
    Template for a custom strategy.
    Implement your logic in the generate() method.
    """

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate a trading signal based on market data.

        Args:
            data: DataFrame containing market history (OHLCV)
            timestamp: Current simulation time

        Returns:
            Signal object with trading decision
        """
        # --- YOUR LOGIC HERE ---

        # Example: Access latest price
        current_price = data.iloc[-1]["close"]

        # Example: Calculate a simple indicator (e.g., RSI, MACD, Bollinger Bands)
        # For this template, we'll just use a random decision
        # Replace this with your actual quantitative logic

        # Placeholder logic
        is_buy_signal = np.random.random() > 0.5
        direction = Direction.LONG if is_buy_signal else Direction.SHORT

        print(
            f"  [Strategy] Analyzing {self.config.model_id}: Price={current_price} -> {direction}"
        )

        return Signal(
            symbol="AAPL",  # Target symbol
            timestamp=timestamp,
            signal_type=SignalType.ENTRY,
            direction=direction,
            probability=0.75,  # Your model's confidence
            expected_return=0.02,  # Expected return for this trade
            score=0.8,  # Overall score
            model_id=self.config.model_id,
            metadata={"price": current_price, "note": "Template strategy"},
            confidence_interval=(0.01, 0.03),
            model_version=self.config.version,
        )


async def main():
    print("[INFO] Starting Strategy Dev Playground...")

    # 1. Initialize Dev Environment
    # This loads the lightweight dev stack (In-Memory Bus, SQLite, Paper Trading)
    settings = load_config("configs/dev.yaml")
    container = initialize(settings)
    print("[INFO] Dev Container Initialized")

    # 1.5 Initialize Database
    db = container.get_database_manager()
    try:
        if db:
            print("[INFO] Initializing Database Connection...")
            await db.initialize()
            print("[INFO] Database Connected")
        else:
            print("[WARNING] No Database Manager found - persistence disabled")

        # 2. Register Your Strategy
        print("[INFO] Registering Custom Strategy...")
        config = ModelConfig(
            model_id="my_custom_strategy_v1", model_type="technical", min_data_points=10
        )
        strategy = MyCustomStrategy(config)

        # Register with the engine
        container.signal_engine._registry.register(strategy)
        print(f"[INFO] Registered models: {container.signal_engine.list_models()}")

        # 3. Create Mock Data
        # In a real scenario, this would come from the Market Data Adapter
        # Here we create a synthetic history for testing
        dates = pd.date_range(start="2025-01-01", periods=20)
        mock_df = pd.DataFrame(
            {
                "open": np.random.normal(150, 2, 20),
                "high": np.random.normal(152, 2, 20),
                "low": np.random.normal(148, 2, 20),
                "close": np.random.normal(150, 2, 20),
                "volume": np.random.randint(1000, 5000, 20),
            },
            index=dates,
        )

        # 4. Run a Single Cycle
        print("\n[INFO] Running Trading Cycle...")

        # Inject data into the pipeline
        cycle_data = {"AAPL": mock_df}

        # Run the full orchestration: Signal -> Risk -> Execution -> Portfolio
        await container.orchestration.run_cycle(data=cycle_data)

        print("[INFO] Cycle Complete. Check logs/dev_audit.log for details.")

    finally:
        # Cleanup
        if db:
            print("[INFO] Shutting down Database Connection...")
            await db.shutdown()
            print("[INFO] Database Disconnected")


if __name__ == "__main__":
    asyncio.run(main())
