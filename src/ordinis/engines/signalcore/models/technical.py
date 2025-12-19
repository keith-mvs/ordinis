from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType


class SMACrossoverModel(Model):
    """
    Simple Moving Average Crossover Strategy.

    Generates buy signals when short SMA crosses above long SMA,
    and sell signals when short SMA crosses below long SMA.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.short_period = config.parameters.get("short_period", 10)
        self.long_period = config.parameters.get("long_period", 50)

    async def validate(self, data: Any) -> bool:
        """Validate if enough data is available."""
        if not isinstance(data, pd.DataFrame):
            return False
        if len(data) < self.long_period:
            return False
        return "close" in data.columns

    async def generate(self, data: Any) -> list[Signal]:
        """Generate signals based on SMA crossover."""
        if not await self.validate(data):
            return []

        df = data.copy()
        # Ensure we have sorted data
        df = df.sort_index()

        # Calculate indicators
        df["sma_short"] = df["close"].rolling(window=self.short_period).mean()
        df["sma_long"] = df["close"].rolling(window=self.long_period).mean()

        signals = []

        # We only look at the last completed bar for signal generation in this simple version
        if len(df) < 2:
            return []

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # Handle symbol extraction safely
        symbol = "UNKNOWN"
        if "symbol" in curr:
            symbol = str(curr["symbol"])
        elif hasattr(data, "symbol"):  # If passed as metadata on dataframe
            symbol = str(data.symbol)

        timestamp = curr.name if isinstance(curr.name, datetime) else datetime.now(UTC)

        # Crossover logic
        # Bullish Crossover: Short crosses above Long
        if prev["sma_short"] <= prev["sma_long"] and curr["sma_short"] > curr["sma_long"]:
            signals.append(
                Signal(
                    symbol=symbol,
                    timestamp=timestamp,
                    signal_type=SignalType.ENTRY,
                    direction=Direction.LONG,
                    probability=0.6,
                    score=0.8,
                    model_id=self.config.model_id,
                    metadata={
                        "strategy": "sma_crossover",
                        "short_sma": float(curr["sma_short"]),
                        "long_sma": float(curr["sma_long"]),
                    },
                )
            )

        # Bearish Crossover: Short crosses below Long
        elif prev["sma_short"] >= prev["sma_long"] and curr["sma_short"] < curr["sma_long"]:
            signals.append(
                Signal(
                    symbol=symbol,
                    timestamp=timestamp,
                    signal_type=SignalType.ENTRY,  # Or EXIT depending on position, assuming ENTRY SHORT for now
                    direction=Direction.SHORT,
                    probability=0.6,
                    score=-0.8,
                    model_id=self.config.model_id,
                    metadata={
                        "strategy": "sma_crossover",
                        "short_sma": float(curr["sma_short"]),
                        "long_sma": float(curr["sma_long"]),
                    },
                )
            )

        return signals
