"""
Fundamental Value Model.

Identifies undervalued securities based on Price-to-Earnings (P/E) ratio.
Based on 'Value-Based Trading Signals' from Knowledge Base.
"""

from datetime import datetime

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType


class FundamentalValueModel(Model):
    """
    Fundamental Value trading model.

    Uses P/E ratio to identify deep value opportunities.

    Parameters:
        value_threshold: P/E below this is considered undervalued (default 15.0)
        deep_value_threshold: P/E below this is deep value (default 10.0)
        overvalued_threshold: P/E above this is overvalued (default 35.0)

    Signals:
        - ENTRY/LONG: P/E < value_threshold
        - ENTRY/SHORT: P/E > overvalued_threshold
    """

    def __init__(self, config: ModelConfig):
        """Initialize Fundamental Value model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.value_threshold = params.get("value_threshold", 15.0)
        self.deep_value_threshold = params.get("deep_value_threshold", 10.0)
        self.overvalued_threshold = params.get("overvalued_threshold", 35.0)

        # Fundamental data doesn't need a long warmup, but we need at least 1 point
        self.config.min_data_points = 1

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate trading signal from Fundamental analysis.

        Args:
            data: Historical OHLCV + Fundamental data
            timestamp: Current timestamp

        Returns:
            Signal with Value prediction
        """
        # Validate data
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        if "pe_ratio" not in data.columns:
            # If fundamental data is missing, we cannot generate a signal
            return None

        if "symbol" in data:
            symbol_data = data["symbol"]
            symbol = symbol_data.iloc[0] if hasattr(symbol_data, "iloc") else str(symbol_data)
        else:
            symbol = "UNKNOWN"

        current_pe = data["pe_ratio"].iloc[-1]
        current_price = data["close"].iloc[-1]

        # Determine signal
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        # Logic
        # Undervalued
        if current_pe < self.value_threshold:
            signal_type = SignalType.ENTRY
            direction = Direction.LONG

            # Score increases as P/E gets lower (closer to 0)
            # If PE is 10 (deep value), score should be high
            dist = self.value_threshold - current_pe
            score = min(dist / 5.0, 1.0)  # Normalize

            probability = 0.6
            expected_return = 0.10  # Value investing expects higher long term returns

            if current_pe < self.deep_value_threshold:
                probability = 0.75
                score = 1.0

        # Overvalued
        elif current_pe > self.overvalued_threshold:
            signal_type = SignalType.ENTRY
            direction = Direction.SHORT

            dist = current_pe - self.overvalued_threshold
            score = -min(dist / 10.0, 1.0)

            probability = 0.6
            expected_return = -0.05

        return Signal(
            model_id=self.config.model_id,
            signal_type=signal_type,
            direction=direction,
            score=score,
            timestamp=timestamp,
            metadata={
                "pe_ratio": float(current_pe),
                "value_threshold": self.value_threshold,
                "current_price": float(current_price),
            },
            symbol=str(symbol),
            probability=probability,
            expected_return=expected_return,
            confidence_interval=(expected_return - 0.02, expected_return + 0.02),
            model_version="1.0",
        )
