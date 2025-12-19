"""
Statistical Mean Reversion Model.

Uses Z-Score of price relative to a moving average to identify overextended moves.
Buy when Z-Score < -Threshold (Oversold).
Sell when Z-Score > +Threshold (Overbought).
"""

from datetime import datetime

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


class StatisticalReversionModel(Model):
    """
    Statistical Mean Reversion trading model.

    Uses Z-Score (Standard Score) to identify statistical anomalies in price.
    Assumes price reverts to the mean after extreme deviations.

    Parameters:
        window: Rolling window for mean/std calculation (default 20)
        entry_threshold: Z-Score threshold for entry (default 2.0)
        exit_threshold: Z-Score threshold for exit (default 0.5)

    Signals:
        - ENTRY/LONG when Z-Score < -entry_threshold
        - ENTRY/SHORT when Z-Score > +entry_threshold
        - EXIT when Z-Score returns within +/- exit_threshold
    """

    def __init__(self, config: ModelConfig):
        """Initialize Statistical Reversion model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.window = params.get("window", 20)
        self.entry_threshold = params.get("entry_threshold", 2.0)
        self.exit_threshold = params.get("exit_threshold", 0.5)

        # Update min data points
        self.config.min_data_points = self.window + 10

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate trading signal from Z-Score analysis.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with Z-Score prediction
        """
        # Validate data
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        if "symbol" in data:
            symbol_data = data["symbol"]
            symbol = symbol_data.iloc[0] if hasattr(symbol_data, "iloc") else str(symbol_data)
        else:
            symbol = "UNKNOWN"

        close = data["close"]

        # Calculate Z-Score
        z_score = TechnicalIndicators.z_score(close, self.window)

        current_z = z_score.iloc[-1]
        current_price = close.iloc[-1]

        # Determine signal
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        # Logic
        # Long Entry: Price is statistically cheap (Z < -2)
        if current_z < -self.entry_threshold:
            signal_type = SignalType.ENTRY
            direction = Direction.LONG
            score = abs(current_z) / 4.0  # Normalize roughly 0-1
            probability = 0.7 + (
                min(abs(current_z), 4.0) / 20.0
            )  # Higher deviation = higher prob (up to a point)
            expected_return = 0.02 * abs(current_z)  # Expect reversion to mean

        # Short Entry: Price is statistically expensive (Z > 2)
        elif current_z > self.entry_threshold:
            signal_type = SignalType.ENTRY
            direction = Direction.SHORT
            score = -abs(current_z) / 4.0
            probability = 0.7 + (min(abs(current_z), 4.0) / 20.0)
            expected_return = -0.02 * abs(current_z)

        # Exit Long: Price returned to mean (Z > -0.5)
        elif current_z > -self.exit_threshold and current_z < self.exit_threshold:
            # This logic is tricky for a stateless model.
            # Ideally we'd know if we are in a position.
            # For now, we signal EXIT if we are near mean, implying "close any reversion trades"
            signal_type = SignalType.EXIT
            direction = Direction.NEUTRAL
            score = 0.0
            probability = 0.5

        # Cap score
        score = max(min(score, 1.0), -1.0)

        return Signal(
            model_id=self.config.model_id,
            signal_type=signal_type,
            direction=direction,
            score=score,
            timestamp=timestamp,
            metadata={
                "window": self.window,
                "z_score": float(current_z),
                "entry_threshold": self.entry_threshold,
                "current_price": float(current_price),
            },
            symbol=str(symbol),
            probability=probability,
            expected_return=expected_return,
            confidence_interval=(expected_return - 0.01, expected_return + 0.01),
            model_version="1.0",
        )
