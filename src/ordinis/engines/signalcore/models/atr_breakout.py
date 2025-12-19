"""
ATR Breakout Model.

Volatility breakout strategy using Keltner Channels (EMA +/- ATR).
Buy when price breaks above upper channel, sell when price breaks below lower channel.
"""

from datetime import datetime

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


class ATRBreakoutModel(Model):
    """
    ATR Breakout (Keltner Channel) trading model.

    Generates signals based on price breakouts from Keltner Channels,
    which are volatility-based bands around an EMA.

    Parameters:
        ema_period: EMA period for the center line (default 20)
        atr_period: ATR calculation period (default 10)
        multiplier: ATR multiplier for channel width (default 2.0)

    Signals:
        - ENTRY/LONG when price closes above upper channel
        - ENTRY/SHORT when price closes below lower channel
        - HOLD when price is within channels
        - Score based on breakout magnitude
    """

    def __init__(self, config: ModelConfig):
        """Initialize ATR Breakout model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.ema_period = params.get("ema_period", 20)
        self.atr_period = params.get("atr_period", 10)
        self.multiplier = params.get("multiplier", 2.0)

        # Update min data points
        self.config.min_data_points = max(self.ema_period, self.atr_period) + 20

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate trading signal from ATR Breakout analysis.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with ATR Breakout prediction
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

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Calculate Indicators
        atr = TechnicalIndicators.atr(high, low, close, self.atr_period)
        ema = TechnicalIndicators.ema(close, self.ema_period)

        # Calculate Channels
        upper = ema + (self.multiplier * atr)
        lower = ema - (self.multiplier * atr)

        # Get current values
        current_price = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]

        # Previous values for crossover detection
        prev_price = close.iloc[-2] if len(close) > 1 else current_price
        prev_upper = upper.iloc[-2] if len(upper) > 1 else current_upper
        prev_lower = lower.iloc[-2] if len(lower) > 1 else current_lower

        # Determine signal
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        # Breakout Logic
        breakout_up = current_price > current_upper
        breakout_down = current_price < current_lower

        # Crossover Logic (fresh breakout)
        crossed_up = prev_price <= prev_upper and current_price > current_upper
        crossed_down = prev_price >= prev_lower and current_price < current_lower

        if crossed_up:
            signal_type = SignalType.ENTRY
            direction = Direction.LONG

            # Score based on breakout strength
            breakout_magnitude = (current_price - current_upper) / current_upper
            score = min(0.5 + (breakout_magnitude * 10), 0.95)
            probability = 0.65 + (score * 0.1)
            expected_return = 0.05 + (score * 0.05)

        elif crossed_down:
            signal_type = SignalType.ENTRY
            direction = Direction.SHORT

            breakout_magnitude = (current_lower - current_price) / current_lower
            score = min(0.5 + (breakout_magnitude * 10), 0.95)
            probability = 0.65 + (score * 0.1)
            expected_return = 0.05 + (score * 0.05)

        elif breakout_up:
            # Continuing trend
            signal_type = SignalType.HOLD
            direction = Direction.LONG
            score = 0.6

        elif breakout_down:
            # Continuing trend
            signal_type = SignalType.HOLD
            direction = Direction.SHORT
            score = 0.6

        # Create signal if actionable
        if signal_type == SignalType.ENTRY:
            return Signal(
                symbol=symbol,
                timestamp=timestamp,
                signal_type=signal_type,
                direction=direction,
                probability=probability,
                expected_return=expected_return,
                score=score,
                model_id=self.config.model_id,
                model_version=self.config.version,
                metadata={
                    "price": float(current_price),
                    "upper_band": float(current_upper),
                    "lower_band": float(current_lower),
                    "atr": float(atr.iloc[-1]),
                    "indicator": "ATR_BREAKOUT",
                },
                confidence_interval=(expected_return * 0.8, expected_return * 1.2),
            )

        return None
