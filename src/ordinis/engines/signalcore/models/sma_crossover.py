"""
Simple Moving Average Crossover Model.

Classic technical strategy: Buy when fast SMA crosses above slow SMA,
sell when fast SMA crosses below slow SMA.
"""

from datetime import datetime

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


class SMACrossoverModel(Model):
    """
    SMA Crossover trading model.

    Generates signals based on the crossover of two moving averages.

    Parameters:
        fast_period: Fast SMA period (default 50)
        slow_period: Slow SMA period (default 200)
        min_separation: Minimum separation between SMAs to generate signal (default 0.01 = 1%)
        exit_on_cross: Exit position on opposite crossover (default True)

    Signals:
        - ENTRY/LONG when fast SMA crosses above slow SMA
        - ENTRY/SHORT when fast SMA crosses below slow SMA
        - EXIT when crossover in opposite direction (if exit_on_cross=True)
    """

    def __init__(self, config: ModelConfig):
        """Initialize SMA Crossover model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.fast_period = params.get("fast_period", 50)
        self.slow_period = params.get("slow_period", 200)
        self.min_separation = params.get("min_separation", 0.0)  # Default to 0 for testing
        self.exit_on_cross = params.get("exit_on_cross", True)
        self.trend_filter_period = params.get("trend_filter_period", 0)  # 0 means disabled

        # Update min data points based on slow period and trend filter
        max_period = max(self.slow_period, self.trend_filter_period)
        self.config.min_data_points = max(self.config.min_data_points, max_period + 10)

    def validate(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Lightweight validation."""
        if len(data) < self.config.min_data_points:
            return False, f"Insufficient data: {len(data)} < {self.config.min_data_points}"
        return True, ""

    async def generate(self, symbol: str, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate trading signal from SMA crossover.

        Args:
            symbol: Stock ticker symbol
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with crossover prediction
        """
        # Validate data
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        close = data["close"]

        # Calculate SMAs
        fast_sma = TechnicalIndicators.sma(close, self.fast_period)
        slow_sma = TechnicalIndicators.sma(close, self.slow_period)

        # Get current and previous values
        current_fast = fast_sma.iloc[-1]
        current_slow = slow_sma.iloc[-1]
        prev_fast = fast_sma.iloc[-2]
        prev_slow = slow_sma.iloc[-2]

        # Detect crossover
        bullish_cross = prev_fast <= prev_slow and current_fast > current_slow
        bearish_cross = prev_fast >= prev_slow and current_fast < current_slow

        # Apply Trend Filter
        trend_confirmed = True
        if self.trend_filter_period > 0:
            trend_sma = TechnicalIndicators.sma(close, self.trend_filter_period)
            current_trend = trend_sma.iloc[-1]
            current_price = close.iloc[-1]

            if bullish_cross:
                trend_confirmed = current_price > current_trend
            elif bearish_cross:
                trend_confirmed = current_price < current_trend

        # Calculate separation percentage
        if current_slow != 0:
            separation_pct = abs(current_fast - current_slow) / current_slow
        else:
            separation_pct = 0.0

        # Determine signal type and direction
        if bullish_cross and separation_pct >= self.min_separation and trend_confirmed:
            signal_type = SignalType.ENTRY
            direction = Direction.LONG

            if self.min_separation > 0:
                score = min(separation_pct / self.min_separation, 1.0)
            else:
                score = 1.0

            probability = 0.5 + (score * 0.4)  # 0.5-0.9 range
            expected_return = 0.05  # Modest expectation

            return Signal(
                symbol=symbol,
                timestamp=timestamp,
                signal_type=signal_type,
                direction=direction,
                probability=probability,
                expected_return=expected_return,
                confidence_interval=(0.0, 0.1),
                score=score,
                model_id=self.config.model_id,
                model_version=self.config.version,
                metadata={"separation_pct": separation_pct},
            )

        if bearish_cross and separation_pct >= self.min_separation and trend_confirmed:
            if self.exit_on_cross:
                signal_type = SignalType.EXIT
                direction = Direction.NEUTRAL
            else:
                signal_type = SignalType.ENTRY
                direction = Direction.SHORT

            if self.min_separation > 0:
                score = min(separation_pct / self.min_separation, 1.0)
            else:
                score = 1.0

            probability = 0.5 + (score * 0.4)  # 0.5-0.9 range

            return Signal(
                symbol=symbol,
                timestamp=timestamp,
                signal_type=signal_type,
                direction=direction,
                probability=probability,
                expected_return=-0.05,
                confidence_interval=(-0.1, 0.0),
                score=-score,
                model_id=self.config.model_id,
                model_version=self.config.version,
                metadata={"separation_pct": separation_pct},
            )

        return None
