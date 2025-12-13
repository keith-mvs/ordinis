"""
Simple Moving Average Crossover Model.

Classic technical strategy: Buy when fast SMA crosses above slow SMA,
sell when fast SMA crosses below slow SMA.
"""

from datetime import datetime, timedelta

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
        self.min_separation = params.get("min_separation", 0.01)
        self.exit_on_cross = params.get("exit_on_cross", True)

        # Update min data points based on slow period
        self.config.min_data_points = max(self.config.min_data_points, self.slow_period + 10)

    def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate trading signal from SMA crossover.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with crossover prediction
        """
        # Validate data
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        symbol = data.get("symbol", ["UNKNOWN"])[0] if "symbol" in data else "UNKNOWN"
        close = data["close"]

        # Calculate SMAs
        fast_sma = TechnicalIndicators.sma(close, self.fast_period)
        slow_sma = TechnicalIndicators.sma(close, self.slow_period)

        # Get current and previous values
        current_fast = fast_sma.iloc[-1]
        current_slow = slow_sma.iloc[-1]
        prev_fast = fast_sma.iloc[-2]
        prev_slow = slow_sma.iloc[-2]

        # Calculate separation (as percentage of price)
        current_price = close.iloc[-1]
        separation_pct = abs(current_fast - current_slow) / current_price

        # Detect crossover
        bullish_cross = prev_fast <= prev_slow and current_fast > current_slow
        bearish_cross = prev_fast >= prev_slow and current_fast < current_slow

        # Determine signal type and direction
        if bullish_cross and separation_pct >= self.min_separation:
            signal_type = SignalType.ENTRY
            direction = Direction.LONG
            score = min(separation_pct / self.min_separation, 1.0)
            probability = 0.5 + (score * 0.2)  # 0.5-0.7 range
            expected_return = 0.05  # Modest expectation
        elif bearish_cross and separation_pct >= self.min_separation:
            if self.exit_on_cross:
                signal_type = SignalType.EXIT
                direction = Direction.NEUTRAL
            else:
                signal_type = SignalType.ENTRY
                direction = Direction.SHORT
            score = -min(separation_pct / self.min_separation, 1.0)
            probability = 0.5 + (abs(score) * 0.2)
            expected_return = -0.05 if signal_type == SignalType.ENTRY else 0.0
        else:
            # No crossover or insufficient separation
            signal_type = SignalType.HOLD
            direction = Direction.NEUTRAL
            score = 0.0
            probability = 0.5
            expected_return = 0.0

        # Calculate confidence interval based on recent volatility
        returns = close.pct_change().dropna()
        recent_vol = returns.tail(20).std()
        confidence_interval = (
            expected_return - 2 * recent_vol,
            expected_return + 2 * recent_vol,
        )

        # Feature contributions for explainability
        feature_contributions = {
            "fast_sma": float(current_fast),
            "slow_sma": float(current_slow),
            "separation_pct": float(separation_pct),
            "bullish_cross": float(bullish_cross),
            "bearish_cross": float(bearish_cross),
        }

        # Data quality check (based on recent data consistency)
        recent_close = close.tail(20)
        data_quality = 1.0 - (recent_close.isnull().sum() / len(recent_close))

        # Staleness
        if isinstance(data.index, pd.DatetimeIndex):
            staleness = timestamp - data.index[-1]
        else:
            staleness = timedelta(seconds=0)

        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=signal_type,
            direction=direction,
            probability=probability,
            expected_return=expected_return,
            confidence_interval=confidence_interval,
            score=score,
            model_id=self.config.model_id,
            model_version=self.config.version,
            feature_contributions=feature_contributions,
            regime="trend" if abs(score) > 0.5 else "ranging",
            data_quality=data_quality,
            staleness=staleness,
            metadata={
                "fast_period": self.fast_period,
                "slow_period": self.slow_period,
                "current_price": float(current_price),
            },
        )
