"""
MACD Model.

Momentum strategy based on Moving Average Convergence Divergence.
Buy on bullish crossovers, sell on bearish crossovers.
"""

from datetime import datetime, timedelta

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


class MACDModel(Model):
    """
    MACD trading model.

    Generates signals based on MACD line and signal line crossovers.

    Parameters:
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)

    Signals:
        - ENTRY/LONG on bullish crossover (MACD crosses above signal line)
        - EXIT on bearish crossover (MACD crosses below signal line)
        - Signal strength based on histogram magnitude and zero line position
    """

    def __init__(self, config: ModelConfig):
        """Initialize MACD model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.fast_period = params.get("fast_period", 12)
        self.slow_period = params.get("slow_period", 26)
        self.signal_period = params.get("signal_period", 9)

        # Update min data points (need enough for slow EMA + signal line)
        self.config.min_data_points = max(
            self.config.min_data_points, self.slow_period + self.signal_period + 20
        )

    def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate trading signal from MACD.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with momentum prediction
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

        # Calculate MACD
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            close, self.fast_period, self.slow_period, self.signal_period
        )

        # Get current and previous values
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]

        prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else current_macd
        prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else current_signal
        prev_histogram = histogram.iloc[-2] if len(histogram) > 1 else current_histogram

        # Detect crossovers
        bullish_crossover = prev_macd <= prev_signal and current_macd > current_signal
        bearish_crossover = prev_macd >= prev_signal and current_macd < current_signal

        # Check position relative to zero line
        macd_above_zero = current_macd > 0
        signal_above_zero = current_signal > 0

        # Histogram momentum (increasing or decreasing)
        histogram_increasing = current_histogram > prev_histogram
        histogram_decreasing = current_histogram < prev_histogram

        # Calculate histogram strength (normalized by price)
        current_price = close.iloc[-1]
        histogram_strength = abs(current_histogram) / current_price if current_price != 0 else 0

        # Determine signal
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        # Bullish crossover - buy signal
        if bullish_crossover:
            signal_type = SignalType.ENTRY
            direction = Direction.LONG

            # Stronger signal if crossover happens above zero line
            if macd_above_zero:
                # Strong bullish - MACD above zero with bullish crossover
                score = 0.7 + min(histogram_strength * 10, 0.3)
                probability = 0.70 + min(histogram_strength * 5, 0.10)  # 0.70-0.80
                expected_return = 0.06 + min(histogram_strength * 2, 0.04)  # 0.06-0.10
            else:
                # Moderate bullish - crossover below zero (potential trend reversal)
                score = 0.5 + min(histogram_strength * 10, 0.2)
                probability = 0.60 + min(histogram_strength * 5, 0.10)  # 0.60-0.70
                expected_return = 0.04 + min(histogram_strength * 2, 0.02)  # 0.04-0.06

        # Bearish crossover - exit signal
        elif bearish_crossover:
            signal_type = SignalType.EXIT
            direction = Direction.NEUTRAL

            if not macd_above_zero:
                # Strong bearish - MACD below zero with bearish crossover
                score = -0.7 - min(histogram_strength * 10, 0.3)
                probability = 0.70 + min(histogram_strength * 5, 0.10)
            else:
                # Moderate bearish - crossover above zero (potential momentum loss)
                score = -0.5 - min(histogram_strength * 10, 0.2)
                probability = 0.60 + min(histogram_strength * 5, 0.10)

        # No crossover - hold with momentum indication
        else:
            signal_type = SignalType.HOLD
            direction = Direction.NEUTRAL

            # Score based on histogram direction and MACD position
            if current_macd > current_signal:
                # Bullish positioning
                base_score = 0.2 if macd_above_zero else 0.1
                score = base_score + (0.1 if histogram_increasing else -0.05)
            else:
                # Bearish positioning
                base_score = -0.2 if not macd_above_zero else -0.1
                score = base_score + (-0.1 if histogram_decreasing else 0.05)

            score = max(min(score, 0.3), -0.3)  # Cap at ±0.3 for hold
            probability = 0.5

        # Calculate confidence interval
        returns = close.pct_change().dropna()
        recent_vol = returns.tail(20).std()
        confidence_interval = (
            expected_return - 2 * recent_vol,
            expected_return + 2 * recent_vol,
        )

        # Feature contributions
        feature_contributions = {
            "macd": float(current_macd),
            "signal": float(current_signal),
            "histogram": float(current_histogram),
            "histogram_strength": float(histogram_strength),
            "macd_above_zero": float(macd_above_zero),
            "bullish_crossover": float(bullish_crossover),
            "bearish_crossover": float(bearish_crossover),
            "histogram_increasing": float(histogram_increasing),
        }

        # Data quality
        recent_close = close.tail(20)
        data_quality = 1.0 - (recent_close.isnull().sum() / len(recent_close))

        # Staleness
        if isinstance(data.index, pd.DatetimeIndex):
            staleness = timestamp - data.index[-1]
        else:
            staleness = timedelta(seconds=0)

        # Regime detection based on MACD characteristics
        if bullish_crossover or bearish_crossover:
            regime = "crossover"
        elif abs(current_histogram) < histogram.tail(20).std() * 0.5:
            regime = "consolidating"
        elif histogram_increasing and current_macd > current_signal:
            regime = "trending_up"
        elif histogram_decreasing and current_macd < current_signal:
            regime = "trending_down"
        else:
            regime = "ranging"

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
            regime=regime,
            data_quality=data_quality,
            staleness=staleness,
            metadata={
                "fast_period": self.fast_period,
                "slow_period": self.slow_period,
                "signal_period": self.signal_period,
                "current_price": float(current_price),
            },
        )
