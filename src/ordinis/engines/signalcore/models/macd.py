"""
MACD Model.

Momentum and trend identification using Moving Average Convergence Divergence (MACD).
Bullish crossovers generate buy signals, bearish crossovers generate sell signals.
"""

from datetime import datetime, timedelta

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


class MACDModel(Model):
    """
    MACD (Moving Average Convergence Divergence) trading model.

    Generates signals based on MACD line crossovers with the signal line,
    which indicates momentum shifts and potential trend changes.

    Parameters:
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line period (default 9)
        min_histogram: Minimum histogram value for signal (default 0.0)

    Signals:
        - ENTRY/LONG when MACD crosses above signal line (bullish crossover)
        - EXIT when MACD crosses below signal line (bearish crossover)
        - HOLD when maintaining trend direction
        - Score based on histogram magnitude (momentum strength)
    """

    def __init__(self, config: ModelConfig):
        """Initialize MACD model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.fast_period = params.get("fast_period", 12)
        self.slow_period = params.get("slow_period", 26)
        self.signal_period = params.get("signal_period", 9)
        self.min_histogram = params.get("min_histogram", 0.0)

        # Update min data points
        self.config.min_data_points = self.slow_period + self.signal_period + 20

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate trading signal from MACD analysis.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with MACD momentum prediction
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

        # Get current values
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]

        # Get previous values for crossover detection
        prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else current_macd
        prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else current_signal
        prev_hist = histogram.iloc[-2] if len(histogram) > 1 else current_hist

        # Detect crossovers
        bullish_cross = prev_macd <= prev_signal and current_macd > current_signal
        bearish_cross = prev_macd >= prev_signal and current_macd < current_signal

        # Calculate histogram momentum
        hist_momentum = current_hist - prev_hist
        hist_strength = abs(current_hist)

        # Determine zero line position
        macd_above_zero = current_macd > 0
        macd_below_zero = current_macd < 0

        # Determine signal
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        # Bullish crossover: MACD crosses above signal line
        if bullish_cross and abs(current_hist) >= self.min_histogram:
            signal_type = SignalType.ENTRY
            direction = Direction.LONG

            # Stronger signal when:
            # 1. Crossing above zero line (strong momentum)
            # 2. Large histogram value
            # 3. Increasing histogram momentum
            zero_line_factor = 1.2 if macd_above_zero else 1.0
            momentum_factor = 1.1 if hist_momentum > 0 else 1.0

            # Normalize histogram strength
            hist_score = min(hist_strength * 100, 1.0)
            score = min(hist_score * zero_line_factor * momentum_factor, 1.0)

            probability = 0.60 + (score * 0.20)  # 0.60-0.80
            expected_return = 0.04 + (score * 0.06)  # 4-10% expected return

        # Bearish crossover: MACD crosses below signal line
        elif bearish_cross:
            signal_type = SignalType.EXIT
            direction = Direction.NEUTRAL

            # Stronger exit signal when crossing below zero
            zero_line_factor = 1.2 if macd_below_zero else 1.0
            momentum_factor = 1.1 if hist_momentum < 0 else 1.0

            hist_score = min(hist_strength * 100, 1.0)
            score = -min(hist_score * zero_line_factor * momentum_factor, 1.0)

            probability = 0.60 + (abs(score) * 0.20)

        # Hold with bullish bias: MACD above signal line
        elif current_macd > current_signal:
            signal_type = SignalType.HOLD
            direction = Direction.LONG

            # Score based on histogram strength and momentum
            hist_score = min(hist_strength * 100, 0.8)
            momentum_boost = 0.1 if hist_momentum > 0 else 0.0
            score = hist_score + momentum_boost
            probability = 0.52 + (score * 0.05)

        # Hold with bearish bias: MACD below signal line
        else:
            signal_type = SignalType.HOLD
            direction = Direction.SHORT

            hist_score = min(hist_strength * 100, 0.8)
            momentum_boost = 0.1 if hist_momentum < 0 else 0.0
            score = -(hist_score + momentum_boost)
            probability = 0.52 + (abs(score) * 0.05)

        # Calculate confidence interval
        returns = close.pct_change().dropna()
        recent_vol = returns.tail(20).std()
        confidence_interval = (
            expected_return - 2 * recent_vol,
            expected_return + 2 * recent_vol,
        )

        # Feature contributions for explainability
        feature_contributions = {
            "macd_line": float(current_macd),
            "signal_line": float(current_signal),
            "histogram": float(current_hist),
            "histogram_momentum": float(hist_momentum),
            "bullish_cross": float(bullish_cross),
            "bearish_cross": float(bearish_cross),
            "macd_above_zero": float(macd_above_zero),
            "histogram_strength": float(hist_strength),
        }

        # Regime detection based on MACD behavior
        # Calculate histogram volatility over recent period
        hist_volatility = histogram.tail(20).std()

        if hist_volatility > 0.015:
            regime = "trending"
        elif hist_volatility > 0.005:
            regime = "ranging"
        else:
            regime = "consolidating"

        # Data quality
        recent_close = close.tail(20)
        data_quality = 1.0 - (recent_close.isnull().sum() / len(recent_close))

        # Staleness
        if isinstance(data.index, pd.DatetimeIndex):
            delta = timestamp - data.index[-1]
            staleness = timedelta(seconds=delta.total_seconds())
        else:
            staleness = timedelta(seconds=0)

        # Determine crossover type
        if bullish_cross:
            crossover_type = "bullish"
        elif bearish_cross:
            crossover_type = "bearish"
        else:
            crossover_type = "none"

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
                "current_price": float(close.iloc[-1]),
                "macd_line": float(current_macd),
                "signal_line": float(current_signal),
                "histogram": float(current_hist),
                "crossover_type": crossover_type,
            },
        )
