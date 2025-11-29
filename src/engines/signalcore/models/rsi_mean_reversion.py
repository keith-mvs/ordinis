"""
RSI Mean Reversion Model.

Mean reversion strategy based on Relative Strength Index (RSI).
Buy when oversold, sell when overbought.
"""

from datetime import datetime, timedelta

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


class RSIMeanReversionModel(Model):
    """
    RSI Mean Reversion trading model.

    Generates signals based on RSI overbought/oversold conditions.

    Parameters:
        rsi_period: RSI calculation period (default 14)
        oversold_threshold: RSI level considered oversold (default 30)
        overbought_threshold: RSI level considered overbought (default 70)
        extreme_oversold: Extreme oversold level for stronger signals (default 20)
        extreme_overbought: Extreme overbought level for stronger signals (default 80)

    Signals:
        - ENTRY/LONG when RSI crosses above oversold threshold
        - EXIT when RSI crosses above overbought threshold
        - Stronger signals at extreme levels
    """

    def __init__(self, config: ModelConfig):
        """Initialize RSI Mean Reversion model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.rsi_period = params.get("rsi_period", 14)
        self.oversold_threshold = params.get("oversold_threshold", 30)
        self.overbought_threshold = params.get("overbought_threshold", 70)
        self.extreme_oversold = params.get("extreme_oversold", 20)
        self.extreme_overbought = params.get("extreme_overbought", 80)

        # Update min data points
        self.config.min_data_points = max(self.config.min_data_points, self.rsi_period + 20)

    def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:  # noqa: PLR0912, PLR0915
        """
        Generate trading signal from RSI mean reversion.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with mean reversion prediction
        """
        # Validate data
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        symbol = data.get("symbol", ["UNKNOWN"])[0] if "symbol" in data else "UNKNOWN"
        close = data["close"]

        # Calculate RSI
        rsi = TechnicalIndicators.rsi(close, self.rsi_period)

        # Get current and previous RSI
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi

        # Detect condition changes
        entering_oversold = (
            prev_rsi > self.oversold_threshold and current_rsi <= self.oversold_threshold
        )
        exiting_oversold = (
            prev_rsi <= self.oversold_threshold and current_rsi > self.oversold_threshold
        )
        entering_overbought = (
            prev_rsi < self.overbought_threshold and current_rsi >= self.overbought_threshold
        )
        exiting_overbought = (
            prev_rsi >= self.overbought_threshold and current_rsi < self.overbought_threshold
        )

        # Check extreme conditions
        is_extreme_oversold = current_rsi <= self.extreme_oversold
        is_extreme_overbought = current_rsi >= self.extreme_overbought

        # Determine signal
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        # Buy signal: RSI crossing up from oversold
        if exiting_oversold or is_extreme_oversold:
            signal_type = SignalType.ENTRY
            direction = Direction.LONG

            # Stronger signal for extreme oversold
            if is_extreme_oversold:
                score_magnitude = (self.extreme_oversold - current_rsi) / self.extreme_oversold
                score = min(score_magnitude, 1.0)
                probability = 0.65 + (score * 0.15)  # 0.65-0.80
                expected_return = 0.08
            else:
                score_magnitude = (self.oversold_threshold - current_rsi) / self.oversold_threshold
                score = min(score_magnitude * 0.7, 0.7)
                probability = 0.55 + (score * 0.1)  # 0.55-0.65
                expected_return = 0.05

        # Exit signal: RSI crossing up into overbought
        elif entering_overbought or is_extreme_overbought:
            signal_type = SignalType.EXIT
            direction = Direction.NEUTRAL

            if is_extreme_overbought:
                score_magnitude = (current_rsi - self.extreme_overbought) / (
                    100 - self.extreme_overbought
                )
                score = -min(score_magnitude, 1.0)
                probability = 0.65 + (abs(score) * 0.15)
            else:
                score_magnitude = (current_rsi - self.overbought_threshold) / (
                    100 - self.overbought_threshold
                )
                score = -min(score_magnitude * 0.7, 0.7)
                probability = 0.55 + (abs(score) * 0.1)

        # Hold in neutral zone
        else:
            signal_type = SignalType.HOLD
            direction = Direction.NEUTRAL
            # Score reflects distance from neutral (50)
            score = (50 - current_rsi) / 50.0
            score = max(min(score, 0.3), -0.3)  # Cap at ±0.3 for hold signals
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
            "rsi": float(current_rsi),
            "rsi_prev": float(prev_rsi),
            "oversold_threshold": float(self.oversold_threshold),
            "overbought_threshold": float(self.overbought_threshold),
            "entering_oversold": float(entering_oversold),
            "exiting_oversold": float(exiting_oversold),
            "entering_overbought": float(entering_overbought),
            "is_extreme": float(is_extreme_oversold or is_extreme_overbought),
        }

        # Data quality
        recent_close = close.tail(20)
        data_quality = 1.0 - (recent_close.isnull().sum() / len(recent_close))

        # Staleness
        if isinstance(data.index, pd.DatetimeIndex):
            staleness = timestamp - data.index[-1]
        else:
            staleness = timedelta(seconds=0)

        # Regime detection
        if current_rsi <= self.oversold_threshold or current_rsi >= self.overbought_threshold:
            regime = "extreme"
        elif 40 <= current_rsi <= 60:
            regime = "neutral"
        else:
            regime = "trending"

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
                "rsi_period": self.rsi_period,
                "current_price": float(close.iloc[-1]),
            },
        )
