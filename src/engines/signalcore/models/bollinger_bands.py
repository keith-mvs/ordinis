"""
Bollinger Bands Model.

Volatility-based trading strategy using Bollinger Bands for mean reversion.
Buy on lower band touches, sell on upper band touches.
"""

from datetime import datetime, timedelta

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


class BollingerBandsModel(Model):
    """
    Bollinger Bands trading model.

    Generates signals based on price interaction with Bollinger Bands,
    which measure volatility and potential overbought/oversold conditions.

    Parameters:
        bb_period: Bollinger Bands period (default 20)
        bb_std: Number of standard deviations (default 2.0)
        min_band_width: Minimum band width for signal generation (default 0.01)

    Signals:
        - ENTRY/LONG when price touches/crosses below lower band
        - EXIT when price reaches middle or upper band
        - Score based on band width (volatility) and distance from bands
    """

    def __init__(self, config: ModelConfig):
        """Initialize Bollinger Bands model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.bb_period = params.get("bb_period", 20)
        self.bb_std = params.get("bb_std", 2.0)
        self.min_band_width = params.get("min_band_width", 0.01)

        # Update min data points
        self.config.min_data_points = max(self.config.min_data_points, self.bb_period + 30)

    def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:  # noqa: PLR0912, PLR0915
        """
        Generate trading signal from Bollinger Bands analysis.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with Bollinger Bands prediction
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

        # Calculate Bollinger Bands
        middle, upper, lower = TechnicalIndicators.bollinger_bands(
            close, self.bb_period, self.bb_std
        )

        # Get current values
        current_price = close.iloc[-1]
        prev_price = close.iloc[-2] if len(close) > 1 else current_price

        lower_val = lower.iloc[-1]
        middle_val = middle.iloc[-1]
        upper_val = upper.iloc[-1]

        prev_lower = lower.iloc[-2] if len(lower) > 1 else lower_val
        prev_upper = upper.iloc[-2] if len(upper) > 1 else upper_val

        # Calculate band width (volatility measure)
        band_width = (upper_val - lower_val) / middle_val if middle_val != 0 else 0.0

        # Calculate position within bands (0 = lower band, 1 = upper band)
        band_range = upper_val - lower_val
        if band_range > 0:
            bb_position = (current_price - lower_val) / band_range
        else:
            bb_position = 0.5

        # Detect band touches and crosses
        touching_lower = current_price <= lower_val * 1.001  # Within 0.1% of lower band
        crossed_below_lower = prev_price > prev_lower and current_price <= lower_val
        touched_or_crossed_lower = touching_lower or crossed_below_lower

        touching_upper = current_price >= upper_val * 0.999  # Within 0.1% of upper band
        crossed_above_upper = prev_price < prev_upper and current_price >= upper_val
        touched_or_crossed_upper = touching_upper or crossed_above_upper

        # Determine signal
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        # Only generate signals if band width is sufficient (avoid low volatility periods)
        if band_width < self.min_band_width:
            # Low volatility - hold
            signal_type = SignalType.HOLD
            direction = Direction.NEUTRAL
            score = 0.0
            probability = 0.5
        # Buy signal: Price at/below lower band
        elif touched_or_crossed_lower:
            signal_type = SignalType.ENTRY
            direction = Direction.LONG

            # Score based on:
            # 1. How far below lower band (stronger signal)
            # 2. Band width (higher volatility = stronger signal)
            distance_below = max(0, lower_val - current_price) / lower_val
            volatility_factor = min(band_width / 0.04, 1.0)  # Normalize to 0.04 = high vol

            score = min((distance_below * 5 + volatility_factor) / 2, 1.0)
            probability = 0.60 + (score * 0.15)  # 0.60-0.75
            expected_return = 0.03 + (score * 0.05)  # 3-8% expected return

        # Sell signal: Price at/above upper band
        elif touched_or_crossed_upper:
            signal_type = SignalType.EXIT
            direction = Direction.NEUTRAL

            # Score based on distance above upper band
            distance_above = max(0, current_price - upper_val) / upper_val
            volatility_factor = min(band_width / 0.04, 1.0)

            score = -min((distance_above * 5 + volatility_factor) / 2, 1.0)
            probability = 0.60 + (abs(score) * 0.15)

        # Price near middle band - potential hold or weak signal
        elif abs(bb_position - 0.5) < 0.2:  # Within 20% of middle
            signal_type = SignalType.HOLD
            direction = Direction.NEUTRAL
            score = 0.0
            probability = 0.5

        # Price in lower half but not at band
        elif bb_position < 0.5:
            signal_type = SignalType.HOLD
            direction = Direction.LONG
            score = (0.5 - bb_position) * 0.5  # Weak positive score
            probability = 0.52

        # Price in upper half but not at band
        else:
            signal_type = SignalType.HOLD
            direction = Direction.SHORT
            score = -(bb_position - 0.5) * 0.5  # Weak negative score
            probability = 0.52

        # Calculate confidence interval
        returns = close.pct_change().dropna()
        recent_vol = returns.tail(20).std()
        confidence_interval = (
            expected_return - 2 * recent_vol,
            expected_return + 2 * recent_vol,
        )

        # Feature contributions for explainability
        feature_contributions = {
            "bb_position": float(bb_position),
            "band_width": float(band_width),
            "distance_to_lower": float((current_price - lower_val) / lower_val),
            "distance_to_upper": float((upper_val - current_price) / upper_val),
            "touched_lower": float(touched_or_crossed_lower),
            "touched_upper": float(touched_or_crossed_upper),
            "volatility": float(band_width),
        }

        # Regime detection based on band width
        if band_width > 0.04:
            regime = "high_volatility"
        elif band_width > 0.02:
            regime = "moderate_volatility"
        else:
            regime = "low_volatility"

        # Data quality
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
            regime=regime,
            data_quality=data_quality,
            staleness=staleness,
            metadata={
                "bb_period": self.bb_period,
                "bb_std": self.bb_std,
                "current_price": float(current_price),
                "lower_band": float(lower_val),
                "middle_band": float(middle_val),
                "upper_band": float(upper_val),
            },
        )
