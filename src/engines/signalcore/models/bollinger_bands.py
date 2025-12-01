"""
Bollinger Bands Model.

Mean reversion strategy based on Bollinger Bands.
Buy when price touches lower band (oversold), sell when price touches upper band (overbought).
"""

from datetime import datetime, timedelta

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


class BollingerBandsModel(Model):
    """
    Bollinger Bands trading model.

    Generates signals based on price position relative to Bollinger Bands.

    Parameters:
        bb_period: Bollinger Bands calculation period (default 20)
        bb_std: Number of standard deviations for bands (default 2.0)
        min_band_width: Minimum band width to avoid low volatility signals (default 0.02)

    Signals:
        - ENTRY/LONG when price touches or crosses below lower band (oversold)
        - EXIT when price touches or crosses above upper band (overbought)
        - Stronger signals when price is further outside bands
    """

    def __init__(self, config: ModelConfig):
        """Initialize Bollinger Bands model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.bb_period = params.get("bb_period", 20)
        self.bb_std = params.get("bb_std", 2.0)
        self.min_band_width = params.get("min_band_width", 0.02)

        # Update min data points
        self.config.min_data_points = max(self.config.min_data_points, self.bb_period + 20)

    def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate trading signal from Bollinger Bands.

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
        current_close = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_middle = middle.iloc[-1]

        # Calculate band width (as percentage of middle band)
        band_width = (current_upper - current_lower) / current_middle if current_middle != 0 else 0

        # Calculate %B (position within bands: 0 = lower, 1 = upper)
        band_range = current_upper - current_lower
        percent_b = (current_close - current_lower) / band_range if band_range != 0 else 0.5

        # Get previous values for crossover detection
        prev_close = close.iloc[-2] if len(close) > 1 else current_close
        prev_upper = upper.iloc[-2] if len(upper) > 1 else current_upper
        prev_lower = lower.iloc[-2] if len(lower) > 1 else current_lower

        # Detect crossovers
        crossing_below_lower = prev_close >= prev_lower and current_close < current_lower
        crossing_above_upper = prev_close <= prev_upper and current_close > current_upper
        touching_lower = current_close <= current_lower
        touching_upper = current_close >= current_upper

        # Determine signal
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        # Check for low volatility (skip signals)
        low_volatility = band_width < self.min_band_width

        if not low_volatility:
            # Buy signal: Price at or below lower band (oversold)
            if touching_lower or crossing_below_lower:
                signal_type = SignalType.ENTRY
                direction = Direction.LONG

                # Calculate signal strength based on how far below lower band
                if current_close < current_lower:
                    # Below lower band - strong signal
                    overshoot = (current_lower - current_close) / current_lower
                    score = min(0.5 + overshoot * 10, 1.0)
                    probability = 0.65 + min(overshoot * 5, 0.15)  # 0.65-0.80
                    expected_return = 0.05 + min(overshoot * 3, 0.03)  # 0.05-0.08
                else:
                    # At lower band
                    score = 0.5
                    probability = 0.60
                    expected_return = 0.03

            # Sell/Exit signal: Price at or above upper band (overbought)
            elif touching_upper or crossing_above_upper:
                signal_type = SignalType.EXIT
                direction = Direction.NEUTRAL

                # Calculate signal strength
                if current_close > current_upper:
                    overshoot = (current_close - current_upper) / current_upper
                    score = -min(0.5 + overshoot * 10, 1.0)
                    probability = 0.65 + min(overshoot * 5, 0.15)
                else:
                    score = -0.5
                    probability = 0.60

            # Hold in neutral zone
            else:
                signal_type = SignalType.HOLD
                direction = Direction.NEUTRAL
                # Score reflects position within bands (0.5 = middle)
                score = (0.5 - percent_b) * 0.6  # Map to -0.3 to 0.3
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
            "percent_b": float(percent_b),
            "band_width": float(band_width),
            "price_vs_middle": float((current_close - current_middle) / current_middle)
            if current_middle != 0
            else 0.0,
            "touching_lower": float(touching_lower),
            "touching_upper": float(touching_upper),
            "crossing_lower": float(crossing_below_lower),
            "crossing_upper": float(crossing_above_upper),
            "low_volatility": float(low_volatility),
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
        if band_width < 0.03:
            regime = "low_volatility"
        elif band_width > 0.08:
            regime = "high_volatility"
        else:
            regime = "moderate_volatility"

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
                "current_price": float(current_close),
                "upper_band": float(current_upper),
                "middle_band": float(current_middle),
                "lower_band": float(current_lower),
            },
        )
