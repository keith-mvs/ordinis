"""
Parabolic SAR Model.

Trend-following system using Parabolic Stop and Reverse for dynamic
support/resistance levels and trend reversals.
"""

from datetime import datetime, timedelta

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType


class ParabolicSARModel(Model):
    """
    Parabolic SAR Trading Model.

    Generates signals based on Parabolic SAR indicator which provides
    dynamic support/resistance levels and trend reversal points.

    Parameters:
        acceleration: Acceleration factor increment (default 0.02)
        maximum: Maximum acceleration factor (default 0.2)
        min_trend_bars: Minimum bars in trend before entry (default 3)

    Signals:
        - ENTRY/LONG when SAR crosses below price (reversal to uptrend)
        - ENTRY/SHORT when SAR crosses above price (reversal to downtrend)
        - EXIT when SAR reverses against position
        - HOLD when maintaining current trend
        - Score based on trend strength and time in trend

    Usage:
        Best in trending markets. Whipsaws in ranging markets.
        Use with volatility filters (ADX, ATR) to avoid false signals.
        SAR acts as trailing stop-loss level.

    Reference:
        Wilder, J. Welles (1978). New Concepts in Technical Trading Systems

    Trading Rules:
        - SAR below price: Bullish trend, SAR is support
        - SAR above price: Bearish trend, SAR is resistance
        - Price crosses SAR: Trend reversal signal
        - SAR accelerates faster with strong trends
    """

    def __init__(self, config: ModelConfig):
        """Initialize Parabolic SAR model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.acceleration = params.get("acceleration", 0.02)
        self.maximum = params.get("maximum", 0.2)
        self.min_trend_bars = params.get("min_trend_bars", 3)

        # Update min data points (need enough for SAR to stabilize)
        self.config.min_data_points = 50

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate trading signal from Parabolic SAR analysis.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with Parabolic SAR trend prediction
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

        # Calculate Parabolic SAR inline
        sar = pd.Series(index=high.index, dtype=float)
        sar.iloc[0] = low.iloc[0]
        trend = 1  # 1 for up, -1 for down
        af = self.acceleration
        ep = high.iloc[0]

        for i in range(1, len(high)):
            prev_sar = sar.iloc[i - 1]
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)

            if trend == 1:  # Uptrend
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i - 1])
                if i > 1:
                    sar.iloc[i] = min(sar.iloc[i], low.iloc[i - 2])

                if low.iloc[i] < sar.iloc[i]:
                    trend = -1
                    sar.iloc[i] = ep
                    ep = low.iloc[i]
                    af = self.acceleration
                elif high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + self.acceleration, self.maximum)
            else:  # Downtrend
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i - 1])
                if i > 1:
                    sar.iloc[i] = max(sar.iloc[i], high.iloc[i - 2])

                if high.iloc[i] > sar.iloc[i]:
                    trend = 1
                    sar.iloc[i] = ep
                    ep = high.iloc[i]
                    af = self.acceleration
                elif low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + self.acceleration, self.maximum)

        # Get current and previous values
        current_sar = sar.iloc[-1]
        current_price = close.iloc[-1]

        # Determine current trend
        if current_price > current_sar:
            current_trend = "bullish"
        else:
            current_trend = "bearish"

        # Check for reversal by looking at previous bars
        reversal_detected = False
        reversal_bar = -1
        trend_bars = 0

        # Look back to find reversal point and count trend bars
        for i in range(len(sar) - 1, 0, -1):
            price = close.iloc[i]
            sar_val = sar.iloc[i]
            prev_price = close.iloc[i - 1]
            prev_sar = sar.iloc[i - 1]

            # Current trend
            trend_now = "bullish" if price > sar_val else "bearish"
            trend_prev = "bullish" if prev_price > prev_sar else "bearish"

            if trend_now == current_trend:
                trend_bars += 1
            else:
                # Found reversal point
                reversal_bar = i
                break

        # Check if reversal just occurred (within last 5 bars for more flexibility)
        # This allows the strategy to catch reversals even if checked infrequently
        if reversal_bar >= len(sar) - 5:
            reversal_detected = True

        # Calculate distance from SAR (as percentage)
        sar_distance = abs(current_price - current_sar) / current_price

        # Generate signal based on SAR analysis
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        if reversal_detected and trend_bars >= self.min_trend_bars:
            # New trend just started with enough confirmation
            signal_type = SignalType.ENTRY

            if current_trend == "bullish":
                direction = Direction.LONG

                # Score based on trend bars and SAR distance
                trend_strength = min(trend_bars / 10, 1.0)  # Cap at 10 bars
                distance_strength = min(sar_distance / 0.05, 1.0)  # Cap at 5%

                score = (trend_strength * 0.6) + (distance_strength * 0.4)
                probability = 0.55 + (score * 0.15)  # 0.55-0.70
                expected_return = 0.03 + (score * 0.05)  # 3-8%

            else:  # bearish
                direction = Direction.SHORT

                trend_strength = min(trend_bars / 10, 1.0)
                distance_strength = min(sar_distance / 0.05, 1.0)

                score = -((trend_strength * 0.6) + (distance_strength * 0.4))
                probability = 0.55 + (abs(score) * 0.15)
                expected_return = 0.0  # Short positions tracked differently

        elif not reversal_detected and trend_bars > 20:
            # Long-running trend might be exhausting
            signal_type = SignalType.HOLD
            direction = Direction.NEUTRAL
            score = 0.0
            probability = 0.5

        elif reversal_detected and trend_bars < self.min_trend_bars:
            # Reversal too recent, wait for confirmation
            signal_type = SignalType.HOLD
            direction = Direction.NEUTRAL
            score = 0.0
            probability = 0.5

        else:
            # Continue holding in existing trend
            signal_type = SignalType.HOLD
            if current_trend == "bullish":
                direction = Direction.LONG
                score = min(trend_bars / 20, 0.5)  # Weak positive
            else:
                direction = Direction.SHORT
                score = -min(trend_bars / 20, 0.5)  # Weak negative
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
            "sar_distance": float(sar_distance),
            "trend_bars": float(trend_bars),
            "reversal_detected": float(reversal_detected),
            "trend_direction": 1.0 if current_trend == "bullish" else -1.0,
        }

        # Regime detection based on trend characteristics
        if trend_bars < 5:
            regime = "new_trend"
        elif trend_bars < 15:
            regime = "established_trend"
        else:
            regime = "mature_trend"

        # Data quality
        recent_close = close.tail(20)
        data_quality = 1.0 - (recent_close.isnull().sum() / len(recent_close))

        # Staleness
        if isinstance(data.index, pd.DatetimeIndex):
            delta = timestamp - data.index[-1]
            staleness = timedelta(seconds=delta.total_seconds())
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
                "current_sar": float(current_sar),
                "current_price": float(current_price),
                "sar_distance": float(sar_distance),
                "trend": current_trend,
                "trend_bars": int(trend_bars),
                "reversal_detected": reversal_detected,
                "reversal_bar": int(reversal_bar) if reversal_bar >= 0 else None,
                "acceleration": self.acceleration,
                "maximum": self.maximum,
            },
        )
