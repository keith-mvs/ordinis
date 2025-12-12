"""
ADX Trend Filter Model.

Uses Average Directional Index (ADX) to filter trades based on trend strength.
Only generates signals when a strong trend is present (ADX > threshold).
"""

from datetime import datetime, timedelta

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType


class ADXTrendModel(Model):
    """
    ADX Trend Strength Model.

    Filters trading signals based on trend strength measured by ADX.
    The model uses +DI and -DI to determine trend direction, and ADX
    to measure trend strength.

    Parameters:
        adx_period: ADX calculation period (default 14)
        adx_threshold: Minimum ADX for strong trend (default 25)
        strong_trend: ADX threshold for very strong trend (default 40)
        di_threshold: Minimum +DI/-DI difference for direction (default 5)

    Signals:
        - ENTRY/LONG when ADX > threshold and +DI > -DI
        - ENTRY/SHORT when ADX > threshold and -DI > +DI
        - HOLD when ADX indicates weak trend
        - Score based on ADX strength and DI divergence

    Usage:
        Use as a filter: Only trade when ADX confirms strong trend.
        Combine with other indicators (RSI, MACD) for entry timing.

    Reference:
        Wilder, J. Welles (1978). New Concepts in Technical Trading Systems
    """

    def __init__(self, config: ModelConfig):
        """Initialize ADX model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.adx_period = params.get("adx_period", 14)
        self.adx_threshold = params.get("adx_threshold", 25)
        self.strong_trend = params.get("strong_trend", 40)
        self.di_threshold = params.get("di_threshold", 5)

        # Update min data points (ADX needs period + smoothing)
        self.config.min_data_points = (self.adx_period * 2) + 30

    def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:  # noqa: PLR0912, PLR0915
        """
        Generate trading signal from ADX analysis.

        Args:
            data: Historical OHLCV data (must include high, low, close)
            timestamp: Current timestamp

        Returns:
            Signal with ADX trend strength prediction
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

        # Calculate ADX using inline calculation to avoid circular imports
        # (TrendIndicators would be ideal but causes import issues)
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate +DM and -DM
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)

        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        # Smooth DM and TR
        plus_dm_smooth = plus_dm.rolling(window=self.adx_period).sum()
        minus_dm_smooth = minus_dm.rolling(window=self.adx_period).sum()
        tr_smooth = tr.rolling(window=self.adx_period).sum()

        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = dx.fillna(0)
        adx = dx.rolling(window=self.adx_period).mean()

        # Get current values
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        current_price = close.iloc[-1]

        # Get previous values for trend detection
        prev_adx = adx.iloc[-2] if len(adx) > 1 else current_adx
        prev_plus_di = plus_di.iloc[-2] if len(plus_di) > 1 else current_plus_di
        prev_minus_di = minus_di.iloc[-2] if len(minus_di) > 1 else current_minus_di

        # Calculate DI difference for trend direction strength
        di_diff = abs(current_plus_di - current_minus_di)

        # Determine trend strength
        if current_adx < self.adx_threshold:
            trend_strength = "weak"
        elif current_adx < self.strong_trend:
            trend_strength = "moderate"
        elif current_adx < 60:
            trend_strength = "strong"
        else:
            trend_strength = "very_strong"

        # Determine signal type and direction
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        # Only generate actionable signals when trend is strong enough
        if current_adx >= self.adx_threshold and di_diff >= self.di_threshold:
            if current_plus_di > current_minus_di:
                # Bullish trend
                signal_type = SignalType.ENTRY
                direction = Direction.LONG

                # Score based on ADX strength and DI divergence
                adx_strength = min((current_adx - self.adx_threshold) / 50, 1.0)
                di_strength = min(di_diff / 30, 1.0)
                score = (adx_strength * 0.6) + (di_strength * 0.4)

                probability = 0.55 + (score * 0.20)  # 0.55-0.75
                expected_return = 0.03 + (score * 0.05)  # 3-8%

            elif current_minus_di > current_plus_di:
                # Bearish trend
                signal_type = SignalType.ENTRY
                direction = Direction.SHORT

                # Score based on ADX strength and DI divergence
                adx_strength = min((current_adx - self.adx_threshold) / 50, 1.0)
                di_strength = min(di_diff / 30, 1.0)
                score = -((adx_strength * 0.6) + (di_strength * 0.4))

                probability = 0.55 + (abs(score) * 0.20)  # 0.55-0.75
                expected_return = 0.0  # Short positions tracked differently

        # Check for trend weakening (potential exit)
        elif current_adx < self.adx_threshold and prev_adx >= self.adx_threshold:
            signal_type = SignalType.EXIT
            direction = Direction.NEUTRAL
            score = -0.3
            probability = 0.60

        # Calculate confidence interval
        returns = close.pct_change().dropna()
        recent_vol = returns.tail(20).std()
        confidence_interval = (
            expected_return - 2 * recent_vol,
            expected_return + 2 * recent_vol,
        )

        # Feature contributions for explainability
        feature_contributions = {
            "adx": float(current_adx),
            "plus_di": float(current_plus_di),
            "minus_di": float(current_minus_di),
            "di_difference": float(di_diff),
            "trend_strength": adx_strength if signal_type == SignalType.ENTRY else 0.0,
        }

        # Regime detection based on ADX
        if current_adx > 40:
            regime = "strong_trend"
        elif current_adx > 25:
            regime = "moderate_trend"
        else:
            regime = "weak_trend"

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
                "adx": float(current_adx),
                "plus_di": float(current_plus_di),
                "minus_di": float(current_minus_di),
                "trend_strength": trend_strength,
                "di_difference": float(di_diff),
                "adx_threshold": self.adx_threshold,
            },
        )
