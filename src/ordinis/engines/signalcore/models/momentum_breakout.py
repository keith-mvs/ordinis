"""
Momentum Breakout Strategy.

Buys breakouts above recent highs, not dips.
This is the OPPOSITE of mean reversion - it follows trends instead of fading them.

Strategy Logic:
- BUY: Price breaks above recent high with volume confirmation
- SELL: Price breaks below recent low with volume confirmation
- Exit on opposite signal or trailing stop

This should perform well in trending markets (like the Nov-Dec 2024 period
that destroyed the mean-reversion strategy).
"""

from datetime import datetime, timedelta

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


class MomentumBreakoutModel(Model):
    """
    Momentum Breakout Strategy.

    Entry Conditions (LONG):
        - Price breaks above N-period high
        - Volume > volume_mult * avg_volume
        - Optional: Price > SMA (trend confirmation)

    Entry Conditions (SHORT):
        - Price breaks below N-period low
        - Volume > volume_mult * avg_volume
        - Optional: Price < SMA (trend confirmation)

    Exit Conditions:
        - Opposite signal triggers
        - Trailing stop hit (handled by execution engine)

    Parameters:
        breakout_period: Lookback for high/low (default 20)
        volume_period: Volume SMA period (default 20)
        volume_mult: Volume multiplier threshold (default 1.5)
        trend_filter_period: SMA period for trend, 0 to disable (default 50)
        require_trend: Only enter in direction of trend (default True)
        enable_shorts: Allow short entries (default True)
        enable_longs: Allow long entries (default True)
    """

    def __init__(self, config: ModelConfig):
        """Initialize Momentum Breakout model."""
        super().__init__(config)

        params = self.config.parameters
        self.breakout_period = params.get("breakout_period", 20)
        self.volume_period = params.get("volume_period", 20)
        self.volume_mult = params.get("volume_mult", 1.5)
        self.trend_filter_period = params.get("trend_filter_period", 50)
        self.require_trend = params.get("require_trend", True)
        self.enable_shorts = params.get("enable_shorts", True)
        self.enable_longs = params.get("enable_longs", True)

        self._in_long = False
        self._in_short = False
        self._entry_price = None

        max_period = max(self.breakout_period, self.volume_period, self.trend_filter_period) + 10
        self.config.min_data_points = max(self.config.min_data_points, max_period)

    def validate(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate data."""
        if len(data) < self.config.min_data_points:
            return False, f"Insufficient data: {len(data)} < {self.config.min_data_points}"

        required_cols = {"high", "low", "close", "volume"}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            return False, f"Missing columns: {missing}"

        return True, ""

    async def generate(self, symbol: str, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """Generate breakout signal."""
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        current_price = close.iloc[-1]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]

        # Calculate breakout levels (exclude current bar)
        recent_high = high.iloc[-(self.breakout_period + 1) : -1].max()
        recent_low = low.iloc[-(self.breakout_period + 1) : -1].min()

        # Volume confirmation
        volume_sma = TechnicalIndicators.sma(volume, self.volume_period)
        avg_volume = volume_sma.iloc[-1]
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        volume_confirmed = volume_ratio >= self.volume_mult

        # Trend filter
        trend_bullish = True
        trend_bearish = True
        current_trend_sma = None
        if self.trend_filter_period > 0:
            trend_sma = TechnicalIndicators.sma(close, self.trend_filter_period)
            current_trend_sma = trend_sma.iloc[-1]
            trend_bullish = current_price > current_trend_sma
            trend_bearish = current_price < current_trend_sma

        # Signal logic
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        # Breakout conditions
        breaks_high = current_high > recent_high
        breaks_low = current_low < recent_low

        # LONG ENTRY - Breakout above recent high
        if self.enable_longs and not self._in_long and not self._in_short:
            trend_ok = trend_bullish if self.require_trend else True
            if breaks_high and volume_confirmed and trend_ok:
                signal_type = SignalType.ENTRY
                direction = Direction.LONG

                # Score based on breakout strength
                breakout_strength = (current_high - recent_high) / recent_high
                volume_strength = min((volume_ratio - 1.0) / 1.0, 1.0)
                score = min((breakout_strength * 100 * 0.5) + (volume_strength * 0.5), 1.0)

                probability = 0.55 + (score * 0.15)
                expected_return = 0.02 + (score * 0.03)

                self._in_long = True
                self._entry_price = current_price

        # SHORT ENTRY - Breakout below recent low
        if (
            self.enable_shorts
            and signal_type == SignalType.HOLD
            and not self._in_short
            and not self._in_long
        ):
            trend_ok = trend_bearish if self.require_trend else True
            if breaks_low and volume_confirmed and trend_ok:
                signal_type = SignalType.ENTRY
                direction = Direction.SHORT

                breakout_strength = (recent_low - current_low) / recent_low
                volume_strength = min((volume_ratio - 1.0) / 1.0, 1.0)
                score = -min((breakout_strength * 100 * 0.5) + (volume_strength * 0.5), 1.0)

                probability = 0.55 + (abs(score) * 0.15)
                expected_return = -0.02 - (abs(score) * 0.03)

                self._in_short = True
                self._entry_price = current_price

        # EXIT LONG - If in long and price breaks below entry or reverses
        if signal_type == SignalType.HOLD and self._in_long:
            # Exit if trend reverses (price falls below SMA)
            if self.trend_filter_period > 0 and not trend_bullish:
                signal_type = SignalType.EXIT
                direction = Direction.NEUTRAL
                self._in_long = False
                self._entry_price = None

        # EXIT SHORT - If in short and price breaks above entry or reverses
        if signal_type == SignalType.HOLD and self._in_short:
            if self.trend_filter_period > 0 and not trend_bearish:
                signal_type = SignalType.EXIT
                direction = Direction.NEUTRAL
                self._in_short = False
                self._entry_price = None

        # Feature contributions
        returns = close.pct_change().dropna()
        recent_vol = returns.tail(20).std() if len(returns) >= 20 else 0.02
        confidence_interval = (expected_return - 2 * recent_vol, expected_return + 2 * recent_vol)

        feature_contributions = {
            "price": float(current_price),
            "recent_high": float(recent_high),
            "recent_low": float(recent_low),
            "breaks_high": float(breaks_high),
            "breaks_low": float(breaks_low),
            "volume_ratio": float(volume_ratio),
            "volume_confirmed": float(volume_confirmed),
            "trend_bullish": float(trend_bullish),
            "trend_bearish": float(trend_bearish),
            "in_long": float(self._in_long),
            "in_short": float(self._in_short),
        }

        data_quality = 1.0 - (close.tail(20).isnull().sum() / 20)

        if isinstance(data.index, pd.DatetimeIndex):
            delta = timestamp - data.index[-1]
            staleness = timedelta(seconds=abs(delta.total_seconds()))
        else:
            staleness = timedelta(seconds=0)

        regime = "bullish" if trend_bullish else ("bearish" if trend_bearish else "neutral")

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
                "breakout_period": self.breakout_period,
                "volume_mult": self.volume_mult,
                "current_price": float(current_price),
            },
        )

    def reset_state(self):
        """Reset position tracking."""
        self._in_long = False
        self._in_short = False
        self._entry_price = None
