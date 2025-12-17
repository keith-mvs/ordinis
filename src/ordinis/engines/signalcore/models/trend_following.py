"""
Trend Following Strategy.

Rides trends in their direction instead of fighting them.
This strategy should have profited from the Nov-Dec 2024 AMD downtrend
that destroyed the mean-reversion strategy.

Strategy Logic:
- LONG: SMA crossover (fast > slow) with momentum confirmation
- SHORT: SMA crossover (fast < slow) with momentum confirmation
- Stays in trend until crossover reverses

Key difference from mean-reversion:
- Mean reversion: Buy oversold, sell overbought (catches falling knives)
- Trend following: Buy uptrends, short downtrends (rides the wave)
"""

from datetime import datetime, timedelta

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


class TrendFollowingModel(Model):
    """
    Trend Following Strategy using SMA crossovers.

    Entry Conditions (LONG):
        - Fast SMA > Slow SMA (uptrend)
        - Price > Fast SMA (momentum)
        - Optional: Volume confirmation

    Entry Conditions (SHORT):
        - Fast SMA < Slow SMA (downtrend)
        - Price < Fast SMA (momentum)
        - Optional: Volume confirmation

    Exit Conditions:
        - Opposite crossover signal
        - Trailing stop hit (handled by execution engine)

    Parameters:
        fast_period: Fast SMA period (default 10)
        slow_period: Slow SMA period (default 30)
        require_momentum: Price must be in direction of trend (default True)
        volume_filter: Require volume confirmation (default False)
        volume_mult: Volume multiplier if volume_filter (default 1.2)
        enable_shorts: Allow short entries (default True)
        enable_longs: Allow long entries (default True)
    """

    def __init__(self, config: ModelConfig):
        """Initialize Trend Following model."""
        super().__init__(config)

        params = self.config.parameters
        self.fast_period = params.get("fast_period", 10)
        self.slow_period = params.get("slow_period", 30)
        self.require_momentum = params.get("require_momentum", True)
        self.volume_filter = params.get("volume_filter", False)
        self.volume_mult = params.get("volume_mult", 1.2)
        self.enable_shorts = params.get("enable_shorts", True)
        self.enable_longs = params.get("enable_longs", True)

        self._in_long = False
        self._in_short = False
        self._entry_price = None
        self._prev_fast_sma = None
        self._prev_slow_sma = None

        max_period = max(self.fast_period, self.slow_period) + 10
        self.config.min_data_points = max(self.config.min_data_points, max_period)

    def validate(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate data."""
        if len(data) < self.config.min_data_points:
            return False, f"Insufficient data: {len(data)} < {self.config.min_data_points}"

        required_cols = {"close", "volume"}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            return False, f"Missing columns: {missing}"

        return True, ""

    async def generate(self, symbol: str, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """Generate trend following signal."""
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        close = data["close"]
        volume = data["volume"]

        current_price = close.iloc[-1]

        # Calculate SMAs
        fast_sma = TechnicalIndicators.sma(close, self.fast_period)
        slow_sma = TechnicalIndicators.sma(close, self.slow_period)

        current_fast = fast_sma.iloc[-1]
        current_slow = slow_sma.iloc[-1]
        prev_fast = fast_sma.iloc[-2] if len(fast_sma) > 1 else current_fast
        prev_slow = slow_sma.iloc[-2] if len(slow_sma) > 1 else current_slow

        # Crossover detection
        is_uptrend = current_fast > current_slow
        is_downtrend = current_fast < current_slow
        bullish_crossover = (current_fast > current_slow) and (prev_fast <= prev_slow)
        bearish_crossover = (current_fast < current_slow) and (prev_fast >= prev_slow)

        # Momentum: price in direction of trend
        price_above_fast = current_price > current_fast
        price_below_fast = current_price < current_fast

        # Volume filter
        volume_confirmed = True
        volume_ratio = 1.0
        if self.volume_filter:
            volume_sma = TechnicalIndicators.sma(volume, 20)
            avg_volume = volume_sma.iloc[-1]
            current_volume = volume.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            volume_confirmed = volume_ratio >= self.volume_mult

        # Trend strength (distance between SMAs as % of price)
        trend_strength = abs(current_fast - current_slow) / current_slow if current_slow > 0 else 0

        # Signal logic
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        # LONG ENTRY - Bullish crossover or already in uptrend
        if self.enable_longs and not self._in_long and not self._in_short:
            momentum_ok = price_above_fast if self.require_momentum else True
            if (bullish_crossover or is_uptrend) and momentum_ok and volume_confirmed:
                signal_type = SignalType.ENTRY
                direction = Direction.LONG

                # Score based on trend strength
                score = min(trend_strength * 20, 1.0)
                if bullish_crossover:
                    score = min(score + 0.2, 1.0)  # Bonus for fresh crossover

                probability = 0.52 + (score * 0.13)
                expected_return = 0.015 + (score * 0.025)

                self._in_long = True
                self._entry_price = current_price

        # SHORT ENTRY - Bearish crossover or already in downtrend
        if (
            self.enable_shorts
            and signal_type == SignalType.HOLD
            and not self._in_short
            and not self._in_long
        ):
            momentum_ok = price_below_fast if self.require_momentum else True
            if (bearish_crossover or is_downtrend) and momentum_ok and volume_confirmed:
                signal_type = SignalType.ENTRY
                direction = Direction.SHORT

                score = -min(trend_strength * 20, 1.0)
                if bearish_crossover:
                    score = -min(abs(score) + 0.2, 1.0)

                probability = 0.52 + (abs(score) * 0.13)
                expected_return = -0.015 - (abs(score) * 0.025)

                self._in_short = True
                self._entry_price = current_price

        # EXIT LONG - Bearish crossover
        if signal_type == SignalType.HOLD and self._in_long:
            if bearish_crossover or (is_downtrend and price_below_fast):
                signal_type = SignalType.EXIT
                direction = Direction.NEUTRAL
                self._in_long = False
                self._entry_price = None

        # EXIT SHORT - Bullish crossover
        if signal_type == SignalType.HOLD and self._in_short:
            if bullish_crossover or (is_uptrend and price_above_fast):
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
            "fast_sma": float(current_fast),
            "slow_sma": float(current_slow),
            "is_uptrend": float(is_uptrend),
            "is_downtrend": float(is_downtrend),
            "bullish_crossover": float(bullish_crossover),
            "bearish_crossover": float(bearish_crossover),
            "trend_strength": float(trend_strength),
            "price_above_fast": float(price_above_fast),
            "price_below_fast": float(price_below_fast),
            "volume_ratio": float(volume_ratio),
            "in_long": float(self._in_long),
            "in_short": float(self._in_short),
        }

        data_quality = 1.0 - (close.tail(20).isnull().sum() / 20)

        if isinstance(data.index, pd.DatetimeIndex):
            delta = timestamp - data.index[-1]
            staleness = timedelta(seconds=abs(delta.total_seconds()))
        else:
            staleness = timedelta(seconds=0)

        regime = "uptrend" if is_uptrend else ("downtrend" if is_downtrend else "neutral")

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
                "trend_strength": float(trend_strength),
            },
        )

    def reset_state(self):
        """Reset position tracking."""
        self._in_long = False
        self._in_short = False
        self._entry_price = None
        self._prev_fast_sma = None
        self._prev_slow_sma = None
