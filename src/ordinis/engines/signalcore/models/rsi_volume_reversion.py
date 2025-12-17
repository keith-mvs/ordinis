"""
RSI Mean Reversion with Volume Confirmation and Trend Filter.

Enhanced mean reversion strategy that combines:
1. RSI oversold/overbought conditions
2. Volume spike confirmation (configurable multiplier of average)
3. Trend filter (price above/below long SMA) - optional

Strategy Modes:
- LONG ONLY: Buy on RSI oversold in uptrends
- SHORT ONLY: Sell on RSI overbought in downtrends
- BIDIRECTIONAL: Both long and short entries

This addresses the 0% win rate issue from pure SMA crossover by:
- Using mean reversion instead of trend following
- Requiring volume confirmation to filter out noise
- Configurable trend filter (can be disabled)
- Support for short selling in overbought conditions
"""

from datetime import datetime, timedelta

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


class RSIVolumeReversionModel(Model):
    """
    RSI Mean Reversion with Volume Confirmation.

    Entry Conditions (LONG):
        - RSI < oversold_threshold (default 30)
        - Volume > volume_mult * avg_volume (default 1.5x)
        - Price > trend_sma (uptrend confirmation, if enabled)

    Entry Conditions (SHORT):
        - RSI > overbought_threshold (default 70)
        - Volume > volume_mult * avg_volume (default 1.5x)
        - Price < trend_sma (downtrend confirmation, if enabled)
        - require_trend_for_shorts: if False, shorts work in any trend

    Exit Conditions:
        - RSI crosses exit_rsi (default 50) - take profit on reversion to mean
        - Stop Loss triggered (passed via parameters)
        - Take Profit triggered (passed via parameters)

    Parameters:
        rsi_period: RSI calculation period (default 14)
        oversold_threshold: RSI level to trigger buy (default 30)
        overbought_threshold: RSI level to trigger sell (default 70)
        exit_rsi: RSI level to exit position (default 50)
        volume_period: Volume SMA period (default 20)
        volume_mult: Volume multiplier threshold (default 1.5, use 1.0 to disable)
        trend_filter_period: SMA period for trend filter, 0 to disable (default 50)
        enable_shorts: Allow short entries (default True)
        enable_longs: Allow long entries (default True)
        require_trend_for_shorts: Require downtrend for shorts (default False)
        require_trend_for_longs: Require uptrend for longs (default False)
    """

    def __init__(self, config: ModelConfig):
        """Initialize RSI Volume Reversion model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.rsi_period = params.get("rsi_period", 14)
        self.oversold_threshold = params.get("oversold_threshold", 30)
        self.overbought_threshold = params.get("overbought_threshold", 70)
        self.exit_rsi = params.get("exit_rsi", 50)
        self.volume_period = params.get("volume_period", 20)
        self.volume_mult = params.get("volume_mult", 1.5)
        self.trend_filter_period = params.get("trend_filter_period", 50)
        self.enable_shorts = params.get("enable_shorts", True)  # Changed default
        self.enable_longs = params.get("enable_longs", True)
        self.require_trend_for_shorts = params.get("require_trend_for_shorts", False)
        self.require_trend_for_longs = params.get("require_trend_for_longs", False)

        # For tracking position state
        self._in_long = False
        self._in_short = False

        # Update min data points based on all indicators
        max_period = max(
            self.rsi_period + 5,
            self.volume_period + 5,
            self.trend_filter_period + 5 if self.trend_filter_period > 0 else 0,
        )
        self.config.min_data_points = max(self.config.min_data_points, max_period)

    def validate(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Lightweight validation."""
        if len(data) < self.config.min_data_points:
            return False, f"Insufficient data: {len(data)} < {self.config.min_data_points}"

        required_cols = {"close", "volume"}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            return False, f"Missing columns: {missing}"

        return True, ""

    async def generate(self, symbol: str, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate trading signal from RSI + Volume reversion.

        Args:
            symbol: Stock ticker symbol
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with mean reversion prediction
        """
        # Validate data
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        close = data["close"]
        volume = data["volume"]

        # ========== INDICATORS ==========
        # RSI
        rsi = TechnicalIndicators.rsi(close, self.rsi_period)
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi

        # Volume SMA and current volume ratio
        volume_sma = TechnicalIndicators.sma(volume, self.volume_period)
        current_volume = volume.iloc[-1]
        avg_volume = volume_sma.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        volume_confirmed = volume_ratio >= self.volume_mult

        # Trend Filter (optional)
        trend_bullish = True  # Default if filter disabled
        trend_bearish = True
        current_trend_sma = None
        if self.trend_filter_period > 0:
            trend_sma = TechnicalIndicators.sma(close, self.trend_filter_period)
            current_trend_sma = trend_sma.iloc[-1]
            current_price = close.iloc[-1]
            trend_bullish = current_price > current_trend_sma
            trend_bearish = current_price < current_trend_sma

        # ========== SIGNAL LOGIC ==========
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        # Define entry conditions
        is_oversold = current_rsi < self.oversold_threshold
        is_overbought = current_rsi > self.overbought_threshold

        # ----- LONG ENTRY -----
        # RSI oversold + volume spike + (optional uptrend)
        if self.enable_longs:
            trend_ok_for_long = trend_bullish if self.require_trend_for_longs else True
            long_entry = (
                is_oversold
                and volume_confirmed
                and trend_ok_for_long
                and not self._in_long
                and not self._in_short
            )

            if long_entry:
                signal_type = SignalType.ENTRY
                direction = Direction.LONG

                # Score based on how oversold (lower RSI = stronger signal)
                rsi_strength = (self.oversold_threshold - current_rsi) / self.oversold_threshold
                volume_strength = (
                    min((volume_ratio - 1.0) / 0.5, 1.0) if volume_ratio > 1.0 else 0.0
                )
                score = (rsi_strength * 0.7) + (volume_strength * 0.3)
                score = min(score, 1.0)

                probability = 0.55 + (score * 0.20)  # 55-75%
                expected_return = 0.02 + (score * 0.03)  # 2-5%

                self._in_long = True

        # ----- SHORT ENTRY -----
        # RSI overbought + volume spike + (optional downtrend)
        if self.enable_shorts and signal_type == SignalType.HOLD:
            trend_ok_for_short = trend_bearish if self.require_trend_for_shorts else True
            short_entry = (
                is_overbought
                and volume_confirmed
                and trend_ok_for_short
                and not self._in_short
                and not self._in_long
            )

            if short_entry:
                signal_type = SignalType.ENTRY
                direction = Direction.SHORT

                rsi_strength = (current_rsi - self.overbought_threshold) / (
                    100 - self.overbought_threshold
                )
                volume_strength = (
                    min((volume_ratio - 1.0) / 0.5, 1.0) if volume_ratio > 1.0 else 0.0
                )
                score = -((rsi_strength * 0.7) + (volume_strength * 0.3))
                score = max(score, -1.0)

                probability = 0.55 + (abs(score) * 0.20)
                expected_return = -0.02 - (abs(score) * 0.03)

                self._in_short = True

        # ----- EXIT LONG -----
        # RSI returns to mean (exit_rsi) - take profit
        if signal_type == SignalType.HOLD and self._in_long and current_rsi >= self.exit_rsi:
            signal_type = SignalType.EXIT
            direction = Direction.NEUTRAL
            score = 0.0
            probability = 0.6
            expected_return = 0.0
            self._in_long = False

        # ----- EXIT SHORT -----
        if signal_type == SignalType.HOLD and self._in_short and current_rsi <= self.exit_rsi:
            signal_type = SignalType.EXIT
            direction = Direction.NEUTRAL
            score = 0.0
            probability = 0.6
            expected_return = 0.0
            self._in_short = False

        # ----- HOLD -----
        if signal_type == SignalType.HOLD:
            signal_type = SignalType.HOLD
            direction = Direction.NEUTRAL
            # Score reflects distance from neutral (50)
            score = (50 - current_rsi) / 100.0
            score = max(min(score, 0.3), -0.3)
            probability = 0.5

        # ========== BUILD SIGNAL ==========
        # Calculate confidence interval
        returns = close.pct_change().dropna()
        recent_vol = returns.tail(20).std() if len(returns) >= 20 else 0.02
        confidence_interval = (
            expected_return - 2 * recent_vol,
            expected_return + 2 * recent_vol,
        )

        # Feature contributions
        feature_contributions = {
            "rsi": float(current_rsi),
            "rsi_prev": float(prev_rsi),
            "volume_ratio": float(volume_ratio),
            "volume_confirmed": float(volume_confirmed),
            "trend_bullish": float(trend_bullish),
            "trend_bearish": float(trend_bearish),
            "trend_sma": float(current_trend_sma) if current_trend_sma else 0.0,
            "is_oversold": float(is_oversold),
            "is_overbought": float(is_overbought),
            "in_long": float(self._in_long),
            "in_short": float(self._in_short),
        }

        # Data quality
        recent_close = close.tail(20)
        data_quality = 1.0 - (recent_close.isnull().sum() / len(recent_close))

        # Staleness
        if isinstance(data.index, pd.DatetimeIndex):
            delta = timestamp - data.index[-1]
            staleness = timedelta(seconds=abs(delta.total_seconds()))
        else:
            staleness = timedelta(seconds=0)

        # Regime detection
        if current_rsi <= self.oversold_threshold:
            regime = "oversold"
        elif current_rsi >= self.overbought_threshold:
            regime = "overbought"
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
                "volume_mult": self.volume_mult,
                "trend_filter_period": self.trend_filter_period,
                "current_price": float(close.iloc[-1]),
            },
        )

    def reset_state(self):
        """Reset position tracking state for new backtest run."""
        self._in_long = False
        self._in_short = False
