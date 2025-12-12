"""
Trend-Following Strategies.

Optimized for trending markets (bull/bear regimes):
- Stay with the trend, don't fight it
- Ride momentum until exhaustion signals
- Use trailing stops to protect profits

Strategies:
- MACrossoverStrategy: Classic dual MA crossover
- BreakoutStrategy: Price/volatility breakouts
- ADXTrendStrategy: ADX-based trend following
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class SignalType(Enum):
    """Trading signal types."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"


@dataclass
class TradingSignal:
    """Signal from a strategy."""

    signal_type: SignalType
    strength: float  # 0-1, confidence in signal
    price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size: float = 1.0  # Fraction of available capital


class TrendFollowingStrategy(ABC):
    """Base class for trend-following strategies."""

    def __init__(self, name: str):
        self.name = name
        self._position = 0  # Current position (shares)
        self._entry_price = 0.0
        self._trailing_stop = None

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate trading signal from current data."""

    def reset(self):
        """Reset strategy state."""
        self._position = 0
        self._entry_price = 0.0
        self._trailing_stop = None

    @property
    def is_long(self) -> bool:
        return self._position > 0

    @property
    def is_flat(self) -> bool:
        return self._position == 0


class MACrossoverStrategy(TrendFollowingStrategy):
    """
    Moving Average Crossover Strategy.

    BULL MARKET OPTIMIZATION:
    - Uses faster MAs for quicker entry
    - Stays invested above 200 MA
    - Only exits on death cross OR price below 200 MA
    - Trailing stop to protect profits

    Entry: Fast MA crosses above slow MA (golden cross)
    Exit: Fast MA crosses below slow MA (death cross) OR trailing stop hit
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        trend_period: int = 200,
        trailing_stop_pct: float = 0.08,
        use_ema: bool = True,
    ):
        super().__init__("MA Crossover (Trend)")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trend_period = trend_period
        self.trailing_stop_pct = trailing_stop_pct
        self.use_ema = use_ema

        self._prev_fast = None
        self._prev_slow = None
        self._highest_since_entry = 0.0

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate signal based on MA crossover."""
        if len(data) < self.trend_period:
            return TradingSignal(SignalType.HOLD, 0.0, data["close"].iloc[-1])

        close = data["close"]
        current_price = close.iloc[-1]

        # Calculate MAs
        if self.use_ema:
            fast_ma = close.ewm(span=self.fast_period, adjust=False).mean().iloc[-1]
            slow_ma = close.ewm(span=self.slow_period, adjust=False).mean().iloc[-1]
        else:
            fast_ma = close.rolling(self.fast_period).mean().iloc[-1]
            slow_ma = close.rolling(self.slow_period).mean().iloc[-1]

        trend_ma = close.rolling(self.trend_period).mean().iloc[-1]

        # Trend filter: only trade in direction of long-term trend
        above_trend = current_price > trend_ma

        # Update tracking
        if self.is_long:
            self._highest_since_entry = max(self._highest_since_entry, current_price)
            trailing_stop = self._highest_since_entry * (1 - self.trailing_stop_pct)

            # Check trailing stop
            if current_price < trailing_stop:
                self._position = 0
                signal = TradingSignal(SignalType.EXIT, 0.8, current_price, stop_loss=trailing_stop)
                self._prev_fast = fast_ma
                self._prev_slow = slow_ma
                return signal

        # Generate crossover signals
        if self._prev_fast is not None and self._prev_slow is not None:
            # Golden cross - bullish
            if self._prev_fast <= self._prev_slow and fast_ma > slow_ma:
                if above_trend and self.is_flat:
                    self._position = 1
                    self._entry_price = current_price
                    self._highest_since_entry = current_price

                    # Calculate signal strength based on MA spread
                    spread = (fast_ma - slow_ma) / slow_ma
                    strength = min(1.0, spread * 20 + 0.5)

                    signal = TradingSignal(
                        SignalType.BUY,
                        strength,
                        current_price,
                        stop_loss=current_price * (1 - self.trailing_stop_pct),
                        position_size=0.95,
                    )
                    self._prev_fast = fast_ma
                    self._prev_slow = slow_ma
                    return signal

            # Death cross - bearish
            elif self._prev_fast >= self._prev_slow and fast_ma < slow_ma:
                if self.is_long:
                    self._position = 0

                    signal = TradingSignal(
                        SignalType.SELL,
                        0.7,
                        current_price,
                    )
                    self._prev_fast = fast_ma
                    self._prev_slow = slow_ma
                    return signal

        self._prev_fast = fast_ma
        self._prev_slow = slow_ma

        return TradingSignal(SignalType.HOLD, 0.0, current_price)

    def reset(self):
        super().reset()
        self._prev_fast = None
        self._prev_slow = None
        self._highest_since_entry = 0.0


class BreakoutStrategy(TrendFollowingStrategy):
    """
    Breakout Strategy.

    TREND MARKET OPTIMIZATION:
    - Enters on new highs (Donchian channel breakout)
    - Uses ATR-based stops for volatility adjustment
    - Pyramids into winning positions

    Entry: Price breaks above N-day high
    Exit: Price breaks below trailing stop (ATR-based)
    """

    def __init__(
        self,
        breakout_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        trend_filter_period: int = 50,
    ):
        super().__init__("Breakout (Trend)")
        self.breakout_period = breakout_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.trend_filter_period = trend_filter_period

        self._atr = None
        self._stop_loss = None

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate signal based on price breakouts."""
        min_bars = max(self.breakout_period, self.atr_period, self.trend_filter_period)
        if len(data) < min_bars:
            return TradingSignal(SignalType.HOLD, 0.0, data["close"].iloc[-1])

        high = data["high"]
        low = data["low"]
        close = data["close"]
        current_price = close.iloc[-1]

        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self._atr = tr.rolling(self.atr_period).mean().iloc[-1]

        # Donchian channels
        upper_channel = high.rolling(self.breakout_period).max().iloc[-1]
        lower_channel = low.rolling(self.exit_period).min().iloc[-1]

        # Trend filter
        trend_ma = close.rolling(self.trend_filter_period).mean().iloc[-1]
        uptrend = current_price > trend_ma

        # Update stop loss for existing position
        if self.is_long and self._stop_loss is not None:
            new_stop = current_price - (self.atr_multiplier * self._atr)
            self._stop_loss = max(self._stop_loss, new_stop)  # Ratchet up only

            # Check stop
            if current_price < self._stop_loss:
                self._position = 0
                return TradingSignal(SignalType.EXIT, 0.8, current_price, stop_loss=self._stop_loss)

        # Entry signal: breakout above channel in uptrend
        if current_price >= upper_channel and uptrend and self.is_flat:
            self._position = 1
            self._entry_price = current_price
            self._stop_loss = current_price - (self.atr_multiplier * self._atr)

            # Signal strength based on breakout magnitude
            breakout_pct = (current_price - upper_channel) / upper_channel
            strength = min(1.0, 0.6 + breakout_pct * 10)

            return TradingSignal(
                SignalType.BUY,
                strength,
                current_price,
                stop_loss=self._stop_loss,
                position_size=0.9,
            )

        # Exit signal: break below exit channel
        if self.is_long and current_price <= lower_channel:
            self._position = 0
            return TradingSignal(
                SignalType.SELL,
                0.7,
                current_price,
            )

        return TradingSignal(SignalType.HOLD, 0.0, current_price)

    def reset(self):
        super().reset()
        self._atr = None
        self._stop_loss = None


class ADXTrendStrategy(TrendFollowingStrategy):
    """
    ADX-Based Trend Strategy.

    Uses ADX to confirm trend strength before entering:
    - Only trades when ADX > threshold (strong trend)
    - Uses +DI/-DI crossovers for direction
    - Scales position size with ADX strength

    Entry: ADX > threshold AND +DI > -DI (for longs)
    Exit: ADX declining OR +DI < -DI
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        adx_strong: float = 40.0,
        trend_ma_period: int = 50,
        trailing_stop_pct: float = 0.06,
    ):
        super().__init__("ADX Trend")
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.adx_strong = adx_strong
        self.trend_ma_period = trend_ma_period
        self.trailing_stop_pct = trailing_stop_pct

        self._prev_adx = None
        self._highest_price = 0.0

    def _calculate_adx_components(self, data: pd.DataFrame) -> tuple:
        """Calculate ADX, +DI, -DI."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smoothed
        atr = tr.rolling(self.adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(self.adx_period).mean()

        return adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate signal based on ADX trend strength."""
        min_bars = max(self.adx_period * 2, self.trend_ma_period)
        if len(data) < min_bars:
            return TradingSignal(SignalType.HOLD, 0.0, data["close"].iloc[-1])

        close = data["close"]
        current_price = close.iloc[-1]

        # Calculate ADX components
        adx, plus_di, minus_di = self._calculate_adx_components(data)

        # Trend MA filter
        trend_ma = close.rolling(self.trend_ma_period).mean().iloc[-1]
        above_trend = current_price > trend_ma

        # Track highest for trailing stop
        if self.is_long:
            self._highest_price = max(self._highest_price, current_price)
            trailing_stop = self._highest_price * (1 - self.trailing_stop_pct)

            # Check stop
            if current_price < trailing_stop:
                self._position = 0
                return TradingSignal(SignalType.EXIT, 0.8, current_price)

            # Check ADX declining (trend weakening)
            if self._prev_adx is not None and adx < self._prev_adx and adx < self.adx_threshold:
                self._position = 0
                self._prev_adx = adx
                return TradingSignal(SignalType.SELL, 0.6, current_price)

            # Check DI crossover (trend reversal)
            if plus_di < minus_di:
                self._position = 0
                self._prev_adx = adx
                return TradingSignal(SignalType.SELL, 0.7, current_price)

        # Entry conditions
        if self.is_flat:
            # Strong trend + bullish direction + above trend MA
            if adx > self.adx_threshold and plus_di > minus_di and above_trend:
                self._position = 1
                self._entry_price = current_price
                self._highest_price = current_price

                # Scale position with ADX strength
                if adx > self.adx_strong:
                    position_size = 0.95
                    strength = 0.9
                else:
                    position_size = 0.7
                    strength = 0.7

                self._prev_adx = adx
                return TradingSignal(
                    SignalType.BUY,
                    strength,
                    current_price,
                    stop_loss=current_price * (1 - self.trailing_stop_pct),
                    position_size=position_size,
                )

        self._prev_adx = adx
        return TradingSignal(SignalType.HOLD, 0.0, current_price)

    def reset(self):
        super().reset()
        self._prev_adx = None
        self._highest_price = 0.0


class TrendFollowingEnsemble:
    """
    Combines multiple trend-following strategies for more robust signals.

    Uses voting/weighting to generate consensus signals.
    """

    def __init__(self):
        self.strategies = [
            MACrossoverStrategy(),
            BreakoutStrategy(),
            ADXTrendStrategy(),
        ]
        self.weights = [0.4, 0.3, 0.3]  # Default weights

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate consensus signal from all strategies."""
        signals = []
        for strategy, weight in zip(self.strategies, self.weights, strict=False):
            signal = strategy.generate_signal(data)
            signals.append((signal, weight))

        # Count weighted votes
        buy_score = sum(w for s, w in signals if s.signal_type == SignalType.BUY)
        sell_score = sum(w for s, w in signals if s.signal_type == SignalType.SELL)
        exit_score = sum(w for s, w in signals if s.signal_type == SignalType.EXIT)

        current_price = data["close"].iloc[-1]

        # Consensus threshold
        threshold = 0.5

        if buy_score >= threshold:
            avg_strength = np.mean(
                [s.strength for s, w in signals if s.signal_type == SignalType.BUY]
            )
            return TradingSignal(SignalType.BUY, avg_strength, current_price, position_size=0.9)

        if sell_score >= threshold or exit_score >= threshold:
            return TradingSignal(SignalType.SELL, max(sell_score, exit_score), current_price)

        return TradingSignal(SignalType.HOLD, 0.0, current_price)

    def reset(self):
        for strategy in self.strategies:
            strategy.reset()
