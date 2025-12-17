"""
Mean-Reversion Strategies.

Optimized for sideways/ranging markets where prices oscillate
within defined bands without clear directional bias.

Key Principles:
- Buy oversold conditions, sell overbought conditions
- Trade toward the mean, not away from it
- Tight stops since moves are expected to be limited
- Quick profit-taking at mean/resistance levels

Strategies:
- BollingerFadeStrategy: Fade moves to band extremes
- RSIReversalStrategy: Trade RSI oversold/overbought reversals
- KeltnerChannelStrategy: ATR-based channel reversion
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .trend_following import SignalType, TradingSignal


class MeanReversionStrategy(ABC):
    """Base class for mean-reversion strategies."""

    def __init__(self, name: str):
        self.name = name
        self._position = 0
        self._entry_price = 0.0
        self._target_price = 0.0
        self._stop_loss = 0.0

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate trading signal."""

    def reset(self):
        """Reset strategy state."""
        self._position = 0
        self._entry_price = 0.0
        self._target_price = 0.0
        self._stop_loss = 0.0

    @property
    def is_long(self) -> bool:
        return self._position > 0

    @property
    def is_flat(self) -> bool:
        return self._position == 0


class BollingerFadeStrategy(MeanReversionStrategy):
    """
    Bollinger Band Fade Strategy.

    SIDEWAYS MARKET OPTIMIZATION:
    - Fade (trade against) moves to band extremes
    - Buy when price touches/breaks lower band
    - Sell when price touches/breaks upper band
    - Target: Middle band (SMA)
    - Stop: Beyond the band + buffer

    Entry: Price closes outside/at Bollinger Band
    Exit: Price reaches middle band OR stop loss
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        entry_threshold: float = 0.0,  # 0 = at band, negative = outside
        stop_buffer: float = 0.005,  # 0.5% beyond band
        take_profit_at_mean: bool = True,
    ):
        super().__init__("Bollinger Fade")
        self.period = period
        self.std_dev = std_dev
        self.entry_threshold = entry_threshold
        self.stop_buffer = stop_buffer
        self.take_profit_at_mean = take_profit_at_mean

    def _calculate_bands(self, close: pd.Series) -> tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        sma = close.rolling(self.period).mean().iloc[-1]
        std = close.rolling(self.period).std().iloc[-1]

        upper = sma + (self.std_dev * std)
        lower = sma - (self.std_dev * std)

        return upper, sma, lower

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate signal based on band extremes."""
        if len(data) < self.period:
            return TradingSignal(SignalType.HOLD, 0.0, data["close"].iloc[-1])

        close = data["close"]
        current_price = close.iloc[-1]

        upper, middle, lower = self._calculate_bands(close)

        # Calculate band position (0 = middle, 1 = upper, -1 = lower)
        band_width = upper - lower
        band_position = (current_price - middle) / (band_width / 2)

        # Check for exit if in position
        if self.is_long:
            # Check stop loss
            if current_price <= self._stop_loss:
                self._position = 0
                return TradingSignal(SignalType.EXIT, 0.8, current_price)

            # Check target (middle band)
            if self.take_profit_at_mean and current_price >= self._target_price:
                self._position = 0
                return TradingSignal(SignalType.SELL, 0.9, current_price)

        # Entry signals
        if self.is_flat:
            # Buy signal: Price at/below lower band
            if band_position <= (-1 + self.entry_threshold):
                self._position = 1
                self._entry_price = current_price
                self._target_price = middle
                self._stop_loss = lower * (1 - self.stop_buffer)

                # Signal strength based on how far outside band
                strength = min(1.0, abs(band_position) / 2 + 0.5)

                return TradingSignal(
                    SignalType.BUY,
                    strength,
                    current_price,
                    stop_loss=self._stop_loss,
                    take_profit=self._target_price,
                    position_size=0.5,  # Smaller size for mean reversion
                )

            # Note: Could add short selling at upper band for full mean reversion
            # For now, we focus on long-only as per typical retail constraints

        return TradingSignal(SignalType.HOLD, 0.0, current_price)


class RSIReversalStrategy(MeanReversionStrategy):
    """
    RSI Reversal Strategy.

    SIDEWAYS MARKET OPTIMIZATION:
    - Trade RSI extremes with confirmation
    - Wait for RSI to turn from extreme (don't catch falling knife)
    - Combine with price action confirmation
    - Quick exits when RSI reaches neutral zone

    Entry: RSI reverses from oversold/overbought
    Exit: RSI reaches neutral OR target profit OR stop loss
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_zone: float = 50.0,
        require_reversal: bool = True,
        stop_percent: float = 0.03,  # 3% stop
    ):
        super().__init__("RSI Reversal")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_zone = exit_zone
        self.require_reversal = require_reversal
        self.stop_percent = stop_percent

        self._prev_rsi = None
        self._was_oversold = False
        self._was_overbought = False

    def _calculate_rsi(self, close: pd.Series) -> float:
        """Calculate RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(self.rsi_period).mean()
        avg_loss = loss.rolling(self.rsi_period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1]

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate signal based on RSI reversals."""
        if len(data) < self.rsi_period + 5:
            return TradingSignal(SignalType.HOLD, 0.0, data["close"].iloc[-1])

        close = data["close"]
        current_price = close.iloc[-1]
        rsi = self._calculate_rsi(close)

        # Track oversold/overbought states
        if rsi < self.oversold:
            self._was_oversold = True
        if rsi > self.overbought:
            self._was_overbought = True

        # Exit logic
        if self.is_long:
            # Stop loss
            if current_price <= self._stop_loss:
                self._position = 0
                self._was_oversold = False
                return TradingSignal(SignalType.EXIT, 0.8, current_price)

            # RSI reaches exit zone (neutral)
            if rsi >= self.exit_zone:
                self._position = 0
                self._was_oversold = False
                return TradingSignal(SignalType.SELL, 0.7, current_price)

            # Take profit if RSI goes overbought
            if rsi >= self.overbought:
                self._position = 0
                self._was_oversold = False
                return TradingSignal(SignalType.SELL, 0.9, current_price)

        # Entry logic
        if self.is_flat:
            # Buy on RSI reversal from oversold
            if self._was_oversold and rsi > self.oversold:
                if self.require_reversal and self._prev_rsi is not None:
                    # Require RSI to be rising
                    if rsi > self._prev_rsi:
                        self._position = 1
                        self._entry_price = current_price
                        self._stop_loss = current_price * (1 - self.stop_percent)
                        self._was_oversold = False

                        # Strength based on how far from oversold
                        strength = min(1.0, (rsi - self.oversold) / 10 + 0.5)

                        self._prev_rsi = rsi
                        return TradingSignal(
                            SignalType.BUY,
                            strength,
                            current_price,
                            stop_loss=self._stop_loss,
                            position_size=0.6,
                        )
                else:
                    # No reversal requirement
                    self._position = 1
                    self._entry_price = current_price
                    self._stop_loss = current_price * (1 - self.stop_percent)
                    self._was_oversold = False

                    self._prev_rsi = rsi  # type: ignore[assignment]
                    return TradingSignal(
                        SignalType.BUY,
                        0.6,
                        current_price,
                        stop_loss=self._stop_loss,
                        position_size=0.6,
                    )

        self._prev_rsi = rsi  # type: ignore[assignment]
        return TradingSignal(SignalType.HOLD, 0.0, current_price)


class KeltnerChannelStrategy(MeanReversionStrategy):
    """
    Keltner Channel Mean Reversion Strategy.

    Uses ATR-based channels (more adaptive than Bollinger):
    - Channels adjust to volatility automatically
    - Less susceptible to squeeze situations
    - Good for consistent mean reversion

    Entry: Price outside Keltner channel
    Exit: Price returns to middle EMA
    """

    def __init__(
        self,
        ema_period: int = 20,
        atr_period: int = 10,
        atr_multiplier: float = 2.0,
        stop_atr_multiplier: float = 1.5,
    ):
        super().__init__("Keltner Channel")
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.stop_atr_multiplier = stop_atr_multiplier

        self._atr = None

    def _calculate_atr(self, data: pd.DataFrame) -> float:
        """Calculate ATR."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean().iloc[-1]

    def _calculate_channels(self, data: pd.DataFrame) -> tuple[float, float, float]:
        """Calculate Keltner Channels."""
        close = data["close"]
        ema = close.ewm(span=self.ema_period, adjust=False).mean().iloc[-1]
        atr = self._calculate_atr(data)
        self._atr = atr  # type: ignore[assignment]

        upper = ema + (self.atr_multiplier * atr)
        lower = ema - (self.atr_multiplier * atr)

        return upper, ema, lower

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate signal based on Keltner channel extremes."""
        min_bars = max(self.ema_period, self.atr_period) + 5
        if len(data) < min_bars:
            return TradingSignal(SignalType.HOLD, 0.0, data["close"].iloc[-1])

        current_price = data["close"].iloc[-1]
        _upper, middle, lower = self._calculate_channels(data)

        # Exit logic
        if self.is_long:
            # Stop loss
            if current_price <= self._stop_loss:
                self._position = 0
                return TradingSignal(SignalType.EXIT, 0.8, current_price)

            # Target: middle EMA
            if current_price >= self._target_price:
                self._position = 0
                return TradingSignal(SignalType.SELL, 0.85, current_price)

        # Entry: Price below lower channel
        if self.is_flat and current_price < lower:
            self._position = 1
            self._entry_price = current_price
            self._target_price = middle
            self._stop_loss = current_price - (self.stop_atr_multiplier * self._atr)  # type: ignore[operator]

            # Strength based on distance outside channel
            distance = (lower - current_price) / self._atr
            strength = min(1.0, distance / 2 + 0.5)

            return TradingSignal(
                SignalType.BUY,
                strength,
                current_price,
                stop_loss=self._stop_loss,
                take_profit=self._target_price,
                position_size=0.5,
            )

        return TradingSignal(SignalType.HOLD, 0.0, current_price)


class StatisticalArbitrageStrategy(MeanReversionStrategy):
    """
    Statistical Arbitrage / Z-Score Mean Reversion.

    More quantitative approach:
    - Calculate z-score of price relative to rolling mean
    - Trade extremes (|z| > threshold)
    - Exit when z-score returns to normal

    Good for pairs trading or single-stock mean reversion.
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_zscore: float = 3.0,
    ):
        super().__init__("Z-Score Reversion")
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_zscore = stop_zscore

    def _calculate_zscore(self, data: pd.Series) -> float:
        """Calculate z-score."""
        mean = data.rolling(self.lookback).mean().iloc[-1]
        std = data.rolling(self.lookback).std().iloc[-1]

        if std < 1e-10:
            return 0.0

        return (data.iloc[-1] - mean) / std

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate signal based on z-score extremes."""
        if len(data) < self.lookback + 5:
            return TradingSignal(SignalType.HOLD, 0.0, data["close"].iloc[-1])

        close = data["close"]
        current_price = close.iloc[-1]
        zscore = self._calculate_zscore(close)

        # Exit logic
        if self.is_long:
            # Stop: Z-score too negative (extreme)
            if zscore <= -self.stop_zscore:
                self._position = 0
                return TradingSignal(SignalType.EXIT, 0.8, current_price)

            # Target: Z-score normalized
            if zscore >= -self.exit_zscore:
                self._position = 0
                return TradingSignal(SignalType.SELL, 0.85, current_price)

        # Entry: Z-score extremely negative (oversold)
        if self.is_flat and zscore <= -self.entry_zscore:
            self._position = 1
            self._entry_price = current_price

            # Strength based on z-score magnitude
            strength = min(1.0, abs(zscore) / 3)

            return TradingSignal(
                SignalType.BUY,
                strength,
                current_price,
                position_size=0.5,
            )

        return TradingSignal(SignalType.HOLD, 0.0, current_price)


class MeanReversionEnsemble:
    """
    Combines multiple mean-reversion strategies for robust signals.

    Uses consensus voting to reduce false signals in ranging markets.
    """

    def __init__(self):
        self.strategies = [
            BollingerFadeStrategy(),
            RSIReversalStrategy(),
            KeltnerChannelStrategy(),
        ]
        self.weights = [0.4, 0.35, 0.25]

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate consensus signal."""
        signals = []
        for strategy, weight in zip(self.strategies, self.weights, strict=False):
            signal = strategy.generate_signal(data)
            signals.append((signal, weight))

        # Count weighted votes
        buy_score = sum(w for s, w in signals if s.signal_type == SignalType.BUY)
        sell_score = sum(
            w for s, w in signals if s.signal_type in [SignalType.SELL, SignalType.EXIT]
        )

        current_price = data["close"].iloc[-1]

        # Need higher threshold for mean reversion (more conservative)
        threshold = 0.6

        if buy_score >= threshold:
            avg_strength = np.mean(
                [s.strength for s, w in signals if s.signal_type == SignalType.BUY]
            )
            return TradingSignal(
                SignalType.BUY,
                avg_strength,
                current_price,
                position_size=0.5,  # Smaller position for mean reversion
            )

        if sell_score >= threshold:
            return TradingSignal(SignalType.SELL, max(sell_score, 0.7), current_price)

        return TradingSignal(SignalType.HOLD, 0.0, current_price)

    def reset(self):
        for strategy in self.strategies:
            strategy.reset()
