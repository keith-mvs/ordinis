"""
Volatility Trading Strategies.

Optimized for high-volatility market conditions where prices
experience sharp swings and expanded ranges.

Key Principles:
- Capitalize on large price swings
- Quick entries and exits (don't overstay)
- Wider stops to accommodate volatility
- Smaller position sizes for risk management
- Consider options for defined risk/reward

Strategies:
- ScalpingStrategy: Quick in-and-out trades on volatility spikes
- VolatilityBreakoutStrategy: Trade breakouts with volatility expansion
- ATRTrailingStrategy: ATR-based adaptive trailing stops
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .trend_following import SignalType, TradingSignal


class VolatilityStrategy(ABC):
    """Base class for volatility trading strategies."""

    def __init__(self, name: str):
        self.name = name
        self._position = 0
        self._entry_price = 0.0
        self._stop_loss = 0.0
        self._take_profit = 0.0
        self._bars_in_trade = 0

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate trading signal."""

    def reset(self):
        """Reset strategy state."""
        self._position = 0
        self._entry_price = 0.0
        self._stop_loss = 0.0
        self._take_profit = 0.0
        self._bars_in_trade = 0

    @property
    def is_long(self) -> bool:
        return self._position > 0

    @property
    def is_flat(self) -> bool:
        return self._position == 0


class ScalpingStrategy(VolatilityStrategy):
    """
    Scalping Strategy for High Volatility.

    VOLATILE MARKET OPTIMIZATION:
    - Quick trades capturing small price movements
    - Tight profit targets (1-2 ATR)
    - Maximum hold time enforced
    - Only trade during high volatility periods

    Entry: Pullback in high-volatility environment
    Exit: Quick profit target OR time stop OR price stop
    """

    def __init__(
        self,
        atr_period: int = 14,
        volatility_threshold: float = 1.5,  # ATR relative to average
        profit_target_atr: float = 1.0,  # Take profit at 1 ATR
        stop_loss_atr: float = 1.5,  # Stop at 1.5 ATR
        max_bars_in_trade: int = 10,  # Time stop
        pullback_threshold: float = 0.3,  # Entry on pullback
    ):
        super().__init__("Scalping (Volatility)")
        self.atr_period = atr_period
        self.volatility_threshold = volatility_threshold
        self.profit_target_atr = profit_target_atr
        self.stop_loss_atr = stop_loss_atr
        self.max_bars_in_trade = max_bars_in_trade
        self.pullback_threshold = pullback_threshold

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ATR series."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def _is_high_volatility(self, atr: pd.Series) -> bool:
        """Check if current volatility is elevated."""
        current_atr = atr.iloc[-1]
        avg_atr = atr.iloc[-50:].mean() if len(atr) >= 50 else atr.mean()
        return current_atr > (avg_atr * self.volatility_threshold)

    def _is_pullback(self, data: pd.DataFrame) -> bool:
        """Check for pullback entry opportunity."""
        close = data["close"]
        high_5 = close.iloc[-5:].max()
        current = close.iloc[-1]

        # Pullback from recent high
        pullback_pct = (high_5 - current) / high_5
        return pullback_pct > self.pullback_threshold * 0.01  # Convert to decimal

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate scalping signal."""
        if len(data) < 50:
            return TradingSignal(SignalType.HOLD, 0.0, data["close"].iloc[-1])

        current_price = data["close"].iloc[-1]
        atr = self._calculate_atr(data)
        current_atr = atr.iloc[-1]

        # Manage existing position
        if self.is_long:
            self._bars_in_trade += 1

            # Time stop
            if self._bars_in_trade >= self.max_bars_in_trade:
                self._position = 0
                self._bars_in_trade = 0
                return TradingSignal(SignalType.EXIT, 0.6, current_price)

            # Price stop
            if current_price <= self._stop_loss:
                self._position = 0
                self._bars_in_trade = 0
                return TradingSignal(SignalType.EXIT, 0.8, current_price)

            # Profit target
            if current_price >= self._take_profit:
                self._position = 0
                self._bars_in_trade = 0
                return TradingSignal(SignalType.SELL, 0.9, current_price)

        # Entry logic
        if self.is_flat:
            # Only trade in high volatility
            if not self._is_high_volatility(atr):
                return TradingSignal(SignalType.HOLD, 0.0, current_price)

            # Look for pullback entry
            if self._is_pullback(data):
                # Check trend is still up (we're buying the dip)
                ema_20 = data["close"].ewm(span=20).mean().iloc[-1]
                if current_price > ema_20 * 0.98:  # Within 2% of EMA
                    self._position = 1
                    self._entry_price = current_price
                    self._stop_loss = current_price - (self.stop_loss_atr * current_atr)
                    self._take_profit = current_price + (self.profit_target_atr * current_atr)
                    self._bars_in_trade = 0

                    return TradingSignal(
                        SignalType.BUY,
                        0.7,
                        current_price,
                        stop_loss=self._stop_loss,
                        take_profit=self._take_profit,
                        position_size=0.3,  # Small size for scalping
                    )

        return TradingSignal(SignalType.HOLD, 0.0, current_price)


class VolatilityBreakoutStrategy(VolatilityStrategy):
    """
    Volatility Breakout Strategy.

    VOLATILE MARKET OPTIMIZATION:
    - Trade breakouts from consolidation
    - Volatility contraction -> expansion pattern
    - Use Bollinger Band squeeze for timing
    - Wider stops for volatile environment

    Entry: Breakout from low-volatility squeeze
    Exit: Trailing stop based on ATR
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        squeeze_percentile: float = 25.0,  # BB width percentile for squeeze
        breakout_confirmation: int = 2,  # Bars above/below band
        atr_trailing_mult: float = 2.0,
    ):
        super().__init__("Volatility Breakout")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.squeeze_percentile = squeeze_percentile
        self.breakout_confirmation = breakout_confirmation
        self.atr_trailing_mult = atr_trailing_mult

        self._squeeze_detected = False
        self._breakout_bars = 0
        self._highest_since_entry = 0.0

    def _calculate_bb(self, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        upper = middle + (self.bb_std * std)
        lower = middle - (self.bb_std * std)
        return upper, middle, lower

    def _calculate_bb_width(
        self, upper: pd.Series, lower: pd.Series, middle: pd.Series
    ) -> pd.Series:
        """Calculate Bollinger Band width."""
        return (upper - lower) / middle * 100

    def _is_squeeze(self, bb_width: pd.Series) -> bool:
        """Check if in squeeze (low volatility)."""
        if len(bb_width) < 100:
            return False

        current_width = bb_width.iloc[-1]
        threshold = bb_width.iloc[-100:].quantile(self.squeeze_percentile / 100)
        return current_width <= threshold

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate current ATR."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate breakout signal."""
        if len(data) < 100:
            return TradingSignal(SignalType.HOLD, 0.0, data["close"].iloc[-1])

        close = data["close"]
        current_price = close.iloc[-1]

        upper, middle, lower = self._calculate_bb(close)
        bb_width = self._calculate_bb_width(upper, lower, middle)

        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]

        # Track squeeze state
        if self._is_squeeze(bb_width):
            self._squeeze_detected = True

        # Manage existing position
        if self.is_long:
            # Update trailing stop
            self._highest_since_entry = max(self._highest_since_entry, current_price)
            atr = self._calculate_atr(data)
            trailing_stop = self._highest_since_entry - (self.atr_trailing_mult * atr)
            self._stop_loss = max(self._stop_loss, trailing_stop)  # Ratchet up

            # Check stop
            if current_price <= self._stop_loss:
                self._position = 0
                self._squeeze_detected = False
                self._breakout_bars = 0
                return TradingSignal(SignalType.EXIT, 0.8, current_price)

        # Entry: Breakout from squeeze
        if self.is_flat and self._squeeze_detected:
            # Check for upside breakout
            if current_price > upper_val:
                self._breakout_bars += 1

                if self._breakout_bars >= self.breakout_confirmation:
                    self._position = 1
                    self._entry_price = current_price
                    self._highest_since_entry = current_price

                    atr = self._calculate_atr(data)
                    self._stop_loss = current_price - (self.atr_trailing_mult * atr)

                    self._squeeze_detected = False
                    self._breakout_bars = 0

                    return TradingSignal(
                        SignalType.BUY,
                        0.8,
                        current_price,
                        stop_loss=self._stop_loss,
                        position_size=0.5,
                    )
            else:
                self._breakout_bars = 0

        return TradingSignal(SignalType.HOLD, 0.0, current_price)


class ATRTrailingStrategy(VolatilityStrategy):
    """
    ATR-Based Trailing Stop Strategy.

    VOLATILE MARKET OPTIMIZATION:
    - Adaptive stops that widen with volatility
    - Stays in trends but protects capital
    - Uses Chandelier Exit concept
    - Multiple ATR multipliers for different risk levels

    Entry: Trend confirmation with ATR expansion
    Exit: Price falls below ATR trailing stop
    """

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 3.0,
        trend_ema: int = 50,
        min_atr_expansion: float = 1.2,  # ATR must be 20% above average
    ):
        super().__init__("ATR Trailing")
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.trend_ema = trend_ema
        self.min_atr_expansion = min_atr_expansion

        self._highest_since_entry = 0.0
        self._chandelier_stop = 0.0

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ATR series."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def _is_atr_expanding(self, atr: pd.Series) -> bool:
        """Check if ATR is expanding."""
        current = atr.iloc[-1]
        avg = atr.iloc[-50:].mean() if len(atr) >= 50 else atr.mean()
        return current > (avg * self.min_atr_expansion)

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate ATR trailing signal."""
        min_bars = max(self.atr_period, self.trend_ema) + 10
        if len(data) < min_bars:
            return TradingSignal(SignalType.HOLD, 0.0, data["close"].iloc[-1])

        close = data["close"]
        high = data["high"]
        current_price = close.iloc[-1]

        atr = self._calculate_atr(data)
        current_atr = atr.iloc[-1]

        ema = close.ewm(span=self.trend_ema).mean().iloc[-1]

        # Manage existing position
        if self.is_long:
            # Update highest high
            current_high = high.iloc[-1]
            if current_high > self._highest_since_entry:
                self._highest_since_entry = current_high
                # Update Chandelier stop
                self._chandelier_stop = self._highest_since_entry - (
                    self.atr_multiplier * current_atr
                )
                self._stop_loss = self._chandelier_stop

            # Check stop
            if current_price <= self._stop_loss:
                self._position = 0
                return TradingSignal(SignalType.EXIT, 0.8, current_price)

        # Entry: Uptrend + ATR expansion
        if self.is_flat:
            uptrend = current_price > ema
            atr_expanding = self._is_atr_expanding(atr)

            if uptrend and atr_expanding:
                # Look for pullback entry
                recent_high = high.iloc[-5:].max()
                if current_price > recent_high * 0.98:  # Near recent high
                    self._position = 1
                    self._entry_price = current_price
                    self._highest_since_entry = high.iloc[-5:].max()
                    self._chandelier_stop = self._highest_since_entry - (
                        self.atr_multiplier * current_atr
                    )
                    self._stop_loss = self._chandelier_stop

                    return TradingSignal(
                        SignalType.BUY,
                        0.75,
                        current_price,
                        stop_loss=self._stop_loss,
                        position_size=0.6,
                    )

        return TradingSignal(SignalType.HOLD, 0.0, current_price)


class VolatilityTradingEnsemble:
    """
    Combines volatility strategies with dynamic selection.

    Selects the most appropriate strategy based on current conditions.
    """

    def __init__(self):
        self.strategies = {
            "scalping": ScalpingStrategy(),
            "breakout": VolatilityBreakoutStrategy(),
            "atr_trailing": ATRTrailingStrategy(),
        }
        self._active_strategy = None

    def _select_strategy(self, data: pd.DataFrame) -> str:
        """Select best strategy for current conditions."""
        close = data["close"]

        # Calculate volatility metrics
        returns = close.pct_change().iloc[-20:]
        current_vol = returns.std() * np.sqrt(252)

        # Bollinger width for squeeze detection
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_width = ((2 * bb_std) / bb_middle).iloc[-1]

        # Strategy selection logic
        if bb_width < 0.03:  # Tight squeeze
            return "breakout"
        if current_vol > 0.30:  # Very high volatility
            return "scalping"
        return "atr_trailing"

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        """Generate signal from selected strategy."""
        if len(data) < 100:
            return TradingSignal(SignalType.HOLD, 0.0, data["close"].iloc[-1])

        strategy_name = self._select_strategy(data)
        strategy = self.strategies[strategy_name]

        # If changing strategies and in position, exit first
        if self._active_strategy and self._active_strategy != strategy_name:
            old_strategy = self.strategies[self._active_strategy]
            if old_strategy.is_long:
                old_strategy.reset()
                self._active_strategy = strategy_name
                return TradingSignal(
                    SignalType.EXIT,
                    0.6,
                    data["close"].iloc[-1],
                )

        self._active_strategy = strategy_name
        return strategy.generate_signal(data)

    def reset(self):
        for strategy in self.strategies.values():
            strategy.reset()
        self._active_strategy = None
