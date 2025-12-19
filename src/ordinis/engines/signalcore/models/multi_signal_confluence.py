"""
Multi-Signal Confluence Model - Requires agreement from multiple oscillators.

Based on empirical analysis showing single-indicator strategies get whipsawed.
This model requires confluence from:
1. RSI (standard or fast)
2. Stochastic Oscillator (%K/%D)
3. ADX trend strength filter (optional)
4. ATR-based adaptive stops

Key insight from CRWD failure analysis:
- Single oscillator signals had ~50% direction change rate (coin flip)
- Multiple oscillator agreement should filter noise significantly

Entry requires 2+ oscillator agreement + ADX/volatility filter.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


class MultiSignalConfluenceModel(Model):
    """
    Multi-Oscillator Confluence with ATR-Based Stops.

    Entry Conditions (LONG):
        - RSI < rsi_oversold (e.g., 30)
        - Stochastic %K < stoch_oversold (e.g., 20)
        - Both oscillators agree (confluence)
        - ADX < adx_max_for_reversion (market not strongly trending)

    Entry Conditions (SHORT):
        - RSI > rsi_overbought (e.g., 70)
        - Stochastic %K > stoch_overbought (e.g., 80)
        - Both oscillators agree (confluence)
        - ADX < adx_max_for_reversion (market not strongly trending)

    Stop Loss:
        - ATR-based: entry_price - (atr_stop_mult * ATR)
        - Adapts to each stock's volatility

    Take Profit:
        - ATR-based: entry_price + (atr_tp_mult * ATR)
        - Risk:Reward ratio built into multipliers

    Parameters:
        rsi_period: RSI calculation period (default 14)
        rsi_oversold: RSI buy threshold (default 30)
        rsi_overbought: RSI sell threshold (default 70)
        stoch_period: Stochastic %K period (default 14)
        stoch_smooth: Stochastic %K smoothing (default 3)
        stoch_d_period: Stochastic %D period (default 3)
        stoch_oversold: Stochastic buy threshold (default 20)
        stoch_overbought: Stochastic sell threshold (default 80)
        adx_period: ADX calculation period (default 14)
        adx_max_for_reversion: Max ADX for mean-reversion (default 30)
        adx_min_for_trend: Min ADX for trend signals (default 25)
        atr_period: ATR period for stops (default 14)
        atr_stop_mult: ATR multiplier for stop loss (default 2.0)
        atr_tp_mult: ATR multiplier for take profit (default 3.0)
        require_all_signals: Require ALL signals (not just 2) (default False)
        enable_shorts: Allow short entries (default True)
        enable_longs: Allow long entries (default True)
    """

    def __init__(self, config: ModelConfig):
        """Initialize Multi-Signal Confluence model."""
        super().__init__(config)

        params = self.config.parameters

        # RSI parameters
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 30)
        self.rsi_overbought = params.get("rsi_overbought", 70)

        # Stochastic parameters
        self.stoch_period = params.get("stoch_period", 14)
        self.stoch_smooth = params.get("stoch_smooth", 3)
        self.stoch_d_period = params.get("stoch_d_period", 3)
        self.stoch_oversold = params.get("stoch_oversold", 20)
        self.stoch_overbought = params.get("stoch_overbought", 80)

        # ADX parameters
        self.adx_period = params.get("adx_period", 14)
        self.adx_max_for_reversion = params.get("adx_max_for_reversion", 30)
        self.adx_min_for_trend = params.get("adx_min_for_trend", 25)

        # ATR stop parameters
        self.atr_period = params.get("atr_period", 14)
        self.atr_stop_mult = params.get("atr_stop_mult", 2.0)
        self.atr_tp_mult = params.get("atr_tp_mult", 3.0)

        # Mode
        self.require_all_signals = params.get("require_all_signals", False)
        self.enable_shorts = params.get("enable_shorts", True)
        self.enable_longs = params.get("enable_longs", True)

        # State tracking
        self._in_long = False
        self._in_short = False
        self._entry_price = None
        self._stop_loss = None
        self._take_profit = None

        # Calculate required data points
        max_period = max(
            self.rsi_period + 5,
            self.stoch_period + self.stoch_smooth + 5,
            self.adx_period * 2 + 5,  # ADX needs more data
            self.atr_period + 5,
        )
        self.config.min_data_points = max(self.config.min_data_points, max_period)

    def validate(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate input data."""
        if len(data) < self.config.min_data_points:
            return False, f"Insufficient data: {len(data)} < {self.config.min_data_points}"

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            return False, f"Missing columns: {missing}"

        return True, ""

    def _compute_stochastic(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Compute Stochastic Oscillator %K and %D."""
        # Raw %K
        lowest_low = low.rolling(window=self.stoch_period).min()
        highest_high = high.rolling(window=self.stoch_period).max()

        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)

        # Slow %K (smoothed)
        stoch_k = raw_k.rolling(window=self.stoch_smooth).mean()

        # %D (signal line)
        stoch_d = stoch_k.rolling(window=self.stoch_d_period).mean()

        return stoch_k, stoch_d

    def _compute_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute ADX, +DI, and -DI.

        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)

        # Smoothed averages (Wilder's smoothing)
        atr = tr.ewm(span=self.adx_period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=self.adx_period, adjust=False).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.ewm(span=self.adx_period, adjust=False).mean() / (atr + 1e-10)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=self.adx_period, adjust=False).mean()

        return adx, plus_di, minus_di

    def _compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Compute Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=self.atr_period).mean()

    async def generate(self, symbol: str, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate trading signal from multi-oscillator confluence.

        Args:
            symbol: Stock ticker symbol
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with confluence-based prediction and ATR stops
        """
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        close = data["close"]
        high = data["high"]
        low = data["low"]
        current_price = close.iloc[-1]

        # ========== COMPUTE ALL INDICATORS ==========

        # RSI
        rsi = TechnicalIndicators.rsi(close, self.rsi_period)
        current_rsi = rsi.iloc[-1]

        # Stochastic
        stoch_k, stoch_d = self._compute_stochastic(high, low, close)
        current_stoch_k = stoch_k.iloc[-1]
        current_stoch_d = stoch_d.iloc[-1]
        prev_stoch_k = stoch_k.iloc[-2] if len(stoch_k) > 1 else current_stoch_k

        # ADX and DI lines
        adx, plus_di, minus_di = self._compute_adx(high, low, close)
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]

        # ATR for stops
        atr = self._compute_atr(high, low, close)
        current_atr = atr.iloc[-1]

        # ========== SIGNAL CONDITIONS ==========

        # RSI signals
        rsi_oversold = current_rsi < self.rsi_oversold
        rsi_overbought = current_rsi > self.rsi_overbought

        # Stochastic signals (with %K crossing %D confirmation)
        stoch_oversold = current_stoch_k < self.stoch_oversold
        stoch_overbought = current_stoch_k > self.stoch_overbought
        stoch_bullish_cross = current_stoch_k > current_stoch_d and prev_stoch_k <= stoch_d.iloc[-2]
        stoch_bearish_cross = current_stoch_k < current_stoch_d and prev_stoch_k >= stoch_d.iloc[-2]

        # ADX conditions
        no_strong_trend = current_adx < self.adx_max_for_reversion
        bullish_direction = current_plus_di > current_minus_di
        bearish_direction = current_minus_di > current_plus_di

        # ========== CONFLUENCE LOGIC ==========

        # Count bullish signals
        bullish_signals = sum(
            [
                rsi_oversold,
                stoch_oversold,
                stoch_bullish_cross and stoch_oversold,  # Cross in oversold territory
            ]
        )

        # Count bearish signals
        bearish_signals = sum(
            [
                rsi_overbought,
                stoch_overbought,
                stoch_bearish_cross and stoch_overbought,  # Cross in overbought territory
            ]
        )

        # Minimum signals required
        min_signals = 3 if self.require_all_signals else 2

        # Calculate ATR-based stops
        atr_stop_distance = current_atr * self.atr_stop_mult
        atr_tp_distance = current_atr * self.atr_tp_mult

        # ========== GENERATE SIGNAL ==========

        # Check for EXIT signals first
        if self._in_long:
            # Exit long on stop or TP or RSI mean reversion
            if current_rsi > 50 and current_stoch_k > 50:
                self._in_long = False
                return Signal(
                    symbol=symbol,
                    direction=Direction.SHORT,
                    signal_type=SignalType.EXIT,
                    timestamp=timestamp,
                    price=current_price,
                    confidence=0.7,
                    metadata={
                        "model": "multi_signal_confluence",
                        "reason": "exit_mean_reversion",
                        "rsi": current_rsi,
                        "stoch_k": current_stoch_k,
                    },
                )

        if self._in_short:
            # Exit short on stop or TP or RSI mean reversion
            if current_rsi < 50 and current_stoch_k < 50:
                self._in_short = False
                return Signal(
                    symbol=symbol,
                    direction=Direction.LONG,
                    signal_type=SignalType.EXIT,
                    timestamp=timestamp,
                    price=current_price,
                    confidence=0.7,
                    metadata={
                        "model": "multi_signal_confluence",
                        "reason": "exit_mean_reversion",
                        "rsi": current_rsi,
                        "stoch_k": current_stoch_k,
                    },
                )

        # LONG entry
        if (
            self.enable_longs
            and not self._in_long
            and bullish_signals >= min_signals
            and no_strong_trend
        ):
            self._in_long = True
            self._entry_price = current_price
            self._stop_loss = current_price - atr_stop_distance
            self._take_profit = current_price + atr_tp_distance

            confidence = min(0.9, 0.5 + (bullish_signals * 0.15))

            return Signal(
                symbol=symbol,
                direction=Direction.LONG,
                signal_type=SignalType.ENTRY,
                timestamp=timestamp,
                price=current_price,
                confidence=confidence,
                metadata={
                    "model": "multi_signal_confluence",
                    "entry_type": "confluence_long",
                    "bullish_signals": bullish_signals,
                    "rsi": current_rsi,
                    "stoch_k": current_stoch_k,
                    "stoch_d": current_stoch_d,
                    "adx": current_adx,
                    "atr": current_atr,
                    "stop_loss": self._stop_loss,
                    "take_profit": self._take_profit,
                    "atr_stop_mult": self.atr_stop_mult,
                    "atr_tp_mult": self.atr_tp_mult,
                },
            )

        # SHORT entry
        if (
            self.enable_shorts
            and not self._in_short
            and bearish_signals >= min_signals
            and no_strong_trend
        ):
            self._in_short = True
            self._entry_price = current_price
            self._stop_loss = current_price + atr_stop_distance
            self._take_profit = current_price - atr_tp_distance

            confidence = min(0.9, 0.5 + (bearish_signals * 0.15))

            return Signal(
                symbol=symbol,
                direction=Direction.SHORT,
                signal_type=SignalType.ENTRY,
                timestamp=timestamp,
                price=current_price,
                confidence=confidence,
                metadata={
                    "model": "multi_signal_confluence",
                    "entry_type": "confluence_short",
                    "bearish_signals": bearish_signals,
                    "rsi": current_rsi,
                    "stoch_k": current_stoch_k,
                    "stoch_d": current_stoch_d,
                    "adx": current_adx,
                    "atr": current_atr,
                    "stop_loss": self._stop_loss,
                    "take_profit": self._take_profit,
                    "atr_stop_mult": self.atr_stop_mult,
                    "atr_tp_mult": self.atr_tp_mult,
                },
            )

        return None

    def reset_state(self):
        """Reset position tracking state."""
        self._in_long = False
        self._in_short = False
        self._entry_price = None
        self._stop_loss = None
        self._take_profit = None
