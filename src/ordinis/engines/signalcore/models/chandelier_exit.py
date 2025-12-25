"""
Chandelier Exit Model.

Implements a trailing stop based on ATR that "hangs" from the highest high
(for longs) or lowest low (for shorts). Used to lock in profits while
allowing trends to run.

Reference: Chuck LeBeau, introduced in the 1990s
"""

from datetime import datetime, timedelta
from enum import Enum

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType


class ExitMode(str, Enum):
    """Exit mode for Chandelier calculation."""
    LONG = "long"  # Exit hangs below highest high
    SHORT = "short"  # Exit hangs above lowest low


class ChandelierExitModel(Model):
    """
    Chandelier Exit Trailing Stop Model.

    Calculates a dynamic trailing stop that:
    - For LONG positions: Highest High (lookback) - ATR × multiplier
    - For SHORT positions: Lowest Low (lookback) + ATR × multiplier

    Parameters:
        atr_period: ATR calculation period (default 22)
        atr_multiplier: ATR multiplier for stop distance (default 3.0)
        lookback: Bars for highest high/lowest low (default 22)
        exit_mode: "long" or "short" (default "long")

    Signals:
        - EXIT when price crosses below/above the chandelier level
        - HOLD when price is safely within the trailing stop
        - Metadata includes chandelier_level for position management

    Usage:
        Activate after an initial profit target is hit. The chandelier
        "hangs" from the extreme price and follows the trend, tightening
        the stop as the trend progresses.

    Reference:
        LeBeau, Chuck. Technical Traders Guide to Computer Analysis
    """

    def __init__(self, config: ModelConfig):
        """Initialize Chandelier Exit model."""
        super().__init__(config)

        params = self.config.parameters
        self.atr_period = params.get("atr_period", 22)
        self.atr_multiplier = params.get("atr_multiplier", 3.0)
        self.lookback = params.get("lookback", 22)
        self.exit_mode = ExitMode(params.get("exit_mode", "long"))

        # Minimum data points
        self.config.min_data_points = max(self.atr_period, self.lookback) + 10

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate exit signal based on Chandelier Exit.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with chandelier exit level and exit recommendation
        """
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        if "symbol" in data:
            symbol_data = data["symbol"]
            symbol = symbol_data.iloc[0] if hasattr(symbol_data, "iloc") else str(symbol_data)
        else:
            symbol = "UNKNOWN"

        def _as_series(col):
            if isinstance(col, pd.DataFrame):
                s = col.iloc[:, 0]
            else:
                s = col
            return pd.to_numeric(s, errors="coerce")

        high = _as_series(data["high"])
        low = _as_series(data["low"])
        close = _as_series(data["close"])

        # Calculate ATR
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()

        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]

        # Calculate Chandelier Exit level
        if self.exit_mode == ExitMode.LONG:
            # For longs: exit hangs below highest high
            highest_high = high.rolling(window=self.lookback).max().iloc[-1]
            chandelier_level = highest_high - (current_atr * self.atr_multiplier)
            
            # Check if price crossed below the chandelier level
            exit_triggered = current_price < chandelier_level
            distance_to_exit = current_price - chandelier_level
            distance_percent = (distance_to_exit / current_price) * 100
            
        else:  # SHORT
            # For shorts: exit hangs above lowest low
            lowest_low = low.rolling(window=self.lookback).min().iloc[-1]
            chandelier_level = lowest_low + (current_atr * self.atr_multiplier)
            
            # Check if price crossed above the chandelier level
            exit_triggered = current_price > chandelier_level
            distance_to_exit = chandelier_level - current_price
            distance_percent = (distance_to_exit / current_price) * 100

        # Determine signal
        if exit_triggered:
            signal_type = SignalType.EXIT
            direction = Direction.NEUTRAL
            score = -0.8  # Strong exit signal
            probability = 0.85
        else:
            signal_type = SignalType.HOLD
            direction = Direction.LONG if self.exit_mode == ExitMode.LONG else Direction.SHORT
            score = 0.0
            probability = 0.5

        # Calculate confidence interval
        returns = close.pct_change().dropna()
        recent_vol = returns.tail(20).std()
        confidence_interval = (-2 * recent_vol, 2 * recent_vol)

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
            expected_return=0.0,
            confidence_interval=confidence_interval,
            score=score,
            model_id=self.config.model_id,
            model_version=self.config.version,
            feature_contributions={
                "atr": float(current_atr),
                "chandelier_level": float(chandelier_level),
                "distance_to_exit": float(distance_to_exit),
                "exit_triggered": 1.0 if exit_triggered else 0.0,
            },
            regime="trailing_stop",
            data_quality=1.0,
            staleness=staleness,
            metadata={
                "chandelier_level": float(chandelier_level),
                "current_price": float(current_price),
                "atr": float(current_atr),
                "atr_multiplier": self.atr_multiplier,
                "lookback": self.lookback,
                "exit_mode": self.exit_mode.value,
                "exit_triggered": exit_triggered,
                "distance_to_exit": float(distance_to_exit),
                "distance_percent": float(distance_percent),
                "highest_high": float(highest_high) if self.exit_mode == ExitMode.LONG else None,
                "lowest_low": float(lowest_low) if self.exit_mode == ExitMode.SHORT else None,
            },
        )

    def calculate_stop_level(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        mode: ExitMode | None = None,
    ) -> float:
        """
        Calculate the current Chandelier stop level.

        This is a utility method for external use without generating a full signal.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            mode: Override exit mode (optional)

        Returns:
            Current Chandelier stop level
        """
        mode = mode or self.exit_mode

        # Calculate ATR
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean().iloc[-1]

        if mode == ExitMode.LONG:
            highest_high = high.rolling(window=self.lookback).max().iloc[-1]
            return highest_high - (atr * self.atr_multiplier)
        else:
            lowest_low = low.rolling(window=self.lookback).min().iloc[-1]
            return lowest_low + (atr * self.atr_multiplier)
