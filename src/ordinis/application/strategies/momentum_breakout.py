"""
Momentum Breakout Strategy.

Trades breakouts from consolidation with momentum confirmation.
"""

from datetime import datetime

import pandas as pd

from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType

from .base import BaseStrategy


class MomentumBreakoutStrategy(BaseStrategy):
    """
    Momentum Breakout Strategy.

    Identifies price breakouts from consolidation periods and enters
    when confirmed by volume and momentum indicators.

    Default Parameters:
    - lookback_period: 20 (period for high/low range)
    - atr_period: 14 (Average True Range period)
    - volume_multiplier: 1.5 (volume threshold for breakout confirmation)
    - min_consolidation_bars: 10 (minimum bars in consolidation)
    - breakout_threshold: 0.02 (2% breakout above high/below low)
    """

    def configure(self):
        """Configure momentum breakout parameters."""
        self.params.setdefault("lookback_period", 20)
        self.params.setdefault("atr_period", 14)
        self.params.setdefault("volume_multiplier", 1.5)
        self.params.setdefault("min_consolidation_bars", 10)
        self.params.setdefault("breakout_threshold", 0.02)
        self.params.setdefault(
            "min_bars", max(self.params["lookback_period"], self.params["atr_period"])
        )

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            data: OHLC data
            period: ATR period

        Returns:
            ATR series
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate momentum breakout signal.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal object or None
        """
        # Validate data
        is_valid, _msg = self.validate_data(data)
        if not is_valid:
            return None

        try:
            close = data["close"]
            high = data["high"]
            low = data["low"]
            volume = data["volume"]

            # Calculate indicators
            lookback = self.params["lookback_period"]
            highest_high = high.rolling(window=lookback).max()
            lowest_low = low.rolling(window=lookback).min()
            avg_volume = volume.rolling(window=lookback).mean()
            atr = self._calculate_atr(data, self.params["atr_period"])

            # Get current values
            current_close = close.iloc[-1]
            current_volume = volume.iloc[-1]
            current_high = highest_high.iloc[-1]
            current_low = lowest_low.iloc[-1]
            current_atr = atr.iloc[-1]
            current_avg_volume = avg_volume.iloc[-1]

            # Calculate range and volatility
            price_range = current_high - current_low
            range_pct = price_range / current_close

            # Volume confirmation
            volume_surge = current_volume > (current_avg_volume * self.params["volume_multiplier"])

            # Breakout detection
            breakout_threshold = self.params["breakout_threshold"]

            # Upside breakout
            if current_close > current_high * (1 + breakout_threshold):
                if volume_surge:
                    # Strong breakout with volume confirmation
                    distance_from_high = (current_close - current_high) / current_high
                    probability = min(0.75, 0.55 + (distance_from_high * 10))

                    # Extract symbol safely
                    if "symbol" in data:
                        symbol_data = data["symbol"]
                        symbol = (
                            symbol_data.iloc[0]
                            if hasattr(symbol_data, "iloc")
                            else str(symbol_data)
                        )
                    else:
                        symbol = "UNKNOWN"

                    return Signal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.ENTRY,
                        direction=Direction.LONG,
                        probability=probability,
                        expected_return=current_atr / current_close,
                        confidence_interval=(
                            -(current_atr / current_close),
                            3 * (current_atr / current_close),
                        ),
                        score=(distance_from_high + (current_volume / current_avg_volume)) * 50,
                        model_id=self.name,
                        model_version="1.0.0",
                        metadata={
                            "strategy": self.name,
                            "breakout_type": "upside",
                            "distance_from_high": distance_from_high,
                            "volume_ratio": current_volume / current_avg_volume,
                            "range_pct": range_pct,
                            "atr": current_atr,
                            "stop_loss": current_close - (2 * current_atr),
                            "take_profit": current_close + (3 * current_atr),
                        },
                    )

            # Downside breakout (short opportunity)
            if current_close < current_low * (1 - breakout_threshold):
                if volume_surge:
                    # Strong breakdown with volume confirmation
                    distance_from_low = (current_low - current_close) / current_low
                    probability = min(0.75, 0.55 + (distance_from_low * 10))

                    # Extract symbol safely
                    if "symbol" in data:
                        symbol_data = data["symbol"]
                        symbol = (
                            symbol_data.iloc[0]
                            if hasattr(symbol_data, "iloc")
                            else str(symbol_data)
                        )
                    else:
                        symbol = "UNKNOWN"

                    return Signal(
                        symbol=symbol,
                        timestamp=timestamp,
                        signal_type=SignalType.ENTRY,
                        direction=Direction.SHORT,
                        probability=probability,
                        expected_return=-(current_atr / current_close),
                        confidence_interval=(
                            -3 * (current_atr / current_close),
                            current_atr / current_close,
                        ),
                        score=(distance_from_low + (current_volume / current_avg_volume)) * 50,
                        model_id=self.name,
                        model_version="1.0.0",
                        metadata={
                            "strategy": self.name,
                            "breakout_type": "downside",
                            "distance_from_low": distance_from_low,
                            "volume_ratio": current_volume / current_avg_volume,
                            "range_pct": range_pct,
                            "atr": current_atr,
                            "stop_loss": current_close + (2 * current_atr),
                            "take_profit": current_close - (3 * current_atr),
                        },
                    )

            # Consolidation detection (no signal, but useful for context)
            if range_pct < 0.02:  # Less than 2% range
                # Extract symbol safely
                if "symbol" in data:
                    symbol_data = data["symbol"]
                    symbol = (
                        symbol_data.iloc[0] if hasattr(symbol_data, "iloc") else str(symbol_data)
                    )
                else:
                    symbol = "UNKNOWN"

                return Signal(
                    symbol=symbol,
                    timestamp=timestamp,
                    signal_type=SignalType.HOLD,
                    direction=Direction.NEUTRAL,
                    probability=0.5,
                    expected_return=0.0,
                    confidence_interval=(-0.01, 0.01),
                    score=0.0,
                    model_id=self.name,
                    model_version="1.0.0",
                    metadata={
                        "strategy": self.name,
                        "market_state": "consolidation",
                        "range_pct": range_pct,
                        "awaiting_breakout": True,
                    },
                )

            return None

        except Exception:
            return None

    def get_description(self) -> str:
        """Get strategy description."""
        return f"""Momentum Breakout Strategy

Trades price breakouts from consolidation with momentum confirmation.

Entry Rules:
- BUY when price breaks above {self.params['lookback_period']}-period high by {self.params['breakout_threshold']*100}%
- SHORT when price breaks below {self.params['lookback_period']}-period low by {self.params['breakout_threshold']*100}%
- Volume must be {self.params['volume_multiplier']}x average volume

Exit Rules:
- Take profit at 3x ATR from entry
- Stop loss at 2x ATR from entry
- Exit on failed breakout (return to range)

Parameters:
- Lookback Period: {self.params['lookback_period']} bars
- ATR Period: {self.params['atr_period']} bars
- Volume Multiplier: {self.params['volume_multiplier']}x
- Breakout Threshold: {self.params['breakout_threshold']*100}%

Best For:
- Range-bound markets transitioning to trends
- High volatility breakouts
- News-driven price movements
- Short to medium-term trades

Risk Considerations:
- False breakouts are common
- Requires tight stops to limit losses
- Can experience whipsaws in choppy markets
- Best combined with broader market trend analysis
"""

    def get_required_bars(self) -> int:
        """Get minimum bars required."""
        return self.params.get(
            "min_bars",
            max(self.params["lookback_period"], self.params["atr_period"]) + 10,
        )
