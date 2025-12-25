"""
Multi-Timeframe Alignment Model.

Provides higher timeframe trend confirmation for lower timeframe signals.
Ensures trades align with the dominant market direction.
"""

from datetime import datetime, timedelta
from enum import Enum, auto

import numpy as np
import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType


class TimeframeAlignment(Enum):
    """Alignment state between timeframes."""
    
    ALIGNED_BULLISH = auto()  # Both timeframes bullish
    ALIGNED_BEARISH = auto()  # Both timeframes bearish
    COUNTER_TREND = auto()    # Trading against higher TF
    NEUTRAL = auto()          # Higher TF unclear


class MTFAlignmentModel(Model):
    """
    Multi-Timeframe Alignment Model.

    Analyzes higher timeframe trend to filter lower timeframe signals:
    - Higher TF trend via SMA (price > SMA = bullish)
    - Higher TF momentum via SMA slope
    - Filters counter-trend signals

    Parameters:
        htf_sma_period: SMA period for higher TF trend (default 50)
        htf_multiplier: Multiplier to create higher TF bars (default 5)
            e.g., if trading on 1-hour, 5x = 5-hour equivalent
        slope_period: Bars for slope calculation (default 5)
        min_slope: Minimum slope for trend confirmation (default 0.0001)
        require_alignment: Reject counter-trend signals (default True)

    Signals:
        - ENTRY when aligned with higher TF
        - HOLD when counter-trend or unclear
        - Metadata includes alignment details

    Usage:
        Use as a filter before taking Fibonacci entries.
        Only take longs when HTF is bullish, shorts when HTF is bearish.

    Example:
        Trading 1-hour Fibonacci entries:
        - HTF = 5-hour (1h * 5)
        - Only long when 5-hour trend is up
        - Only short when 5-hour trend is down
    """

    def __init__(self, config: ModelConfig):
        """Initialize MTF Alignment model."""
        super().__init__(config)

        params = self.config.parameters
        self.htf_sma_period = params.get("htf_sma_period", 50)
        self.htf_multiplier = params.get("htf_multiplier", 5)
        self.slope_period = params.get("slope_period", 5)
        self.min_slope = params.get("min_slope", 0.0001)
        self.require_alignment = params.get("require_alignment", True)

        # Need enough data for HTF construction
        self.config.min_data_points = (self.htf_sma_period + self.slope_period) * self.htf_multiplier + 10

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate signal with MTF alignment analysis.

        Args:
            data: Historical OHLCV data (lower timeframe)
            timestamp: Current timestamp

        Returns:
            Signal with alignment metadata
        """
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        symbol = self._extract_symbol(data)

        close = pd.to_numeric(data["close"].squeeze(), errors="coerce")
        high = pd.to_numeric(data["high"].squeeze(), errors="coerce")
        low = pd.to_numeric(data["low"].squeeze(), errors="coerce")

        current_price = float(close.iloc[-1])

        # Construct higher timeframe data
        htf_data = self._construct_htf_data(data)
        
        if htf_data is None or len(htf_data) < self.htf_sma_period + self.slope_period:
            return self._create_neutral_signal(symbol, timestamp, current_price, "Insufficient HTF data")

        htf_close = htf_data["close"]

        # Calculate HTF SMA
        htf_sma = htf_close.rolling(window=self.htf_sma_period).mean()
        current_htf_sma = float(htf_sma.iloc[-1])
        current_htf_price = float(htf_close.iloc[-1])

        # Calculate HTF SMA slope
        if len(htf_sma.dropna()) >= self.slope_period:
            sma_values = htf_sma.dropna().tail(self.slope_period)
            slope = (sma_values.iloc[-1] - sma_values.iloc[0]) / len(sma_values) / current_htf_price
        else:
            slope = 0.0

        # Determine HTF trend
        htf_bullish = current_htf_price > current_htf_sma and slope > self.min_slope
        htf_bearish = current_htf_price < current_htf_sma and slope < -self.min_slope
        htf_neutral = not htf_bullish and not htf_bearish

        # Determine LTF trend (simple: recent close vs short SMA)
        ltf_sma = close.rolling(window=20).mean()
        ltf_bullish = current_price > float(ltf_sma.iloc[-1]) if not pd.isna(ltf_sma.iloc[-1]) else False
        ltf_bearish = current_price < float(ltf_sma.iloc[-1]) if not pd.isna(ltf_sma.iloc[-1]) else False

        # Determine alignment
        if htf_bullish and ltf_bullish:
            alignment = TimeframeAlignment.ALIGNED_BULLISH
            direction = Direction.LONG
            signal_type = SignalType.ENTRY
            score = 0.8
        elif htf_bearish and ltf_bearish:
            alignment = TimeframeAlignment.ALIGNED_BEARISH
            direction = Direction.SHORT
            signal_type = SignalType.ENTRY
            score = -0.8
        elif htf_neutral:
            alignment = TimeframeAlignment.NEUTRAL
            direction = Direction.NEUTRAL
            signal_type = SignalType.HOLD
            score = 0.0
        else:
            # Counter-trend
            alignment = TimeframeAlignment.COUNTER_TREND
            direction = Direction.NEUTRAL if self.require_alignment else (Direction.LONG if ltf_bullish else Direction.SHORT)
            signal_type = SignalType.HOLD if self.require_alignment else SignalType.ENTRY
            score = 0.3 if ltf_bullish else -0.3

        # Calculate probability
        alignment_strength = abs(current_htf_price - current_htf_sma) / current_htf_sma
        probability = 0.5 + min(alignment_strength * 5, 0.3)  # 0.5 to 0.8

        # Feature contributions
        feature_contributions = {
            "htf_trend": 1.0 if htf_bullish else (-1.0 if htf_bearish else 0.0),
            "ltf_trend": 1.0 if ltf_bullish else (-1.0 if ltf_bearish else 0.0),
            "htf_sma_distance": float((current_htf_price - current_htf_sma) / current_htf_sma),
            "htf_slope": float(slope),
            "alignment_score": 1.0 if alignment in [TimeframeAlignment.ALIGNED_BULLISH, TimeframeAlignment.ALIGNED_BEARISH] else 0.0,
        }

        # Regime
        if htf_bullish:
            regime = "htf_bullish"
        elif htf_bearish:
            regime = "htf_bearish"
        else:
            regime = "htf_neutral"

        # Data quality
        data_quality = 1.0 - (close.isnull().sum() / len(close))

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
            expected_return=0.02 if alignment in [TimeframeAlignment.ALIGNED_BULLISH, TimeframeAlignment.ALIGNED_BEARISH] else 0.0,
            confidence_interval=(-0.03, 0.05),
            score=score,
            model_id=self.config.model_id,
            model_version=self.config.version,
            feature_contributions=feature_contributions,
            regime=regime,
            data_quality=data_quality,
            staleness=staleness,
            metadata={
                "alignment": alignment.name,
                "is_aligned": alignment in [TimeframeAlignment.ALIGNED_BULLISH, TimeframeAlignment.ALIGNED_BEARISH],
                "htf_bullish": htf_bullish,
                "htf_bearish": htf_bearish,
                "htf_neutral": htf_neutral,
                "ltf_bullish": ltf_bullish,
                "ltf_bearish": ltf_bearish,
                "htf_sma": current_htf_sma,
                "htf_price": current_htf_price,
                "htf_slope": float(slope),
                "current_price": current_price,
                "htf_multiplier": self.htf_multiplier,
                "htf_bars_used": len(htf_data),
            },
        )

    def _construct_htf_data(self, data: pd.DataFrame) -> pd.DataFrame | None:
        """
        Construct higher timeframe OHLCV from lower timeframe data.

        Groups every N bars together where N = htf_multiplier.

        Args:
            data: Lower timeframe OHLCV data

        Returns:
            Higher timeframe OHLCV data or None if insufficient data
        """
        n = len(data)
        multiplier = self.htf_multiplier

        if n < multiplier * (self.htf_sma_period + 5):
            return None

        # Create HTF bars by grouping
        htf_bars = []
        
        for i in range(0, n - multiplier + 1, multiplier):
            chunk = data.iloc[i:i + multiplier]
            
            htf_bar = {
                "open": float(pd.to_numeric(chunk["open"].iloc[0], errors="coerce")),
                "high": float(pd.to_numeric(chunk["high"], errors="coerce").max()),
                "low": float(pd.to_numeric(chunk["low"], errors="coerce").min()),
                "close": float(pd.to_numeric(chunk["close"].iloc[-1], errors="coerce")),
            }
            
            if "volume" in chunk.columns:
                htf_bar["volume"] = float(pd.to_numeric(chunk["volume"], errors="coerce").sum())
            
            htf_bars.append(htf_bar)

        if not htf_bars:
            return None

        htf_df = pd.DataFrame(htf_bars)
        return htf_df

    def _create_neutral_signal(
        self, symbol: str, timestamp: datetime, current_price: float, reason: str
    ) -> Signal:
        """Create a neutral/hold signal."""
        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=SignalType.HOLD,
            direction=Direction.NEUTRAL,
            probability=0.5,
            expected_return=0.0,
            confidence_interval=(-0.02, 0.02),
            score=0.0,
            model_id=self.config.model_id,
            model_version=self.config.version,
            feature_contributions={},
            regime="unknown",
            data_quality=1.0,
            staleness=timedelta(seconds=0),
            metadata={
                "alignment": TimeframeAlignment.NEUTRAL.name,
                "is_aligned": False,
                "reason": reason,
                "current_price": current_price,
            },
        )

    def _extract_symbol(self, data: pd.DataFrame) -> str:
        """Extract symbol from data."""
        if "symbol" in data:
            symbol_data = data["symbol"]
            return symbol_data.iloc[0] if hasattr(symbol_data, "iloc") else str(symbol_data)
        return "UNKNOWN"

    def check_alignment(
        self, data: pd.DataFrame, desired_direction: Direction
    ) -> tuple[bool, TimeframeAlignment]:
        """
        Synchronous helper to check if a direction aligns with HTF trend.

        Args:
            data: OHLCV data
            desired_direction: Direction to check (LONG or SHORT)

        Returns:
            Tuple of (is_aligned: bool, alignment: TimeframeAlignment)
        """
        close = pd.to_numeric(data["close"].squeeze(), errors="coerce")
        current_price = float(close.iloc[-1])

        htf_data = self._construct_htf_data(data)
        
        if htf_data is None or len(htf_data) < self.htf_sma_period:
            return False, TimeframeAlignment.NEUTRAL

        htf_close = htf_data["close"]
        htf_sma = htf_close.rolling(window=self.htf_sma_period).mean()
        
        if pd.isna(htf_sma.iloc[-1]):
            return False, TimeframeAlignment.NEUTRAL

        current_htf_sma = float(htf_sma.iloc[-1])
        current_htf_price = float(htf_close.iloc[-1])

        htf_bullish = current_htf_price > current_htf_sma
        htf_bearish = current_htf_price < current_htf_sma

        if desired_direction == Direction.LONG:
            if htf_bullish:
                return True, TimeframeAlignment.ALIGNED_BULLISH
            elif htf_bearish:
                return False, TimeframeAlignment.COUNTER_TREND
            else:
                return False, TimeframeAlignment.NEUTRAL
        elif desired_direction == Direction.SHORT:
            if htf_bearish:
                return True, TimeframeAlignment.ALIGNED_BEARISH
            elif htf_bullish:
                return False, TimeframeAlignment.COUNTER_TREND
            else:
                return False, TimeframeAlignment.NEUTRAL
        else:
            return False, TimeframeAlignment.NEUTRAL
