"""
Fractal Swing Detection Model.

Implements fractal-based swing high/low detection for more robust
identification of significant price pivots used in Fibonacci analysis.
"""

from datetime import datetime, timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType


class SwingPoint(NamedTuple):
    """A detected swing high or low point."""
    
    index: int  # Bar index in the data
    price: float  # Price level
    is_high: bool  # True for swing high, False for swing low
    strength: int  # Number of bars on each side that confirm the swing
    timestamp: datetime | None = None


class FractalSwingModel(Model):
    """
    Fractal-Based Swing Detection Model.

    Uses Williams Fractals and extensions to identify significant swing
    highs and lows that form the basis for Fibonacci retracement levels.

    A fractal high is formed when:
        - N bars before have lower highs
        - N bars after have lower highs
        - The center bar is the local maximum

    Parameters:
        fractal_period: Bars on each side for fractal confirmation (default 2)
        min_swing_bars: Minimum bars between valid swings (default 5)
        confirmation_bars: Extra bars to wait for confirmation (default 0)
        strength_lookback: Extended lookback for strength scoring (default 10)
        min_swing_pct: Minimum swing size as % of price (default 0.02)

    Signals:
        - Provides swing high/low points for Fibonacci calculation
        - Higher strength = more significant swing

    Usage:
        Use output swing points as inputs to Fibonacci retracement model.
        More reliable than simple rolling min/max.

    Reference:
        Williams Fractals - Bill Williams
        Swing Point Analysis - Technical Analysis foundations
    """

    def __init__(self, config: ModelConfig):
        """Initialize Fractal Swing model."""
        super().__init__(config)

        params = self.config.parameters
        self.fractal_period = params.get("fractal_period", 2)
        self.min_swing_bars = params.get("min_swing_bars", 5)
        self.confirmation_bars = params.get("confirmation_bars", 0)
        self.strength_lookback = params.get("strength_lookback", 10)
        self.min_swing_pct = params.get("min_swing_pct", 0.02)

        # Need enough bars for fractal detection + confirmation
        self.config.min_data_points = (
            self.fractal_period * 2 + self.confirmation_bars + self.strength_lookback + 10
        )

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate signal with detected swing points.

        The signal metadata contains the most recent swing high and low,
        which can be used for Fibonacci level calculation.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with swing point metadata
        """
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        symbol = self._extract_symbol(data)

        high = pd.to_numeric(data["high"].squeeze(), errors="coerce")
        low = pd.to_numeric(data["low"].squeeze(), errors="coerce")
        close = pd.to_numeric(data["close"].squeeze(), errors="coerce")

        current_price = float(close.iloc[-1])

        # Detect all swing points
        swing_highs = self._detect_fractal_highs(high, data.index if isinstance(data.index, pd.DatetimeIndex) else None)
        swing_lows = self._detect_fractal_lows(low, data.index if isinstance(data.index, pd.DatetimeIndex) else None)

        # Filter by minimum swing size
        valid_highs = [s for s in swing_highs if self._is_significant_swing(s, swing_lows, current_price)]
        valid_lows = [s for s in swing_lows if self._is_significant_swing(s, swing_highs, current_price)]

        # Get most recent valid swing high and low
        recent_high = valid_highs[-1] if valid_highs else None
        recent_low = valid_lows[-1] if valid_lows else None

        # Determine trend based on swing sequence
        if recent_high and recent_low:
            if recent_high.index > recent_low.index:
                trend = "bullish"
                direction = Direction.LONG
            else:
                trend = "bearish"
                direction = Direction.SHORT
            swing_range = recent_high.price - recent_low.price
            swing_pct = swing_range / recent_low.price if recent_low.price > 0 else 0
        else:
            trend = "undefined"
            direction = Direction.NEUTRAL
            swing_range = 0.0
            swing_pct = 0.0

        # Calculate average swing strength
        avg_strength = 0.0
        if recent_high and recent_low:
            avg_strength = (recent_high.strength + recent_low.strength) / 2.0

        # Score based on swing quality
        score = min(avg_strength / self.strength_lookback, 1.0)  # Normalize to 0-1

        # Feature contributions
        feature_contributions = {
            "swing_high_strength": float(recent_high.strength) if recent_high else 0.0,
            "swing_low_strength": float(recent_low.strength) if recent_low else 0.0,
            "swing_range_pct": float(swing_pct),
            "num_highs": float(len(valid_highs)),
            "num_lows": float(len(valid_lows)),
        }

        # Regime based on swing characteristics
        if swing_pct > 0.15:
            regime = "high_volatility"
        elif swing_pct > 0.05:
            regime = "trending"
        else:
            regime = "consolidation"

        # Data quality
        data_quality = 1.0 - (high.isnull().sum() / len(high))

        # Staleness
        if isinstance(data.index, pd.DatetimeIndex):
            delta = timestamp - data.index[-1]
            staleness = timedelta(seconds=delta.total_seconds())
        else:
            staleness = timedelta(seconds=0)

        # Convert timestamps for metadata
        high_ts = recent_high.timestamp.isoformat() if recent_high and recent_high.timestamp else None
        low_ts = recent_low.timestamp.isoformat() if recent_low and recent_low.timestamp else None

        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=SignalType.HOLD,  # This model provides data, not trading signals
            direction=direction,
            probability=0.5 + (score * 0.2),
            expected_return=0.0,
            confidence_interval=(-0.02, 0.02),
            score=score if direction == Direction.LONG else -score,
            model_id=self.config.model_id,
            model_version=self.config.version,
            feature_contributions=feature_contributions,
            regime=regime,
            data_quality=data_quality,
            staleness=staleness,
            metadata={
                "swing_high": float(recent_high.price) if recent_high else None,
                "swing_high_index": int(recent_high.index) if recent_high else None,
                "swing_high_strength": int(recent_high.strength) if recent_high else None,
                "swing_high_timestamp": high_ts,
                "swing_low": float(recent_low.price) if recent_low else None,
                "swing_low_index": int(recent_low.index) if recent_low else None,
                "swing_low_strength": int(recent_low.strength) if recent_low else None,
                "swing_low_timestamp": low_ts,
                "swing_range": float(swing_range),
                "swing_pct": float(swing_pct),
                "trend": trend,
                "current_price": float(current_price),
                "num_swing_highs": len(valid_highs),
                "num_swing_lows": len(valid_lows),
                "all_swing_highs": [{"index": s.index, "price": s.price, "strength": s.strength} for s in valid_highs[-5:]],
                "all_swing_lows": [{"index": s.index, "price": s.price, "strength": s.strength} for s in valid_lows[-5:]],
            },
        )

    def _detect_fractal_highs(
        self, high: pd.Series, index: pd.DatetimeIndex | None = None
    ) -> list[SwingPoint]:
        """
        Detect fractal highs in the data.

        A fractal high at bar i requires:
            high[i] > high[i-1], high[i-2], ..., high[i-n]
            high[i] > high[i+1], high[i+2], ..., high[i+n]

        Args:
            high: High price series
            index: Optional datetime index for timestamps

        Returns:
            List of SwingPoint objects for detected highs
        """
        swing_highs = []
        n = len(high)
        period = self.fractal_period

        # Need confirmation bars after the swing
        end_idx = n - period - self.confirmation_bars

        for i in range(period, end_idx):
            is_fractal = True
            strength = 0

            # Check bars before
            for j in range(1, period + 1):
                if high.iloc[i] <= high.iloc[i - j]:
                    is_fractal = False
                    break
                strength += 1

            if not is_fractal:
                continue

            # Check bars after
            for j in range(1, period + 1):
                if high.iloc[i] <= high.iloc[i + j]:
                    is_fractal = False
                    break
                strength += 1

            if is_fractal:
                # Calculate extended strength
                extended_strength = self._calculate_extended_strength(high, i, is_high=True)
                
                ts = index[i] if index is not None else None
                swing_highs.append(SwingPoint(
                    index=i,
                    price=float(high.iloc[i]),
                    is_high=True,
                    strength=extended_strength,
                    timestamp=ts,
                ))

        # Filter out swings too close together
        return self._filter_nearby_swings(swing_highs, is_high=True)

    def _detect_fractal_lows(
        self, low: pd.Series, index: pd.DatetimeIndex | None = None
    ) -> list[SwingPoint]:
        """
        Detect fractal lows in the data.

        A fractal low at bar i requires:
            low[i] < low[i-1], low[i-2], ..., low[i-n]
            low[i] < low[i+1], low[i+2], ..., low[i+n]

        Args:
            low: Low price series
            index: Optional datetime index for timestamps

        Returns:
            List of SwingPoint objects for detected lows
        """
        swing_lows = []
        n = len(low)
        period = self.fractal_period

        end_idx = n - period - self.confirmation_bars

        for i in range(period, end_idx):
            is_fractal = True
            strength = 0

            # Check bars before
            for j in range(1, period + 1):
                if low.iloc[i] >= low.iloc[i - j]:
                    is_fractal = False
                    break
                strength += 1

            if not is_fractal:
                continue

            # Check bars after
            for j in range(1, period + 1):
                if low.iloc[i] >= low.iloc[i + j]:
                    is_fractal = False
                    break
                strength += 1

            if is_fractal:
                extended_strength = self._calculate_extended_strength(low, i, is_high=False)
                
                ts = index[i] if index is not None else None
                swing_lows.append(SwingPoint(
                    index=i,
                    price=float(low.iloc[i]),
                    is_high=False,
                    strength=extended_strength,
                    timestamp=ts,
                ))

        return self._filter_nearby_swings(swing_lows, is_high=False)

    def _calculate_extended_strength(
        self, series: pd.Series, idx: int, is_high: bool
    ) -> int:
        """
        Calculate extended strength by counting how many bars the swing dominates.

        Args:
            series: Price series (high for swing highs, low for swing lows)
            idx: Index of the swing point
            is_high: True for swing high, False for swing low

        Returns:
            Strength score (higher = more significant swing)
        """
        strength = 0
        pivot_price = series.iloc[idx]

        # Check backwards
        for i in range(1, min(self.strength_lookback, idx) + 1):
            if is_high:
                if series.iloc[idx - i] < pivot_price:
                    strength += 1
                else:
                    break
            else:
                if series.iloc[idx - i] > pivot_price:
                    strength += 1
                else:
                    break

        # Check forwards
        for i in range(1, min(self.strength_lookback, len(series) - idx - 1) + 1):
            if is_high:
                if series.iloc[idx + i] < pivot_price:
                    strength += 1
                else:
                    break
            else:
                if series.iloc[idx + i] > pivot_price:
                    strength += 1
                else:
                    break

        return strength

    def _filter_nearby_swings(
        self, swings: list[SwingPoint], is_high: bool
    ) -> list[SwingPoint]:
        """
        Filter out swings that are too close together, keeping the stronger one.

        Args:
            swings: List of detected swings
            is_high: True for swing highs, False for swing lows

        Returns:
            Filtered list of swings
        """
        if not swings or len(swings) < 2:
            return swings

        filtered = []
        i = 0

        while i < len(swings):
            current = swings[i]
            
            # Look ahead for nearby swings
            j = i + 1
            best = current
            
            while j < len(swings) and swings[j].index - current.index < self.min_swing_bars:
                if swings[j].strength > best.strength:
                    best = swings[j]
                elif swings[j].strength == best.strength:
                    # Same strength: keep the more extreme price
                    if is_high and swings[j].price > best.price:
                        best = swings[j]
                    elif not is_high and swings[j].price < best.price:
                        best = swings[j]
                j += 1

            filtered.append(best)
            i = j if j > i + 1 else i + 1

        return filtered

    def _is_significant_swing(
        self, swing: SwingPoint, opposite_swings: list[SwingPoint], current_price: float
    ) -> bool:
        """
        Check if a swing is significant enough based on distance from opposite swings.

        Args:
            swing: The swing to check
            opposite_swings: List of opposite swing type
            current_price: Current market price

        Returns:
            True if swing is significant
        """
        if not opposite_swings:
            return True

        # Find nearest opposite swing
        nearest_opposite = min(
            opposite_swings,
            key=lambda s: abs(s.index - swing.index)
        )

        swing_range = abs(swing.price - nearest_opposite.price)
        swing_pct = swing_range / min(swing.price, nearest_opposite.price)

        return swing_pct >= self.min_swing_pct

    def _extract_symbol(self, data: pd.DataFrame) -> str:
        """Extract symbol from data."""
        if "symbol" in data:
            symbol_data = data["symbol"]
            return symbol_data.iloc[0] if hasattr(symbol_data, "iloc") else str(symbol_data)
        return "UNKNOWN"

    def get_swing_points(
        self, data: pd.DataFrame
    ) -> tuple[list[SwingPoint], list[SwingPoint]]:
        """
        Synchronous helper to get swing points without generating a full signal.

        Args:
            data: OHLCV data

        Returns:
            Tuple of (swing_highs, swing_lows)
        """
        high = pd.to_numeric(data["high"].squeeze(), errors="coerce")
        low = pd.to_numeric(data["low"].squeeze(), errors="coerce")
        close = pd.to_numeric(data["close"].squeeze(), errors="coerce")

        current_price = float(close.iloc[-1])

        swing_highs = self._detect_fractal_highs(high)
        swing_lows = self._detect_fractal_lows(low)

        # Filter by significance
        valid_highs = [s for s in swing_highs if self._is_significant_swing(s, swing_lows, current_price)]
        valid_lows = [s for s in swing_lows if self._is_significant_swing(s, swing_highs, current_price)]

        return valid_highs, valid_lows
