"""
Fibonacci Retracement Model.

Identifies potential entry points at key Fibonacci retracement levels
during pullbacks in established trends.
"""

from datetime import datetime, timedelta

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType


class FibonacciRetracementModel(Model):
    """
    Fibonacci Retracement Trading Model.

    Generates buy signals when price retraces to key Fibonacci levels
    (38.2%, 50%, 61.8%) after an uptrend, or sell signals after retracements
    in a downtrend.

    Parameters:
        swing_lookback: Bars to look back for swing high/low (default 50)
        key_levels: Fibonacci levels to watch (default [0.382, 0.500, 0.618])
        tolerance: Price tolerance near level (default 0.01 = 1%)
        min_swing_size: Minimum swing range as % (default 0.05 = 5%)

    Signals:
        - ENTRY/LONG when price near key Fib level in uptrend
        - ENTRY/SHORT when price near key Fib level in downtrend
        - HOLD when price not near any key level
        - Score based on level significance and trend strength

    Usage:
        Best used in trending markets with clear swings.
        Combine with trend indicators (ADX, MA) for confirmation.

    Reference:
        Fibonacci ratios derive from golden ratio φ ≈ 1.618
        38.2% = 1 - (1/φ), 61.8% = 1/φ, 50% is common but not Fibonacci
    """

    def __init__(self, config: ModelConfig):
        """Initialize Fibonacci Retracement model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.swing_lookback = params.get("swing_lookback", 50)
        self.key_levels = params.get("key_levels", [0.382, 0.500, 0.618])
        self.tolerance = params.get("tolerance", 0.01)  # 1%
        self.min_swing_size = params.get("min_swing_size", 0.05)  # 5%

        # Update min data points
        self.config.min_data_points = self.swing_lookback + 20

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate trading signal from Fibonacci retracement analysis.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with Fibonacci retracement prediction
        """
        # Validate data
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        if "symbol" in data:
            symbol_data = data["symbol"]
            symbol = symbol_data.iloc[0] if hasattr(symbol_data, "iloc") else str(symbol_data)
        else:
            symbol = "UNKNOWN"

        # Normalize numeric series
        def _as_series(col):
            if isinstance(col, pd.DataFrame):
                if col.shape[1] >= 1:
                    s = col.iloc[:, 0]
                else:
                    raise ValueError("Column is empty DataFrame")
            else:
                s = col
            s = pd.to_numeric(s, errors="coerce")
            if s.isnull().all():
                raise ValueError("Column contains only nulls or non-numeric values")
            return s

        close = _as_series(data["close"])
        high = _as_series(data["high"])
        low = _as_series(data["low"])

        # Find recent swing high and low (inline to avoid imports)
        start_index = max(0, len(data) - self.swing_lookback)
        window = data.iloc[start_index:]

        # Work with normalized window columns
        window_high = _as_series(window["high"]) if "high" in window.columns else high.iloc[start_index:]
        window_low = _as_series(window["low"]) if "low" in window.columns else low.iloc[start_index:]

        swing_high = float(window_high.max())
        swing_low = float(window_low.min())

        # Get positional index instead of index value
        # Use idxmax/idxmin then convert to integer positions if possible
        try:
            high_idx_pos = window_high.idxmax()
            low_idx_pos = window_low.idxmin()
            # Convert to integer positions relative to original data
            high_idx_pos = window.index.get_loc(high_idx_pos)
            low_idx_pos = window.index.get_loc(low_idx_pos)
        except Exception:
            high_idx_pos = int(window_high.argmax())
            low_idx_pos = int(window_low.argmin())

        # Convert to simple comparison (higher position = more recent)
        high_idx = start_index + high_idx_pos
        low_idx = start_index + low_idx_pos

        current_price = close.iloc[-1]

        # Calculate swing range
        swing_range = swing_high - swing_low
        swing_percent = swing_range / swing_low if swing_low > 0 else 0

        # Determine trend direction based on swing sequence
        if high_idx > low_idx:
            trend_direction = "bullish"
        else:
            trend_direction = "bearish"

        # Calculate Fibonacci levels inline
        diff = swing_high - swing_low
        fib_level_dict = {
            "0.0%": swing_high,
            "23.6%": swing_high - (diff * 0.236),
            "38.2%": swing_high - (diff * 0.382),
            "50.0%": swing_high - (diff * 0.500),
            "61.8%": swing_high - (diff * 0.618),
            "78.6%": swing_high - (diff * 0.786),
            "100.0%": swing_low,
        }

        # Create FibonacciLevels-like object (just use dict)
        class FibLevels:
            def __init__(self, levels):
                self.levels = levels

        fib_levels = FibLevels(fib_level_dict)

        # Check if swing is significant enough
        if swing_percent < self.min_swing_size:
            # Swing too small, don't trade
            return self._create_hold_signal(
                symbol,
                timestamp,
                close,
                current_price,
                {"reason": "Swing range too small", "swing_percent": float(swing_percent)},
            )

        # Find nearest Fibonacci level
        nearest_level_name = None
        nearest_level_price = None
        nearest_distance = float("inf")

        for level_name, level_price in fib_levels.levels.items():
            # Only check key levels
            level_value = float(level_name.strip("%").replace(",", "")) / 100.0
            if level_value not in self.key_levels:
                continue

            distance = abs(current_price - level_price) / current_price
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_level_name = level_name
                nearest_level_price = level_price

        # Generate signal based on proximity to key level
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        if nearest_distance <= self.tolerance and nearest_level_name:
            # Price is near a key Fibonacci level
            level_value = float(nearest_level_name.strip("%").replace(",", "")) / 100.0

            # Score based on level significance and distance
            # 61.8% and 38.2% are stronger than 50%
            if level_value == 0.618:
                level_strength = 1.0  # Golden ratio - strongest
            elif level_value == 0.382:
                level_strength = 0.9  # Also golden ratio derived
            elif level_value == 0.500:
                level_strength = 0.7  # Not Fibonacci but commonly used
            else:
                level_strength = 0.5

            # Distance factor (closer = better)
            distance_factor = 1.0 - (nearest_distance / self.tolerance)

            # Swing size factor (larger swings = more reliable)
            swing_factor = min(swing_percent / 0.10, 1.0)  # Cap at 10%

            if trend_direction == "bullish":
                signal_type = SignalType.ENTRY
                direction = Direction.LONG
                score = (level_strength * 0.5) + (distance_factor * 0.3) + (swing_factor * 0.2)
                probability = 0.55 + (score * 0.15)  # 0.55-0.70
                expected_return = 0.03 + (score * 0.05)  # 3-8%

            else:  # bearish
                signal_type = SignalType.ENTRY
                direction = Direction.SHORT
                score = -((level_strength * 0.5) + (distance_factor * 0.3) + (swing_factor * 0.2))
                probability = 0.55 + (abs(score) * 0.15)  # 0.55-0.70
                expected_return = 0.0  # Short positions tracked differently

        # Calculate confidence interval
        returns = close.pct_change().dropna()
        recent_vol = returns.tail(20).std()
        confidence_interval = (
            expected_return - 2 * recent_vol,
            expected_return + 2 * recent_vol,
        )

        # Feature contributions for explainability
        feature_contributions = {
            "nearest_level": float(nearest_level_name.strip("%").replace(",", "")) / 100.0
            if nearest_level_name
            else 0.0,
            "distance_to_level": float(nearest_distance),
            "swing_size": float(swing_percent),
            "trend_direction": 1.0 if trend_direction == "bullish" else -1.0,
        }

        # Regime detection
        if swing_percent > 0.15:
            regime = "high_volatility"
        elif swing_percent > 0.08:
            regime = "moderate_volatility"
        else:
            regime = "low_volatility"

        # Data quality
        recent_close = close.tail(20)
        data_quality = 1.0 - (recent_close.isnull().sum() / len(recent_close))

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
                "swing_high": float(swing_high),
                "swing_low": float(swing_low),
                "swing_range": float(swing_range),
                "current_price": float(current_price),
                "nearest_level": nearest_level_name if nearest_level_name else "none",
                "nearest_price": float(nearest_level_price) if nearest_level_price else 0.0,
                "distance": float(nearest_distance),
                "trend": trend_direction,
                "all_levels": {k: float(v) for k, v in fib_levels.levels.items()},
            },
        )

    def _create_hold_signal(
        self,
        symbol: str,
        timestamp: datetime,
        close: pd.Series,
        current_price: float,
        metadata: dict,
    ) -> Signal:
        """Create a HOLD signal with minimal metadata."""
        returns = close.pct_change().dropna()
        recent_vol = returns.tail(20).std()

        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=SignalType.HOLD,
            direction=Direction.NEUTRAL,
            probability=0.5,
            expected_return=0.0,
            confidence_interval=(-2 * recent_vol, 2 * recent_vol),
            score=0.0,
            model_id=self.config.model_id,
            model_version=self.config.version,
            feature_contributions={},
            regime="neutral",
            data_quality=1.0,
            staleness=timedelta(seconds=0),
            metadata=metadata,
        )
