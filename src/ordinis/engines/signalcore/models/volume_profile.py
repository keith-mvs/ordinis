"""
Volume Profile Confirmation Model.

Analyzes volume patterns during price pullbacks to confirm exhaustion
and potential reversal points for Fibonacci retracement entries.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType


class VolumeProfileModel(Model):
    """
    Volume Profile Confirmation Model.

    Validates Fibonacci retracement entries by analyzing volume patterns:
    - Declining volume on pullback (sellers exhausting)
    - Increasing volume on bounce (buyers returning)
    - Volume climax detection at swing points

    Parameters:
        lookback: Bars for volume analysis (default 20)
        pullback_decline_threshold: Min volume decline % during pullback (default 0.2)
        bounce_increase_threshold: Min volume increase % on bounce (default 0.3)
        volume_ma_period: Period for volume moving average (default 20)
        climax_threshold: Multiplier for climax detection (default 2.0)

    Signals:
        - ENTRY when volume confirms reversal pattern
        - HOLD when volume pattern inconclusive
        - Metadata includes volume analysis details

    Usage:
        Use in conjunction with Fibonacci levels to confirm entries.
        Volume confirmation significantly improves win rate on retracement trades.

    Reference:
        Volume Spread Analysis (VSA) - Tom Williams
        Wyckoff Method - Richard Wyckoff
    """

    def __init__(self, config: ModelConfig):
        """Initialize Volume Profile model."""
        super().__init__(config)

        params = self.config.parameters
        self.lookback = params.get("lookback", 20)
        self.pullback_decline_threshold = params.get("pullback_decline_threshold", 0.2)
        self.bounce_increase_threshold = params.get("bounce_increase_threshold", 0.3)
        self.volume_ma_period = params.get("volume_ma_period", 20)
        self.climax_threshold = params.get("climax_threshold", 2.0)

        self.config.min_data_points = max(self.lookback, self.volume_ma_period) + 10

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate volume confirmation signal.

        Args:
            data: Historical OHLCV data with 'volume' column
            timestamp: Current timestamp

        Returns:
            Signal with volume confirmation analysis
        """
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        if "volume" not in data.columns:
            raise ValueError("Volume column required for volume profile analysis")

        symbol = self._extract_symbol(data)

        # Extract series
        close = pd.to_numeric(data["close"].squeeze(), errors="coerce")
        volume = pd.to_numeric(data["volume"].squeeze(), errors="coerce")

        current_price = float(close.iloc[-1])
        current_volume = float(volume.iloc[-1])

        # Calculate volume moving average
        vol_ma = volume.rolling(window=self.volume_ma_period).mean()
        current_vol_ma = float(vol_ma.iloc[-1]) if not pd.isna(vol_ma.iloc[-1]) else current_volume

        # Relative volume (current vs average)
        relative_volume = current_volume / current_vol_ma if current_vol_ma > 0 else 1.0

        # Analyze recent pullback volume pattern
        pullback_analysis = self._analyze_pullback_volume(close, volume, self.lookback)
        
        # Detect volume climax
        climax_detected = relative_volume > self.climax_threshold

        # Determine if volume confirms entry
        volume_confirms = False
        confirmation_strength = 0.0
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL

        if pullback_analysis["pattern_detected"]:
            if pullback_analysis["pullback_declining"] and pullback_analysis["bounce_increasing"]:
                # Classic confirmation: declining pullback + increasing bounce
                volume_confirms = True
                confirmation_strength = 0.9
                signal_type = SignalType.ENTRY
                direction = pullback_analysis["implied_direction"]
            elif pullback_analysis["pullback_declining"]:
                # Partial confirmation: declining pullback only
                volume_confirms = True
                confirmation_strength = 0.6
                signal_type = SignalType.ENTRY
                direction = pullback_analysis["implied_direction"]
            elif pullback_analysis["bounce_increasing"]:
                # Partial confirmation: increasing bounce only
                volume_confirms = True
                confirmation_strength = 0.5
                signal_type = SignalType.ENTRY
                direction = pullback_analysis["implied_direction"]

        # Calculate probability based on confirmation strength
        probability = 0.5 + (confirmation_strength * 0.25)  # 0.5 to 0.725
        score = confirmation_strength if direction == Direction.LONG else -confirmation_strength

        # Feature contributions
        feature_contributions = {
            "relative_volume": float(relative_volume),
            "pullback_decline": float(pullback_analysis.get("pullback_decline_pct", 0)),
            "bounce_increase": float(pullback_analysis.get("bounce_increase_pct", 0)),
            "climax_factor": float(relative_volume / self.climax_threshold) if relative_volume > 1 else 0.0,
            "confirmation_strength": float(confirmation_strength),
        }

        # Regime based on volume characteristics
        if relative_volume > 2.0:
            regime = "high_volume"
        elif relative_volume > 1.0:
            regime = "normal_volume"
        else:
            regime = "low_volume"

        # Data quality
        recent_vol = volume.tail(20)
        data_quality = 1.0 - (recent_vol.isnull().sum() / len(recent_vol))

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
            expected_return=0.02 * confirmation_strength,
            confidence_interval=(-0.02, 0.04),
            score=score,
            model_id=self.config.model_id,
            model_version=self.config.version,
            feature_contributions=feature_contributions,
            regime=regime,
            data_quality=data_quality,
            staleness=staleness,
            metadata={
                "volume_confirms": volume_confirms,
                "confirmation_strength": confirmation_strength,
                "relative_volume": relative_volume,
                "volume_ma": current_vol_ma,
                "current_volume": current_volume,
                "climax_detected": climax_detected,
                "pullback_declining": pullback_analysis.get("pullback_declining", False),
                "bounce_increasing": pullback_analysis.get("bounce_increasing", False),
                "pullback_decline_pct": pullback_analysis.get("pullback_decline_pct", 0),
                "bounce_increase_pct": pullback_analysis.get("bounce_increase_pct", 0),
                "pattern_detected": pullback_analysis.get("pattern_detected", False),
            },
        )

    def _analyze_pullback_volume(
        self, close: pd.Series, volume: pd.Series, lookback: int
    ) -> dict:
        """
        Analyze volume pattern during price pullback.

        Returns dict with:
            - pattern_detected: Whether a pullback/bounce pattern is found
            - pullback_declining: Volume declining during pullback
            - bounce_increasing: Volume increasing on bounce
            - pullback_decline_pct: Percentage decline in pullback volume
            - bounce_increase_pct: Percentage increase in bounce volume
            - implied_direction: Direction implied by the pattern
        """
        result = {
            "pattern_detected": False,
            "pullback_declining": False,
            "bounce_increasing": False,
            "pullback_decline_pct": 0.0,
            "bounce_increase_pct": 0.0,
            "implied_direction": Direction.NEUTRAL,
        }

        if len(close) < lookback:
            return result

        recent_close = close.tail(lookback).values
        recent_volume = volume.tail(lookback).values

        # Find local high and low in the lookback period
        local_high_idx = np.argmax(recent_close)
        local_low_idx = np.argmin(recent_close)

        # Determine if we're in an uptrend pullback or downtrend pullback
        if local_high_idx < local_low_idx:
            # High came first, then low → potential bullish reversal (pullback in uptrend)
            # Pullback phase: from high to low
            # Bounce phase: from low to end
            if local_low_idx > 0 and local_low_idx < len(recent_close) - 1:
                pullback_vol = recent_volume[local_high_idx:local_low_idx + 1]
                bounce_vol = recent_volume[local_low_idx:]

                if len(pullback_vol) >= 2 and len(bounce_vol) >= 2:
                    result["pattern_detected"] = True
                    result["implied_direction"] = Direction.LONG

                    # Check if pullback volume is declining
                    pullback_avg_first = np.mean(pullback_vol[:len(pullback_vol)//2 + 1])
                    pullback_avg_last = np.mean(pullback_vol[len(pullback_vol)//2:])
                    
                    if pullback_avg_first > 0:
                        decline_pct = (pullback_avg_first - pullback_avg_last) / pullback_avg_first
                        result["pullback_decline_pct"] = decline_pct
                        result["pullback_declining"] = decline_pct >= self.pullback_decline_threshold

                    # Check if bounce volume is increasing
                    bounce_avg_first = np.mean(bounce_vol[:len(bounce_vol)//2 + 1])
                    bounce_avg_last = np.mean(bounce_vol[len(bounce_vol)//2:])
                    
                    if bounce_avg_first > 0:
                        increase_pct = (bounce_avg_last - bounce_avg_first) / bounce_avg_first
                        result["bounce_increase_pct"] = increase_pct
                        result["bounce_increasing"] = increase_pct >= self.bounce_increase_threshold

        elif local_low_idx < local_high_idx:
            # Low came first, then high → potential bearish reversal (pullback in downtrend)
            if local_high_idx > 0 and local_high_idx < len(recent_close) - 1:
                pullback_vol = recent_volume[local_low_idx:local_high_idx + 1]
                bounce_vol = recent_volume[local_high_idx:]

                if len(pullback_vol) >= 2 and len(bounce_vol) >= 2:
                    result["pattern_detected"] = True
                    result["implied_direction"] = Direction.SHORT

                    # Check if pullback (rally) volume is declining
                    pullback_avg_first = np.mean(pullback_vol[:len(pullback_vol)//2 + 1])
                    pullback_avg_last = np.mean(pullback_vol[len(pullback_vol)//2:])
                    
                    if pullback_avg_first > 0:
                        decline_pct = (pullback_avg_first - pullback_avg_last) / pullback_avg_first
                        result["pullback_decline_pct"] = decline_pct
                        result["pullback_declining"] = decline_pct >= self.pullback_decline_threshold

                    # Check if breakdown volume is increasing
                    bounce_avg_first = np.mean(bounce_vol[:len(bounce_vol)//2 + 1])
                    bounce_avg_last = np.mean(bounce_vol[len(bounce_vol)//2:])
                    
                    if bounce_avg_first > 0:
                        increase_pct = (bounce_avg_last - bounce_avg_first) / bounce_avg_first
                        result["bounce_increase_pct"] = increase_pct
                        result["bounce_increasing"] = increase_pct >= self.bounce_increase_threshold

        return result

    def _extract_symbol(self, data: pd.DataFrame) -> str:
        """Extract symbol from data."""
        if "symbol" in data:
            symbol_data = data["symbol"]
            return symbol_data.iloc[0] if hasattr(symbol_data, "iloc") else str(symbol_data)
        return "UNKNOWN"

    def check_confirmation(
        self,
        data: pd.DataFrame,
        direction: Direction,
    ) -> tuple[bool, float]:
        """
        Synchronous helper to check if volume confirms a direction.

        Args:
            data: OHLCV data
            direction: Expected trade direction

        Returns:
            Tuple of (confirms: bool, strength: float)
        """
        if "volume" not in data.columns or len(data) < self.lookback:
            return False, 0.0

        close = pd.to_numeric(data["close"].squeeze(), errors="coerce")
        volume = pd.to_numeric(data["volume"].squeeze(), errors="coerce")

        analysis = self._analyze_pullback_volume(close, volume, self.lookback)

        if not analysis["pattern_detected"]:
            return False, 0.0

        if analysis["implied_direction"] != direction:
            return False, 0.0

        strength = 0.0
        if analysis["pullback_declining"]:
            strength += 0.5
        if analysis["bounce_increasing"]:
            strength += 0.5

        return strength > 0, strength
