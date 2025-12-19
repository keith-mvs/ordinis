"""Advanced technical models: Ichimoku, Chart Patterns, Volume Profile, Options."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType


@dataclass
class IchimokuValues:
    """Ichimoku Cloud components."""

    tenkan: float  # Conversion line
    kijun: float  # Base line
    senkou_a: float  # Leading span A
    senkou_b: float  # Leading span B
    chikou: float  # Lagging span


class IchimokuModel(Model):
    """Ichimoku Cloud technical analysis model."""

    def __init__(self, config: ModelConfig | None = None):
        """Initialize model.

        Args:
            config: Model configuration
        """
        if config is None:
            config = ModelConfig(
                model_id="ichimoku_cloud",
                model_type="technical",
                version="1.0.0",
                parameters={
                    "tenkan_period": 9,
                    "kijun_period": 26,
                    "senkou_b_period": 52,
                },
                min_data_points=52,
                lookback_period=52,
            )

        super().__init__(config)

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """Generate signal from Ichimoku.

        Args:
            data: OHLCV data
            timestamp: Signal timestamp

        Returns:
            Signal
        """
        # Calculate Ichimoku components
        tenkan_period = self.config.parameters.get("tenkan_period", 9)
        kijun_period = self.config.parameters.get("kijun_period", 26)
        senkou_b_period = self.config.parameters.get("senkou_b_period", 52)

        # Tenkan-sen (Conversion Line)
        tenkan_high = data["high"].rolling(window=tenkan_period).max()
        tenkan_low = data["low"].rolling(window=tenkan_period).min()
        tenkan = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = data["high"].rolling(window=kijun_period).max()
        kijun_low = data["low"].rolling(window=kijun_period).min()
        kijun = (kijun_high + kijun_low) / 2

        # Senkou Span B (Leading Span B)
        senkou_b_high = data["high"].rolling(window=senkou_b_period).max()
        senkou_b_low = data["low"].rolling(window=senkou_b_period).min()
        senkou_b = (senkou_b_high + senkou_b_low) / 2

        # Get latest values
        latest_close = data["close"].iloc[-1]
        latest_tenkan = tenkan.iloc[-1]
        latest_kijun = kijun.iloc[-1]
        latest_senkou_b = senkou_b.iloc[-1]

        # Cloud thickness (Senkou A is average of Tenkan and Kijun)
        latest_senkou_a = (latest_tenkan + latest_kijun) / 2

        # Ichimoku signals
        cloud_top = max(latest_senkou_a, latest_senkou_b)
        cloud_bottom = min(latest_senkou_a, latest_senkou_b)

        # Signal logic
        score = 0.0
        direction = Direction.NEUTRAL

        # Price above cloud = bullish
        if latest_close > cloud_top:
            score += 0.3
            direction = Direction.LONG

        # Price below cloud = bearish
        elif latest_close < cloud_bottom:
            score -= 0.3
            direction = Direction.SHORT

        # Tenkan above Kijun = bullish
        if latest_tenkan > latest_kijun:
            score += 0.25
        else:
            score -= 0.25

        # Cloud trend (Senkou A trending up)
        if latest_senkou_a > senkou_b.iloc[-26]:
            score += 0.15
        else:
            score -= 0.15

        # Normalize score
        score = np.clip(score, -1.0, 1.0)
        probability = 0.5 + abs(score) * 0.4

        return Signal(
            symbol="UNKNOWN",  # Will be set by engine
            timestamp=timestamp,
            signal_type=SignalType.ENTRY if abs(score) > 0.3 else SignalType.HOLD,
            direction=direction,
            probability=probability,
            expected_return=score * 0.02,  # Estimate 2% per point of score
            confidence_interval=(-0.01, 0.01),
            score=score,
            model_id=self.config.model_id,
            model_version=self.config.version,
            metadata={
                "tenkan": float(latest_tenkan),
                "kijun": float(latest_kijun),
                "senkou_a": float(latest_senkou_a),
                "senkou_b": float(latest_senkou_b),
                "cloud_thickness": float(cloud_top - cloud_bottom),
            },
        )


class ChartPatternModel(Model):
    """Chart pattern recognition (Head & Shoulders, Triangles, Flags)."""

    def __init__(self, config: ModelConfig | None = None):
        """Initialize model."""
        if config is None:
            config = ModelConfig(
                model_id="chart_patterns",
                model_type="technical",
                version="1.0.0",
                parameters={
                    "min_bars_for_pattern": 20,
                    "tolerance_pct": 0.02,
                },
                min_data_points=50,
                lookback_period=50,
            )

        super().__init__(config)

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """Detect chart patterns.

        Args:
            data: OHLCV data
            timestamp: Signal timestamp

        Returns:
            Signal
        """
        score = 0.0
        patterns_found = []

        # Head & Shoulders detection (simplified)
        if len(data) >= 20:
            recent = data["close"].iloc[-20:].values
            peaks = self._find_peaks(recent)

            # Head & Shoulders: left shoulder, head, right shoulder
            if len(peaks) >= 3:
                if recent[peaks[0]] < recent[peaks[1]] > recent[peaks[2]]:
                    if recent[peaks[0]] > recent[peaks[2]] * 0.95:  # Shoulders similar height
                        patterns_found.append("head_and_shoulders")
                        score -= 0.4  # Bearish

        # Triangle pattern (converging highs and lows)
        if len(data) >= 15:
            recent_high = data["high"].iloc[-15:].values
            recent_low = data["low"].iloc[-15:].values

            high_trend = np.polyfit(range(len(recent_high)), recent_high, 1)[0]
            low_trend = np.polyfit(range(len(recent_low)), recent_low, 1)[0]

            # Converging = triangle
            if high_trend < 0 and low_trend > 0:
                patterns_found.append("triangle")
                score += 0.3  # Bullish (breakout expected)

        # Flag pattern (small consolidation after sharp move)
        if len(data) >= 20:
            recent = data["close"].iloc[-20:]
            volatility_early = recent.iloc[:5].std()
            volatility_late = recent.iloc[-5:].std()

            if volatility_early > volatility_late * 2:
                patterns_found.append("flag")
                score += 0.25

        # Normalize
        score = np.clip(score, -1.0, 1.0)
        probability = 0.5 + abs(score) * 0.3
        direction = (
            Direction.LONG if score > 0 else (Direction.SHORT if score < 0 else Direction.NEUTRAL)
        )

        return Signal(
            symbol="UNKNOWN",
            timestamp=timestamp,
            signal_type=SignalType.ENTRY if abs(score) > 0.2 else SignalType.HOLD,
            direction=direction,
            probability=probability,
            expected_return=score * 0.015,
            confidence_interval=(-0.01, 0.01),
            score=score,
            model_id=self.config.model_id,
            model_version=self.config.version,
            metadata={"patterns": patterns_found},
        )

    @staticmethod
    def _find_peaks(arr: np.ndarray, min_distance: int = 2) -> list[int]:
        """Find local peaks in array."""
        peaks = []

        for i in range(min_distance, len(arr) - min_distance):
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                peaks.append(i)

        return peaks


class VolumeProfileModel(Model):
    """Volume Profile analysis for support/resistance."""

    def __init__(self, config: ModelConfig | None = None):
        """Initialize model."""
        if config is None:
            config = ModelConfig(
                model_id="volume_profile",
                model_type="technical",
                version="1.0.0",
                parameters={"bins": 10},
                min_data_points=100,
                lookback_period=100,
            )

        super().__init__(config)

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """Analyze volume profile for support/resistance.

        Args:
            data: OHLCV data
            timestamp: Signal timestamp

        Returns:
            Signal
        """
        bins = self.config.parameters.get("bins", 10)

        # Create volume profile
        price_bins = np.linspace(data["low"].min(), data["high"].max(), bins + 1)
        profile = np.zeros(bins)

        for i in range(len(data)):
            price = data["close"].iloc[i]
            vol = data["volume"].iloc[i]

            # Find bin for this price
            bin_idx = np.searchsorted(price_bins, price) - 1
            bin_idx = np.clip(bin_idx, 0, bins - 1)
            profile[bin_idx] += vol

        # Find POC (Point of Control)
        poc_idx = np.argmax(profile)
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2

        # Find VAH (Value Area High) and VAL (Value Area Low)
        sorted_indices = np.argsort(profile)[::-1]
        va_volume = np.sum(profile) * 0.70  # 70% of volume
        va_indices = sorted_indices[
            : np.searchsorted(np.cumsum(profile[sorted_indices]), va_volume)
        ]
        va_high = price_bins[np.max(va_indices) + 1]
        va_low = price_bins[np.min(va_indices)]

        # Current price relative to volume profile
        current_price = data["close"].iloc[-1]
        score = 0.0

        # If price below VAL, likely to move up (bullish)
        if current_price < va_low:
            score = 0.4
            direction = Direction.LONG

        # If price above VAH, likely to move down (bearish)
        elif current_price > va_high:
            score = -0.4
            direction = Direction.SHORT

        # If price in value area, equilibrium (neutral)
        else:
            score = 0.0
            direction = Direction.NEUTRAL

        probability = 0.5 + abs(score) * 0.35

        return Signal(
            symbol="UNKNOWN",
            timestamp=timestamp,
            signal_type=SignalType.ENTRY if abs(score) > 0.2 else SignalType.HOLD,
            direction=direction,
            probability=probability,
            expected_return=score * 0.01,
            confidence_interval=(-0.01, 0.01),
            score=score,
            model_id=self.config.model_id,
            model_version=self.config.version,
            metadata={
                "poc": float(poc_price),
                "va_high": float(va_high),
                "va_low": float(va_low),
            },
        )


class OptionsSignalsModel(Model):
    """Options market signals (implied volatility, skew, open interest)."""

    def __init__(self, config: ModelConfig | None = None):
        """Initialize model."""
        if config is None:
            config = ModelConfig(
                model_id="options_signals",
                model_type="options",
                version="1.0.0",
                parameters={
                    "iv_percentile_threshold": 0.7,
                    "oi_increase_threshold": 0.2,
                },
                min_data_points=20,
                lookback_period=20,
            )

        super().__init__(config)

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """Generate signals from options market data.

        Args:
            data: OHLCV data (should include IV and OI if available)
            timestamp: Signal timestamp

        Returns:
            Signal
        """
        # Placeholder: In production, would fetch from options API
        # This demonstrates the model interface

        score = 0.0

        # Typically would check:
        # - Implied Volatility rank (high IV = sell premium, low IV = buy premium)
        # - Put/Call ratio (extreme ratios signal reversals)
        # - Open Interest changes (large increases signal institutional interest)
        # - Skew (put skew can indicate fear/hedging)

        # Simplified example: assume IV is available in metadata
        iv_percentile = 0.5  # Would calculate from historical IV

        if iv_percentile > 0.7:
            # High IV = expect reversion lower
            score = -0.2
            direction = Direction.SHORT
        elif iv_percentile < 0.3:
            # Low IV = expect expansion (buy premium)
            score = 0.2
            direction = Direction.LONG
        else:
            direction = Direction.NEUTRAL

        probability = 0.5 + abs(score) * 0.25

        return Signal(
            symbol="UNKNOWN",
            timestamp=timestamp,
            signal_type=SignalType.ENTRY if abs(score) > 0.15 else SignalType.HOLD,
            direction=direction,
            probability=probability,
            expected_return=score * 0.01,
            confidence_interval=(-0.01, 0.01),
            score=score,
            model_id=self.config.model_id,
            model_version=self.config.version,
            metadata={
                "iv_percentile": iv_percentile,
                "options_data_available": True,
            },
        )
