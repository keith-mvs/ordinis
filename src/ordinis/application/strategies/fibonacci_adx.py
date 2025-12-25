"""
Fibonacci + ADX Combined Strategy.

Combines Fibonacci retracement levels with ADX trend filter for high-probability entries.
Enhanced with volume confirmation, fractal swing detection, and multi-timeframe alignment.
"""

from datetime import datetime

import pandas as pd

from ordinis.engines.signalcore import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType
from ordinis.engines.signalcore.models import (
    ADXTrendModel,
    FibonacciRetracementModel,
    FractalSwingModel,
    MTFAlignmentModel,
    VolumeProfileModel,
)

from .base import BaseStrategy


class FibonacciADXStrategy(BaseStrategy):
    """
    Fibonacci Retracement + ADX Filter Strategy.

    Combines Fibonacci levels with ADX trend confirmation:
    - ADX confirms strong trend (ADX > 25)
    - Fibonacci identifies entry levels on pullbacks
    - Enters at key levels (38.2%, 50%, 61.8%)
    - Optional: Require trend acceleration (ADX slope positive)
    - Optional: Fractal swing detection for robust levels
    - Optional: Volume confirmation for pullback exhaustion
    - Optional: Multi-timeframe alignment filter

    Parameters:
        adx_period: ADX calculation period (default 14)
        adx_threshold: Minimum ADX for trend (default 25)
        swing_lookback: Bars for swing identification (default 50)
        fib_levels: Key Fibonacci levels (default [0.382, 0.5, 0.618])
        tolerance: Price tolerance near level (default 0.01 = 1%)
        require_trend_accelerating: Gate entries on positive ADX slope (default False)
        slope_lookback: Bars for ADX slope calculation (default 5)
        use_fractal_swings: Use fractal detection for swings (default False)
        fractal_period: Fractal confirmation period (default 2)
        require_volume_confirmation: Gate entries on volume patterns (default False)
        volume_lookback: Volume analysis lookback (default 20)
        require_mtf_alignment: Gate entries on higher TF trend (default False)
        htf_multiplier: Higher timeframe multiplier (default 5)

    Best Markets:
        - Trending markets with clear swings
        - Medium to high volatility
        - Avoid in choppy markets

    Risk Management:
        - Tiered stops based on entry level
        - Multi-target take profits (swing high, 127.2%, 161.8% extensions)
        - Maximum 3% position size
    """

    def __init__(
        self,
        name: str,
        adx_period: int = 14,
        adx_threshold: float = 25,
        swing_lookback: int = 50,
        fib_levels: list[float] | None = None,
        tolerance: float = 0.01,
        require_trend_accelerating: bool = False,
        slope_lookback: int = 5,
        use_fractal_swings: bool = False,
        fractal_period: int = 2,
        require_volume_confirmation: bool = False,
        volume_lookback: int = 20,
        require_mtf_alignment: bool = False,
        htf_multiplier: int = 5,
        htf_sma_period: int = 50,
        **kwargs,
    ):
        params = {
            "adx_period": adx_period,
            "adx_threshold": adx_threshold,
            "swing_lookback": swing_lookback,
            "fib_levels": fib_levels if fib_levels is not None else [0.382, 0.5, 0.618],
            "tolerance": tolerance,
            "require_trend_accelerating": require_trend_accelerating,
            "slope_lookback": slope_lookback,
            "use_fractal_swings": use_fractal_swings,
            "fractal_period": fractal_period,
            "require_volume_confirmation": require_volume_confirmation,
            "volume_lookback": volume_lookback,
            "require_mtf_alignment": require_mtf_alignment,
            "htf_multiplier": htf_multiplier,
            "htf_sma_period": htf_sma_period,
            **kwargs,
        }
        # conservative min bars - account for MTF if enabled
        base_min = max(adx_period * 2 + 30 + slope_lookback, swing_lookback + 20)
        if require_mtf_alignment:
            base_min = max(base_min, (htf_sma_period + 10) * htf_multiplier)
        params.setdefault("min_bars", base_min)
        super().__init__(name, **params)

    def configure(self):
        """Configure strategy parameters and underlying models."""
        # Set defaults
        self.params.setdefault("adx_period", 14)
        self.params.setdefault("adx_threshold", 25)
        self.params.setdefault("swing_lookback", 50)
        self.params.setdefault("fib_levels", [0.382, 0.5, 0.618])
        self.params.setdefault("tolerance", 0.01)
        self.params.setdefault("require_trend_accelerating", False)
        self.params.setdefault("slope_lookback", 5)
        self.params.setdefault("use_fractal_swings", False)
        self.params.setdefault("fractal_period", 2)
        self.params.setdefault("require_volume_confirmation", False)
        self.params.setdefault("volume_lookback", 20)
        self.params.setdefault("require_mtf_alignment", False)
        self.params.setdefault("htf_multiplier", 5)
        self.params.setdefault("htf_sma_period", 50)

        base_min = max(
            self.params["adx_period"] * 2 + 30 + self.params["slope_lookback"],
            self.params["swing_lookback"] + 20,
        )
        if self.params["require_mtf_alignment"]:
            base_min = max(base_min, (self.params["htf_sma_period"] + 10) * self.params["htf_multiplier"])
        self.params.setdefault("min_bars", base_min)

        self.adx_period = int(self.params["adx_period"])
        self.adx_threshold = float(self.params["adx_threshold"])
        self.swing_lookback = int(self.params["swing_lookback"])
        self.fib_levels = list(self.params["fib_levels"])
        self.tolerance = float(self.params["tolerance"])
        self.require_trend_accelerating = bool(self.params["require_trend_accelerating"])
        self.slope_lookback = int(self.params["slope_lookback"])
        self.use_fractal_swings = bool(self.params["use_fractal_swings"])
        self.fractal_period = int(self.params["fractal_period"])
        self.require_volume_confirmation = bool(self.params["require_volume_confirmation"])
        self.volume_lookback = int(self.params["volume_lookback"])
        self.require_mtf_alignment = bool(self.params["require_mtf_alignment"])
        self.htf_multiplier = int(self.params["htf_multiplier"])
        self.htf_sma_period = int(self.params["htf_sma_period"])

        # Initialize ADX model with slope parameters
        adx_config = ModelConfig(
            model_id=f"{self.name}-adx",
            model_type="trend",
            version="1.2.0",
            parameters={
                "adx_period": self.adx_period,
                "adx_threshold": self.adx_threshold,
                "slope_lookback": self.slope_lookback,
            },
        )
        self.adx_model = ADXTrendModel(adx_config)

        # Initialize Fibonacci model
        fib_config = ModelConfig(
            model_id=f"{self.name}-fibonacci",
            model_type="static_level",
            version="1.0.0",
            parameters={
                "swing_lookback": self.swing_lookback,
                "key_levels": self.fib_levels,
                "tolerance": self.tolerance,
            },
        )
        self.fib_model = FibonacciRetracementModel(fib_config)

        # Initialize optional models
        self.fractal_model = None
        if self.use_fractal_swings:
            fractal_config = ModelConfig(
                model_id=f"{self.name}-fractal",
                model_type="swing_detection",
                version="1.0.0",
                parameters={
                    "fractal_period": self.fractal_period,
                    "min_swing_bars": 5,
                    "strength_lookback": 10,
                },
            )
            self.fractal_model = FractalSwingModel(fractal_config)

        self.volume_model = None
        if self.require_volume_confirmation:
            volume_config = ModelConfig(
                model_id=f"{self.name}-volume",
                model_type="volume_profile",
                version="1.0.0",
                parameters={
                    "lookback": self.volume_lookback,
                    "pullback_decline_threshold": 0.2,
                    "bounce_increase_threshold": 0.3,
                },
            )
            self.volume_model = VolumeProfileModel(volume_config)

        self.mtf_model = None
        if self.require_mtf_alignment:
            mtf_config = ModelConfig(
                model_id=f"{self.name}-mtf",
                model_type="mtf_alignment",
                version="1.0.0",
                parameters={
                    "htf_sma_period": self.htf_sma_period,
                    "htf_multiplier": self.htf_multiplier,
                    "require_alignment": True,
                },
            )
            self.mtf_model = MTFAlignmentModel(mtf_config)

    async def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """Asynchronously generate trading signal combining ADX + Fibonacci + optional filters."""
        # Validate data
        is_valid, _msg = self.validate_data(data)
        if not is_valid:
            return None

        min_required = max(
            self.adx_model.config.min_data_points, self.fib_model.config.min_data_points
        )
        if self.fractal_model:
            min_required = max(min_required, self.fractal_model.config.min_data_points)
        if self.mtf_model:
            min_required = max(min_required, self.mtf_model.config.min_data_points)
        if len(data) < min_required:
            return None

        try:
            adx_signal = await self.adx_model.generate(data, timestamp)
        except Exception:
            return None

        # Only proceed if ADX shows strong trend
        if adx_signal.metadata.get("adx", 0) < self.adx_threshold:
            return None

        # Optional: Require trend to be accelerating (positive ADX slope)
        if self.require_trend_accelerating:
            trend_accelerating = adx_signal.metadata.get("trend_accelerating", False)
            if not trend_accelerating:
                return None

        # Get swing points from fractal model if enabled
        fractal_metadata = {}
        if self.fractal_model:
            try:
                fractal_signal = await self.fractal_model.generate(data, timestamp)
                fractal_metadata = {
                    "fractal_swing_high": fractal_signal.metadata.get("swing_high"),
                    "fractal_swing_low": fractal_signal.metadata.get("swing_low"),
                    "fractal_high_strength": fractal_signal.metadata.get("swing_high_strength"),
                    "fractal_low_strength": fractal_signal.metadata.get("swing_low_strength"),
                }
            except Exception:
                pass  # Fall back to regular swing detection

        try:
            fib_signal = await self.fib_model.generate(data, timestamp)
        except Exception:
            return None

        # Check if Fibonacci and ADX agree on direction
        if not (
            adx_signal.direction == fib_signal.direction
            and fib_signal.signal_type == SignalType.ENTRY
        ):
            return None

        # Optional: Check multi-timeframe alignment
        mtf_metadata = {}
        if self.mtf_model:
            try:
                mtf_signal = await self.mtf_model.generate(data, timestamp)
                mtf_metadata = {
                    "mtf_alignment": mtf_signal.metadata.get("alignment"),
                    "mtf_is_aligned": mtf_signal.metadata.get("is_aligned"),
                    "htf_bullish": mtf_signal.metadata.get("htf_bullish"),
                    "htf_bearish": mtf_signal.metadata.get("htf_bearish"),
                    "htf_sma": mtf_signal.metadata.get("htf_sma"),
                }
                # Reject if not aligned with higher timeframe
                if not mtf_signal.metadata.get("is_aligned", False):
                    return None
            except Exception:
                # If MTF check fails and required, reject
                if self.require_mtf_alignment:
                    return None

        # Optional: Check volume confirmation
        volume_metadata = {}
        if self.volume_model and "volume" in data.columns:
            try:
                volume_signal = await self.volume_model.generate(data, timestamp)
                volume_metadata = {
                    "volume_confirms": volume_signal.metadata.get("volume_confirms"),
                    "volume_confirmation_strength": volume_signal.metadata.get("confirmation_strength"),
                    "pullback_declining": volume_signal.metadata.get("pullback_declining"),
                    "bounce_increasing": volume_signal.metadata.get("bounce_increasing"),
                    "relative_volume": volume_signal.metadata.get("relative_volume"),
                }
                # Reject if volume doesn't confirm
                if not volume_signal.metadata.get("volume_confirms", False):
                    return None
            except Exception:
                # If volume check fails and required, reject
                if self.require_volume_confirmation:
                    return None

        # Combine scores (ADX weighted 40%, Fibonacci 60%)
        combined_score = (adx_signal.score * 0.4) + (fib_signal.score * 0.6)
        combined_prob = (adx_signal.probability * 0.4) + (fib_signal.probability * 0.6)

        # Boost probability if multiple confirmations pass
        confirmation_count = 0
        if self.require_trend_accelerating and adx_signal.metadata.get("trend_accelerating"):
            confirmation_count += 1
        if mtf_metadata.get("mtf_is_aligned"):
            confirmation_count += 1
        if volume_metadata.get("volume_confirms"):
            confirmation_count += 1
        if fractal_metadata.get("fractal_swing_high"):
            confirmation_count += 1

        # Small probability boost per confirmation (up to 0.1)
        combined_prob = min(combined_prob + (confirmation_count * 0.025), 0.85)

        # Calculate stop loss and take profit with tiered logic
        swing_high = fib_signal.metadata.get("swing_high")
        swing_low = fib_signal.metadata.get("swing_low")
        
        # Override with fractal swings if available (more robust)
        if fractal_metadata.get("fractal_swing_high"):
            swing_high = fractal_metadata["fractal_swing_high"]
        if fractal_metadata.get("fractal_swing_low"):
            swing_low = fractal_metadata["fractal_swing_low"]

        current_price = fib_signal.metadata.get("current_price")
        nearest_level = fib_signal.metadata.get("nearest_level", "")
        all_levels = fib_signal.metadata.get("all_levels", {})

        # Calculate Fibonacci extension levels for multi-target exits
        swing_range = (swing_high - swing_low) if swing_high and swing_low else 0
        ext_1272 = swing_high + (swing_range * 0.272) if swing_high else None  # 127.2%
        ext_1618 = swing_high + (swing_range * 0.618) if swing_high else None  # 161.8%

        # Tiered stop-loss based on entry level
        stop_loss = None
        take_profit = None
        take_profit_2 = None
        take_profit_3 = None

        if fib_signal.direction == Direction.LONG:
            # Tiered stops: stop just below the next Fibonacci level
            if nearest_level == "38.2%":
                # Entry at 38.2% → stop below 50% level
                stop_loss = all_levels.get("50.0%", swing_low) * 0.995 if all_levels else swing_low * 0.98
            elif nearest_level == "50.0%":
                # Entry at 50% → stop below 61.8% level
                stop_loss = all_levels.get("61.8%", swing_low) * 0.995 if all_levels else swing_low * 0.98
            elif nearest_level == "61.8%":
                # Entry at 61.8% → stop below swing low
                stop_loss = swing_low * 0.98 if swing_low else None
            else:
                # Fallback
                stop_loss = swing_low * 0.98 if swing_low else None

            # Multi-target take profits
            take_profit = swing_high  # TP1: swing high (100% retracement)
            take_profit_2 = ext_1272  # TP2: 127.2% extension
            take_profit_3 = ext_1618  # TP3: 161.8% extension

        else:  # SHORT
            # For short: tiered stops above Fibonacci levels
            if nearest_level == "38.2%":
                stop_loss = all_levels.get("23.6%", swing_high) * 1.005 if all_levels else swing_high * 1.02
            elif nearest_level == "50.0%":
                stop_loss = all_levels.get("38.2%", swing_high) * 1.005 if all_levels else swing_high * 1.02
            elif nearest_level == "61.8%":
                stop_loss = all_levels.get("50.0%", swing_high) * 1.005 if all_levels else swing_high * 1.02
            else:
                stop_loss = swing_high * 1.02 if swing_high else None

            # Multi-target take profits for shorts (extensions below swing low)
            take_profit = swing_low
            take_profit_2 = swing_low - (swing_range * 0.272) if swing_low else None
            take_profit_3 = swing_low - (swing_range * 0.618) if swing_low else None

        # Combine metadata
        metadata = {
            **fib_signal.metadata,
            "adx": adx_signal.metadata.get("adx", 0),
            "plus_di": adx_signal.metadata.get("plus_di", 0),
            "minus_di": adx_signal.metadata.get("minus_di", 0),
            "adx_slope": adx_signal.metadata.get("adx_slope", 0),
            "trend_accelerating": adx_signal.metadata.get("trend_accelerating", False),
            "trend_decelerating": adx_signal.metadata.get("trend_decelerating", False),
            "strategy": self.name,
            "entry_level": nearest_level,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "take_profit_2": take_profit_2,
            "take_profit_3": take_profit_3,
            "extension_1272": ext_1272,
            "extension_1618": ext_1618,
            "risk_reward_ratio": (
                abs(take_profit - current_price) / abs(current_price - stop_loss)
                if take_profit and stop_loss and current_price and stop_loss != current_price
                else None
            ),
            "confirmation_count": confirmation_count,
            **fractal_metadata,
            **mtf_metadata,
            **volume_metadata,
        }

        # Include feature contributions from all models
        feature_contributions = {
            **adx_signal.feature_contributions,
            **fib_signal.feature_contributions,
        }

        return Signal(
            symbol=fib_signal.symbol,
            timestamp=timestamp,
            signal_type=SignalType.ENTRY,
            direction=fib_signal.direction,
            probability=combined_prob,
            expected_return=(fib_signal.expected_return * 1.3)
            if fib_signal.expected_return is not None
            else None,
            confidence_interval=fib_signal.confidence_interval,
            score=combined_score,
            model_id=f"{self.name}-combined",
            model_version="1.4.0",
            feature_contributions=feature_contributions,
            regime=f"{adx_signal.regime}_{fib_signal.regime}",
            data_quality=min(adx_signal.data_quality, fib_signal.data_quality),
            staleness=max(adx_signal.staleness, fib_signal.staleness),
            metadata=metadata,
        )

    def get_description(self) -> str:
        return (
            "Fibonacci Retracement + ADX Trend Filter strategy: combines Fibonacci entries "
            "with ADX trend confirmation. Entries at key Fibonacci levels with ADX trend "
            "confirmation and basic stop/take-profit management."
        )

    @property
    def required_bars(self) -> int:
        """Minimum bars required for signal generation."""
        return max(
            self.adx_model.config.min_data_points,
            self.fib_model.config.min_data_points,
        )
