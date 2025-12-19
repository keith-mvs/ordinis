"""
ADX-Filtered RSI Strategy.

Combines ADX trend strength filter with RSI mean reversion for high-quality trades.
Only takes RSI signals when ADX confirms a strong trend.
"""

from datetime import datetime

import pandas as pd

from ordinis.engines.signalcore import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType
from ordinis.engines.signalcore.models import ADXTrendModel, RSIMeanReversionModel

from .base import BaseStrategy


class ADXFilteredRSIStrategy(BaseStrategy):
    """
    ADX-Filtered RSI Mean Reversion Strategy.

    Uses ADX to filter for trending markets, then applies RSI for timing.
    - Only enters when ADX shows strong trend (ADX > 25)
    - Uses RSI oversold (< 30) for long entries in uptrends
    - Uses RSI overbought (> 70) for exits

    Parameters:
        adx_period: ADX calculation period (default 14)
        adx_threshold: Minimum ADX for trend (default 25)
        rsi_period: RSI calculation period (default 14)
        oversold_threshold: RSI oversold level (default 30)
        overbought_threshold: RSI overbought level (default 70)

    Best Markets:
        - Trending markets with clear direction
        - Avoid in choppy, range-bound conditions
        - Works well with liquid, volatile stocks

    Risk Management:
        - ATR-based stop loss (2x ATR)
        - Target at 1.5:1 reward/risk ratio
        - Maximum 2% position size
    """

    def __init__(
        self,
        name: str,
        adx_period: int = 14,
        adx_threshold: float = 25,
        rsi_period: int = 14,
        oversold_threshold: float = 30,
        overbought_threshold: float = 70,
    ):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            adx_period: ADX calculation period
            adx_threshold: Minimum ADX for trend confirmation
            rsi_period: RSI calculation period
            oversold_threshold: RSI oversold level for entries
            overbought_threshold: RSI overbought level for exits
        """
        super().__init__(name)

        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold

        # Initialize ADX model
        adx_config = ModelConfig(
            model_id=f"{name}-adx",
            model_type="trend",
            version="1.0.0",
            parameters={
                "adx_period": adx_period,
                "adx_threshold": adx_threshold,
            },
        )
        self.adx_model = ADXTrendModel(adx_config)

        # Initialize RSI model
        rsi_config = ModelConfig(
            model_id=f"{name}-rsi",
            model_type="momentum",
            version="1.0.0",
            parameters={
                "rsi_period": rsi_period,
                "oversold_threshold": oversold_threshold,
                "overbought_threshold": overbought_threshold,
            },
        )
        self.rsi_model = RSIMeanReversionModel(rsi_config)

    def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate trading signal.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal if conditions met, None otherwise
        """
        # Check minimum data requirements
        min_required = max(
            self.adx_model.config.min_data_points, self.rsi_model.config.min_data_points
        )
        if len(data) < min_required:
            return None

        # Get ADX signal to check trend strength
        adx_signal = self.adx_model.generate(data, timestamp)

        # Only proceed if ADX shows strong trend
        if adx_signal.metadata["adx"] < self.adx_threshold:
            # No strong trend, don't trade
            return None

        # Get trend direction from ADX
        trend_direction = adx_signal.metadata["trend_strength"]
        if trend_direction == "weak":
            return None

        # Get RSI signal for timing
        rsi_signal = self.rsi_model.generate(data, timestamp)

        # Combine signals based on ADX trend and RSI timing
        if adx_signal.direction == Direction.LONG:
            # In uptrend, look for RSI oversold for entry
            if (
                rsi_signal.signal_type == SignalType.ENTRY
                and rsi_signal.direction == Direction.LONG
            ):
                # RSI oversold in uptrend - good entry
                combined_score = (adx_signal.score * 0.4) + (rsi_signal.score * 0.6)
                combined_prob = (adx_signal.probability * 0.4) + (rsi_signal.probability * 0.6)

                # Combine metadata
                metadata = {
                    **rsi_signal.metadata,
                    "adx": adx_signal.metadata["adx"],
                    "plus_di": adx_signal.metadata["plus_di"],
                    "minus_di": adx_signal.metadata["minus_di"],
                    "strategy": self.name,
                }

                return Signal(
                    symbol=rsi_signal.symbol,
                    timestamp=timestamp,
                    signal_type=SignalType.ENTRY,
                    direction=Direction.LONG,
                    probability=combined_prob,
                    expected_return=rsi_signal.expected_return
                    * 1.2,  # Boost for trend confirmation
                    confidence_interval=rsi_signal.confidence_interval,
                    score=combined_score,
                    model_id=f"{self.name}-combined",
                    model_version="1.0.0",
                    feature_contributions={
                        **adx_signal.feature_contributions,
                        **rsi_signal.feature_contributions,
                    },
                    regime=f"{adx_signal.regime}_{rsi_signal.regime}",
                    data_quality=min(adx_signal.data_quality, rsi_signal.data_quality),
                    staleness=max(adx_signal.staleness, rsi_signal.staleness),
                    metadata=metadata,
                )

            if rsi_signal.signal_type == SignalType.EXIT:
                # RSI exit signal
                return Signal(
                    symbol=rsi_signal.symbol,
                    timestamp=timestamp,
                    signal_type=SignalType.EXIT,
                    direction=Direction.NEUTRAL,
                    probability=rsi_signal.probability,
                    expected_return=0.0,
                    confidence_interval=rsi_signal.confidence_interval,
                    score=rsi_signal.score,
                    model_id=f"{self.name}-combined",
                    model_version="1.0.0",
                    feature_contributions=rsi_signal.feature_contributions,
                    regime=rsi_signal.regime,
                    data_quality=rsi_signal.data_quality,
                    staleness=rsi_signal.staleness,
                    metadata={**rsi_signal.metadata, "strategy": self.name},
                )

        # No qualifying signal
        return None

    @property
    def required_bars(self) -> int:
        """Minimum bars required for signal generation."""
        return max(
            self.adx_model.config.min_data_points,
            self.rsi_model.config.min_data_points,
        )
