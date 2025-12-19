"""
Mean Reversion Model.

Strategy:
1. Trend Filter: Trade only in direction of long-term trend (e.g. 200 SMA).
2. Entry Signal:
   - RSI Oversold (e.g. < 30) in Uptrend -> Long
   - RSI Overbought (e.g. > 70) in Downtrend -> Short
   - OR Bollinger Band Breach (Close < Lower in Uptrend, Close > Upper in Downtrend)
3. Volume Confirmation: Volume > Average Volume * Factor
"""

from datetime import datetime
import logging

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class MeanReversionModel(Model):
    """
    Mean Reversion trading model.

    Parameters:
        rsi_period: RSI period (default 14)
        rsi_oversold: RSI oversold threshold (default 30)
        rsi_overbought: RSI overbought threshold (default 70)
        bb_period: Bollinger Bands period (default 20)
        bb_std: Bollinger Bands std dev (default 2.0)
        volume_period: Volume SMA period (default 20)
        volume_factor: Volume threshold factor (default 1.5)
        trend_filter_period: Trend SMA period (default 200)
    """

    def __init__(self, config: ModelConfig):
        """Initialize Mean Reversion model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 30)
        self.rsi_overbought = params.get("rsi_overbought", 70)

        self.bb_period = params.get("bb_period", 20)
        self.bb_std = params.get("bb_std", 2.0)

        self.volume_period = params.get("volume_period", 20)
        self.volume_factor = params.get("volume_factor", 1.5)

        self.trend_filter_period = params.get("trend_filter_period", 200)

        # Update min data points
        max_period = max(
            self.rsi_period, self.bb_period, self.volume_period, self.trend_filter_period
        )
        self.config.min_data_points = max(self.config.min_data_points, max_period + 10)

    async def generate(self, symbol: str, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """Generate signals from data."""
        if len(data) < self.config.min_data_points:
            return None

        # Calculate Indicators
        close = data["close"]
        volume = data["volume"]

        # 1. Trend Filter
        if self.trend_filter_period > 0:
            trend_sma = TechnicalIndicators.sma(close, self.trend_filter_period)
        else:
            trend_sma = pd.Series(0, index=close.index)  # Dummy

        # 2. RSI
        rsi = TechnicalIndicators.rsi(close, self.rsi_period)

        # 3. Bollinger Bands
        _, bb_upper, bb_lower = TechnicalIndicators.bollinger_bands(
            close, self.bb_period, self.bb_std
        )

        # 4. Volume SMA
        vol_sma = TechnicalIndicators.sma(volume, self.volume_period)

        # We look at the last completed bar
        i = -1

        current_close = close.iloc[i]
        current_vol = volume.iloc[i]
        current_vol_sma = vol_sma.iloc[i]
        current_rsi = rsi.iloc[i]
        current_trend = trend_sma.iloc[i]
        current_lower = bb_lower.iloc[i]
        current_upper = bb_upper.iloc[i]

        # Check Volume Confirmation
        if current_vol < (current_vol_sma * self.volume_factor):
            logger.debug(
                f"No volume confirmation: {current_vol} < {current_vol_sma * self.volume_factor}"
            )
            return None  # No volume confirmation

        # Determine Trend
        is_uptrend = True
        is_downtrend = True
        if self.trend_filter_period > 0:
            is_uptrend = current_close > current_trend
            is_downtrend = current_close < current_trend

        logger.debug(
            f"Trend: Up={is_uptrend}, Down={is_downtrend}, Close={current_close}, TrendSMA={current_trend}"
        )

        # Logic
        signal_direction = None
        reason = ""

        # Long Entry
        if is_uptrend:
            if current_rsi < self.rsi_oversold:
                signal_direction = Direction.LONG
                reason = f"RSI Oversold ({current_rsi:.2f} < {self.rsi_oversold})"
            elif current_close < current_lower:
                signal_direction = Direction.LONG
                reason = f"BB Lower Breach ({current_close:.2f} < {current_lower:.2f})"

        # Short Entry
        if is_downtrend:
            if current_rsi > self.rsi_overbought:
                signal_direction = Direction.SHORT
                reason = f"RSI Overbought ({current_rsi:.2f} > {self.rsi_overbought})"
            elif current_close > current_upper:
                signal_direction = Direction.SHORT
                reason = f"BB Upper Breach ({current_close:.2f} > {current_upper:.2f})"

        if signal_direction:
            logger.info(
                f"SIGNAL GENERATED: {signal_direction} {symbol} at {timestamp}. Reason: {reason}. Vol: {current_vol:.0f} > {current_vol_sma * self.volume_factor:.0f}"
            )
            return Signal(
                signal_type=SignalType.ENTRY,
                direction=signal_direction,
                symbol=symbol,
                timestamp=timestamp,
                price=current_close,
                confidence=1.0,
                model_id=self.config.model_id,
                model_version=self.config.version,
                metadata={
                    "rsi": float(current_rsi),
                    "bb_lower": float(current_lower),
                    "bb_upper": float(current_upper),
                    "trend_sma": float(current_trend),
                    "reason": reason,
                    "volume_ratio": float(current_vol / current_vol_sma)
                    if current_vol_sma > 0
                    else 0,
                },
            )

        return None

        return None
