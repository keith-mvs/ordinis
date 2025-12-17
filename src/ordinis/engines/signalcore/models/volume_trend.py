"""
Volume Trend Model.

Combines Price vs VWAP and On-Balance Volume (OBV) trend to identify
volume-supported price moves.
"""

from datetime import datetime

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType
from ..features.technical import TechnicalIndicators


class VolumeTrendModel(Model):
    """
    Volume Trend trading model.

    Uses VWAP (Volume Weighted Average Price) as a dynamic benchmark
    and OBV (On-Balance Volume) to confirm trend strength.

    Parameters:
        vwap_window: Rolling window for VWAP (default 20)
        obv_sma_window: Window for OBV smoothing (default 20)

    Signals:
        - ENTRY/LONG: Price > VWAP and OBV > SMA(OBV)
        - ENTRY/SHORT: Price < VWAP and OBV < SMA(OBV)
        - HOLD: Mixed signals
    """

    def __init__(self, config: ModelConfig):
        """Initialize Volume Trend model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.vwap_window = params.get("vwap_window", 20)
        self.obv_sma_window = params.get("obv_sma_window", 20)

        # Update min data points
        self.config.min_data_points = max(self.vwap_window, self.obv_sma_window) + 10

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate trading signal from Volume analysis.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal with Volume Trend prediction
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

        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # Calculate Indicators
        # Note: Using rolling VWAP for short-term trend
        vwap = TechnicalIndicators.vwap(high, low, close, volume, window=self.vwap_window)

        obv = TechnicalIndicators.obv(close, volume)
        obv_sma = TechnicalIndicators.sma(obv, self.obv_sma_window)

        current_price = close.iloc[-1]
        current_vwap = vwap.iloc[-1]
        current_obv = obv.iloc[-1]
        current_obv_sma = obv_sma.iloc[-1]

        # Determine signal
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        # Logic
        # Bullish: Price above institutional average (VWAP) AND Volume confirming (OBV > SMA)
        if current_price > current_vwap and current_obv > current_obv_sma:
            signal_type = SignalType.ENTRY
            direction = Direction.LONG

            # Score based on distance from VWAP (momentum)
            dist_pct = (current_price - current_vwap) / current_vwap
            score = min(dist_pct * 10, 1.0)  # 10% above VWAP = score 1.0
            probability = 0.65
            expected_return = 0.01

        # Bearish: Price below VWAP AND Volume confirming down move
        elif current_price < current_vwap and current_obv < current_obv_sma:
            signal_type = SignalType.ENTRY
            direction = Direction.SHORT

            dist_pct = (current_vwap - current_price) / current_vwap
            score = -min(dist_pct * 10, 1.0)
            probability = 0.65
            expected_return = -0.01

        return Signal(
            model_id=self.config.model_id,
            signal_type=signal_type,
            direction=direction,
            score=score,
            timestamp=timestamp,
            metadata={
                "vwap": float(current_vwap),
                "obv": float(current_obv),
                "obv_sma": float(current_obv_sma),
                "current_price": float(current_price),
            },
            symbol=str(symbol),
            probability=probability,
            expected_return=expected_return,
            confidence_interval=(expected_return - 0.005, expected_return + 0.005),
            model_version="1.0",
        )
