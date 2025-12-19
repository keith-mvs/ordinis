"""
Parabolic SAR Trend Following Strategy.

Pure trend-following system using Parabolic SAR for entries and trailing stops.
"""

from datetime import datetime

import pandas as pd

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Signal
from ordinis.engines.signalcore.models import ParabolicSARModel

from .base import BaseStrategy


class ParabolicSARStrategy(BaseStrategy):
    """
    Parabolic SAR Trend Following Strategy.

    Follows trends using Parabolic SAR as both entry signal and trailing stop.
    - Enter on SAR reversal (new trend starts)
    - Exit when SAR reverses against position

    Default Parameters:
    - acceleration: 0.02 (AF increment)
    - maximum: 0.2 (Max AF)
    - min_trend_bars: 3 (Minimum bars before entry)

    Best Markets:
        - Strongly trending markets
        - Avoid in ranging/choppy conditions
        - Works with all timeframes

    Risk Management:
        - SAR acts as dynamic trailing stop
        - No fixed stop loss needed
        - SAR accelerates with trend strength
    """

    def configure(self):
        """Configure Parabolic SAR parameters."""
        # Set default parameters
        self.params.setdefault("acceleration", 0.02)
        self.params.setdefault("maximum", 0.2)
        self.params.setdefault("min_trend_bars", 3)
        self.params.setdefault("min_bars", 50)

        # Create underlying signal model
        model_config = ModelConfig(
            model_id=f"{self.name}-psar-model",
            model_type="trend",
            parameters={
                "acceleration": self.params["acceleration"],
                "maximum": self.params["maximum"],
                "min_trend_bars": self.params["min_trend_bars"],
            },
        )

        self.model = ParabolicSARModel(model_config)

    def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate Parabolic SAR signal.

        Args:
            data: Historical OHLCV data with DatetimeIndex
            timestamp: Current timestamp

        Returns:
            Signal object or None
        """
        # Validate data
        is_valid, _msg = self.validate_data(data)
        if not is_valid:
            return None

        try:
            # Generate signal using Parabolic SAR model
            signal = self.model.generate(data, timestamp)

            # Enrich signal metadata with strategy info
            if signal:
                signal.metadata["strategy"] = self.name

                # Add SAR as stop loss level
                sar_value = signal.metadata.get("current_sar", 0)
                current_price = signal.metadata.get("current_price", 0)

                if signal.direction.value == "long":
                    signal.metadata["stop_loss"] = sar_value
                    signal.metadata["take_profit"] = current_price * 1.15  # 15% target
                elif signal.direction.value == "short":
                    signal.metadata["stop_loss"] = sar_value
                    signal.metadata["take_profit"] = current_price * 0.85  # 15% target

            return signal

        except Exception as e:
            # Log error but don't crash
            print(f"Error generating Parabolic SAR signal: {e}")
            return None
