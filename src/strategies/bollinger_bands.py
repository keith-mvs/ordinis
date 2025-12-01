"""
Bollinger Bands Strategy.

Volatility-based mean reversion strategy using Bollinger Bands.
"""

from datetime import datetime

import pandas as pd

from engines.signalcore.core.model import ModelConfig
from engines.signalcore.core.signal import Signal
from engines.signalcore.models import BollingerBandsModel

from .base import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy.

    Trades mean reversion opportunities based on Bollinger Bands,
    which measure price volatility using standard deviations.

    Default Parameters:
    - bb_period: 20 (Bollinger Bands calculation period)
    - bb_std: 2.0 (Number of standard deviations)
    - min_band_width: 0.01 (Minimum volatility for signal generation)
    """

    def configure(self):
        """Configure Bollinger Bands parameters."""
        # Set default parameters if not provided
        self.params.setdefault("bb_period", 20)
        self.params.setdefault("bb_std", 2.0)
        self.params.setdefault("min_band_width", 0.01)
        self.params.setdefault("min_bars", self.params["bb_period"] + 30)

        # Create underlying signal model
        model_config = ModelConfig(
            model_id=f"{self.name}-bb-model",
            model_type="volatility",
            parameters={
                "bb_period": self.params["bb_period"],
                "bb_std": self.params["bb_std"],
                "min_band_width": self.params.get("min_band_width", 0.01),
            },
        )

        self.model = BollingerBandsModel(model_config)

    def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate Bollinger Bands signal.

        Args:
            data: Historical OHLCV data with DatetimeIndex
            timestamp: Current timestamp

        Returns:
            Signal object or None
        """
        # Validate data
        is_valid, msg = self.validate_data(data)
        if not is_valid:
            return None

        try:
            # Generate signal using Bollinger Bands model
            signal = self.model.generate(data, timestamp)
            return signal
        except Exception:
            return None

    def get_description(self) -> str:
        """Get strategy description."""
        return f"""Bollinger Bands Volatility Strategy

Trades mean reversion opportunities using Bollinger Bands as dynamic
support and resistance levels based on price volatility.

Entry Rules:
- BUY when price touches or crosses below the lower band (oversold)
- Higher conviction in high volatility environments
- Requires minimum band width of {self.params.get('min_band_width', 0.01):.1%}

Exit Rules:
- SELL when price reaches middle or upper band
- Exit when price touches or crosses above upper band (overbought)

Parameters:
- BB Period: {self.params['bb_period']} bars
- Standard Deviations: {self.params['bb_std']}
- Min Band Width: {self.params.get('min_band_width', 0.01):.1%}

Best For:
- Range-bound markets
- Mean-reverting assets
- Markets with cyclical volatility
- Sideways trending periods

Risk Considerations:
- Weak in strong trending markets
- Can give false signals in low volatility
- Requires proper position sizing
- Works best with volume confirmation
"""

    def get_required_bars(self) -> int:
        """Get minimum bars required."""
        return self.params.get("min_bars", self.params["bb_period"] + 30)
