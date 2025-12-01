"""
Bollinger Bands Strategy.

Trades mean reversion signals using Bollinger Bands indicator.
"""

from datetime import datetime

import pandas as pd

from engines.signalcore.core.model import ModelConfig
from engines.signalcore.core.signal import Signal
from engines.signalcore.models import BollingerBandsModel

from .base import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Mean Reversion Strategy.

    Enters long positions when price touches lower band (oversold)
    and exits when price touches upper band (overbought).

    Default Parameters:
    - bb_period: 20 (standard BB calculation period)
    - bb_std: 2.0 (number of standard deviations)
    - min_band_width: 0.02 (minimum band width to avoid low volatility)
    """

    def configure(self):
        """Configure Bollinger Bands parameters."""
        # Set default parameters if not provided
        self.params.setdefault("bb_period", 20)
        self.params.setdefault("bb_std", 2.0)
        self.params.setdefault("min_band_width", 0.02)
        self.params.setdefault("min_bars", self.params["bb_period"] + 20)

        # Create underlying signal model
        model_config = ModelConfig(
            model_id=f"{self.name}-bb-model",
            model_type="mean_reversion",
            parameters={
                "bb_period": self.params["bb_period"],
                "bb_std": self.params["bb_std"],
                "min_band_width": self.params["min_band_width"],
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
            # Generate signal using BB model
            signal = self.model.generate(data, timestamp)
            return signal
        except Exception:
            return None

    def get_description(self) -> str:
        """Get strategy description."""
        return f"""Bollinger Bands Mean Reversion Strategy

Trades mean reversion opportunities using Bollinger Bands.

Entry Rules:
- BUY when price touches or crosses below lower band (oversold)
- Higher conviction when price moves further below lower band

Exit Rules:
- SELL when price touches or crosses above upper band (overbought)
- Higher conviction when price moves further above upper band

Parameters:
- BB Period: {self.params['bb_period']} bars
- Standard Deviations: {self.params['bb_std']}
- Min Band Width: {self.params['min_band_width']} (filters low volatility)

Best For:
- Range-bound markets
- Mean-reverting assets
- Counter-trend trading
- Moderate volatility environments

Risk Considerations:
- Can struggle in strong trends (band walking)
- May experience drawdowns during breakouts
- Avoid low volatility periods (compressed bands)
"""

    def get_required_bars(self) -> int:
        """Get minimum bars required."""
        return self.params.get("min_bars", self.params["bb_period"] + 20)
