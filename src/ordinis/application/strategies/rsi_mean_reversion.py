"""
RSI Mean Reversion Strategy.

Trades mean reversion signals using the Relative Strength Index (RSI) indicator.
"""

from datetime import datetime

import pandas as pd

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Signal
from ordinis.engines.signalcore.models import RSIMeanReversionModel

from .base import BaseStrategy


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy.

    Enters long positions when RSI indicates oversold conditions
    and exits when price reverts to mean or RSI indicates overbought.

    Default Parameters:
    - rsi_period: 14 (standard RSI calculation period)
    - oversold_threshold: 30 (RSI level indicating oversold)
    - overbought_threshold: 70 (RSI level indicating overbought)
    - extreme_oversold: 20 (extreme oversold for high conviction)
    - extreme_overbought: 80 (extreme overbought for exit signals)
    """

    def configure(self):
        """Configure RSI mean reversion parameters."""
        # Set default parameters if not provided
        self.params.setdefault("rsi_period", 14)
        self.params.setdefault("oversold_threshold", 30)
        self.params.setdefault("overbought_threshold", 70)
        self.params.setdefault("extreme_oversold", 20)
        self.params.setdefault("extreme_overbought", 80)
        self.params.setdefault("min_bars", self.params["rsi_period"] + 20)

        # Create underlying signal model
        model_config = ModelConfig(
            model_id=f"{self.name}-rsi-model",
            model_type="mean_reversion",
            parameters={
                "rsi_period": self.params["rsi_period"],
                "oversold_threshold": self.params["oversold_threshold"],
                "overbought_threshold": self.params["overbought_threshold"],
                "extreme_oversold": self.params.get("extreme_oversold", 20),
                "extreme_overbought": self.params.get("extreme_overbought", 80),
            },
        )

        self.model = RSIMeanReversionModel(model_config)

    async def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate RSI mean reversion signal.

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
            # Generate signal using RSI model
            signal = await self.model.generate(data, timestamp)
            return signal
        except Exception:
            return None

    def get_description(self) -> str:
        """Get strategy description."""
        return f"""RSI Mean Reversion Strategy

Trades mean reversion opportunities using the Relative Strength Index.

Entry Rules:
- BUY when RSI < {self.params['oversold_threshold']} (oversold)
- Higher conviction when RSI < {self.params.get('extreme_oversold', 20)} (extreme oversold)

Exit Rules:
- SELL when RSI > {self.params['overbought_threshold']} (overbought)
- Take profit when RSI > {self.params.get('extreme_overbought', 80)} (extreme overbought)

Parameters:
- RSI Period: {self.params['rsi_period']} bars
- Oversold Threshold: {self.params['oversold_threshold']}
- Overbought Threshold: {self.params['overbought_threshold']}

Best For:
- Range-bound markets
- Mean-reverting assets
- Counter-trend trading
- Lower volatility environments

Risk Considerations:
- Can struggle in strong trends
- May experience drawdowns during breakouts
- Requires proper risk management for trend changes
"""

    def get_required_bars(self) -> int:
        """Get minimum bars required."""
        return self.params.get("min_bars", self.params["rsi_period"] + 20)
