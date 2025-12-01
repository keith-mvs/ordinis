"""
MACD Strategy.

Trades momentum signals using MACD (Moving Average Convergence Divergence) indicator.
"""

from datetime import datetime

import pandas as pd

from engines.signalcore.core.model import ModelConfig
from engines.signalcore.core.signal import Signal
from engines.signalcore.models import MACDModel

from .base import BaseStrategy


class MACDStrategy(BaseStrategy):
    """
    MACD Momentum Strategy.

    Enters long positions on bullish MACD crossovers
    and exits on bearish crossovers.

    Default Parameters:
    - fast_period: 12 (fast EMA period)
    - slow_period: 26 (slow EMA period)
    - signal_period: 9 (signal line period)
    """

    def configure(self):
        """Configure MACD parameters."""
        # Set default parameters if not provided
        self.params.setdefault("fast_period", 12)
        self.params.setdefault("slow_period", 26)
        self.params.setdefault("signal_period", 9)
        self.params.setdefault(
            "min_bars",
            self.params["slow_period"] + self.params["signal_period"] + 20,
        )

        # Create underlying signal model
        model_config = ModelConfig(
            model_id=f"{self.name}-macd-model",
            model_type="momentum",
            parameters={
                "fast_period": self.params["fast_period"],
                "slow_period": self.params["slow_period"],
                "signal_period": self.params["signal_period"],
            },
        )

        self.model = MACDModel(model_config)

    def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate MACD signal.

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
            # Generate signal using MACD model
            signal = self.model.generate(data, timestamp)
            return signal
        except Exception:
            return None

    def get_description(self) -> str:
        """Get strategy description."""
        return f"""MACD Momentum Strategy

Trades momentum opportunities using MACD crossovers.

Entry Rules:
- BUY on bullish crossover (MACD crosses above signal line)
- Higher conviction when crossover occurs above zero line
- Histogram strength adds to conviction

Exit Rules:
- SELL on bearish crossover (MACD crosses below signal line)
- Higher conviction when crossover occurs below zero line

Parameters:
- Fast Period: {self.params['fast_period']} bars
- Slow Period: {self.params['slow_period']} bars
- Signal Period: {self.params['signal_period']} bars

Best For:
- Trending markets
- Momentum-driven assets
- Medium-term trading
- Trend confirmation

Risk Considerations:
- Lagging indicator (may miss early moves)
- Can generate false signals in ranging markets
- Works best with trend confirmation
"""

    def get_required_bars(self) -> int:
        """Get minimum bars required."""
        return self.params.get(
            "min_bars",
            self.params["slow_period"] + self.params["signal_period"] + 20,
        )
