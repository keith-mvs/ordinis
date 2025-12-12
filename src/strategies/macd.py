"""
MACD Strategy.

Momentum and trend identification strategy using MACD indicator.
"""

from datetime import datetime

import pandas as pd

from src.engines.signalcore.core.model import ModelConfig
from src.engines.signalcore.core.signal import Signal
from src.engines.signalcore.models import MACDModel

from .base import BaseStrategy


class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy.

    Trades momentum shifts and trend changes based on MACD line crossovers
    with the signal line. Designed for trending markets.

    Default Parameters:
    - fast_period: 12 (Fast EMA period)
    - slow_period: 26 (Slow EMA period)
    - signal_period: 9 (Signal line period)
    - min_histogram: 0.0 (Minimum histogram value for entry)
    """

    def configure(self):
        """Configure MACD parameters."""
        # Set default parameters if not provided
        self.params.setdefault("fast_period", 12)
        self.params.setdefault("slow_period", 26)
        self.params.setdefault("signal_period", 9)
        self.params.setdefault("min_histogram", 0.0)
        self.params.setdefault(
            "min_bars", self.params["slow_period"] + self.params["signal_period"] + 20
        )

        # Create underlying signal model
        model_config = ModelConfig(
            model_id=f"{self.name}-macd-model",
            model_type="momentum",
            parameters={
                "fast_period": self.params["fast_period"],
                "slow_period": self.params["slow_period"],
                "signal_period": self.params["signal_period"],
                "min_histogram": self.params.get("min_histogram", 0.0),
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

            # Enrich signal metadata with strategy info
            if signal:
                signal.metadata["strategy"] = self.name

                # Add stop loss and take profit based on MACD
                current_price = signal.metadata.get("current_price", 0)
                histogram = signal.metadata.get("histogram", 0)

                # Use histogram size to set risk/reward
                atr_proxy = abs(histogram) * 100 if histogram != 0 else current_price * 0.02

                # Set stop loss and take profit
                if signal.direction.value == "long":
                    signal.metadata["stop_loss"] = current_price - (atr_proxy * 2)
                    signal.metadata["take_profit"] = current_price + (atr_proxy * 3)
                elif signal.direction.value == "short":
                    signal.metadata["stop_loss"] = current_price + (atr_proxy * 2)
                    signal.metadata["take_profit"] = current_price - (atr_proxy * 3)
                else:
                    signal.metadata["stop_loss"] = current_price * 0.98
                    signal.metadata["take_profit"] = current_price * 1.02

            return signal
        except Exception:
            return None

    def get_description(self) -> str:
        """Get strategy description."""
        return f"""MACD Momentum Strategy

Trades trend changes and momentum shifts using the Moving Average
Convergence Divergence (MACD) indicator.

Entry Rules:
- BUY when MACD line crosses above signal line (bullish crossover)
- Higher conviction when crossing above zero line
- Requires minimum histogram of {self.params.get('min_histogram', 0.0):.4f}
- Stronger signals with increasing histogram momentum

Exit Rules:
- SELL when MACD line crosses below signal line (bearish crossover)
- Exit priority when crossing below zero line

Parameters:
- Fast EMA: {self.params['fast_period']} periods
- Slow EMA: {self.params['slow_period']} periods
- Signal Line: {self.params['signal_period']} periods
- Min Histogram: {self.params.get('min_histogram', 0.0):.4f}

Best For:
- Trending markets
- Momentum-driven assets
- Trend following strategies
- Medium to long-term trades

Risk Considerations:
- Lagging indicator (uses EMAs)
- Can whipsaw in ranging markets
- False signals during consolidation
- Best combined with trend filters
"""

    def get_required_bars(self) -> int:
        """Get minimum bars required."""
        return self.params.get(
            "min_bars", self.params["slow_period"] + self.params["signal_period"] + 20
        )
