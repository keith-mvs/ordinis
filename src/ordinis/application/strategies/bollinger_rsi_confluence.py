"""
Bollinger RSI Confluence Strategy.

A multi-indicator mean reversion strategy combining Bollinger Bands and RSI
using Goldman Sachs gs-quant adapted indicators for production-grade analytics.

This strategy wraps BollingerRSIConfluenceModel for SignalCore integration.
"""

from datetime import datetime

import pandas as pd

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Signal
from ordinis.engines.signalcore.models import BollingerRSIConfluenceModel

from .base import BaseStrategy


class BollingerRSIConfluenceStrategy(BaseStrategy):
    """
    Bollinger Bands + RSI Confluence Strategy.

    Combines two mean reversion indicators for higher-conviction signals:
    - Bollinger Bands: Price deviation from moving average
    - RSI: Momentum oscillator for overbought/oversold conditions

    Entry signals require BOTH indicators to confirm:
    - LONG: Price below lower band AND RSI oversold
    - SHORT: Price above upper band AND RSI overbought

    Uses gs-quant adapted indicators for production-grade calculations.

    Default Parameters:
    - bb_period: 20 (Bollinger Bands lookback)
    - bb_std: 2.0 (Standard deviations for bands)
    - rsi_period: 14 (RSI calculation period)
    - rsi_oversold: 30 (RSI oversold threshold)
    - rsi_overbought: 70 (RSI overbought threshold)
    - vol_lookback: 22 (Volatility calculation window)
    - min_volatility: 5.0 (Minimum annualized vol % for trading)
    """

    def configure(self) -> None:
        """Configure strategy parameters."""
        # Bollinger Bands parameters
        self.params.setdefault("bb_period", 20)
        self.params.setdefault("bb_std", 2.0)

        # RSI parameters
        self.params.setdefault("rsi_period", 14)
        self.params.setdefault("rsi_oversold", 30)
        self.params.setdefault("rsi_overbought", 70)

        # Extreme thresholds for higher conviction
        self.params.setdefault("rsi_extreme_oversold", 20)
        self.params.setdefault("rsi_extreme_overbought", 80)

        # Volatility filter
        self.params.setdefault("vol_lookback", 22)
        self.params.setdefault("min_volatility", 5.0)  # Minimum 5% annualized vol

        # Minimum bars required
        self.params.setdefault(
            "min_bars",
            max(self.params["bb_period"], self.params["rsi_period"]) + 30,
        )

        # Create underlying SignalCore model
        model_config = ModelConfig(
            model_id=f"{self.name}-bb-rsi-confluence-model",
            model_type="confluence",
            parameters={
                "bb_period": self.params["bb_period"],
                "bb_std": self.params["bb_std"],
                "rsi_period": self.params["rsi_period"],
                "rsi_oversold": self.params["rsi_oversold"],
                "rsi_overbought": self.params["rsi_overbought"],
                "rsi_extreme_oversold": self.params["rsi_extreme_oversold"],
                "rsi_extreme_overbought": self.params["rsi_extreme_overbought"],
                "min_volatility": self.params["min_volatility"],
                "vol_lookback": self.params["vol_lookback"],
            },
        )

        self.model = BollingerRSIConfluenceModel(model_config)

    async def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate confluence signal from Bollinger Bands and RSI.

        Args:
            data: Historical OHLCV data with DatetimeIndex
            timestamp: Current timestamp

        Returns:
            Signal object or None if no signal
        """
        # Validate data
        is_valid, msg = self.validate_data(data)
        if not is_valid:
            return None

        try:
            # Generate signal using the underlying model
            signal = await self.model.generate(data, timestamp)

            # Enrich signal metadata with strategy info
            if signal:
                signal.metadata["strategy"] = self.name

                # Add stop loss and take profit based on bands
                middle_band = signal.metadata.get("middle_band", 0)
                lower_band = signal.metadata.get("lower_band", 0)
                upper_band = signal.metadata.get("upper_band", 0)
                current_price = signal.metadata.get("current_price", 0)

                if signal.direction.value == "long":
                    signal.metadata["stop_loss"] = lower_band * 0.99
                    signal.metadata["take_profit"] = middle_band
                elif signal.direction.value == "short":
                    signal.metadata["stop_loss"] = upper_band * 1.01
                    signal.metadata["take_profit"] = middle_band
                else:
                    signal.metadata["stop_loss"] = current_price * 0.98
                    signal.metadata["take_profit"] = current_price * 1.02

            return signal

        except Exception:
            return None

    def get_description(self) -> str:
        """Get human-readable strategy description."""
        return f"""Bollinger RSI Confluence Strategy

Multi-indicator mean reversion strategy using Goldman Sachs gs-quant analytics.
Requires BOTH Bollinger Bands and RSI to confirm signals for higher conviction.

Entry Rules:
- LONG: Price < Lower Band ({self.params['bb_std']}σ) AND RSI < {self.params['rsi_oversold']}
- SHORT: Price > Upper Band ({self.params['bb_std']}σ) AND RSI > {self.params['rsi_overbought']}

High Conviction Signals:
- Extreme oversold: RSI < {self.params['rsi_extreme_oversold']}
- Extreme overbought: RSI > {self.params['rsi_extreme_overbought']}

Volatility Filter:
- Minimum volatility: {self.params['min_volatility']}% annualized
- Higher volatility reduces conviction

Parameters:
- Bollinger Period: {self.params['bb_period']} bars
- Bollinger Width: {self.params['bb_std']} standard deviations
- RSI Period: {self.params['rsi_period']} bars
- Volatility Lookback: {self.params['vol_lookback']} bars

Best For:
- Range-bound markets
- Mean-reverting assets
- Avoiding false breakouts

Risk Considerations:
- May miss trending opportunities
- Requires proper stop-loss for trend changes
- Confluence reduces signal frequency but improves quality
"""

    def get_required_bars(self) -> int:
        """Get minimum bars required for signal generation."""
        return self.params.get(
            "min_bars",
            max(self.params["bb_period"], self.params["rsi_period"]) + 30,
        )
