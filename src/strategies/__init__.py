"""
Pre-built Trading Strategies Library.

This module provides ready-to-use trading strategies that can be easily
integrated into the Intelligent Investor system.

Available Strategies:
- RSIMeanReversionStrategy: Mean reversion using RSI indicator
- MovingAverageCrossoverStrategy: Trend following with MA crossovers
- MomentumBreakoutStrategy: Breakout trading with momentum confirmation
- BollingerBandsStrategy: Mean reversion using Bollinger Bands
- MACDStrategy: Momentum trading using MACD crossovers
"""

from .base import BaseStrategy
from .bollinger_bands import BollingerBandsStrategy
from .macd import MACDStrategy
from .momentum_breakout import MomentumBreakoutStrategy
from .moving_average_crossover import MovingAverageCrossoverStrategy
from .rsi_mean_reversion import RSIMeanReversionStrategy

__all__ = [
    "BaseStrategy",
    "RSIMeanReversionStrategy",
    "MovingAverageCrossoverStrategy",
    "MomentumBreakoutStrategy",
    "BollingerBandsStrategy",
    "MACDStrategy",
]
