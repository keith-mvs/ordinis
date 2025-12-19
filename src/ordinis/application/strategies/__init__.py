"""
Pre-built Trading Strategies Library.

This module provides ready-to-use trading strategies that can be easily
integrated into the Ordinis trading system.

Available Strategies:
- ADXFilteredRSIStrategy: RSI strategy filtered by ADX trend strength
- BollingerBandsStrategy: Volatility-based mean reversion using Bollinger Bands
- BollingerRSIConfluenceStrategy: Multi-indicator confluence using gs-quant analytics
- FibonacciADXStrategy: Fibonacci retracement with ADX confirmation
- MACDStrategy: Momentum and trend identification using MACD
- MomentumBreakoutStrategy: Breakout trading with momentum confirmation
- MovingAverageCrossoverStrategy: Trend following with MA crossovers
- ParabolicSARStrategy: Trend following using Parabolic SAR
- RSIMeanReversionStrategy: Mean reversion using RSI indicator
"""

from .adx_filtered_rsi import ADXFilteredRSIStrategy
from .base import BaseStrategy
from .bollinger_bands import BollingerBandsStrategy
from .bollinger_rsi_confluence import BollingerRSIConfluenceStrategy
from .fibonacci_adx import FibonacciADXStrategy
from .macd import MACDStrategy
from .momentum_breakout import MomentumBreakoutStrategy
from .moving_average_crossover import MovingAverageCrossoverStrategy
from .parabolic_sar_trend import ParabolicSARStrategy
from .rsi_mean_reversion import RSIMeanReversionStrategy

__all__ = [
    "ADXFilteredRSIStrategy",
    "BaseStrategy",
    "BollingerBandsStrategy",
    "BollingerRSIConfluenceStrategy",
    "FibonacciADXStrategy",
    "MACDStrategy",
    "MomentumBreakoutStrategy",
    "MovingAverageCrossoverStrategy",
    "ParabolicSARStrategy",
    "RSIMeanReversionStrategy",
]
