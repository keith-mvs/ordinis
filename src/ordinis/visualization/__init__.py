"""
Visualization module for trading system.

Provides interactive charts and visual analytics for
trading strategies, backtests, and performance monitoring.

Available Components:
- IndicatorChart: Technical indicator visualizations (BB, MACD, RSI)
- ChartUtils: Utilities for chart theming, export, and comparison
"""

from .charts import ChartUtils
from .indicators import IndicatorChart

__all__ = [
    "ChartUtils",
    "IndicatorChart",
]
