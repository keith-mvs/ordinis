"""
Visualization module for trading system.

Provides interactive charts, dashboards, and visual analytics for
trading strategies, backtests, and performance monitoring.

Available Components:
- IndicatorChart: Technical indicator visualizations (BB, MACD, RSI)
- ChartUtils: Utilities for chart theming, export, and comparison
- PerformanceDashboard: Interactive KPI and performance dashboard
"""

from .charts import ChartUtils
from .dashboard import PerformanceDashboard
from .indicators import IndicatorChart

__all__ = [
    "ChartUtils",
    "IndicatorChart",
    "PerformanceDashboard",
]
