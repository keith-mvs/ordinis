"""
Quant integrations for Ordinis.

Includes adapters for:
- NVIDIA blueprint integrations (Quantitative Portfolio Optimization)
- Goldman Sachs gs-quant timeseries analytics (standalone, no API required)
"""

from .gs_quant_adapter import (
    AnnualizationFactor,
    # Types
    Returns,
    Window,
    beta,
    bollinger_bands,
    correlation,
    exponential_moving_average,
    # Utilities
    generate_series,
    macd,
    max_drawdown,
    # Technical Indicators
    moving_average,
    percentiles,
    prices,
    # Returns/Prices
    returns,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    rsi,
    sharpe_ratio,
    # Risk Metrics
    volatility,
    # Statistics
    zscores,
)
from .qpo_adapter import (
    DEFAULT_QPO_SRC,
    QPOEnvironmentError,
    QPOPortfolioOptimizer,
    QPOScenarioGenerator,
)

__all__ = [
    # QPO Adapter
    "DEFAULT_QPO_SRC",
    "QPOEnvironmentError",
    "QPOPortfolioOptimizer",
    "QPOScenarioGenerator",
    # GS Quant Adapter - Types
    "Returns",
    "Window",
    "AnnualizationFactor",
    # GS Quant Adapter - Returns/Prices
    "returns",
    "prices",
    # GS Quant Adapter - Technical Indicators
    "moving_average",
    "exponential_moving_average",
    "bollinger_bands",
    "rsi",
    "macd",
    # GS Quant Adapter - Risk Metrics
    "volatility",
    "sharpe_ratio",
    "max_drawdown",
    "correlation",
    "beta",
    # GS Quant Adapter - Statistics
    "zscores",
    "percentiles",
    "rolling_std",
    "rolling_mean",
    "rolling_min",
    "rolling_max",
    # GS Quant Adapter - Utilities
    "generate_series",
]
