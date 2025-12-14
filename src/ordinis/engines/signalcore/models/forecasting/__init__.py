"""
Forecasting models for price and volatility.

- Price forecasting: Nixtla statsforecast (ARIMA, AutoARIMA)
- Volatility estimation: arch library (GARCH, EGARCH, TGARCH)
"""

from .statsforecast_model import (
    ARIMAForecastModel,
    AutoARIMAForecastModel,
    ForecastResult,
)
from .volatility_model import (
    EGARCHVolatilityModel,
    GARCHVolatilityModel,
    TGARCHVolatilityModel,
    VolatilityForecast,
)

__all__ = [
    # Price forecasting
    "ARIMAForecastModel",
    "AutoARIMAForecastModel",
    "EGARCHVolatilityModel",
    "ForecastResult",
    # Volatility estimation
    "GARCHVolatilityModel",
    "TGARCHVolatilityModel",
    "VolatilityForecast",
]
