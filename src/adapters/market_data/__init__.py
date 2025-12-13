"""
Market data adapters for various data providers.

Provides adapters for:
- AlphaVantage
- Finnhub
- IEX Cloud
- Polygon.io
- Twelve Data
"""

from .alphavantage import AlphaVantageDataPlugin
from .finnhub import FinnhubDataPlugin
from .iex import IEXDataPlugin
from .polygon import PolygonDataPlugin
from .twelvedata import TwelveDataPlugin

__all__ = [
    "AlphaVantageDataPlugin",
    "FinnhubDataPlugin",
    "PolygonDataPlugin",
    "TwelveDataPlugin",
    "IEXDataPlugin",
]
