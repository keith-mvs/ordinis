"""
Market data adapters for various data providers.

Provides adapters for:
- AlphaVantage
- Finnhub
- IEX Cloud
- Polygon.io
- Twelve Data
- Yahoo Finance
"""

from ordinis.adapters.market_data.alphavantage import AlphaVantageDataPlugin
from ordinis.adapters.market_data.finnhub import FinnhubDataPlugin
from ordinis.adapters.market_data.iex import IEXDataPlugin
from ordinis.adapters.market_data.polygon import PolygonDataPlugin
from ordinis.adapters.market_data.twelvedata import TwelveDataPlugin
from ordinis.adapters.market_data.yahoo import YahooDataPlugin

__all__ = [
    "AlphaVantageDataPlugin",
    "FinnhubDataPlugin",
    "IEXDataPlugin",
    "PolygonDataPlugin",
    "TwelveDataPlugin",
    "YahooDataPlugin",
]
