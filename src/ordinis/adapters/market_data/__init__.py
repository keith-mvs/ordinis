"""
Market data adapters for various data providers.

Provides adapters for:
- AlphaVantage
- Finnhub
- IEX Cloud
- Twelve Data
- Yahoo Finance
- Data Aggregator (multi-source consensus)
"""

from ordinis.adapters.market_data.aggregator import (
    AggregatedQuote,
    AggregationConfig,
    AggregationMethod,
    DataAggregator,
    ProviderResult,
    ProviderStats,
    ProviderWeight,
)
from ordinis.adapters.market_data.alphavantage import AlphaVantageDataPlugin
from ordinis.adapters.market_data.finnhub import FinnhubDataPlugin
from ordinis.adapters.market_data.fmp import FMPDataPlugin
from ordinis.adapters.market_data.iex import IEXDataPlugin
from ordinis.adapters.market_data.massive import MassiveDataPlugin
from ordinis.adapters.market_data.newsapi import NewsAPIDataPlugin
from ordinis.adapters.market_data.twelvedata import TwelveDataPlugin
from ordinis.adapters.market_data.yahoo import YahooDataPlugin

__all__ = [
    # Aggregator
    "AggregatedQuote",
    "AggregationConfig",
    "AggregationMethod",
    # Providers
    "AlphaVantageDataPlugin",
    "DataAggregator",
    "FMPDataPlugin",
    "FinnhubDataPlugin",
    "IEXDataPlugin",
    "MassiveDataPlugin",
    "NewsAPIDataPlugin",
    "ProviderResult",
    "ProviderStats",
    "ProviderWeight",
    "TwelveDataPlugin",
    "YahooDataPlugin",
]
