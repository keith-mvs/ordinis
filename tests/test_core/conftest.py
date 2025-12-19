"""Fixtures for core module tests."""

from datetime import datetime

import pytest


@pytest.fixture
def valid_quote_data():
    """Valid quote data for testing."""
    return {
        "symbol": "AAPL",
        "bid": 150.0,
        "ask": 150.10,
        "last": 150.05,
        "volume": 1000000,
        "timestamp": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def valid_ohlc_bar():
    """Valid OHLC bar data for testing."""
    return {
        "symbol": "AAPL",
        "open": 149.0,
        "high": 151.0,
        "low": 148.5,
        "close": 150.0,
        "volume": 1000000,
        "timestamp": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def invalid_ohlc_bar():
    """Invalid OHLC bar data for testing (high < low)."""
    return {
        "symbol": "AAPL",
        "open": 150.0,
        "high": 148.0,  # High < Low (invalid)
        "low": 149.0,
        "close": 149.5,
        "volume": 1000000,
        "timestamp": datetime.utcnow().isoformat(),
    }
