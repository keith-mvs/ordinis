"""
Tests for market data domain models.
Tests cover:
- Bar model
"""

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from ordinis.domain.market_data import Bar


class TestBar:
    """Tests for Bar model."""

    @pytest.mark.unit
    def test_bar_creation(self):
        """Test creating a Bar."""
        now = datetime.now(timezone.utc)
        bar = Bar(
            symbol="AAPL",
            timestamp=now,
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.50,
            volume=1000000.0,
        )

        assert bar.symbol == "AAPL"
        assert bar.timestamp == now
        assert bar.open == 150.00
        assert bar.high == 152.00
        assert bar.low == 149.00
        assert bar.close == 151.50
        assert bar.volume == 1000000.0
        assert bar.vwap is None
        assert bar.trade_count is None

    @pytest.mark.unit
    def test_bar_with_optional_fields(self):
        """Test Bar with optional fields."""
        now = datetime.now(timezone.utc)
        bar = Bar(
            symbol="AAPL",
            timestamp=now,
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.50,
            volume=1000000.0,
            vwap=150.75,
            trade_count=5000,
        )

        assert bar.vwap == 150.75
        assert bar.trade_count == 5000

    @pytest.mark.unit
    def test_bar_is_frozen(self):
        """Test Bar is immutable (frozen)."""
        now = datetime.now(timezone.utc)
        bar = Bar(
            symbol="AAPL",
            timestamp=now,
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.50,
            volume=1000000.0,
        )

        with pytest.raises(ValidationError):
            bar.close = 155.00

    @pytest.mark.unit
    def test_bar_equality(self):
        """Test Bar equality."""
        now = datetime.now(timezone.utc)
        bar1 = Bar(
            symbol="AAPL",
            timestamp=now,
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.50,
            volume=1000000.0,
        )
        bar2 = Bar(
            symbol="AAPL",
            timestamp=now,
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.50,
            volume=1000000.0,
        )

        assert bar1 == bar2

    @pytest.mark.unit
    def test_bar_inequality(self):
        """Test Bar inequality."""
        now = datetime.now(timezone.utc)
        bar1 = Bar(
            symbol="AAPL",
            timestamp=now,
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.50,
            volume=1000000.0,
        )
        bar2 = Bar(
            symbol="GOOGL",
            timestamp=now,
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.50,
            volume=1000000.0,
        )

        assert bar1 != bar2

    @pytest.mark.unit
    def test_bar_hashable(self):
        """Test Bar is hashable (can be used in sets/dicts)."""
        now = datetime.now(timezone.utc)
        bar = Bar(
            symbol="AAPL",
            timestamp=now,
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.50,
            volume=1000000.0,
        )

        # Should be able to hash and use in set
        bar_set = {bar}
        assert bar in bar_set

    @pytest.mark.unit
    def test_bar_dict_conversion(self):
        """Test Bar conversion to dict."""
        now = datetime.now(timezone.utc)
        bar = Bar(
            symbol="AAPL",
            timestamp=now,
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.50,
            volume=1000000.0,
        )

        d = bar.model_dump()
        assert d["symbol"] == "AAPL"
        assert d["open"] == 150.00
        assert d["close"] == 151.50
