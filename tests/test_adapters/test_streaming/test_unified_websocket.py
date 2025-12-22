"""Tests for UnifiedMarketDataStream and related classes.

Tests cover:
- DataProvider enum
- DataType enum
- ProviderConfig dataclass
- MarketDataEvent dataclass
- SubscriptionRequest dataclass
- ProviderStats dataclass and methods
- UnifiedMarketDataStream initialization and basic methods
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ordinis.adapters.streaming.unified_websocket import (
    DataProvider,
    DataType,
    ProviderConfig,
    MarketDataEvent,
    SubscriptionRequest,
    ProviderStats,
    UnifiedMarketDataStream,
)
from ordinis.adapters.streaming.stream_protocol import StreamStatus, StreamQuote


class TestDataProvider:
    """Tests for DataProvider enum."""

    @pytest.mark.unit
    def test_all_providers_defined(self):
        """Test all expected providers are defined."""
        assert DataProvider.ALPACA is not None
        assert DataProvider.FINNHUB is not None
        assert DataProvider.MASSIVE is not None
        assert DataProvider.POLYGON is not None

    @pytest.mark.unit
    def test_provider_count(self):
        """Test correct number of providers."""
        assert len(DataProvider) == 4


class TestDataType:
    """Tests for DataType enum."""

    @pytest.mark.unit
    def test_all_types_defined(self):
        """Test all expected data types are defined."""
        assert DataType.QUOTE is not None
        assert DataType.TRADE is not None
        assert DataType.BAR is not None
        assert DataType.ORDERBOOK is not None

    @pytest.mark.unit
    def test_type_count(self):
        """Test correct number of data types."""
        assert len(DataType) == 4


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""

    @pytest.mark.unit
    def test_create_minimal_config(self):
        """Test creating config with minimal required fields."""
        config = ProviderConfig(
            provider=DataProvider.ALPACA,
            api_key="test_key",
        )

        assert config.provider == DataProvider.ALPACA
        assert config.api_key == "test_key"
        assert config.api_secret is None
        assert config.enabled is True

    @pytest.mark.unit
    def test_create_full_config(self):
        """Test creating config with all fields."""
        config = ProviderConfig(
            provider=DataProvider.FINNHUB,
            api_key="key123",
            api_secret="secret456",
            base_url="https://api.example.com",
            ws_url="wss://ws.example.com",
            enabled=True,
            priority=1,
            rate_limit_per_second=50,
            supports_quotes=True,
            supports_trades=False,
            supports_bars=True,
            supports_orderbook=True,
        )

        assert config.provider == DataProvider.FINNHUB
        assert config.api_secret == "secret456"
        assert config.priority == 1
        assert config.rate_limit_per_second == 50
        assert config.supports_trades is False
        assert config.supports_orderbook is True

    @pytest.mark.unit
    def test_default_values(self):
        """Test default values are set correctly."""
        config = ProviderConfig(
            provider=DataProvider.MASSIVE,
            api_key="test",
        )

        assert config.base_url == ""
        assert config.ws_url == ""
        assert config.priority == 0
        assert config.rate_limit_per_second == 100
        assert config.supports_quotes is True
        assert config.supports_trades is True
        assert config.supports_bars is True
        assert config.supports_orderbook is False


class TestMarketDataEvent:
    """Tests for MarketDataEvent dataclass."""

    @pytest.fixture
    def sample_quote(self):
        """Create sample quote for testing."""
        return StreamQuote(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bid=150.0,
            ask=150.05,
            bid_size=100,
            ask_size=200,
            provider="test",
        )

    @pytest.mark.unit
    def test_create_event(self, sample_quote):
        """Test creating a market data event."""
        event = MarketDataEvent(
            symbol="AAPL",
            data_type=DataType.QUOTE,
            provider=DataProvider.ALPACA,
            timestamp=datetime.now(timezone.utc),
            data=sample_quote,
        )

        assert event.symbol == "AAPL"
        assert event.data_type == DataType.QUOTE
        assert event.provider == DataProvider.ALPACA
        assert event.latency_ms == 0.0
        assert event.sequence_id == 0

    @pytest.mark.unit
    def test_event_with_latency(self, sample_quote):
        """Test event with latency tracking."""
        event = MarketDataEvent(
            symbol="MSFT",
            data_type=DataType.TRADE,
            provider=DataProvider.FINNHUB,
            timestamp=datetime.now(timezone.utc),
            data=sample_quote,
            latency_ms=5.5,
            sequence_id=123,
        )

        assert event.latency_ms == 5.5
        assert event.sequence_id == 123


class TestSubscriptionRequest:
    """Tests for SubscriptionRequest dataclass."""

    @pytest.mark.unit
    def test_create_minimal_request(self):
        """Test creating request with minimal fields."""
        request = SubscriptionRequest(
            symbols=["AAPL", "MSFT"],
        )

        assert request.symbols == ["AAPL", "MSFT"]
        assert DataType.QUOTE in request.data_types
        assert DataType.TRADE in request.data_types
        assert request.preferred_provider is None

    @pytest.mark.unit
    def test_create_full_request(self):
        """Test creating request with all fields."""
        request = SubscriptionRequest(
            symbols=["GOOGL"],
            data_types=[DataType.BAR],
            preferred_provider=DataProvider.POLYGON,
        )

        assert request.symbols == ["GOOGL"]
        assert request.data_types == [DataType.BAR]
        assert request.preferred_provider == DataProvider.POLYGON


class TestProviderStats:
    """Tests for ProviderStats dataclass."""

    @pytest.mark.unit
    def test_create_stats(self):
        """Test creating provider stats."""
        stats = ProviderStats(provider=DataProvider.ALPACA)

        assert stats.provider == DataProvider.ALPACA
        assert stats.status == StreamStatus.DISCONNECTED
        assert stats.messages_received == 0
        assert stats.errors_count == 0
        assert stats.avg_latency_ms == 0.0
        assert stats.last_message_time is None

    @pytest.mark.unit
    def test_update_latency_first_message(self):
        """Test updating latency with first message."""
        stats = ProviderStats(provider=DataProvider.FINNHUB)
        stats.update_latency(10.0)

        assert stats.avg_latency_ms == 10.0

    @pytest.mark.unit
    def test_update_latency_ema(self):
        """Test updating latency uses EMA."""
        stats = ProviderStats(provider=DataProvider.MASSIVE)
        stats.messages_received = 1  # Simulate already received messages
        stats.avg_latency_ms = 10.0

        stats.update_latency(20.0)

        # EMA with alpha=0.1: 0.1*20 + 0.9*10 = 2 + 9 = 11
        assert abs(stats.avg_latency_ms - 11.0) < 0.001

    @pytest.mark.unit
    def test_symbols_subscribed_default(self):
        """Test symbols_subscribed defaults to empty set."""
        stats = ProviderStats(provider=DataProvider.POLYGON)
        assert stats.symbols_subscribed == set()


class TestUnifiedMarketDataStreamInit:
    """Tests for UnifiedMarketDataStream initialization."""

    @pytest.mark.unit
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        stream = UnifiedMarketDataStream()

        assert stream._dedup_window_ms == 100.0
        assert stream._failover_threshold == 5
        assert stream._health_check_interval == 30.0
        assert stream._running is False
        assert stream._primary_provider is None

    @pytest.mark.unit
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        stream = UnifiedMarketDataStream(
            dedup_window_ms=200.0,
            failover_threshold_errors=10,
            health_check_interval_seconds=60.0,
        )

        assert stream._dedup_window_ms == 200.0
        assert stream._failover_threshold == 10
        assert stream._health_check_interval == 60.0

    @pytest.mark.unit
    def test_empty_providers(self):
        """Test stream starts with no providers."""
        stream = UnifiedMarketDataStream()

        assert len(stream._providers) == 0
        assert len(stream._provider_configs) == 0
        assert len(stream._provider_stats) == 0


class TestUnifiedMarketDataStreamHandlers:
    """Tests for handler management."""

    @pytest.fixture
    def stream(self):
        """Create stream instance."""
        return UnifiedMarketDataStream()

    @pytest.mark.unit
    def test_add_handler(self, stream):
        """Test adding a handler."""

        def my_handler(event):
            pass

        stream.add_handler(my_handler, DataType.QUOTE)

        assert my_handler in stream._handlers[DataType.QUOTE]

    @pytest.mark.unit
    def test_add_multiple_handlers(self, stream):
        """Test adding multiple handlers for same type."""

        def handler1(event):
            pass

        def handler2(event):
            pass

        stream.add_handler(handler1, DataType.TRADE)
        stream.add_handler(handler2, DataType.TRADE)

        assert len(stream._handlers[DataType.TRADE]) == 2

    @pytest.mark.unit
    def test_remove_handler(self, stream):
        """Test removing a handler."""

        def my_handler(event):
            pass

        stream.add_handler(my_handler, DataType.BAR)
        stream.remove_handler(my_handler, DataType.BAR)

        assert my_handler not in stream._handlers[DataType.BAR]

    @pytest.mark.unit
    def test_remove_nonexistent_handler(self, stream):
        """Test removing a handler that doesn't exist doesn't raise."""

        def my_handler(event):
            pass

        # Should not raise
        stream.remove_handler(my_handler, DataType.ORDERBOOK)


class TestUnifiedMarketDataStreamProviders:
    """Tests for provider management."""

    @pytest.fixture
    def stream(self):
        """Create stream instance."""
        return UnifiedMarketDataStream()

    @pytest.mark.unit
    def test_add_disabled_provider_skipped(self, stream):
        """Test that disabled providers are skipped."""
        config = ProviderConfig(
            provider=DataProvider.POLYGON,
            api_key="test",
            enabled=False,
        )

        stream.add_provider(config)

        assert DataProvider.POLYGON not in stream._provider_configs
        assert DataProvider.POLYGON not in stream._provider_stats
