"""Tests for stream protocol definitions."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from ordinis.adapters.streaming.stream_protocol import (
    BufferedStreamHandler,
    CallbackStreamHandler,
    StreamBar,
    StreamConfig,
    StreamEventType,
    StreamProvider,
    StreamQuote,
    StreamStatus,
    StreamTrade,
)


class TestStreamEventType:
    """Tests for StreamEventType enum."""

    def test_event_types_exist(self) -> None:
        """Test all event types are defined."""
        assert StreamEventType.QUOTE
        assert StreamEventType.TRADE
        assert StreamEventType.BAR
        assert StreamEventType.STATUS
        assert StreamEventType.ERROR
        assert StreamEventType.HEARTBEAT

    def test_event_types_unique(self) -> None:
        """Test event types have unique values."""
        values = [e.value for e in StreamEventType]
        assert len(values) == len(set(values))


class TestStreamStatus:
    """Tests for StreamStatus enum."""

    def test_status_types_exist(self) -> None:
        """Test all status types are defined."""
        assert StreamStatus.DISCONNECTED
        assert StreamStatus.CONNECTING
        assert StreamStatus.CONNECTED
        assert StreamStatus.RECONNECTING
        assert StreamStatus.ERROR
        assert StreamStatus.CLOSED

    def test_status_types_unique(self) -> None:
        """Test status types have unique values."""
        values = [s.value for s in StreamStatus]
        assert len(values) == len(set(values))


class TestStreamQuote:
    """Tests for StreamQuote dataclass."""

    @pytest.fixture
    def sample_quote(self) -> StreamQuote:
        """Create sample quote."""
        return StreamQuote(
            symbol="AAPL",
            bid=150.00,
            ask=150.05,
            bid_size=100,
            ask_size=200,
            timestamp=datetime.now(UTC),
            provider="test",
        )

    def test_quote_creation(self, sample_quote: StreamQuote) -> None:
        """Test quote creation."""
        assert sample_quote.symbol == "AAPL"
        assert sample_quote.bid == 150.00
        assert sample_quote.ask == 150.05
        assert sample_quote.bid_size == 100
        assert sample_quote.ask_size == 200
        assert sample_quote.provider == "test"

    def test_mid_price(self, sample_quote: StreamQuote) -> None:
        """Test mid price calculation."""
        assert sample_quote.mid == 150.025

    def test_spread(self, sample_quote: StreamQuote) -> None:
        """Test spread calculation."""
        assert abs(sample_quote.spread - 0.05) < 1e-9

    def test_spread_bps(self, sample_quote: StreamQuote) -> None:
        """Test spread in basis points."""
        expected_bps = (0.05 / 150.025) * 10000
        assert abs(sample_quote.spread_bps - expected_bps) < 0.01

    def test_spread_bps_zero_mid(self) -> None:
        """Test spread bps with zero mid price."""
        quote = StreamQuote(
            symbol="TEST",
            bid=0.0,
            ask=0.0,
            bid_size=0,
            ask_size=0,
            timestamp=datetime.now(UTC),
            provider="test",
        )
        assert quote.spread_bps == 0.0

    def test_quote_immutable(self, sample_quote: StreamQuote) -> None:
        """Test quote is immutable."""
        with pytest.raises(AttributeError):
            sample_quote.symbol = "MSFT"  # type: ignore[misc]


class TestStreamTrade:
    """Tests for StreamTrade dataclass."""

    @pytest.fixture
    def sample_trade(self) -> StreamTrade:
        """Create sample trade."""
        return StreamTrade(
            symbol="AAPL",
            price=150.00,
            size=100,
            timestamp=datetime.now(UTC),
            provider="test",
            exchange="NASDAQ",
            conditions=("@", "F"),
        )

    def test_trade_creation(self, sample_trade: StreamTrade) -> None:
        """Test trade creation."""
        assert sample_trade.symbol == "AAPL"
        assert sample_trade.price == 150.00
        assert sample_trade.size == 100
        assert sample_trade.exchange == "NASDAQ"
        assert sample_trade.conditions == ("@", "F")

    def test_notional_value(self, sample_trade: StreamTrade) -> None:
        """Test notional value calculation."""
        assert sample_trade.notional == 15000.00

    def test_trade_default_exchange(self) -> None:
        """Test trade with default exchange."""
        trade = StreamTrade(
            symbol="AAPL",
            price=150.00,
            size=100,
            timestamp=datetime.now(UTC),
            provider="test",
        )
        assert trade.exchange == ""
        assert trade.conditions == ()


class TestStreamBar:
    """Tests for StreamBar dataclass."""

    @pytest.fixture
    def bullish_bar(self) -> StreamBar:
        """Create bullish bar."""
        return StreamBar(
            symbol="AAPL",
            open=150.00,
            high=152.00,
            low=149.50,
            close=151.50,
            volume=1000000,
            timestamp=datetime.now(UTC),
            provider="test",
            vwap=150.75,
        )

    @pytest.fixture
    def bearish_bar(self) -> StreamBar:
        """Create bearish bar."""
        return StreamBar(
            symbol="AAPL",
            open=152.00,
            high=152.50,
            low=149.00,
            close=149.50,
            volume=1500000,
            timestamp=datetime.now(UTC),
            provider="test",
        )

    def test_bar_creation(self, bullish_bar: StreamBar) -> None:
        """Test bar creation."""
        assert bullish_bar.symbol == "AAPL"
        assert bullish_bar.open == 150.00
        assert bullish_bar.high == 152.00
        assert bullish_bar.low == 149.50
        assert bullish_bar.close == 151.50
        assert bullish_bar.volume == 1000000
        assert bullish_bar.vwap == 150.75

    def test_bar_range(self, bullish_bar: StreamBar) -> None:
        """Test bar range calculation."""
        assert bullish_bar.range == 2.50

    def test_bar_body(self, bullish_bar: StreamBar) -> None:
        """Test bar body calculation."""
        assert bullish_bar.body == 1.50

    def test_is_bullish(self, bullish_bar: StreamBar, bearish_bar: StreamBar) -> None:
        """Test bullish detection."""
        assert bullish_bar.is_bullish is True
        assert bearish_bar.is_bullish is False

    def test_bar_no_vwap(self, bearish_bar: StreamBar) -> None:
        """Test bar without vwap."""
        assert bearish_bar.vwap is None


class TestStreamConfig:
    """Tests for StreamConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default config values."""
        config = StreamConfig(api_key="test_key")
        assert config.api_key == "test_key"
        assert config.reconnect_enabled is True
        assert config.reconnect_delay_seconds == 1.0
        assert config.reconnect_delay_max_seconds == 60.0
        assert config.max_reconnect_attempts == 10
        assert config.connection_timeout_seconds == 30.0
        assert config.heartbeat_interval_seconds == 30.0
        assert config.buffer_size == 1000

    def test_custom_config(self) -> None:
        """Test custom config values."""
        config = StreamConfig(
            api_key="custom_key",
            reconnect_enabled=False,
            reconnect_delay_seconds=5.0,
            max_reconnect_attempts=5,
        )
        assert config.api_key == "custom_key"
        assert config.reconnect_enabled is False
        assert config.reconnect_delay_seconds == 5.0
        assert config.max_reconnect_attempts == 5


class TestStreamProvider:
    """Tests for StreamProvider protocol."""

    def test_protocol_attributes(self) -> None:
        """Test protocol defines required methods."""
        assert hasattr(StreamProvider, "status")
        assert hasattr(StreamProvider, "provider_name")
        assert hasattr(StreamProvider, "connect")
        assert hasattr(StreamProvider, "disconnect")
        assert hasattr(StreamProvider, "subscribe")
        assert hasattr(StreamProvider, "unsubscribe")


class TestCallbackStreamHandler:
    """Tests for CallbackStreamHandler."""

    @pytest.fixture
    def handler_with_callbacks(self) -> tuple[CallbackStreamHandler, dict]:
        """Create handler with tracking callbacks."""
        results: dict = {"quotes": [], "trades": [], "bars": [], "status": [], "errors": []}

        def on_quote(q: StreamQuote) -> None:
            results["quotes"].append(q)

        def on_trade(t: StreamTrade) -> None:
            results["trades"].append(t)

        def on_bar(b: StreamBar) -> None:
            results["bars"].append(b)

        def on_status(s: StreamStatus, m: str) -> None:
            results["status"].append((s, m))

        def on_error(e: Exception) -> None:
            results["errors"].append(e)

        handler = CallbackStreamHandler(
            on_quote_callback=on_quote,
            on_trade_callback=on_trade,
            on_bar_callback=on_bar,
            on_status_callback=on_status,
            on_error_callback=on_error,
        )
        return handler, results

    @pytest.mark.asyncio
    async def test_quote_callback(
        self, handler_with_callbacks: tuple[CallbackStreamHandler, dict]
    ) -> None:
        """Test quote callback invocation."""
        handler, results = handler_with_callbacks
        quote = StreamQuote(
            symbol="AAPL",
            bid=150.0,
            ask=150.05,
            bid_size=100,
            ask_size=200,
            timestamp=datetime.now(UTC),
            provider="test",
        )
        await handler.on_quote(quote)
        assert len(results["quotes"]) == 1
        assert results["quotes"][0] == quote

    @pytest.mark.asyncio
    async def test_trade_callback(
        self, handler_with_callbacks: tuple[CallbackStreamHandler, dict]
    ) -> None:
        """Test trade callback invocation."""
        handler, results = handler_with_callbacks
        trade = StreamTrade(
            symbol="AAPL",
            price=150.0,
            size=100,
            timestamp=datetime.now(UTC),
            provider="test",
        )
        await handler.on_trade(trade)
        assert len(results["trades"]) == 1
        assert results["trades"][0] == trade

    @pytest.mark.asyncio
    async def test_async_callback(self) -> None:
        """Test async callback support."""
        results: list[StreamQuote] = []

        async def async_on_quote(q: StreamQuote) -> None:
            await asyncio.sleep(0.001)
            results.append(q)

        handler = CallbackStreamHandler(on_quote_callback=async_on_quote)
        quote = StreamQuote(
            symbol="AAPL",
            bid=150.0,
            ask=150.05,
            bid_size=100,
            ask_size=200,
            timestamp=datetime.now(UTC),
            provider="test",
        )
        await handler.on_quote(quote)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_no_callback(self) -> None:
        """Test handler with no callbacks."""
        handler = CallbackStreamHandler()
        quote = StreamQuote(
            symbol="AAPL",
            bid=150.0,
            ask=150.05,
            bid_size=100,
            ask_size=200,
            timestamp=datetime.now(UTC),
            provider="test",
        )
        # Should not raise
        await handler.on_quote(quote)


class TestBufferedStreamHandler:
    """Tests for BufferedStreamHandler."""

    @pytest.fixture
    def handler(self) -> BufferedStreamHandler:
        """Create buffered handler."""
        return BufferedStreamHandler(max_buffer_size=10)

    def _make_quote(self, symbol: str, bid: float) -> StreamQuote:
        """Create a quote for testing."""
        return StreamQuote(
            symbol=symbol,
            bid=bid,
            ask=bid + 0.05,
            bid_size=100,
            ask_size=200,
            timestamp=datetime.now(UTC),
            provider="test",
        )

    def _make_trade(self, symbol: str, price: float) -> StreamTrade:
        """Create a trade for testing."""
        return StreamTrade(
            symbol=symbol,
            price=price,
            size=100,
            timestamp=datetime.now(UTC),
            provider="test",
        )

    @pytest.mark.asyncio
    async def test_buffer_quotes(self, handler: BufferedStreamHandler) -> None:
        """Test buffering quotes."""
        quote = self._make_quote("AAPL", 150.0)
        await handler.on_quote(quote)

        quotes = await handler.get_quotes()
        assert len(quotes) == 1
        assert quotes[0] == quote

    @pytest.mark.asyncio
    async def test_buffer_max_size(self, handler: BufferedStreamHandler) -> None:
        """Test buffer respects max size."""
        for i in range(15):
            quote = self._make_quote("AAPL", 150.0 + i)
            await handler.on_quote(quote)

        quotes = await handler.get_quotes()
        assert len(quotes) == 10
        # Should keep the last 10
        assert quotes[0].bid == 155.0  # 150 + 5
        assert quotes[-1].bid == 164.0  # 150 + 14

    @pytest.mark.asyncio
    async def test_get_quotes_clear(self, handler: BufferedStreamHandler) -> None:
        """Test clearing quotes on get."""
        await handler.on_quote(self._make_quote("AAPL", 150.0))
        await handler.on_quote(self._make_quote("AAPL", 151.0))

        quotes = await handler.get_quotes(clear=True)
        assert len(quotes) == 2

        quotes_after = await handler.get_quotes()
        assert len(quotes_after) == 0

    @pytest.mark.asyncio
    async def test_get_latest_quote(self, handler: BufferedStreamHandler) -> None:
        """Test getting latest quote for symbol."""
        await handler.on_quote(self._make_quote("AAPL", 150.0))
        await handler.on_quote(self._make_quote("MSFT", 300.0))
        await handler.on_quote(self._make_quote("AAPL", 151.0))

        latest = await handler.get_latest_quote("AAPL")
        assert latest is not None
        assert latest.bid == 151.0

        latest_msft = await handler.get_latest_quote("MSFT")
        assert latest_msft is not None
        assert latest_msft.bid == 300.0

        latest_unknown = await handler.get_latest_quote("UNKNOWN")
        assert latest_unknown is None

    @pytest.mark.asyncio
    async def test_buffer_trades(self, handler: BufferedStreamHandler) -> None:
        """Test buffering trades."""
        trade = self._make_trade("AAPL", 150.0)
        await handler.on_trade(trade)

        trades = await handler.get_trades()
        assert len(trades) == 1
        assert trades[0] == trade

    @pytest.mark.asyncio
    async def test_get_latest_trade(self, handler: BufferedStreamHandler) -> None:
        """Test getting latest trade for symbol."""
        await handler.on_trade(self._make_trade("AAPL", 150.0))
        await handler.on_trade(self._make_trade("AAPL", 151.0))

        latest = await handler.get_latest_trade("AAPL")
        assert latest is not None
        assert latest.price == 151.0

    @pytest.mark.asyncio
    async def test_buffer_bars(self, handler: BufferedStreamHandler) -> None:
        """Test buffering bars."""
        bar = StreamBar(
            symbol="AAPL",
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000,
            timestamp=datetime.now(UTC),
            provider="test",
        )
        await handler.on_bar(bar)

        bars = await handler.get_bars()
        assert len(bars) == 1
        assert bars[0] == bar

    @pytest.mark.asyncio
    async def test_buffer_status(self, handler: BufferedStreamHandler) -> None:
        """Test buffering status events."""
        await handler.on_status(StreamStatus.CONNECTING, "Connecting...")
        await handler.on_status(StreamStatus.CONNECTED, "Connected")

        # Status history is internal, verify no errors
        assert handler._status_history[-1][0] == StreamStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_buffer_errors(self, handler: BufferedStreamHandler) -> None:
        """Test buffering error events."""
        error = ConnectionError("Test error")
        await handler.on_error(error)

        assert len(handler._errors) == 1
        assert handler._errors[0][0] == error

    @pytest.mark.asyncio
    async def test_clear_all(self, handler: BufferedStreamHandler) -> None:
        """Test clearing all buffers."""
        await handler.on_quote(self._make_quote("AAPL", 150.0))
        await handler.on_trade(self._make_trade("AAPL", 150.0))
        await handler.on_status(StreamStatus.CONNECTED, "OK")
        await handler.on_error(Exception("Test"))

        await handler.clear()

        assert await handler.get_quotes() == []
        assert await handler.get_trades() == []
        assert await handler.get_bars() == []

    @pytest.mark.asyncio
    async def test_concurrent_access(self, handler: BufferedStreamHandler) -> None:
        """Test concurrent buffer access."""

        async def add_quotes() -> None:
            for i in range(100):
                await handler.on_quote(self._make_quote("AAPL", 150.0 + i))

        async def read_quotes() -> None:
            for _ in range(50):
                await handler.get_quotes()
                await asyncio.sleep(0.001)

        await asyncio.gather(add_quotes(), read_quotes())

        # Should not raise and maintain buffer size
        quotes = await handler.get_quotes()
        assert len(quotes) <= 10
