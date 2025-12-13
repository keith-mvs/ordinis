"""Tests for streaming providers (Polygon, Finnhub)."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from unittest.mock import AsyncMock

import pytest

from ordinis.adapters.streaming.finnhub_stream import FinnhubStream
from ordinis.adapters.streaming.polygon_stream import PolygonMarket, PolygonStream
from ordinis.adapters.streaming.stream_protocol import (
    BufferedStreamHandler,
    StreamConfig,
)


class TestPolygonStream:
    """Tests for Polygon.io streaming provider."""

    @pytest.fixture
    def config(self) -> StreamConfig:
        """Create stream config."""
        return StreamConfig(api_key="test_api_key")

    @pytest.fixture
    def polygon(self, config: StreamConfig) -> PolygonStream:
        """Create Polygon stream."""
        return PolygonStream(config)

    def test_provider_name(self, polygon: PolygonStream) -> None:
        """Test provider name."""
        assert polygon.provider_name == "polygon-stocks"

    def test_provider_name_crypto(self, config: StreamConfig) -> None:
        """Test provider name for crypto."""
        polygon = PolygonStream(config, market=PolygonMarket.CRYPTO)
        assert polygon.provider_name == "polygon-crypto"

    def test_websocket_url_stocks(self, polygon: PolygonStream) -> None:
        """Test WebSocket URL for stocks."""
        assert polygon.websocket_url == "wss://socket.polygon.io/stocks"

    def test_websocket_url_delayed(self, config: StreamConfig) -> None:
        """Test WebSocket URL for delayed data."""
        polygon = PolygonStream(config, delayed=True)
        assert polygon.websocket_url == "wss://delayed.polygon.io/stocks"

    def test_websocket_url_forex(self, config: StreamConfig) -> None:
        """Test WebSocket URL for forex."""
        polygon = PolygonStream(config, market=PolygonMarket.FOREX)
        assert polygon.websocket_url == "wss://socket.polygon.io/forex"

    def test_set_channels(self, polygon: PolygonStream) -> None:
        """Test setting channels."""
        polygon.set_channels({PolygonStream.CHANNEL_TRADES, "INVALID"})
        assert polygon._channels == {PolygonStream.CHANNEL_TRADES}

    @pytest.mark.asyncio
    async def test_authenticate_success(self, polygon: PolygonStream) -> None:
        """Test successful authentication."""
        mock_ws = AsyncMock()
        mock_ws.recv.return_value = json.dumps([{"ev": "status", "status": "auth_success"}])
        polygon._ws = mock_ws

        await polygon._authenticate()

        assert polygon._authenticated is True
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["action"] == "auth"
        assert sent_data["params"] == "test_api_key"

    @pytest.mark.asyncio
    async def test_authenticate_failure(self, polygon: PolygonStream) -> None:
        """Test failed authentication."""
        mock_ws = AsyncMock()
        mock_ws.recv.return_value = json.dumps(
            [{"ev": "status", "status": "auth_failed", "message": "Invalid key"}]
        )
        polygon._ws = mock_ws

        with pytest.raises(ConnectionError, match="Authentication failed"):
            await polygon._authenticate()

    @pytest.mark.asyncio
    async def test_send_subscribe(self, polygon: PolygonStream) -> None:
        """Test sending subscription."""
        mock_ws = AsyncMock()
        polygon._ws = mock_ws
        polygon._authenticated = True

        await polygon._send_subscribe(["AAPL", "MSFT"])

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["action"] == "subscribe"
        # Should include both channels for both symbols
        params = sent_data["params"].split(",")
        assert "Q.AAPL" in params
        assert "T.AAPL" in params
        assert "Q.MSFT" in params
        assert "T.MSFT" in params

    @pytest.mark.asyncio
    async def test_send_unsubscribe(self, polygon: PolygonStream) -> None:
        """Test sending unsubscription."""
        mock_ws = AsyncMock()
        polygon._ws = mock_ws
        polygon._authenticated = True

        await polygon._send_unsubscribe(["AAPL"])

        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["action"] == "unsubscribe"

    @pytest.mark.asyncio
    async def test_process_quote_message(self, polygon: PolygonStream) -> None:
        """Test processing quote message."""
        handler = BufferedStreamHandler()
        polygon.add_handler(handler)

        # Polygon quote message format
        msg = json.dumps(
            [
                {
                    "ev": "Q",
                    "sym": "AAPL",
                    "bp": 150.00,
                    "ap": 150.05,
                    "bs": 100,
                    "as": 200,
                    "t": int(datetime.now(UTC).timestamp() * 1e9),
                }
            ]
        )

        await polygon._process_message(msg)

        quotes = await handler.get_quotes()
        assert len(quotes) == 1
        assert quotes[0].symbol == "AAPL"
        assert quotes[0].bid == 150.00
        assert quotes[0].ask == 150.05

    @pytest.mark.asyncio
    async def test_process_trade_message(self, polygon: PolygonStream) -> None:
        """Test processing trade message."""
        handler = BufferedStreamHandler()
        polygon.add_handler(handler)

        msg = json.dumps(
            [
                {
                    "ev": "T",
                    "sym": "AAPL",
                    "p": 150.02,
                    "s": 50,
                    "x": "NASDAQ",
                    "c": ["@", "F"],
                    "t": int(datetime.now(UTC).timestamp() * 1e9),
                }
            ]
        )

        await polygon._process_message(msg)

        trades = await handler.get_trades()
        assert len(trades) == 1
        assert trades[0].symbol == "AAPL"
        assert trades[0].price == 150.02
        assert trades[0].size == 50
        assert trades[0].exchange == "NASDAQ"
        assert trades[0].conditions == ("@", "F")

    @pytest.mark.asyncio
    async def test_process_bar_message(self, polygon: PolygonStream) -> None:
        """Test processing aggregate/bar message."""
        handler = BufferedStreamHandler()
        polygon.add_handler(handler)
        polygon.set_channels({PolygonStream.CHANNEL_MINUTE_AGG})

        msg = json.dumps(
            [
                {
                    "ev": "AM",
                    "sym": "AAPL",
                    "o": 150.00,
                    "h": 152.00,
                    "l": 149.50,
                    "c": 151.50,
                    "v": 1000000,
                    "vw": 150.75,
                    "s": int(datetime.now(UTC).timestamp() * 1000),
                }
            ]
        )

        await polygon._process_message(msg)

        bars = await handler.get_bars()
        assert len(bars) == 1
        assert bars[0].symbol == "AAPL"
        assert bars[0].open == 150.00
        assert bars[0].close == 151.50
        assert bars[0].vwap == 150.75

    @pytest.mark.asyncio
    async def test_parse_quote_empty(self, polygon: PolygonStream) -> None:
        """Test parsing empty quote data returns defaults."""
        result = polygon._parse_quote({})
        # Returns quote with defaults for missing fields
        assert result is not None
        assert result.symbol == ""
        assert result.bid == 0.0

    @pytest.mark.asyncio
    async def test_parse_trade_empty(self, polygon: PolygonStream) -> None:
        """Test parsing empty trade data returns defaults."""
        result = polygon._parse_trade({})
        # Returns trade with defaults for missing fields
        assert result is not None
        assert result.symbol == ""
        assert result.price == 0.0


class TestFinnhubStream:
    """Tests for Finnhub streaming provider."""

    @pytest.fixture
    def config(self) -> StreamConfig:
        """Create stream config."""
        return StreamConfig(api_key="test_api_key")

    @pytest.fixture
    def finnhub(self, config: StreamConfig) -> FinnhubStream:
        """Create Finnhub stream."""
        return FinnhubStream(config)

    def test_provider_name(self, finnhub: FinnhubStream) -> None:
        """Test provider name."""
        assert finnhub.provider_name == "finnhub"

    def test_websocket_url(self, finnhub: FinnhubStream) -> None:
        """Test WebSocket URL includes token."""
        assert finnhub.websocket_url == "wss://ws.finnhub.io?token=test_api_key"

    @pytest.mark.asyncio
    async def test_subscribe_forex(self, finnhub: FinnhubStream) -> None:
        """Test forex subscription formatting."""
        mock_ws = AsyncMock()
        finnhub._ws = mock_ws

        await finnhub.subscribe_forex("EUR/USD")

        # Should format as OANDA:EUR_USD
        assert "OANDA:EUR_USD" in finnhub._subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_crypto(self, finnhub: FinnhubStream) -> None:
        """Test crypto subscription formatting."""
        mock_ws = AsyncMock()
        finnhub._ws = mock_ws

        await finnhub.subscribe_crypto("BTCUSDT", "BINANCE")

        assert "BINANCE:BTCUSDT" in finnhub._subscriptions

    @pytest.mark.asyncio
    async def test_send_subscribe(self, finnhub: FinnhubStream) -> None:
        """Test sending subscription."""
        mock_ws = AsyncMock()
        finnhub._ws = mock_ws

        await finnhub._send_subscribe(["AAPL", "MSFT"])

        assert mock_ws.send.call_count == 2
        # Check each message
        for call in mock_ws.send.call_args_list:
            sent_data = json.loads(call[0][0])
            assert sent_data["type"] == "subscribe"
            assert sent_data["symbol"] in ["AAPL", "MSFT"]

    @pytest.mark.asyncio
    async def test_process_trade_message(self, finnhub: FinnhubStream) -> None:
        """Test processing trade message."""
        handler = BufferedStreamHandler()
        finnhub.add_handler(handler)

        msg = json.dumps(
            {
                "type": "trade",
                "data": [
                    {
                        "s": "AAPL",
                        "p": 150.02,
                        "v": 50,
                        "t": int(datetime.now(UTC).timestamp() * 1000),
                        "c": ["1", "12"],
                    }
                ],
            }
        )

        await finnhub._process_message(msg)

        trades = await handler.get_trades()
        assert len(trades) == 1
        assert trades[0].symbol == "AAPL"
        assert trades[0].price == 150.02
        assert trades[0].size == 50
        assert trades[0].conditions == ("1", "12")

    @pytest.mark.asyncio
    async def test_process_ping_message(self, finnhub: FinnhubStream) -> None:
        """Test processing ping message."""
        mock_ws = AsyncMock()
        finnhub._ws = mock_ws

        msg = json.dumps({"type": "ping"})
        await finnhub._process_message(msg)

        # Should respond with pong
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "pong"

    @pytest.mark.asyncio
    async def test_process_multiple_trades(self, finnhub: FinnhubStream) -> None:
        """Test processing multiple trades in single message."""
        handler = BufferedStreamHandler()
        finnhub.add_handler(handler)

        msg = json.dumps(
            {
                "type": "trade",
                "data": [
                    {"s": "AAPL", "p": 150.00, "v": 100, "t": 1000000000000},
                    {"s": "AAPL", "p": 150.01, "v": 50, "t": 1000000001000},
                ],
            }
        )

        await finnhub._process_message(msg)

        trades = await handler.get_trades()
        assert len(trades) == 2

    @pytest.mark.asyncio
    async def test_parse_trade_empty(self, finnhub: FinnhubStream) -> None:
        """Test parsing empty trade data returns defaults."""
        result = finnhub._parse_trade({})
        # Returns trade with defaults for missing fields
        assert result is not None
        assert result.symbol == ""
        assert result.price == 0.0

    @pytest.mark.asyncio
    async def test_authenticate_no_op(self, finnhub: FinnhubStream) -> None:
        """Test authentication is no-op (uses URL token)."""
        # Should not raise
        await finnhub._authenticate()
