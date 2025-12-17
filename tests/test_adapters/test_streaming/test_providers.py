"""Tests for streaming providers (Finnhub)."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from unittest.mock import AsyncMock

import pytest

from ordinis.adapters.streaming.finnhub_stream import FinnhubStream
from ordinis.adapters.streaming.stream_protocol import (
    BufferedStreamHandler,
    StreamConfig,
)


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
