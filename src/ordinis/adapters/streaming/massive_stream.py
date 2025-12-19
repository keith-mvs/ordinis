"""
Massive (Polygon) WebSocket streaming adapter for real-time market data.

Provides real-time quotes, trades, and minute bars via WebSocket.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from typing import Any

from ordinis.adapters.streaming.stream_protocol import (
    StreamBar,
    StreamConfig,
    StreamQuote,
    StreamTrade,
)
from ordinis.adapters.streaming.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


class MassiveStreamManager(WebSocketManager):
    """
    Massive/Polygon WebSocket streaming manager.

    Provides real-time market data streaming:
    - Quotes (Q.*): Bid/ask updates
    - Trades (T.*): Individual trade executions
    - Minute Bars (AM.*): Aggregated minute bars

    Usage:
        config = StreamConfig(api_key="your_api_key")
        stream = MassiveStreamManager(config)

        handler = CallbackStreamHandler(
            on_bar_callback=my_bar_handler,
            on_quote_callback=my_quote_handler,
        )
        stream.add_handler(handler)

        await stream.connect()
        await stream.subscribe(["AAPL", "AMD", "TSLA"])

        # ... trading loop ...

        await stream.disconnect()
    """

    # Polygon/Massive WebSocket endpoints
    WS_URL_STOCKS = "wss://socket.polygon.io/stocks"
    WS_URL_OPTIONS = "wss://socket.polygon.io/options"
    WS_URL_CRYPTO = "wss://socket.polygon.io/crypto"

    def __init__(
        self,
        config: StreamConfig,
        feed_type: str = "stocks",
    ) -> None:
        """
        Initialize Massive stream manager.

        Args:
            config: Stream configuration with API key
            feed_type: Type of feed - "stocks", "options", or "crypto"
        """
        super().__init__(config)
        self._feed_type = feed_type
        self._authenticated = False

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"massive-{self._feed_type}"

    @property
    def websocket_url(self) -> str:
        """Get WebSocket URL based on feed type."""
        urls = {
            "stocks": self.WS_URL_STOCKS,
            "options": self.WS_URL_OPTIONS,
            "crypto": self.WS_URL_CRYPTO,
        }
        return urls.get(self._feed_type, self.WS_URL_STOCKS)

    async def _authenticate(self) -> None:
        """Authenticate with Polygon/Massive."""
        if not self._ws:
            return

        # First, wait for the initial "connected" status
        response = await self._ws.recv()
        data = json.loads(response)

        if isinstance(data, list):
            for msg in data:
                if msg.get("ev") == "status" and msg.get("status") == "connected":
                    logger.info("%s: WebSocket connected, sending auth", self.provider_name)
                    break

        # Now send authentication
        auth_msg = {"action": "auth", "params": self._config.api_key}
        await self._ws.send(json.dumps(auth_msg))

        # Wait for auth response
        response = await self._ws.recv()
        data = json.loads(response)

        if isinstance(data, list):
            for msg in data:
                if msg.get("ev") == "status":
                    if msg.get("status") == "auth_success":
                        self._authenticated = True
                        logger.info("%s: Authentication successful", self.provider_name)
                        return
                    if msg.get("status") == "auth_failed":
                        raise RuntimeError(f"Authentication failed: {msg.get('message')}")

        raise RuntimeError(f"Unexpected auth response: {data}")

    async def _send_subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols."""
        if not self._ws or not self._authenticated:
            return

        # Subscribe to quotes, trades, second aggregates, and minute bars
        channels = []
        for symbol in symbols:
            channels.extend(
                [
                    f"Q.{symbol}",  # Quotes
                    f"T.{symbol}",  # Trades
                    f"A.{symbol}",  # Per-second aggregates (for fast signals)
                    f"AM.{symbol}",  # Minute aggregates (for indicator calc)
                ]
            )

        msg = {"action": "subscribe", "params": ",".join(channels)}
        await self._ws.send(json.dumps(msg))
        logger.info("%s: Subscribed to %d symbols", self.provider_name, len(symbols))

    async def _send_unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols."""
        if not self._ws:
            return

        channels = []
        for symbol in symbols:
            channels.extend(
                [
                    f"Q.{symbol}",
                    f"T.{symbol}",
                    f"A.{symbol}",
                    f"AM.{symbol}",
                ]
            )

        msg = {"action": "unsubscribe", "params": ",".join(channels)}
        await self._ws.send(json.dumps(msg))
        logger.info("%s: Unsubscribed from %d symbols", self.provider_name, len(symbols))

    async def _send_heartbeat(self) -> None:
        """Send ping/heartbeat."""
        if self._ws:
            # Polygon doesn't require explicit heartbeat, but we can send empty message
            pass

    async def _process_message(self, message: str | bytes) -> None:
        """Process incoming WebSocket message."""
        if isinstance(message, bytes):
            message = message.decode("utf-8")

        data = json.loads(message)

        # Messages come as arrays
        if not isinstance(data, list):
            data = [data]

        # Log first few messages for debugging
        if self._message_count < 10:
            logger.info("%s: Message #%d: %s", self.provider_name, self._message_count, data[:3] if len(data) > 3 else data)

        for msg in data:
            event_type = msg.get("ev")

            if event_type == "Q":
                await self._handle_quote(msg)
            elif event_type == "T":
                await self._handle_trade(msg)
            elif event_type == "A":
                # Per-second aggregate - treat same as bar for fast signals
                await self._handle_bar(msg)
            elif event_type == "AM":
                await self._handle_bar(msg)
            elif event_type == "status":
                self._handle_status(msg)

    async def _handle_quote(self, msg: dict[str, Any]) -> None:
        """Handle quote message."""
        try:
            quote = StreamQuote(
                symbol=msg.get("sym", ""),
                bid=float(msg.get("bp", 0)),
                ask=float(msg.get("ap", 0)),
                bid_size=int(msg.get("bs", 0)),
                ask_size=int(msg.get("as", 0)),
                timestamp=datetime.fromtimestamp(msg.get("t", 0) / 1000, tz=UTC),
                provider=self.provider_name,
            )
            await self._notify_quote(quote)
        except Exception as e:
            logger.debug("Error parsing quote: %s", e)

    async def _handle_trade(self, msg: dict[str, Any]) -> None:
        """Handle trade message."""
        try:
            trade = StreamTrade(
                symbol=msg.get("sym", ""),
                price=float(msg.get("p", 0)),
                size=int(msg.get("s", 0)),
                timestamp=datetime.fromtimestamp(msg.get("t", 0) / 1000, tz=UTC),
                provider=self.provider_name,
                exchange=str(msg.get("x", "")),
                conditions=tuple(msg.get("c", [])),
            )
            await self._notify_trade(trade)
        except Exception as e:
            logger.debug("Error parsing trade: %s", e)

    async def _handle_bar(self, msg: dict[str, Any]) -> None:
        """Handle minute aggregate bar message."""
        try:
            bar = StreamBar(
                symbol=msg.get("sym", ""),
                open=float(msg.get("o", 0)),
                high=float(msg.get("h", 0)),
                low=float(msg.get("l", 0)),
                close=float(msg.get("c", 0)),
                volume=int(msg.get("v", 0)),
                timestamp=datetime.fromtimestamp(msg.get("s", 0) / 1000, tz=UTC),
                provider=self.provider_name,
                vwap=float(msg.get("vw", 0)) if msg.get("vw") else None,
            )
            await self._notify_bar(bar)
        except Exception as e:
            logger.debug("Error parsing bar: %s", e)

    def _handle_status(self, msg: dict[str, Any]) -> None:
        """Handle status message."""
        status = msg.get("status", "")
        message = msg.get("message", "")
        logger.debug("%s: Status - %s: %s", self.provider_name, status, message)
