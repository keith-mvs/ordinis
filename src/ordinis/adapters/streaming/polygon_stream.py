"""
Polygon.io WebSocket streaming provider.

Supports real-time quotes, trades, and aggregates for stocks, options, forex, and crypto.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
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


class PolygonMarket(Enum):
    """Polygon market types."""

    STOCKS = "stocks"
    OPTIONS = "options"
    FOREX = "forex"
    CRYPTO = "crypto"


class PolygonStream(WebSocketManager):
    """Polygon.io WebSocket stream provider."""

    # Channel prefixes
    CHANNEL_QUOTES = "Q"
    CHANNEL_TRADES = "T"
    CHANNEL_SECOND_AGG = "A"
    CHANNEL_MINUTE_AGG = "AM"

    def __init__(
        self,
        config: StreamConfig,
        market: PolygonMarket = PolygonMarket.STOCKS,
        delayed: bool = False,
    ) -> None:
        """Initialize Polygon stream."""
        super().__init__(config)
        self._market = market
        self._delayed = delayed
        self._authenticated = False
        self._channels: set[str] = {self.CHANNEL_QUOTES, self.CHANNEL_TRADES}

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return f"polygon-{self._market.value}"

    @property
    def websocket_url(self) -> str:
        """Get WebSocket URL."""
        market = self._market.value
        if self._delayed:
            return f"wss://delayed.polygon.io/{market}"
        return f"wss://socket.polygon.io/{market}"

    def set_channels(self, channels: set[str]) -> None:
        """Set which channels to subscribe to."""
        valid = {
            self.CHANNEL_QUOTES,
            self.CHANNEL_TRADES,
            self.CHANNEL_SECOND_AGG,
            self.CHANNEL_MINUTE_AGG,
        }
        self._channels = channels & valid

    async def _authenticate(self) -> None:
        """Authenticate with Polygon."""
        if not self._ws:
            return

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
                        logger.info("%s: Authenticated successfully", self.provider_name)
                        return
                    if msg.get("status") == "auth_failed":
                        raise ConnectionError(
                            f"Authentication failed: {msg.get('message', 'Unknown error')}"
                        )

    async def _send_subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols."""
        if not self._ws or not self._authenticated:
            return

        # Build subscription params for each channel
        params = []
        for channel in self._channels:
            for symbol in symbols:
                params.append(f"{channel}.{symbol}")

        if params:
            msg = {"action": "subscribe", "params": ",".join(params)}
            await self._ws.send(json.dumps(msg))
            logger.debug("%s: Subscribed to %s", self.provider_name, params)

    async def _send_unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols."""
        if not self._ws or not self._authenticated:
            return

        params = []
        for channel in self._channels:
            for symbol in symbols:
                params.append(f"{channel}.{symbol}")

        if params:
            msg = {"action": "unsubscribe", "params": ",".join(params)}
            await self._ws.send(json.dumps(msg))
            logger.debug("%s: Unsubscribed from %s", self.provider_name, params)

    async def _send_heartbeat(self) -> None:
        """Send ping frame."""
        if self._ws:
            await self._ws.ping()

    async def _process_message(self, message: str | bytes) -> None:
        """Process incoming Polygon message."""
        if isinstance(message, bytes):
            message = message.decode("utf-8")

        data = json.loads(message)

        if not isinstance(data, list):
            data = [data]

        for msg in data:
            event_type = msg.get("ev")

            if event_type == self.CHANNEL_QUOTES:
                quote = self._parse_quote(msg)
                if quote:
                    await self._notify_quote(quote)

            elif event_type == self.CHANNEL_TRADES:
                trade = self._parse_trade(msg)
                if trade:
                    await self._notify_trade(trade)

            elif event_type in (self.CHANNEL_SECOND_AGG, self.CHANNEL_MINUTE_AGG):
                bar = self._parse_bar(msg)
                if bar:
                    await self._notify_bar(bar)

    def _parse_quote(self, data: dict[str, Any]) -> StreamQuote | None:
        """Parse quote message."""
        try:
            symbol = data.get("sym", "")
            timestamp_ns = data.get("t", 0)
            timestamp = datetime.fromtimestamp(timestamp_ns / 1e9, tz=UTC)

            return StreamQuote(
                symbol=symbol,
                bid=float(data.get("bp", 0)),
                ask=float(data.get("ap", 0)),
                bid_size=int(data.get("bs", 0)),
                ask_size=int(data.get("as", 0)),
                timestamp=timestamp,
                provider=self.provider_name,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("Failed to parse quote: %s - %s", data, e)
            return None

    def _parse_trade(self, data: dict[str, Any]) -> StreamTrade | None:
        """Parse trade message."""
        try:
            symbol = data.get("sym", "")
            timestamp_ns = data.get("t", 0)
            timestamp = datetime.fromtimestamp(timestamp_ns / 1e9, tz=UTC)

            conditions = data.get("c", [])
            if isinstance(conditions, list):
                conditions = tuple(str(c) for c in conditions)
            else:
                conditions = ()

            return StreamTrade(
                symbol=symbol,
                price=float(data.get("p", 0)),
                size=int(data.get("s", 0)),
                timestamp=timestamp,
                provider=self.provider_name,
                exchange=str(data.get("x", "")),
                conditions=conditions,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("Failed to parse trade: %s - %s", data, e)
            return None

    def _parse_bar(self, data: dict[str, Any]) -> StreamBar | None:
        """Parse aggregate/bar message."""
        try:
            symbol = data.get("sym", "")
            # Aggregate timestamps are in milliseconds
            timestamp_ms = data.get("s", data.get("e", 0))
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)

            return StreamBar(
                symbol=symbol,
                open=float(data.get("o", 0)),
                high=float(data.get("h", 0)),
                low=float(data.get("l", 0)),
                close=float(data.get("c", 0)),
                volume=int(data.get("v", 0)),
                timestamp=timestamp,
                provider=self.provider_name,
                vwap=float(data["vw"]) if data.get("vw") is not None else None,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("Failed to parse bar: %s - %s", data, e)
            return None
