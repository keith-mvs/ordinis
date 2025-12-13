"""
Finnhub WebSocket streaming provider.

Supports real-time trades for stocks, forex (OANDA), and crypto (Binance).
Note: Finnhub only provides trade data, not quotes or aggregates.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
from typing import Any

from ordinis.adapters.streaming.stream_protocol import StreamConfig, StreamTrade
from ordinis.adapters.streaming.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


class FinnhubStream(WebSocketManager):
    """Finnhub WebSocket stream provider."""

    def __init__(self, config: StreamConfig) -> None:
        """Initialize Finnhub stream."""
        super().__init__(config)

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "finnhub"

    @property
    def websocket_url(self) -> str:
        """Get WebSocket URL."""
        return f"wss://ws.finnhub.io?token={self._config.api_key}"

    async def subscribe_forex(self, pair: str) -> None:
        """Subscribe to forex pair (e.g., 'EUR/USD')."""
        formatted = f"OANDA:{pair.replace('/', '_')}"
        await self.subscribe([formatted])

    async def subscribe_crypto(self, symbol: str, exchange: str = "BINANCE") -> None:
        """Subscribe to crypto symbol (e.g., 'BTCUSDT')."""
        formatted = f"{exchange}:{symbol}"
        await self.subscribe([formatted])

    async def _authenticate(self) -> None:
        """Finnhub authenticates via URL token, no additional auth needed."""
        logger.info("%s: Connected (token in URL)", self.provider_name)

    async def _send_subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols."""
        if not self._ws:
            return

        for symbol in symbols:
            msg = {"type": "subscribe", "symbol": symbol}
            await self._ws.send(json.dumps(msg))
            logger.debug("%s: Subscribed to %s", self.provider_name, symbol)

    async def _send_unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols."""
        if not self._ws:
            return

        for symbol in symbols:
            msg = {"type": "unsubscribe", "symbol": symbol}
            await self._ws.send(json.dumps(msg))
            logger.debug("%s: Unsubscribed from %s", self.provider_name, symbol)

    async def _send_heartbeat(self) -> None:
        """Send ping frame."""
        if self._ws:
            await self._ws.ping()

    async def _process_message(self, message: str | bytes) -> None:
        """Process incoming Finnhub message."""
        if isinstance(message, bytes):
            message = message.decode("utf-8")

        data = json.loads(message)

        msg_type = data.get("type")

        if msg_type == "trade":
            trades = data.get("data", [])
            for trade_data in trades:
                trade = self._parse_trade(trade_data)
                if trade:
                    await self._notify_trade(trade)

        elif msg_type == "ping":
            # Finnhub sends ping messages, we can respond with pong
            if self._ws:
                await self._ws.send(json.dumps({"type": "pong"}))

    def _parse_trade(self, data: dict[str, Any]) -> StreamTrade | None:
        """Parse trade message."""
        try:
            symbol = data.get("s", "")
            # Finnhub timestamps are in milliseconds
            timestamp_ms = data.get("t", 0)
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)

            # Parse conditions if present
            conditions = data.get("c", [])
            if isinstance(conditions, list):
                conditions = tuple(str(c) for c in conditions)
            else:
                conditions = ()

            return StreamTrade(
                symbol=symbol,
                price=float(data.get("p", 0)),
                size=int(data.get("v", 0)),
                timestamp=timestamp,
                provider=self.provider_name,
                conditions=conditions,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("Failed to parse trade: %s - %s", data, e)
            return None
