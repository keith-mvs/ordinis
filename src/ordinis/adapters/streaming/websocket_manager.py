"""
WebSocket connection manager with automatic reconnection.

Provides base class for WebSocket stream providers with:
- Connection lifecycle management
- Automatic reconnection with exponential backoff
- Heartbeat monitoring
- Subscription management
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import contextlib
from datetime import UTC, datetime
import logging
from typing import TYPE_CHECKING, Any

from ordinis.adapters.streaming.stream_protocol import (
    BaseStreamHandler,
    StreamBar,
    StreamConfig,
    StreamQuote,
    StreamStatus,
    StreamTrade,
)

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)


class WebSocketManager(ABC):
    """Abstract WebSocket manager with reconnection logic."""

    def __init__(self, config: StreamConfig) -> None:
        """Initialize WebSocket manager."""
        self._config = config
        self._status = StreamStatus.DISCONNECTED
        self._ws: ClientConnection | None = None
        self._handlers: list[BaseStreamHandler] = []
        self._subscriptions: set[str] = set()
        self._reconnect_attempts = 0
        self._current_reconnect_delay = config.reconnect_delay_seconds
        self._receive_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._last_message_time: datetime | None = None
        self._connected_at: datetime | None = None
        self._message_count = 0
        self._error_count = 0

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name."""

    @property
    @abstractmethod
    def websocket_url(self) -> str:
        """Get WebSocket URL to connect to."""

    @property
    def status(self) -> StreamStatus:
        """Get current connection status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._status == StreamStatus.CONNECTED

    @property
    def subscriptions(self) -> set[str]:
        """Get current subscriptions."""
        return self._subscriptions.copy()

    @property
    def stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        return {
            "provider": self.provider_name,
            "status": self._status.name,
            "connected_at": self._connected_at.isoformat() if self._connected_at else None,
            "last_message": self._last_message_time.isoformat()
            if self._last_message_time
            else None,
            "message_count": self._message_count,
            "error_count": self._error_count,
            "subscription_count": len(self._subscriptions),
            "reconnect_attempts": self._reconnect_attempts,
        }

    def add_handler(self, handler: BaseStreamHandler) -> None:
        """Add an event handler."""
        if handler not in self._handlers:
            self._handlers.append(handler)

    def remove_handler(self, handler: BaseStreamHandler) -> None:
        """Remove an event handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    async def connect(self) -> None:
        """Connect to WebSocket."""
        if self._status in (StreamStatus.CONNECTED, StreamStatus.CONNECTING):
            return

        await self._set_status(StreamStatus.CONNECTING, "Initiating connection")
        await self._connect_internal()

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        await self._set_status(StreamStatus.CLOSED, "Disconnecting")

        # Cancel tasks
        for task in [self._receive_task, self._heartbeat_task, self._reconnect_task]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        self._receive_task = None
        self._heartbeat_task = None
        self._reconnect_task = None

        # Close WebSocket
        if self._ws:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

        await self._set_status(StreamStatus.DISCONNECTED, "Disconnected")

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols."""
        new_symbols = set(symbols) - self._subscriptions
        if not new_symbols:
            return

        self._subscriptions.update(new_symbols)

        if self.is_connected:
            await self._send_subscribe(list(new_symbols))

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols."""
        existing = set(symbols) & self._subscriptions
        if not existing:
            return

        self._subscriptions -= existing

        if self.is_connected:
            await self._send_unsubscribe(list(existing))

    async def _connect_internal(self) -> None:
        """Internal connection logic."""
        try:
            import websockets

            self._ws = await asyncio.wait_for(
                websockets.connect(self.websocket_url),
                timeout=self._config.connection_timeout_seconds,
            )

            await self._authenticate()
            self._connected_at = datetime.now(UTC)
            self._reconnect_attempts = 0
            self._current_reconnect_delay = self._config.reconnect_delay_seconds

            await self._set_status(StreamStatus.CONNECTED, "Connected")

            # Resubscribe if we have subscriptions
            if self._subscriptions:
                await self._send_subscribe(list(self._subscriptions))

            # Start background tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        except TimeoutError:
            logger.warning("%s: Connection timed out", self.provider_name)
            await self._handle_connection_failure("Connection timed out")
        except Exception as e:
            logger.error("%s: Connection failed: %s", self.provider_name, e)
            await self._handle_connection_failure(str(e))

    async def _receive_loop(self) -> None:
        """Receive and process messages."""
        try:
            if not self._ws:
                return

            async for message in self._ws:
                self._last_message_time = datetime.now(UTC)
                self._message_count += 1

                try:
                    await self._process_message(message)
                except Exception as e:
                    logger.debug("%s: Error processing message: %s", self.provider_name, e)
                    self._error_count += 1

        except Exception as e:
            if self._status not in (StreamStatus.CLOSED, StreamStatus.DISCONNECTED):
                logger.warning("%s: Receive loop error: %s", self.provider_name, e)
                await self._handle_connection_failure(str(e))

    async def _heartbeat_loop(self) -> None:
        """Monitor connection health."""
        try:
            while self.is_connected:
                await asyncio.sleep(self._config.heartbeat_interval_seconds)

                if not self._last_message_time:
                    continue

                elapsed = (datetime.now(UTC) - self._last_message_time).total_seconds()
                if elapsed > self._config.heartbeat_interval_seconds * 2:
                    logger.warning(
                        "%s: No messages for %.1f seconds",
                        self.provider_name,
                        elapsed,
                    )
                    await self._send_heartbeat()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("%s: Heartbeat error: %s", self.provider_name, e)

    async def _handle_connection_failure(self, reason: str) -> None:
        """Handle connection failure with reconnection."""
        await self._set_status(StreamStatus.ERROR, reason)
        await self._notify_error(ConnectionError(reason))

        if self._config.reconnect_enabled:
            await self._schedule_reconnect()
        else:
            await self._set_status(StreamStatus.DISCONNECTED, "Reconnection disabled")

    async def _schedule_reconnect(self) -> None:
        """Schedule a reconnection attempt."""
        if self._reconnect_attempts >= self._config.max_reconnect_attempts:
            await self._set_status(
                StreamStatus.DISCONNECTED,
                f"Max reconnection attempts ({self._config.max_reconnect_attempts}) reached",
            )
            return

        self._reconnect_attempts += 1
        await self._set_status(
            StreamStatus.RECONNECTING,
            f"Reconnecting in {self._current_reconnect_delay:.1f}s "
            f"(attempt {self._reconnect_attempts}/{self._config.max_reconnect_attempts})",
        )

        await asyncio.sleep(self._current_reconnect_delay)

        # Exponential backoff
        self._current_reconnect_delay = min(
            self._current_reconnect_delay * 2,
            self._config.reconnect_delay_max_seconds,
        )

        await self._connect_internal()

    async def _set_status(self, status: StreamStatus, message: str) -> None:
        """Update status and notify handlers."""
        old_status = self._status
        self._status = status

        if old_status != status:
            logger.info(
                "%s: %s -> %s: %s", self.provider_name, old_status.name, status.name, message
            )

        for handler in self._handlers:
            try:
                await handler.on_status(status, message)
            except Exception as e:
                logger.debug("Handler error on status: %s", e)

    async def _notify_quote(self, quote: StreamQuote) -> None:
        """Notify handlers of quote."""
        for handler in self._handlers:
            try:
                await handler.on_quote(quote)
            except Exception as e:
                logger.debug("Handler error on quote: %s", e)

    async def _notify_trade(self, trade: StreamTrade) -> None:
        """Notify handlers of trade."""
        for handler in self._handlers:
            try:
                await handler.on_trade(trade)
            except Exception as e:
                logger.debug("Handler error on trade: %s", e)

    async def _notify_bar(self, bar: StreamBar) -> None:
        """Notify handlers of bar."""
        for handler in self._handlers:
            try:
                await handler.on_bar(bar)
            except Exception as e:
                logger.debug("Handler error on bar: %s", e)

    async def _notify_error(self, error: Exception) -> None:
        """Notify handlers of error."""
        for handler in self._handlers:
            try:
                await handler.on_error(error)
            except Exception as e:
                logger.debug("Handler error on error: %s", e)

    @abstractmethod
    async def _authenticate(self) -> None:
        """Authenticate with the server."""

    @abstractmethod
    async def _send_subscribe(self, symbols: list[str]) -> None:
        """Send subscription request."""

    @abstractmethod
    async def _send_unsubscribe(self, symbols: list[str]) -> None:
        """Send unsubscription request."""

    @abstractmethod
    async def _send_heartbeat(self) -> None:
        """Send heartbeat/ping."""

    @abstractmethod
    async def _process_message(self, message: str | bytes) -> None:
        """Process incoming message."""
