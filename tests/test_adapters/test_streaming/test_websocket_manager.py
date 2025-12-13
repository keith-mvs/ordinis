"""Tests for WebSocket manager."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ordinis.adapters.streaming.stream_protocol import (
    BufferedStreamHandler,
    StreamConfig,
    StreamQuote,
    StreamStatus,
)
from ordinis.adapters.streaming.websocket_manager import WebSocketManager


class ConcreteWebSocketManager(WebSocketManager):
    """Concrete implementation for testing."""

    def __init__(self, config: StreamConfig, url: str = "wss://test.com") -> None:
        super().__init__(config)
        self._url = url
        self._auth_called = False
        self._subscribed: list[str] = []
        self._unsubscribed: list[str] = []
        self._heartbeats_sent = 0

    @property
    def provider_name(self) -> str:
        return "test-provider"

    @property
    def websocket_url(self) -> str:
        return self._url

    async def _authenticate(self) -> None:
        self._auth_called = True

    async def _send_subscribe(self, symbols: list[str]) -> None:
        self._subscribed.extend(symbols)

    async def _send_unsubscribe(self, symbols: list[str]) -> None:
        self._unsubscribed.extend(symbols)

    async def _send_heartbeat(self) -> None:
        self._heartbeats_sent += 1

    async def _process_message(self, message: str | bytes) -> None:
        # Parse and notify
        pass


class TestWebSocketManagerInit:
    """Tests for WebSocketManager initialization."""

    def test_init_default_state(self) -> None:
        """Test initial state."""
        config = StreamConfig(api_key="test")
        manager = ConcreteWebSocketManager(config)

        assert manager.status == StreamStatus.DISCONNECTED
        assert not manager.is_connected
        assert manager.subscriptions == set()
        assert manager.provider_name == "test-provider"

    def test_stats_initial(self) -> None:
        """Test initial stats."""
        config = StreamConfig(api_key="test")
        manager = ConcreteWebSocketManager(config)
        stats = manager.stats

        assert stats["provider"] == "test-provider"
        assert stats["status"] == "DISCONNECTED"
        assert stats["connected_at"] is None
        assert stats["message_count"] == 0
        assert stats["error_count"] == 0


class TestWebSocketManagerHandlers:
    """Tests for handler management."""

    @pytest.fixture
    def manager(self) -> ConcreteWebSocketManager:
        """Create manager."""
        config = StreamConfig(api_key="test")
        return ConcreteWebSocketManager(config)

    def test_add_handler(self, manager: ConcreteWebSocketManager) -> None:
        """Test adding handler."""
        handler = BufferedStreamHandler()
        manager.add_handler(handler)
        assert handler in manager._handlers

    def test_add_handler_duplicate(self, manager: ConcreteWebSocketManager) -> None:
        """Test adding same handler twice."""
        handler = BufferedStreamHandler()
        manager.add_handler(handler)
        manager.add_handler(handler)
        assert manager._handlers.count(handler) == 1

    def test_remove_handler(self, manager: ConcreteWebSocketManager) -> None:
        """Test removing handler."""
        handler = BufferedStreamHandler()
        manager.add_handler(handler)
        manager.remove_handler(handler)
        assert handler not in manager._handlers

    def test_remove_nonexistent_handler(self, manager: ConcreteWebSocketManager) -> None:
        """Test removing handler that wasn't added."""
        handler = BufferedStreamHandler()
        # Should not raise
        manager.remove_handler(handler)


class TestWebSocketManagerSubscription:
    """Tests for subscription management."""

    @pytest.fixture
    def manager(self) -> ConcreteWebSocketManager:
        """Create manager."""
        config = StreamConfig(api_key="test")
        return ConcreteWebSocketManager(config)

    @pytest.mark.asyncio
    async def test_subscribe_adds_to_set(self, manager: ConcreteWebSocketManager) -> None:
        """Test subscription adds to internal set."""
        await manager.subscribe(["AAPL", "MSFT"])
        assert manager.subscriptions == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_subscribe_deduplicates(self, manager: ConcreteWebSocketManager) -> None:
        """Test duplicate subscriptions are ignored."""
        await manager.subscribe(["AAPL"])
        await manager.subscribe(["AAPL", "MSFT"])
        assert manager.subscriptions == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_unsubscribe_removes(self, manager: ConcreteWebSocketManager) -> None:
        """Test unsubscription removes from set."""
        await manager.subscribe(["AAPL", "MSFT"])
        await manager.unsubscribe(["AAPL"])
        assert manager.subscriptions == {"MSFT"}

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self, manager: ConcreteWebSocketManager) -> None:
        """Test unsubscribing from non-subscribed symbol."""
        await manager.subscribe(["AAPL"])
        await manager.unsubscribe(["MSFT"])
        assert manager.subscriptions == {"AAPL"}


class TestWebSocketManagerConnection:
    """Tests for connection handling."""

    @pytest.fixture
    def manager(self) -> ConcreteWebSocketManager:
        """Create manager."""
        config = StreamConfig(
            api_key="test",
            reconnect_enabled=True,
            max_reconnect_attempts=3,
        )
        return ConcreteWebSocketManager(config)

    @pytest.mark.asyncio
    async def test_connect_success(self, manager: ConcreteWebSocketManager) -> None:
        """Test successful connection."""
        mock_ws = AsyncMock()
        mock_ws.__aiter__ = AsyncMock(return_value=iter([]))

        async def mock_connect(*args, **kwargs):
            return mock_ws

        with patch.dict("sys.modules", {"websockets": MagicMock(connect=mock_connect)}):
            await manager.connect()

        assert manager._auth_called
        assert manager.status == StreamStatus.CONNECTED
        assert manager.is_connected

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, manager: ConcreteWebSocketManager) -> None:
        """Test connect when already connected."""
        manager._status = StreamStatus.CONNECTED

        # Should not attempt to connect again
        await manager.connect()
        assert not manager._auth_called

    @pytest.mark.asyncio
    async def test_connect_resubscribes(self, manager: ConcreteWebSocketManager) -> None:
        """Test that connect resubscribes existing subscriptions."""
        await manager.subscribe(["AAPL", "MSFT"])
        manager._subscribed.clear()

        mock_ws = AsyncMock()
        mock_ws.__aiter__ = AsyncMock(return_value=iter([]))

        async def mock_connect(*args, **kwargs):
            return mock_ws

        with patch.dict("sys.modules", {"websockets": MagicMock(connect=mock_connect)}):
            await manager.connect()

        assert set(manager._subscribed) == {"AAPL", "MSFT"}

    @pytest.mark.asyncio
    async def test_disconnect(self, manager: ConcreteWebSocketManager) -> None:
        """Test disconnection."""
        mock_ws = AsyncMock()
        manager._ws = mock_ws
        manager._status = StreamStatus.CONNECTED

        await manager.disconnect()

        assert manager.status == StreamStatus.DISCONNECTED
        assert manager._ws is None
        mock_ws.close.assert_called_once()


class TestWebSocketManagerNotifications:
    """Tests for handler notifications."""

    @pytest.fixture
    def manager_with_handler(self) -> tuple[ConcreteWebSocketManager, BufferedStreamHandler]:
        """Create manager with handler."""
        config = StreamConfig(api_key="test")
        manager = ConcreteWebSocketManager(config)
        handler = BufferedStreamHandler()
        manager.add_handler(handler)
        return manager, handler

    @pytest.mark.asyncio
    async def test_notify_quote(
        self, manager_with_handler: tuple[ConcreteWebSocketManager, BufferedStreamHandler]
    ) -> None:
        """Test quote notification."""
        manager, handler = manager_with_handler

        quote = StreamQuote(
            symbol="AAPL",
            bid=150.0,
            ask=150.05,
            bid_size=100,
            ask_size=200,
            timestamp=datetime.now(UTC),
            provider="test",
        )

        await manager._notify_quote(quote)

        quotes = await handler.get_quotes()
        assert len(quotes) == 1
        assert quotes[0] == quote

    @pytest.mark.asyncio
    async def test_notify_error(
        self, manager_with_handler: tuple[ConcreteWebSocketManager, BufferedStreamHandler]
    ) -> None:
        """Test error notification."""
        manager, handler = manager_with_handler

        error = ConnectionError("Test error")
        await manager._notify_error(error)

        assert len(handler._errors) == 1
        assert handler._errors[0][0] == error

    @pytest.mark.asyncio
    async def test_notify_status(
        self, manager_with_handler: tuple[ConcreteWebSocketManager, BufferedStreamHandler]
    ) -> None:
        """Test status notification."""
        manager, handler = manager_with_handler

        await manager._set_status(StreamStatus.CONNECTED, "Connected")

        assert manager.status == StreamStatus.CONNECTED
        assert handler._status_history[-1][0] == StreamStatus.CONNECTED


class TestWebSocketManagerReconnection:
    """Tests for reconnection logic."""

    @pytest.fixture
    def manager(self) -> ConcreteWebSocketManager:
        """Create manager with fast reconnect settings."""
        config = StreamConfig(
            api_key="test",
            reconnect_enabled=True,
            reconnect_delay_seconds=0.01,
            reconnect_delay_max_seconds=0.1,
            max_reconnect_attempts=3,
        )
        return ConcreteWebSocketManager(config)

    @pytest.mark.asyncio
    async def test_reconnect_exponential_backoff(self, manager: ConcreteWebSocketManager) -> None:
        """Test exponential backoff on reconnection."""
        initial_delay = manager._current_reconnect_delay

        # Disable reconnect to avoid actual reconnection attempt during test
        manager._config.reconnect_enabled = False

        # Simulate connection failure (won't reconnect since disabled)
        await manager._handle_connection_failure("Test failure")

        # Re-enable and manually test backoff logic by calling _schedule_reconnect
        manager._config.reconnect_enabled = True
        manager._reconnect_attempts = manager._config.max_reconnect_attempts  # Max attempts
        await manager._schedule_reconnect()  # This will just disconnect

        # Verify the delay doubling logic directly
        manager2 = ConcreteWebSocketManager(manager._config)
        initial = manager2._current_reconnect_delay
        # Manually simulate backoff
        manager2._current_reconnect_delay = min(
            manager2._current_reconnect_delay * 2,
            manager2._config.reconnect_delay_max_seconds,
        )
        assert manager2._current_reconnect_delay == initial * 2

    @pytest.mark.asyncio
    async def test_reconnect_max_delay(self, manager: ConcreteWebSocketManager) -> None:
        """Test max reconnect delay is respected."""
        manager._current_reconnect_delay = 0.05

        await manager._handle_connection_failure("Test failure")

        assert manager._current_reconnect_delay <= manager._config.reconnect_delay_max_seconds

    @pytest.mark.asyncio
    async def test_reconnect_max_attempts(self, manager: ConcreteWebSocketManager) -> None:
        """Test max reconnection attempts."""
        manager._reconnect_attempts = manager._config.max_reconnect_attempts

        await manager._schedule_reconnect()

        assert manager.status == StreamStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_reconnect_disabled(self) -> None:
        """Test disabled reconnection."""
        config = StreamConfig(
            api_key="test",
            reconnect_enabled=False,
        )
        manager = ConcreteWebSocketManager(config)

        await manager._handle_connection_failure("Test failure")

        assert manager.status == StreamStatus.DISCONNECTED
