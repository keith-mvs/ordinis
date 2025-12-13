"""
Stream protocol definitions for real-time market data.

Defines data types, protocols, and handlers for WebSocket streaming.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Protocol, runtime_checkable


class StreamEventType(Enum):
    """Types of streaming events."""

    QUOTE = auto()
    TRADE = auto()
    BAR = auto()
    STATUS = auto()
    ERROR = auto()
    HEARTBEAT = auto()


class StreamStatus(Enum):
    """WebSocket connection status."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    ERROR = auto()
    CLOSED = auto()


@dataclass(frozen=True)
class StreamQuote:
    """Real-time quote data."""

    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    timestamp: datetime
    provider: str

    @property
    def mid(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        if self.mid == 0:
            return 0.0
        return (self.spread / self.mid) * 10000


@dataclass(frozen=True)
class StreamTrade:
    """Real-time trade data."""

    symbol: str
    price: float
    size: int
    timestamp: datetime
    provider: str
    exchange: str = ""
    conditions: tuple[str, ...] = ()

    @property
    def notional(self) -> float:
        """Calculate trade notional value."""
        return self.price * self.size


@dataclass(frozen=True)
class StreamBar:
    """Real-time bar/candle data."""

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime
    provider: str
    vwap: float | None = None

    @property
    def range(self) -> float:
        """Calculate bar range."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Calculate candle body size."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """Check if bar is bullish (close >= open)."""
        return self.close >= self.open


@dataclass
class StreamConfig:
    """Configuration for stream connections."""

    api_key: str
    reconnect_enabled: bool = True
    reconnect_delay_seconds: float = 1.0
    reconnect_delay_max_seconds: float = 60.0
    max_reconnect_attempts: int = 10
    connection_timeout_seconds: float = 30.0
    heartbeat_interval_seconds: float = 30.0
    buffer_size: int = 1000


@runtime_checkable
class StreamProvider(Protocol):
    """Protocol for stream providers."""

    @property
    def status(self) -> StreamStatus:
        """Get current connection status."""
        ...

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        ...

    async def connect(self) -> None:
        """Connect to the stream."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the stream."""
        ...

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to symbols."""
        ...

    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols."""
        ...

    def add_handler(self, handler: BaseStreamHandler) -> None:
        """Add an event handler."""
        ...

    def remove_handler(self, handler: BaseStreamHandler) -> None:
        """Remove an event handler."""
        ...


class BaseStreamHandler(ABC):
    """Base class for stream event handlers."""

    @abstractmethod
    async def on_quote(self, quote: StreamQuote) -> None:
        """Handle quote event."""

    @abstractmethod
    async def on_trade(self, trade: StreamTrade) -> None:
        """Handle trade event."""

    @abstractmethod
    async def on_bar(self, bar: StreamBar) -> None:
        """Handle bar event."""

    @abstractmethod
    async def on_status(self, status: StreamStatus, message: str) -> None:
        """Handle status change event."""

    @abstractmethod
    async def on_error(self, error: Exception) -> None:
        """Handle error event."""


# Type aliases for callbacks
QuoteCallback = Callable[[StreamQuote], None] | Callable[[StreamQuote], Awaitable[None]]
TradeCallback = Callable[[StreamTrade], None] | Callable[[StreamTrade], Awaitable[None]]
BarCallback = Callable[[StreamBar], None] | Callable[[StreamBar], Awaitable[None]]
StatusCallback = (
    Callable[[StreamStatus, str], None] | Callable[[StreamStatus, str], Awaitable[None]]
)
ErrorCallback = Callable[[Exception], None] | Callable[[Exception], Awaitable[None]]


class CallbackStreamHandler(BaseStreamHandler):
    """Stream handler that dispatches to callbacks."""

    def __init__(
        self,
        on_quote_callback: QuoteCallback | None = None,
        on_trade_callback: TradeCallback | None = None,
        on_bar_callback: BarCallback | None = None,
        on_status_callback: StatusCallback | None = None,
        on_error_callback: ErrorCallback | None = None,
    ) -> None:
        """Initialize with optional callbacks."""
        self._on_quote = on_quote_callback
        self._on_trade = on_trade_callback
        self._on_bar = on_bar_callback
        self._on_status = on_status_callback
        self._on_error = on_error_callback

    async def _call(self, callback: Callable[..., object] | None, *args: object) -> None:
        """Call a callback, handling both sync and async."""
        if callback is None:
            return
        result = callback(*args)
        if asyncio.iscoroutine(result):
            await result

    async def on_quote(self, quote: StreamQuote) -> None:
        """Dispatch quote to callback."""
        await self._call(self._on_quote, quote)

    async def on_trade(self, trade: StreamTrade) -> None:
        """Dispatch trade to callback."""
        await self._call(self._on_trade, trade)

    async def on_bar(self, bar: StreamBar) -> None:
        """Dispatch bar to callback."""
        await self._call(self._on_bar, bar)

    async def on_status(self, status: StreamStatus, message: str) -> None:
        """Dispatch status to callback."""
        await self._call(self._on_status, status, message)

    async def on_error(self, error: Exception) -> None:
        """Dispatch error to callback."""
        await self._call(self._on_error, error)


@dataclass
class BufferedStreamHandler(BaseStreamHandler):
    """Stream handler that buffers events for batch processing."""

    max_buffer_size: int = 1000
    _quotes: list[StreamQuote] = field(default_factory=list)
    _trades: list[StreamTrade] = field(default_factory=list)
    _bars: list[StreamBar] = field(default_factory=list)
    _status_history: list[tuple[StreamStatus, str, datetime]] = field(default_factory=list)
    _errors: list[tuple[Exception, datetime]] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def on_quote(self, quote: StreamQuote) -> None:
        """Buffer quote event."""
        async with self._lock:
            self._quotes.append(quote)
            if len(self._quotes) > self.max_buffer_size:
                self._quotes = self._quotes[-self.max_buffer_size :]

    async def on_trade(self, trade: StreamTrade) -> None:
        """Buffer trade event."""
        async with self._lock:
            self._trades.append(trade)
            if len(self._trades) > self.max_buffer_size:
                self._trades = self._trades[-self.max_buffer_size :]

    async def on_bar(self, bar: StreamBar) -> None:
        """Buffer bar event."""
        async with self._lock:
            self._bars.append(bar)
            if len(self._bars) > self.max_buffer_size:
                self._bars = self._bars[-self.max_buffer_size :]

    async def on_status(self, status: StreamStatus, message: str) -> None:
        """Buffer status event."""
        async with self._lock:
            self._status_history.append((status, message, datetime.now(UTC)))
            if len(self._status_history) > 100:
                self._status_history = self._status_history[-100:]

    async def on_error(self, error: Exception) -> None:
        """Buffer error event."""
        async with self._lock:
            self._errors.append((error, datetime.now(UTC)))
            if len(self._errors) > 100:
                self._errors = self._errors[-100:]

    async def get_quotes(self, clear: bool = False) -> list[StreamQuote]:
        """Get buffered quotes."""
        async with self._lock:
            quotes = list(self._quotes)
            if clear:
                self._quotes.clear()
            return quotes

    async def get_trades(self, clear: bool = False) -> list[StreamTrade]:
        """Get buffered trades."""
        async with self._lock:
            trades = list(self._trades)
            if clear:
                self._trades.clear()
            return trades

    async def get_bars(self, clear: bool = False) -> list[StreamBar]:
        """Get buffered bars."""
        async with self._lock:
            bars = list(self._bars)
            if clear:
                self._bars.clear()
            return bars

    async def get_latest_quote(self, symbol: str) -> StreamQuote | None:
        """Get latest quote for a symbol."""
        async with self._lock:
            for quote in reversed(self._quotes):
                if quote.symbol == symbol:
                    return quote
            return None

    async def get_latest_trade(self, symbol: str) -> StreamTrade | None:
        """Get latest trade for a symbol."""
        async with self._lock:
            for trade in reversed(self._trades):
                if trade.symbol == symbol:
                    return trade
            return None

    async def clear(self) -> None:
        """Clear all buffers."""
        async with self._lock:
            self._quotes.clear()
            self._trades.clear()
            self._bars.clear()
            self._status_history.clear()
            self._errors.clear()
