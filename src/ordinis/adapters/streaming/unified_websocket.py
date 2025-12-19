"""
Unified WebSocket Market Data Integration.

Provides a unified interface for real-time market data from multiple providers:
- Alpaca (stocks, crypto)
- Finnhub (stocks, forex)
- Massive (intraday)
- Polygon (stocks, options, crypto)

Step 1 of Trade Enhancement Roadmap.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
import logging
from typing import TYPE_CHECKING, Any, Callable

from ordinis.adapters.streaming.stream_protocol import (
    StreamBar,
    StreamConfig,
    StreamQuote,
    StreamStatus,
    StreamTrade,
)
from ordinis.adapters.streaming.websocket_manager import WebSocketManager

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DataProvider(Enum):
    """Supported market data providers."""
    
    ALPACA = auto()
    FINNHUB = auto()
    MASSIVE = auto()
    POLYGON = auto()


class DataType(Enum):
    """Types of market data."""
    
    QUOTE = auto()
    TRADE = auto()
    BAR = auto()
    ORDERBOOK = auto()


@dataclass
class ProviderConfig:
    """Configuration for a data provider."""
    
    provider: DataProvider
    api_key: str
    api_secret: str | None = None
    base_url: str = ""
    ws_url: str = ""
    enabled: bool = True
    priority: int = 0  # Lower = higher priority for failover
    rate_limit_per_second: int = 100
    supports_quotes: bool = True
    supports_trades: bool = True
    supports_bars: bool = True
    supports_orderbook: bool = False


@dataclass
class MarketDataEvent:
    """Unified market data event."""
    
    symbol: str
    data_type: DataType
    provider: DataProvider
    timestamp: datetime
    data: StreamQuote | StreamTrade | StreamBar
    latency_ms: float = 0.0
    sequence_id: int = 0


@dataclass
class SubscriptionRequest:
    """Request to subscribe to market data."""
    
    symbols: list[str]
    data_types: list[DataType] = field(default_factory=lambda: [DataType.QUOTE, DataType.TRADE])
    preferred_provider: DataProvider | None = None


@dataclass
class ProviderStats:
    """Statistics for a provider."""
    
    provider: DataProvider
    status: StreamStatus = StreamStatus.DISCONNECTED
    messages_received: int = 0
    errors_count: int = 0
    avg_latency_ms: float = 0.0
    last_message_time: datetime | None = None
    symbols_subscribed: set[str] = field(default_factory=set)
    
    def update_latency(self, latency_ms: float) -> None:
        """Update running average latency."""
        if self.messages_received == 0:
            self.avg_latency_ms = latency_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms


class UnifiedMarketDataStream:
    """
    Unified market data stream aggregating multiple providers.
    
    Features:
    - Automatic failover between providers
    - Deduplication of messages
    - Latency tracking and provider scoring
    - Seamless reconnection
    - Priority-based provider selection
    
    Example:
        >>> stream = UnifiedMarketDataStream()
        >>> stream.add_provider(ProviderConfig(
        ...     provider=DataProvider.ALPACA,
        ...     api_key="...",
        ...     api_secret="...",
        ... ))
        >>> stream.add_handler(my_quote_handler, DataType.QUOTE)
        >>> await stream.connect()
        >>> await stream.subscribe(["AAPL", "MSFT"], [DataType.QUOTE, DataType.TRADE])
    """
    
    def __init__(
        self,
        dedup_window_ms: float = 100.0,
        failover_threshold_errors: int = 5,
        health_check_interval_seconds: float = 30.0,
    ) -> None:
        """
        Initialize unified market data stream.
        
        Args:
            dedup_window_ms: Window for deduplicating messages across providers
            failover_threshold_errors: Errors before triggering failover
            health_check_interval_seconds: Health check interval
        """
        self._providers: dict[DataProvider, WebSocketManager] = {}
        self._provider_configs: dict[DataProvider, ProviderConfig] = {}
        self._provider_stats: dict[DataProvider, ProviderStats] = {}
        self._handlers: dict[DataType, list[Callable[[MarketDataEvent], Any]]] = defaultdict(list)
        self._subscriptions: dict[str, set[DataProvider]] = defaultdict(set)
        self._dedup_cache: dict[str, datetime] = {}  # symbol:seqid -> timestamp
        self._dedup_window_ms = dedup_window_ms
        self._failover_threshold = failover_threshold_errors
        self._health_check_interval = health_check_interval_seconds
        self._sequence_counter = 0
        self._running = False
        self._health_check_task: asyncio.Task | None = None
        self._primary_provider: DataProvider | None = None
        
    def add_provider(self, config: ProviderConfig) -> None:
        """
        Add a data provider.
        
        Args:
            config: Provider configuration
        """
        if not config.enabled:
            logger.info(f"Provider {config.provider.name} disabled, skipping")
            return
            
        self._provider_configs[config.provider] = config
        self._provider_stats[config.provider] = ProviderStats(provider=config.provider)
        
        # Create provider-specific WebSocket manager
        manager = self._create_provider_manager(config)
        if manager:
            self._providers[config.provider] = manager
            
        logger.info(f"Added provider: {config.provider.name} (priority={config.priority})")
        
    def _create_provider_manager(self, config: ProviderConfig) -> WebSocketManager | None:
        """Create provider-specific WebSocket manager."""
        stream_config = StreamConfig(
            api_key=config.api_key,
            reconnect_enabled=True,
            reconnect_delay_seconds=1.0,
            max_reconnect_attempts=10,
        )
        
        if config.provider == DataProvider.ALPACA:
            from ordinis.adapters.streaming.alpaca_stream import AlpacaWebSocketManager
            return AlpacaWebSocketManager(stream_config, api_secret=config.api_secret)
        elif config.provider == DataProvider.FINNHUB:
            from ordinis.adapters.streaming.finnhub_stream import FinnhubWebSocketManager
            return FinnhubWebSocketManager(stream_config)
        elif config.provider == DataProvider.MASSIVE:
            from ordinis.adapters.streaming.massive_stream import MassiveWebSocketManager
            return MassiveWebSocketManager(stream_config)
        else:
            logger.warning(f"No manager implementation for {config.provider.name}")
            return None
            
    def add_handler(
        self,
        handler: Callable[[MarketDataEvent], Any],
        data_type: DataType,
    ) -> None:
        """
        Add event handler for specific data type.
        
        Args:
            handler: Callback function
            data_type: Type of data to handle
        """
        self._handlers[data_type].append(handler)
        
    def remove_handler(
        self,
        handler: Callable[[MarketDataEvent], Any],
        data_type: DataType,
    ) -> None:
        """Remove event handler."""
        if handler in self._handlers[data_type]:
            self._handlers[data_type].remove(handler)
            
    async def connect(self) -> bool:
        """
        Connect to all enabled providers.
        
        Returns:
            True if at least one provider connected successfully
        """
        if self._running:
            return True
            
        connected_count = 0
        
        # Connect in priority order
        sorted_providers = sorted(
            self._provider_configs.items(),
            key=lambda x: x[1].priority,
        )
        
        for provider, config in sorted_providers:
            if provider not in self._providers:
                continue
                
            manager = self._providers[provider]
            try:
                await manager.connect()
                self._provider_stats[provider].status = StreamStatus.CONNECTED
                connected_count += 1
                
                # Set primary provider (highest priority that connects)
                if self._primary_provider is None:
                    self._primary_provider = provider
                    
                logger.info(f"Connected to {provider.name}")
            except Exception as e:
                logger.error(f"Failed to connect to {provider.name}: {e}")
                self._provider_stats[provider].status = StreamStatus.ERROR
                self._provider_stats[provider].errors_count += 1
                
        if connected_count > 0:
            self._running = True
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
        return connected_count > 0
        
    async def disconnect(self) -> None:
        """Disconnect from all providers."""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
                
        for provider, manager in self._providers.items():
            try:
                await manager.disconnect()
                self._provider_stats[provider].status = StreamStatus.DISCONNECTED
            except Exception as e:
                logger.error(f"Error disconnecting from {provider.name}: {e}")
                
    async def subscribe(
        self,
        symbols: list[str],
        data_types: list[DataType] | None = None,
        preferred_provider: DataProvider | None = None,
    ) -> dict[str, DataProvider]:
        """
        Subscribe to market data for symbols.
        
        Args:
            symbols: List of symbols to subscribe
            data_types: Types of data to receive (default: quotes and trades)
            preferred_provider: Preferred provider (uses primary if not specified)
            
        Returns:
            Dict mapping symbol to provider handling it
        """
        data_types = data_types or [DataType.QUOTE, DataType.TRADE]
        provider = preferred_provider or self._primary_provider
        
        if provider is None:
            raise RuntimeError("No providers connected")
            
        symbol_to_provider: dict[str, DataProvider] = {}
        
        if provider in self._providers:
            manager = self._providers[provider]
            await manager.subscribe(symbols)
            
            for symbol in symbols:
                self._subscriptions[symbol].add(provider)
                self._provider_stats[provider].symbols_subscribed.add(symbol)
                symbol_to_provider[symbol] = provider
                
            logger.info(f"Subscribed to {len(symbols)} symbols via {provider.name}")
            
        return symbol_to_provider
        
    async def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribe from symbols across all providers."""
        for symbol in symbols:
            providers = self._subscriptions.pop(symbol, set())
            for provider in providers:
                if provider in self._providers:
                    await self._providers[provider].unsubscribe([symbol])
                    self._provider_stats[provider].symbols_subscribed.discard(symbol)
                    
    def _is_duplicate(self, symbol: str, timestamp: datetime) -> bool:
        """Check if message is duplicate within dedup window."""
        key = f"{symbol}:{timestamp.timestamp()}"
        
        now = datetime.now(UTC)
        if key in self._dedup_cache:
            return True
            
        # Clean old entries
        cutoff = now.timestamp() - (self._dedup_window_ms / 1000)
        self._dedup_cache = {
            k: v for k, v in self._dedup_cache.items()
            if v.timestamp() > cutoff
        }
        
        self._dedup_cache[key] = now
        return False
        
    async def _dispatch_event(self, event: MarketDataEvent) -> None:
        """Dispatch event to registered handlers."""
        # Skip duplicates
        if self._is_duplicate(event.symbol, event.timestamp):
            return
            
        self._sequence_counter += 1
        event.sequence_id = self._sequence_counter
        
        # Update stats
        stats = self._provider_stats.get(event.provider)
        if stats:
            stats.messages_received += 1
            stats.last_message_time = datetime.now(UTC)
            stats.update_latency(event.latency_ms)
            
        # Call handlers
        handlers = self._handlers.get(event.data_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Handler error for {event.data_type.name}: {e}")
                
    async def _health_check_loop(self) -> None:
        """Periodic health check and failover logic."""
        while self._running:
            await asyncio.sleep(self._health_check_interval)
            
            for provider, stats in self._provider_stats.items():
                # Check for stale connections
                if stats.last_message_time:
                    age = (datetime.now(UTC) - stats.last_message_time).total_seconds()
                    if age > self._health_check_interval * 2:
                        logger.warning(f"Provider {provider.name} stale ({age:.1f}s)")
                        stats.status = StreamStatus.ERROR
                        
                # Check error threshold for failover
                if stats.errors_count >= self._failover_threshold:
                    if provider == self._primary_provider:
                        await self._trigger_failover(provider)
                        
    async def _trigger_failover(self, failed_provider: DataProvider) -> None:
        """Trigger failover from failed provider."""
        logger.warning(f"Triggering failover from {failed_provider.name}")
        
        # Find next best provider
        candidates = [
            (p, c) for p, c in self._provider_configs.items()
            if p != failed_provider
            and p in self._providers
            and self._provider_stats[p].status == StreamStatus.CONNECTED
        ]
        
        if not candidates:
            logger.error("No failover candidates available!")
            return
            
        # Sort by priority
        candidates.sort(key=lambda x: x[1].priority)
        new_primary = candidates[0][0]
        
        # Migrate subscriptions
        symbols = list(self._provider_stats[failed_provider].symbols_subscribed)
        if symbols:
            await self.subscribe(symbols, preferred_provider=new_primary)
            
        self._primary_provider = new_primary
        logger.info(f"Failed over to {new_primary.name}")
        
    def get_stats(self) -> dict[str, ProviderStats]:
        """Get statistics for all providers."""
        return {p.name: s for p, s in self._provider_stats.items()}
        
    def get_best_provider(self, symbol: str | None = None) -> DataProvider | None:
        """
        Get best provider based on latency and reliability.
        
        Args:
            symbol: Optional symbol to check subscription
            
        Returns:
            Best provider or None
        """
        candidates = []
        
        for provider, stats in self._provider_stats.items():
            if stats.status != StreamStatus.CONNECTED:
                continue
            if symbol and symbol not in stats.symbols_subscribed:
                continue
                
            # Score = low latency + low errors + priority
            config = self._provider_configs[provider]
            score = (
                stats.avg_latency_ms * 0.5
                + stats.errors_count * 10
                + config.priority * 100
            )
            candidates.append((provider, score))
            
        if not candidates:
            return self._primary_provider
            
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]


# =============================================================================
# Alpaca-specific WebSocket Manager
# =============================================================================


class AlpacaWebSocketManager(WebSocketManager):
    """Alpaca WebSocket manager for stocks and crypto."""
    
    def __init__(
        self,
        config: StreamConfig,
        api_secret: str | None = None,
        feed: str = "iex",  # "iex" or "sip"
    ) -> None:
        """Initialize Alpaca WebSocket manager."""
        super().__init__(config)
        self._api_secret = api_secret
        self._feed = feed
        
    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "Alpaca"
        
    @property
    def websocket_url(self) -> str:
        """Get WebSocket URL."""
        return f"wss://stream.data.alpaca.markets/v2/{self._feed}"
        
    async def _authenticate(self) -> bool:
        """Authenticate with Alpaca."""
        if not self._ws:
            return False
            
        auth_message = {
            "action": "auth",
            "key": self._config.api_key,
            "secret": self._api_secret,
        }
        
        await self._ws.send(str(auth_message))
        # Wait for auth response
        response = await self._ws.recv()
        # Parse and validate response
        return True
        
    async def _send_subscribe(self, symbols: list[str]) -> None:
        """Send subscription message."""
        if not self._ws:
            return
            
        subscribe_message = {
            "action": "subscribe",
            "trades": symbols,
            "quotes": symbols,
            "bars": symbols,
        }
        
        await self._ws.send(str(subscribe_message))
        
    async def _send_unsubscribe(self, symbols: list[str]) -> None:
        """Send unsubscription message."""
        if not self._ws:
            return
            
        unsubscribe_message = {
            "action": "unsubscribe",
            "trades": symbols,
            "quotes": symbols,
            "bars": symbols,
        }
        
        await self._ws.send(str(unsubscribe_message))
        
    async def _handle_message(self, message: str) -> None:
        """Handle incoming message."""
        import json
        
        try:
            data = json.loads(message)
            
            if isinstance(data, list):
                for item in data:
                    await self._process_item(item)
            else:
                await self._process_item(data)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
            
    async def _process_item(self, item: dict) -> None:
        """Process a single message item."""
        msg_type = item.get("T")
        
        if msg_type == "t":  # Trade
            trade = StreamTrade(
                symbol=item["S"],
                price=float(item["p"]),
                size=int(item["s"]),
                timestamp=datetime.fromisoformat(item["t"].replace("Z", "+00:00")),
                provider=self.provider_name,
                exchange=item.get("x", ""),
                conditions=tuple(item.get("c", [])),
            )
            await self._emit_trade(trade)
            
        elif msg_type == "q":  # Quote
            quote = StreamQuote(
                symbol=item["S"],
                bid=float(item["bp"]),
                ask=float(item["ap"]),
                bid_size=int(item["bs"]),
                ask_size=int(item["as"]),
                timestamp=datetime.fromisoformat(item["t"].replace("Z", "+00:00")),
                provider=self.provider_name,
            )
            await self._emit_quote(quote)
            
        elif msg_type == "b":  # Bar
            bar = StreamBar(
                symbol=item["S"],
                open=float(item["o"]),
                high=float(item["h"]),
                low=float(item["l"]),
                close=float(item["c"]),
                volume=int(item["v"]),
                timestamp=datetime.fromisoformat(item["t"].replace("Z", "+00:00")),
                provider=self.provider_name,
                vwap=item.get("vw"),
            )
            await self._emit_bar(bar)
