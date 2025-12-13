"""
Base plugin classes and interfaces.
"""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """Plugin lifecycle status."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class PluginCapability(Enum):
    """Plugin capabilities."""

    READ = "read"
    WRITE = "write"
    STREAM = "stream"
    HISTORICAL = "historical"
    REALTIME = "realtime"


@dataclass
class PluginConfig:
    """Base plugin configuration."""

    name: str
    enabled: bool = True
    api_key: str | None = None
    api_secret: str | None = None
    base_url: str | None = None
    timeout_seconds: int = 30
    max_retries: int = 3
    rate_limit_per_minute: int = 60
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginHealth:
    """Plugin health status."""

    status: PluginStatus
    last_check: datetime
    latency_ms: float
    error_count: int = 0
    last_error: str | None = None
    message: str | None = None


class Plugin(ABC):
    """
    Abstract base class for all plugins.

    Plugins are modular components that provide specific functionality
    such as market data, news, broker connectivity, etc.
    """

    # Class attributes to be overridden
    name: str = "base_plugin"
    version: str = "1.0.0"
    description: str = "Base plugin"
    capabilities: list[PluginCapability] = []

    def __init__(self, config: PluginConfig):
        self.config = config
        self._status = PluginStatus.UNINITIALIZED
        self._health = PluginHealth(
            status=PluginStatus.UNINITIALIZED,
            last_check=datetime.utcnow(),
            latency_ms=0.0,
        )
        self._rate_limiter = RateLimiter(config.rate_limit_per_minute)

    @property
    def status(self) -> PluginStatus:
        """Get current plugin status."""
        return self._status

    @property
    def health(self) -> PluginHealth:
        """Get plugin health information."""
        return self._health

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the plugin.

        Returns:
            True if initialization successful, False otherwise.
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the plugin gracefully."""

    @abstractmethod
    async def health_check(self) -> PluginHealth:
        """
        Perform health check.

        Returns:
            Current health status.
        """

    async def _set_status(self, status: PluginStatus) -> None:
        """Update plugin status."""
        old_status = self._status
        self._status = status
        self._health.status = status
        logger.info(f"Plugin {self.name}: {old_status.value} -> {status.value}")

    async def _handle_error(self, error: Exception) -> None:
        """Handle plugin error."""
        self._health.error_count += 1
        self._health.last_error = str(error)
        logger.error(f"Plugin {self.name} error: {error}")

        if self._health.error_count >= self.config.max_retries:
            await self._set_status(PluginStatus.ERROR)


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = datetime.utcnow()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """
        Acquire a rate limit token.

        Returns:
            True if token acquired, False if rate limited.
        """
        async with self._lock:
            now = datetime.utcnow()
            elapsed = (now - self.last_update).total_seconds()

            # Replenish tokens based on elapsed time
            tokens_to_add = elapsed * (self.requests_per_minute / 60)
            self.tokens = float(min(self.requests_per_minute, self.tokens + tokens_to_add))
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            return False

    async def wait_for_token(self) -> None:
        """Wait until a token is available."""
        while not await self.acquire():
            await asyncio.sleep(0.1)


class DataPlugin(Plugin):
    """
    Base class for data provider plugins.

    Provides market data, news, fundamentals, etc.
    """

    capabilities = [PluginCapability.READ]

    @abstractmethod
    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get real-time quote for a symbol."""

    @abstractmethod
    async def get_historical(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1d"
    ) -> list[dict[str, Any]]:
        """Get historical OHLCV data."""

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists."""
        try:
            await self.get_quote(symbol)
            return True
        except Exception:
            return False


class BrokerPlugin(Plugin):
    """
    Base class for broker connectivity plugins.

    Handles order execution, account management, etc.
    """

    capabilities = [PluginCapability.READ, PluginCapability.WRITE]

    @abstractmethod
    async def get_account(self) -> dict[str, Any]:
        """Get account information."""

    @abstractmethod
    async def get_positions(self) -> list[dict[str, Any]]:
        """Get current positions."""

    @abstractmethod
    async def submit_order(self, order: dict[str, Any]) -> str:
        """
        Submit an order.

        Returns:
            Order ID from broker.
        """

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""

    @abstractmethod
    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Get order status."""


class NewsPlugin(Plugin):
    """
    Base class for news and sentiment plugins.
    """

    capabilities = [PluginCapability.READ, PluginCapability.STREAM]

    @abstractmethod
    async def get_news(
        self, symbols: list[str] | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get recent news articles."""

    @abstractmethod
    async def get_sentiment(self, symbol: str) -> dict[str, Any]:
        """Get sentiment analysis for symbol."""


class FundamentalsPlugin(Plugin):
    """
    Base class for fundamental data plugins.
    """

    capabilities = [PluginCapability.READ, PluginCapability.HISTORICAL]

    @abstractmethod
    async def get_financials(self, symbol: str, period: str = "quarterly") -> dict[str, Any]:
        """Get financial statements."""

    @abstractmethod
    async def get_metrics(self, symbol: str) -> dict[str, Any]:
        """Get key financial metrics."""

    @abstractmethod
    async def get_estimates(self, symbol: str) -> dict[str, Any]:
        """Get analyst estimates."""


class OptionsPlugin(Plugin):
    """
    Base class for options data plugins.
    """

    capabilities = [PluginCapability.READ, PluginCapability.REALTIME]

    @abstractmethod
    async def get_options_chain(self, symbol: str, expiration: str | None = None) -> dict[str, Any]:
        """Get options chain for underlying."""

    @abstractmethod
    async def get_greeks(
        self, symbol: str, strike: float, expiration: str, option_type: str
    ) -> dict[str, Any]:
        """Get option Greeks."""


class AlternativeDataPlugin(Plugin):
    """
    Base class for alternative data plugins.

    Covers satellite imagery, web traffic, sentiment, etc.
    """

    capabilities = [PluginCapability.READ]

    @abstractmethod
    async def get_data(self, symbol: str, data_type: str, **kwargs) -> dict[str, Any]:
        """Get alternative data."""
