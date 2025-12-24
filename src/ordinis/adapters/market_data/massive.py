"""
Massive market data plugin.

Provides real-time and historical market data from Massive (formerly Polygon).
"""

from datetime import datetime
import logging
from typing import Any

import aiohttp

from ordinis.plugins.base import (
    DataPlugin,
    PluginCapability,
    PluginConfig,
    PluginHealth,
    PluginStatus,
)

logger = logging.getLogger(__name__)


class MassiveDataPlugin(DataPlugin):
    """
    Massive market data plugin.

    Provides:
    - Real-time quotes
    - Historical OHLCV data
    - Reference data
    """

    name = "massive"
    version = "1.0.0"
    description = "Massive market data provider"
    capabilities = [
        PluginCapability.READ,
        PluginCapability.REALTIME,
        PluginCapability.HISTORICAL,
    ]

    # Placeholder URL - update if specific endpoint is known
    BASE_URL = "https://api.massive.com/v1"

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self._session: aiohttp.ClientSession | None = None
        self._api_key = config.extra.get("api_key", "")

    async def initialize(self) -> bool:
        """Initialize the Massive connection."""
        await self._set_status(PluginStatus.INITIALIZING)

        if not self._api_key:
            logger.error("Massive API key not provided")
            await self._set_status(PluginStatus.ERROR)
            return False

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

            # Test API key
            # Assuming a standard ticker details endpoint for validation
            test_result = await self._make_request("reference/tickers/AAPL")

            if test_result:
                await self._set_status(PluginStatus.READY)
                logger.info("Massive plugin initialized successfully")
                return True

            logger.error("Massive API key validation failed")
            await self._set_status(PluginStatus.ERROR)
            return False

        except Exception as e:
            await self._handle_error(e)
            return False

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        if self._session:
            await self._session.close()

        await self._set_status(PluginStatus.STOPPED)
        logger.info("Massive plugin shutdown complete")

    async def health_check(self) -> PluginHealth:
        """Check plugin health."""
        start_time = datetime.utcnow()

        try:
            await self._make_request("reference/tickers/AAPL")
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            self._health = PluginHealth(
                status=PluginStatus.READY,
                last_check=datetime.utcnow(),
                latency_ms=latency,
                message="Massive API healthy",
            )

        except Exception as e:
            self._health = PluginHealth(
                status=PluginStatus.ERROR,
                last_check=datetime.utcnow(),
                error=str(e),
                message="Massive API unreachable",
            )

        return self._health

    async def _make_request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Make request to Massive API."""
        if not self._session:
            raise RuntimeError("Plugin not initialized")

        url = f"{self.BASE_URL}/{endpoint}"
        request_params = {"apiKey": self._api_key}
        if params:
            request_params.update(params)

        async with self._session.get(url, params=request_params) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Massive API error {response.status}: {text}")
            return await response.json()

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get real-time quote."""
        # Assuming a standard quote endpoint
        return await self._make_request(f"last/quote/{symbol}")

    async def get_historical_bars(
        self, symbol: str, multiplier: int, timespan: str, from_date: str, to_date: str
    ) -> dict[str, Any]:
        """Get historical bars."""
        # Assuming a standard aggregates endpoint
        endpoint = f"aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        return await self._make_request(endpoint)
