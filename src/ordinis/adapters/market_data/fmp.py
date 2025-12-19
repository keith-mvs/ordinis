"""
Financial Modeling Prep (FMP) market data plugin.

Provides fundamental data, real-time quotes, and historical data.
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


class FMPDataPlugin(DataPlugin):
    """
    Financial Modeling Prep market data plugin.

    Provides:
    - Real-time quotes
    - Historical OHLCV data
    - Fundamental data (Ratios, Financial Statements)
    """

    name = "fmp"
    version = "1.0.0"
    description = "Financial Modeling Prep data provider"
    capabilities = [
        PluginCapability.READ,
        PluginCapability.REALTIME,
        PluginCapability.HISTORICAL,
    ]

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self._session: aiohttp.ClientSession | None = None
        self._api_key = config.options.get("api_key", "")

    async def initialize(self) -> bool:
        """Initialize the FMP connection."""
        await self._set_status(PluginStatus.INITIALIZING)

        if not self._api_key:
            logger.error("FMP API key not provided")
            await self._set_status(PluginStatus.ERROR)
            return False

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

            # Test API key
            test_result = await self._make_request("quote/AAPL")

            if test_result and isinstance(test_result, list) and len(test_result) > 0:
                await self._set_status(PluginStatus.READY)
                logger.info("FMP plugin initialized successfully")
                return True

            logger.error(f"FMP API key validation failed: {test_result}")
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
        logger.info("FMP plugin shutdown complete")

    async def health_check(self) -> PluginHealth:
        """Check plugin health."""
        start_time = datetime.utcnow()

        try:
            await self._make_request("quote/AAPL")
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            self._health = PluginHealth(
                status=PluginStatus.READY,
                last_check=datetime.utcnow(),
                latency_ms=latency,
                message="FMP API healthy",
            )

        except Exception as e:
            self._health = PluginHealth(
                status=PluginStatus.ERROR,
                last_check=datetime.utcnow(),
                error=str(e),
                message="FMP API unreachable",
            )

        return self._health

    async def _make_request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Make request to FMP API."""
        if not self._session:
            raise RuntimeError("Plugin not initialized")

        url = f"{self.BASE_URL}/{endpoint}"
        request_params = {"apikey": self._api_key}
        if params:
            request_params.update(params)

        async with self._session.get(url, params=request_params) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"FMP API error {response.status}: {text}")
            return await response.json()

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get real-time quote."""
        data = await self._make_request(f"quote/{symbol}")
        if not data:
            return {}
        return data[0]

    async def get_ratios(self, symbol: str) -> dict[str, Any]:
        """Get financial ratios (including P/E)."""
        # Get TTM ratios
        data = await self._make_request(f"ratios-ttm/{symbol}")
        if not data:
            return {}
        return data[0]

    async def get_historical_price(
        self, symbol: str, from_date: str, to_date: str
    ) -> list[dict[str, Any]]:
        """Get historical price data."""
        params = {"from": from_date, "to": to_date}
        return await self._make_request(f"historical-price-full/{symbol}", params=params)
