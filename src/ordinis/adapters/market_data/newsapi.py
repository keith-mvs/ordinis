"""
NewsAPI data plugin.

Provides news articles for sentiment analysis.
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


class NewsAPIDataPlugin(DataPlugin):
    """
    NewsAPI data plugin.

    Provides:
    - News articles by keyword/symbol
    - Top headlines
    """

    name = "newsapi"
    version = "1.0.0"
    description = "NewsAPI provider"
    capabilities = [
        PluginCapability.READ,
        PluginCapability.REALTIME,  # News is effectively real-time
    ]

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self._session: aiohttp.ClientSession | None = None
        self._api_key = config.extra.get("api_key", "")

    async def initialize(self) -> bool:
        """Initialize the NewsAPI connection."""
        await self._set_status(PluginStatus.INITIALIZING)

        if not self._api_key:
            logger.error("NewsAPI key not provided")
            await self._set_status(PluginStatus.ERROR)
            return False

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

            # Test API key
            test_result = await self._make_request(
                "top-headlines", {"country": "us", "pageSize": 1}
            )

            if test_result and test_result.get("status") == "ok":
                await self._set_status(PluginStatus.READY)
                logger.info("NewsAPI plugin initialized successfully")
                return True

            logger.error(f"NewsAPI key validation failed: {test_result}")
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
        logger.info("NewsAPI plugin shutdown complete")

    async def health_check(self) -> PluginHealth:
        """Check plugin health."""
        start_time = datetime.utcnow()

        try:
            await self._make_request("top-headlines", {"country": "us", "pageSize": 1})
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            self._health = PluginHealth(
                status=PluginStatus.READY,
                last_check=datetime.utcnow(),
                latency_ms=latency,
                message="NewsAPI healthy",
            )

        except Exception as e:
            self._health = PluginHealth(
                status=PluginStatus.ERROR,
                last_check=datetime.utcnow(),
                error=str(e),
                message="NewsAPI unreachable",
            )

        return self._health

    async def _make_request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Make request to NewsAPI."""
        if not self._session:
            raise RuntimeError("Plugin not initialized")

        url = f"{self.BASE_URL}/{endpoint}"
        request_params = {"apiKey": self._api_key}
        if params:
            request_params.update(params)

        async with self._session.get(url, params=request_params) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"NewsAPI error {response.status}: {text}")
            return await response.json()

    async def get_news(
        self, query: str, from_date: str | None = None, to_date: str | None = None
    ) -> list[dict[str, Any]]:
        """Get news articles."""
        params = {"q": query, "sortBy": "publishedAt"}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        data = await self._make_request("everything", params)
        return data.get("articles", [])
