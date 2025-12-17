"""
Massive market data adapter for FlowRoute engine.
"""

from datetime import datetime, timedelta
import logging
from typing import Any

from ordinis.adapters.market_data.massive import MassiveDataPlugin
from ordinis.plugins.base import PluginConfig

logger = logging.getLogger(__name__)


class MassiveMarketDataAdapter:
    """Adapter for Massive market data."""

    def __init__(self, api_key: str):
        self.config = PluginConfig(name="massive", enabled=True, options={"api_key": api_key})
        self.plugin = MassiveDataPlugin(self.config)
        self._initialized = False

    async def ensure_initialized(self):
        if not self._initialized:
            await self.plugin.initialize()
            self._initialized = True

    async def get_price_history(self, symbol: str, periods: int, timeframe: str) -> list[float]:
        await self.ensure_initialized()
        # Map timeframe "1Min" to Massive format
        multiplier = 1
        timespan = "minute"
        if timeframe == "1Min":
            multiplier = 1
            timespan = "minute"
        elif timeframe == "1Hour":
            multiplier = 1
            timespan = "hour"
        elif timeframe == "1Day":
            multiplier = 1
            timespan = "day"

        # Calculate dates
        to_date = datetime.now().strftime("%Y-%m-%d")
        # Fetch enough data
        days_back = 5
        if timespan == "day":
            days_back = periods * 2
        elif timespan == "hour":
            days_back = (periods // 6) + 2

        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        try:
            data = await self.plugin.get_historical_bars(
                symbol, multiplier, timespan, from_date, to_date
            )
            if "results" in data:
                # Extract closing prices
                closes = [bar["c"] for bar in data["results"]]
                return closes[-periods:] if len(closes) > periods else closes
            return []
        except Exception as e:
            logger.error(f"Error getting price history: {e}")
            return []

    def is_market_open(self) -> bool:
        # Simple check for now
        # TODO: Implement proper market status check
        return True

    async def get_latest_trade(self, symbol: str) -> dict[str, Any] | None:
        await self.ensure_initialized()
        try:
            # Using internal method for now as plugin doesn't expose it yet
            data = await self.plugin._make_request(f"last/trade/{symbol}")
            if "results" in data:
                res = data["results"]
                return {"price": res.get("p"), "size": res.get("s"), "timestamp": res.get("t")}
            return None
        except Exception as e:
            logger.error(f"Error getting latest trade: {e}")
            return None

    async def get_latest_quote(self, symbol: str) -> dict[str, Any] | None:
        await self.ensure_initialized()
        try:
            data = await self.plugin.get_quote(symbol)
            if "results" in data:
                res = data["results"]
                return {
                    "bid": res.get("p"),  # Assuming p=bid
                    "ask": res.get("P"),  # Assuming P=ask
                    "timestamp": res.get("t"),
                }
            return None
        except Exception as e:
            logger.error(f"Error getting latest quote: {e}")
            return None
