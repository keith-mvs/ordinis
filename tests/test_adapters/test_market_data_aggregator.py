from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from ordinis.adapters.market_data.aggregator import (
    AggregationConfig,
    AggregationMethod,
    DataAggregator,
)
from ordinis.plugins.base import DataPlugin, PluginConfig, PluginHealth, PluginStatus


class MockDataPlugin(DataPlugin):
    def __init__(self, name, price=100.0):
        config = PluginConfig(name=name, enabled=True)
        super().__init__(config)
        self.name = name
        self.mock_price = price
        self.should_fail = False

        # Overwrite with AsyncMock for tracking
        self.initialize = AsyncMock(return_value=True)
        self.shutdown = AsyncMock(return_value=None)
        self.validate_symbol = AsyncMock(return_value=True)
        self.get_historical = AsyncMock(return_value=[])

    # Implement abstract methods to satisfy ABC check at instantiation
    async def initialize(self) -> bool:
        return True

    async def shutdown(self) -> None:
        pass

    async def validate_symbol(self, symbol: str) -> bool:
        return True

    async def get_historical(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1d"
    ):
        return []

    async def get_quote(self, symbol: str):
        if self.should_fail:
            raise Exception("Provider failed")
        return {
            "symbol": symbol,
            "price": self.mock_price,
            "bid": self.mock_price - 0.1,
            "ask": self.mock_price + 0.1,
            "volume": 1000,
            "timestamp": datetime.now(UTC),
        }

    async def health_check(self):
        return PluginHealth(
            status=PluginStatus.READY, last_check=datetime.now(UTC), latency_ms=10.0, message="OK"
        )


@pytest.fixture
def aggregator_config():
    return PluginConfig(name="aggregator", enabled=True)


@pytest.fixture
def mock_providers():
    p1 = MockDataPlugin("provider1", 100.0)
    p2 = MockDataPlugin("provider2", 101.0)
    p3 = MockDataPlugin("provider3", 102.0)
    return [p1, p2, p3]


@pytest.mark.asyncio
async def test_aggregator_initialization(aggregator_config, mock_providers):
    aggregator = DataAggregator(aggregator_config, providers=mock_providers)

    success = await aggregator.initialize()
    assert success is True
    assert aggregator.status == PluginStatus.READY

    # Verify all providers were initialized
    for p in mock_providers:
        p.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_aggregator_get_quote_median(aggregator_config, mock_providers):
    # Prices: 100, 101, 102 -> Median should be 101
    aggregator = DataAggregator(aggregator_config, providers=mock_providers)
    await aggregator.initialize()

    quote = await aggregator.get_quote("AAPL")

    assert quote["symbol"] == "AAPL"
    assert quote["price"] == 101.0
    assert quote["aggregation"]["provider_count"] == 3


@pytest.mark.asyncio
async def test_aggregator_get_quote_mean(aggregator_config, mock_providers):
    # Prices: 100, 101, 102 -> Mean should be 101
    agg_config = AggregationConfig(method=AggregationMethod.MEAN)
    aggregator = DataAggregator(
        aggregator_config, providers=mock_providers, aggregation_config=agg_config
    )
    await aggregator.initialize()

    quote = await aggregator.get_quote("AAPL")

    assert quote["price"] == 101.0


@pytest.mark.asyncio
async def test_aggregator_provider_failure(aggregator_config, mock_providers):
    # Make one provider fail
    mock_providers[0].should_fail = True

    aggregator = DataAggregator(aggregator_config, providers=mock_providers)
    await aggregator.initialize()

    quote = await aggregator.get_quote("AAPL")

    # Should still work with remaining providers (101, 102) -> Median 101.5
    assert quote["price"] == 101.5
    assert quote["aggregation"]["provider_count"] == 2

    # Check stats
    stats = aggregator.provider_stats
    assert stats["provider1"].error_count == 1
    assert stats["provider2"].success_count == 1


@pytest.mark.asyncio
async def test_aggregator_outlier_detection(aggregator_config):
    # 100, 100, 100, 200 (outlier)
    p1 = MockDataPlugin("p1", 100.0)
    p2 = MockDataPlugin("p2", 100.0)
    p3 = MockDataPlugin("p3", 100.0)
    p4 = MockDataPlugin("p4", 200.0)  # Outlier

    agg_config = AggregationConfig(
        method=AggregationMethod.MEAN,
        outlier_detection=True,
        outlier_threshold=1.0,  # Lower threshold to catch 200 (Z=1.5)
    )

    aggregator = DataAggregator(
        aggregator_config, providers=[p1, p2, p3, p4], aggregation_config=agg_config
    )
    await aggregator.initialize()

    quote = await aggregator.get_quote("AAPL")

    # Should exclude 200 and average the rest -> 100
    assert quote["price"] == 100.0
    assert "p4" in quote["aggregation"]["outliers_excluded"]


@pytest.mark.asyncio
async def test_add_remove_provider(aggregator_config):
    aggregator = DataAggregator(aggregator_config)
    p1 = MockDataPlugin("p1", 100.0)

    aggregator.add_provider(p1)
    assert len(aggregator.providers) == 1

    aggregator.remove_provider(p1)
    assert len(aggregator.providers) == 0


@pytest.mark.asyncio
async def test_historical_data_fallback(aggregator_config, mock_providers):
    aggregator = DataAggregator(aggregator_config, providers=mock_providers)

    # Mock historical data return
    mock_providers[0].get_historical = AsyncMock(side_effect=Exception("Fail"))
    mock_providers[1].get_historical = AsyncMock(return_value=[{"price": 100}])

    data = await aggregator.get_historical("AAPL", datetime.now(UTC), datetime.now(UTC))

    assert len(data) == 1
    assert data[0]["price"] == 100
    # Should have tried p1 then p2
    mock_providers[0].get_historical.assert_called_once()
    mock_providers[1].get_historical.assert_called_once()
