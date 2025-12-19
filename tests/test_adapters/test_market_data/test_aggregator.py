"""Tests for DataAggregator."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ordinis.adapters.market_data.aggregator import (
    AggregatedQuote,
    AggregationConfig,
    AggregationMethod,
    DataAggregator,
    ProviderResult,
    ProviderStats,
    ProviderWeight,
)
from ordinis.plugins.base import (
    DataPlugin,
    PluginConfig,
    PluginHealth,
    PluginStatus,
)


class MockDataPlugin(DataPlugin):
    """Mock data plugin for testing."""

    def __init__(
        self,
        plugin_name: str,
        quote_price: float | None = None,
        should_fail: bool = False,
    ) -> None:
        """Initialize mock plugin."""
        super().__init__(PluginConfig(name=plugin_name))
        self._plugin_name = plugin_name
        self._quote_price = quote_price
        self._should_fail = should_fail

    @property
    def name(self) -> str:
        """Get plugin name."""
        return self._plugin_name

    async def initialize(self) -> bool:
        """Initialize plugin."""
        if self._should_fail:
            return False
        await self._set_status(PluginStatus.READY)
        return True

    async def shutdown(self) -> None:
        """Shutdown plugin."""
        await self._set_status(PluginStatus.STOPPED)

    async def health_check(self) -> PluginHealth:
        """Check health."""
        return PluginHealth(
            status=PluginStatus.ERROR if self._should_fail else PluginStatus.READY,
            last_check=datetime.now(UTC),
            latency_ms=10.0,
        )

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get quote."""
        if self._should_fail:
            raise ConnectionError("Mock failure")
        if self._quote_price is None:
            raise ValueError("No quote configured")
        return {
            "symbol": symbol,
            "price": self._quote_price,
            "bid": self._quote_price - 0.01,
            "ask": self._quote_price + 0.01,
            "volume": 1000000,
        }

    async def get_historical(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1d"
    ) -> list[dict[str, Any]]:
        """Get historical data."""
        if self._should_fail:
            raise ConnectionError("Mock failure")
        return [{"date": start.isoformat(), "close": self._quote_price or 100.0}]


class TestProviderStats:
    """Tests for ProviderStats dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        stats = ProviderStats(provider_name="test")
        assert stats.request_count == 0
        assert stats.success_count == 0
        assert stats.error_count == 0
        assert stats.total_latency_ms == 0.0

    def test_success_rate_zero_requests(self) -> None:
        """Test success rate with zero requests."""
        stats = ProviderStats(provider_name="test")
        assert stats.success_rate == 0.0

    def test_success_rate_calculation(self) -> None:
        """Test success rate calculation."""
        stats = ProviderStats(
            provider_name="test",
            request_count=10,
            success_count=8,
        )
        assert stats.success_rate == 0.8

    def test_avg_latency_zero_success(self) -> None:
        """Test average latency with zero successes."""
        stats = ProviderStats(provider_name="test")
        assert stats.avg_latency_ms == 0.0

    def test_avg_latency_calculation(self) -> None:
        """Test average latency calculation."""
        stats = ProviderStats(
            provider_name="test",
            success_count=5,
            total_latency_ms=500.0,
        )
        assert stats.avg_latency_ms == 100.0


class TestProviderResult:
    """Tests for ProviderResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful result."""
        result = ProviderResult(
            provider_name="test",
            data={"price": 100.0},
            latency_ms=50.0,
            timestamp=datetime.now(UTC),
            success=True,
        )
        assert result.success is True
        assert result.error is None

    def test_failure_result(self) -> None:
        """Test failed result."""
        result = ProviderResult(
            provider_name="test",
            data={},
            latency_ms=100.0,
            timestamp=datetime.now(UTC),
            success=False,
            error="Connection failed",
        )
        assert result.success is False
        assert result.error == "Connection failed"


class TestAggregatedQuote:
    """Tests for AggregatedQuote dataclass."""

    def test_source_summary(self) -> None:
        """Test source summary generation."""
        quote = AggregatedQuote(
            symbol="AAPL",
            price=150.0,
            bid=149.95,
            ask=150.05,
            volume=1000000,
            timestamp=datetime.now(UTC),
            method=AggregationMethod.MEDIAN,
            provider_count=3,
            providers_used=["a", "b", "c"],
            outliers_excluded=["d"],
            confidence=0.85,
        )
        assert quote.source_summary == "3 providers, 1 outliers excluded"


class TestAggregationConfig:
    """Tests for AggregationConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = AggregationConfig()
        assert config.method == AggregationMethod.MEDIAN
        assert config.outlier_detection is True
        assert config.outlier_threshold == 2.0
        assert config.min_providers == 1
        assert config.timeout_seconds == 10.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = AggregationConfig(
            method=AggregationMethod.WEIGHTED_MEAN,
            outlier_threshold=3.0,
            min_providers=2,
            weights=[ProviderWeight("a", 2.0), ProviderWeight("b", 1.0)],
        )
        assert config.method == AggregationMethod.WEIGHTED_MEAN
        assert config.outlier_threshold == 3.0
        assert len(config.weights) == 2


class TestDataAggregatorInit:
    """Tests for DataAggregator initialization."""

    def test_init_no_providers(self) -> None:
        """Test initialization without providers."""
        config = PluginConfig(name="aggregator")
        aggregator = DataAggregator(config)
        assert len(aggregator.providers) == 0

    def test_init_with_providers(self) -> None:
        """Test initialization with providers."""
        config = PluginConfig(name="aggregator")
        providers = [
            MockDataPlugin("p1", 100.0),
            MockDataPlugin("p2", 100.0),
        ]
        aggregator = DataAggregator(config, providers=providers)
        assert len(aggregator.providers) == 2

    def test_add_provider(self) -> None:
        """Test adding a provider."""
        config = PluginConfig(name="aggregator")
        aggregator = DataAggregator(config)
        provider = MockDataPlugin("test", 100.0)

        aggregator.add_provider(provider)

        assert len(aggregator.providers) == 1
        assert "test" in aggregator.provider_stats

    def test_remove_provider(self) -> None:
        """Test removing a provider."""
        config = PluginConfig(name="aggregator")
        provider = MockDataPlugin("test", 100.0)
        aggregator = DataAggregator(config, providers=[provider])

        aggregator.remove_provider(provider)

        assert len(aggregator.providers) == 0


class TestDataAggregatorLifecycle:
    """Tests for DataAggregator lifecycle methods."""

    @pytest.fixture
    def aggregator_with_providers(self) -> DataAggregator:
        """Create aggregator with mock providers."""
        config = PluginConfig(name="aggregator")
        providers = [
            MockDataPlugin("p1", 100.0),
            MockDataPlugin("p2", 100.5),
        ]
        return DataAggregator(config, providers=providers)

    @pytest.mark.asyncio
    async def test_initialize_success(self, aggregator_with_providers: DataAggregator) -> None:
        """Test successful initialization."""
        result = await aggregator_with_providers.initialize()
        assert result is True
        assert aggregator_with_providers.status == PluginStatus.READY

    @pytest.mark.asyncio
    async def test_initialize_no_providers(self) -> None:
        """Test initialization without providers fails."""
        config = PluginConfig(name="aggregator")
        aggregator = DataAggregator(config)

        result = await aggregator.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_below_min_providers(self) -> None:
        """Test initialization fails when below min providers."""
        config = PluginConfig(name="aggregator")
        agg_config = AggregationConfig(min_providers=3)
        providers = [MockDataPlugin("p1", 100.0)]
        aggregator = DataAggregator(config, providers=providers, aggregation_config=agg_config)

        result = await aggregator.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_shutdown(self, aggregator_with_providers: DataAggregator) -> None:
        """Test shutdown."""
        await aggregator_with_providers.initialize()
        await aggregator_with_providers.shutdown()

        assert aggregator_with_providers.status == PluginStatus.STOPPED

    @pytest.mark.asyncio
    async def test_health_check(self, aggregator_with_providers: DataAggregator) -> None:
        """Test health check."""
        await aggregator_with_providers.initialize()
        health = await aggregator_with_providers.health_check()

        assert health.status == PluginStatus.READY
        assert "2/2 providers healthy" in health.message


class TestDataAggregatorQuotes:
    """Tests for DataAggregator quote aggregation."""

    @pytest.fixture
    def aggregator(self) -> DataAggregator:
        """Create aggregator with multiple providers."""
        config = PluginConfig(name="aggregator")
        providers = [
            MockDataPlugin("p1", 100.00),
            MockDataPlugin("p2", 100.02),
            MockDataPlugin("p3", 100.01),
        ]
        return DataAggregator(config, providers=providers)

    @pytest.mark.asyncio
    async def test_get_quote_median(self, aggregator: DataAggregator) -> None:
        """Test getting aggregated quote with median."""
        await aggregator.initialize()
        quote = await aggregator.get_quote("AAPL")

        assert quote["symbol"] == "AAPL"
        # Median of 100.00, 100.01, 100.02 = 100.01
        assert quote["price"] == 100.01
        assert quote["aggregation"]["method"] == "median"
        assert quote["aggregation"]["provider_count"] == 3

    @pytest.mark.asyncio
    async def test_get_quote_mean(self) -> None:
        """Test getting aggregated quote with mean."""
        config = PluginConfig(name="aggregator")
        agg_config = AggregationConfig(method=AggregationMethod.MEAN)
        providers = [
            MockDataPlugin("p1", 100.00),
            MockDataPlugin("p2", 100.03),
        ]
        aggregator = DataAggregator(config, providers=providers, aggregation_config=agg_config)
        await aggregator.initialize()

        quote = await aggregator.get_quote("AAPL")

        # Mean of 100.00, 100.03 = 100.015
        assert abs(quote["price"] - 100.015) < 0.001

    @pytest.mark.asyncio
    async def test_get_quote_min(self) -> None:
        """Test getting aggregated quote with min."""
        config = PluginConfig(name="aggregator")
        agg_config = AggregationConfig(method=AggregationMethod.MIN)
        providers = [
            MockDataPlugin("p1", 100.00),
            MockDataPlugin("p2", 99.95),
            MockDataPlugin("p3", 100.05),
        ]
        aggregator = DataAggregator(config, providers=providers, aggregation_config=agg_config)
        await aggregator.initialize()

        quote = await aggregator.get_quote("AAPL")

        assert quote["price"] == 99.95

    @pytest.mark.asyncio
    async def test_get_quote_max(self) -> None:
        """Test getting aggregated quote with max."""
        config = PluginConfig(name="aggregator")
        agg_config = AggregationConfig(method=AggregationMethod.MAX)
        providers = [
            MockDataPlugin("p1", 100.00),
            MockDataPlugin("p2", 100.10),
        ]
        aggregator = DataAggregator(config, providers=providers, aggregation_config=agg_config)
        await aggregator.initialize()

        quote = await aggregator.get_quote("AAPL")

        assert quote["price"] == 100.10

    @pytest.mark.asyncio
    async def test_get_quote_weighted_mean(self) -> None:
        """Test getting aggregated quote with weighted mean."""
        config = PluginConfig(name="aggregator")
        agg_config = AggregationConfig(
            method=AggregationMethod.WEIGHTED_MEAN,
            weights=[
                ProviderWeight("p1", weight=2.0),
                ProviderWeight("p2", weight=1.0),
            ],
        )
        providers = [
            MockDataPlugin("p1", 100.00),  # weight 2
            MockDataPlugin("p2", 103.00),  # weight 1
        ]
        aggregator = DataAggregator(config, providers=providers, aggregation_config=agg_config)
        await aggregator.initialize()

        quote = await aggregator.get_quote("AAPL")

        # Weighted: (100 * 2 + 103 * 1) / 3 = 101.0
        assert abs(quote["price"] - 101.0) < 0.001

    @pytest.mark.asyncio
    async def test_get_quote_trimmed_mean(self) -> None:
        """Test getting aggregated quote with trimmed mean."""
        config = PluginConfig(name="aggregator")
        agg_config = AggregationConfig(method=AggregationMethod.TRIMMED_MEAN)
        # With 5 providers and 10% trim, should remove 1 from each end
        providers = [
            MockDataPlugin("p1", 95.00),  # Will be trimmed
            MockDataPlugin("p2", 100.00),
            MockDataPlugin("p3", 100.01),
            MockDataPlugin("p4", 100.02),
            MockDataPlugin("p5", 105.00),  # Will be trimmed
        ]
        aggregator = DataAggregator(config, providers=providers, aggregation_config=agg_config)
        await aggregator.initialize()

        quote = await aggregator.get_quote("AAPL")

        # Mean of 100.00, 100.01, 100.02 = 100.01
        assert abs(quote["price"] - 100.01) < 0.001


class TestDataAggregatorOutlierDetection:
    """Tests for outlier detection."""

    @pytest.mark.asyncio
    async def test_outlier_excluded(self) -> None:
        """Test that outliers are excluded from aggregation."""
        config = PluginConfig(name="aggregator")
        agg_config = AggregationConfig(
            method=AggregationMethod.MEAN,
            outlier_detection=True,
            outlier_threshold=2.0,
        )
        # Provider p6 is a clear outlier (200 vs ~100)
        providers = [
            MockDataPlugin("p1", 100.00),
            MockDataPlugin("p2", 100.01),
            MockDataPlugin("p3", 100.02),
            MockDataPlugin("p4", 99.99),
            MockDataPlugin("p5", 100.03),
            MockDataPlugin("outlier", 200.00),  # Outlier
        ]
        aggregator = DataAggregator(config, providers=providers, aggregation_config=agg_config)
        await aggregator.initialize()

        quote = await aggregator.get_quote("AAPL")

        # Outlier should be excluded
        assert "outlier" in quote["aggregation"]["outliers_excluded"]
        # Price should be around 100, not skewed by 200
        assert 99.9 < quote["price"] < 100.1

    @pytest.mark.asyncio
    async def test_outlier_detection_disabled(self) -> None:
        """Test outlier detection can be disabled."""
        config = PluginConfig(name="aggregator")
        agg_config = AggregationConfig(
            method=AggregationMethod.MEAN,
            outlier_detection=False,
        )
        providers = [
            MockDataPlugin("p1", 100.00),
            MockDataPlugin("p2", 200.00),  # Would be outlier if detection enabled
        ]
        aggregator = DataAggregator(config, providers=providers, aggregation_config=agg_config)
        await aggregator.initialize()

        quote = await aggregator.get_quote("AAPL")

        # Mean of 100 and 200 = 150
        assert quote["price"] == 150.0
        assert quote["aggregation"]["outliers_excluded"] == []


class TestDataAggregatorFailureHandling:
    """Tests for failure handling."""

    @pytest.mark.asyncio
    async def test_partial_provider_failure(self) -> None:
        """Test aggregation continues with partial failures."""
        config = PluginConfig(name="aggregator")
        providers = [
            MockDataPlugin("good1", 100.00),
            MockDataPlugin("fail", should_fail=True),
            MockDataPlugin("good2", 100.02),
        ]
        aggregator = DataAggregator(config, providers=providers)
        await aggregator.initialize()

        quote = await aggregator.get_quote("AAPL")

        # Should use the 2 successful providers
        assert quote["aggregation"]["provider_count"] == 2
        assert "good1" in quote["aggregation"]["providers_used"]
        assert "good2" in quote["aggregation"]["providers_used"]

    @pytest.mark.asyncio
    async def test_all_providers_fail(self) -> None:
        """Test error when all providers fail."""
        config = PluginConfig(name="aggregator")
        providers = [
            MockDataPlugin("fail1", should_fail=True),
            MockDataPlugin("fail2", should_fail=True),
        ]
        aggregator = DataAggregator(config, providers=providers)
        await aggregator.initialize()

        with pytest.raises(ValueError, match="No successful quotes"):
            await aggregator.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_disabled_provider_not_used(self) -> None:
        """Test disabled provider is not queried."""
        config = PluginConfig(name="aggregator")
        agg_config = AggregationConfig(
            weights=[
                ProviderWeight("enabled", 1.0, enabled=True),
                ProviderWeight("disabled", 1.0, enabled=False),
            ]
        )
        providers = [
            MockDataPlugin("enabled", 100.00),
            MockDataPlugin("disabled", 200.00),
        ]
        aggregator = DataAggregator(config, providers=providers, aggregation_config=agg_config)
        await aggregator.initialize()

        quote = await aggregator.get_quote("AAPL")

        # Only enabled provider should be used
        assert quote["aggregation"]["provider_count"] == 1
        assert quote["aggregation"]["providers_used"] == ["enabled"]


class TestDataAggregatorHistorical:
    """Tests for historical data retrieval."""

    @pytest.mark.asyncio
    async def test_get_historical_success(self) -> None:
        """Test getting historical data."""
        config = PluginConfig(name="aggregator")
        providers = [MockDataPlugin("p1", 100.0)]
        aggregator = DataAggregator(config, providers=providers)
        await aggregator.initialize()

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 31, tzinfo=UTC)
        data = await aggregator.get_historical("AAPL", start, end)

        assert len(data) == 1

    @pytest.mark.asyncio
    async def test_get_historical_falls_back(self) -> None:
        """Test historical fallback to next provider."""
        config = PluginConfig(name="aggregator")
        providers = [
            MockDataPlugin("fail", should_fail=True),
            MockDataPlugin("good", 100.0),
        ]
        aggregator = DataAggregator(config, providers=providers)
        await aggregator.initialize()

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 31, tzinfo=UTC)
        data = await aggregator.get_historical("AAPL", start, end)

        assert len(data) == 1

    @pytest.mark.asyncio
    async def test_get_historical_all_fail(self) -> None:
        """Test error when all providers fail for historical."""
        config = PluginConfig(name="aggregator")
        providers = [MockDataPlugin("fail", should_fail=True)]
        aggregator = DataAggregator(config, providers=providers)
        await aggregator.initialize()

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 1, 31, tzinfo=UTC)

        with pytest.raises(ValueError, match="No provider could fetch historical"):
            await aggregator.get_historical("AAPL", start, end)


class TestDataAggregatorValidation:
    """Tests for symbol validation."""

    @pytest.mark.asyncio
    async def test_validate_symbol_exists(self) -> None:
        """Test validating existing symbol."""
        config = PluginConfig(name="aggregator")
        providers = [MockDataPlugin("p1", 100.0)]
        aggregator = DataAggregator(config, providers=providers)
        await aggregator.initialize()

        result = await aggregator.validate_symbol("AAPL")

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_symbol_not_exists(self) -> None:
        """Test validating non-existing symbol."""
        config = PluginConfig(name="aggregator")
        providers = [MockDataPlugin("fail", should_fail=True)]
        aggregator = DataAggregator(config, providers=providers)
        await aggregator.initialize()

        result = await aggregator.validate_symbol("INVALID")

        assert result is False


class TestDataAggregatorConfidence:
    """Tests for confidence scoring."""

    @pytest.mark.asyncio
    async def test_confidence_increases_with_providers(self) -> None:
        """Test confidence increases with more providers."""
        config = PluginConfig(name="aggregator")

        # Single provider
        providers_1 = [MockDataPlugin("p1", 100.0)]
        agg1 = DataAggregator(config, providers=providers_1)
        await agg1.initialize()
        quote1 = await agg1.get_quote("AAPL")

        # Multiple providers
        providers_3 = [
            MockDataPlugin("p1", 100.0),
            MockDataPlugin("p2", 100.0),
            MockDataPlugin("p3", 100.0),
        ]
        agg3 = DataAggregator(config, providers=providers_3)
        await agg3.initialize()
        quote3 = await agg3.get_quote("AAPL")

        # More providers = higher confidence
        assert quote3["aggregation"]["confidence"] > quote1["aggregation"]["confidence"]

    @pytest.mark.asyncio
    async def test_confidence_higher_with_agreement(self) -> None:
        """Test confidence is higher when prices agree."""
        config = PluginConfig(name="aggregator")
        agg_config = AggregationConfig(outlier_detection=False)

        # Providers with agreeing prices
        providers_agree = [
            MockDataPlugin("p1", 100.00),
            MockDataPlugin("p2", 100.01),
            MockDataPlugin("p3", 100.02),
        ]
        agg_agree = DataAggregator(config, providers=providers_agree, aggregation_config=agg_config)
        await agg_agree.initialize()
        quote_agree = await agg_agree.get_quote("AAPL")

        # Providers with disagreeing prices
        providers_disagree = [
            MockDataPlugin("p1", 100.00),
            MockDataPlugin("p2", 110.00),
            MockDataPlugin("p3", 120.00),
        ]
        agg_disagree = DataAggregator(
            config, providers=providers_disagree, aggregation_config=agg_config
        )
        await agg_disagree.initialize()
        quote_disagree = await agg_disagree.get_quote("AAPL")

        # Agreement = higher confidence
        assert (
            quote_agree["aggregation"]["confidence"] > quote_disagree["aggregation"]["confidence"]
        )
