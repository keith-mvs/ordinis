"""
Data aggregator for multi-source consensus pricing.

Combines data from multiple providers with:
- Outlier detection via z-score
- Multiple aggregation methods
- Provider weighting and confidence scoring
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import logging
import statistics
from typing import Any

from ordinis.plugins.base import (
    DataPlugin,
    PluginCapability,
    PluginConfig,
    PluginHealth,
    PluginStatus,
)

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Methods for aggregating prices from multiple sources."""

    MEDIAN = "median"
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MIN = "min"
    MAX = "max"
    TRIMMED_MEAN = "trimmed_mean"


@dataclass
class ProviderWeight:
    """Weight configuration for a provider."""

    provider_name: str
    weight: float = 1.0
    enabled: bool = True


@dataclass
class ProviderResult:
    """Result from a single provider."""

    provider_name: str
    data: dict[str, Any]
    latency_ms: float
    timestamp: datetime
    success: bool = True
    error: str | None = None


@dataclass
class AggregatedQuote:
    """Aggregated quote from multiple providers."""

    symbol: str
    price: float
    bid: float | None
    ask: float | None
    volume: int | None
    timestamp: datetime
    method: AggregationMethod
    provider_count: int
    providers_used: list[str]
    outliers_excluded: list[str]
    confidence: float
    spread_pct: float | None = None

    @property
    def source_summary(self) -> str:
        """Get summary of sources used."""
        return f"{self.provider_count} providers, {len(self.outliers_excluded)} outliers excluded"


@dataclass
class ProviderStats:
    """Statistics for a provider."""

    provider_name: str
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    last_success: datetime | None = None
    last_error: datetime | None = None
    outlier_count: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.request_count == 0:
            return 0.0
        return self.success_count / self.request_count

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.success_count == 0:
            return 0.0
        return self.total_latency_ms / self.success_count


@dataclass
class AggregationConfig:
    """Configuration for data aggregation."""

    method: AggregationMethod = AggregationMethod.MEDIAN
    outlier_detection: bool = True
    outlier_threshold: float = 2.0  # Z-score threshold
    min_providers: int = 1
    timeout_seconds: float = 10.0
    weights: list[ProviderWeight] = field(default_factory=list)
    trimmed_mean_pct: float = 0.1  # Trim 10% from each end


class DataAggregator(DataPlugin):
    """
    Aggregates market data from multiple providers.

    Features:
    - Parallel data fetching from multiple providers
    - Outlier detection using z-score
    - Multiple aggregation methods
    - Provider weighting
    - Confidence scoring
    """

    name = "data_aggregator"
    version = "1.0.0"
    description = "Multi-source data aggregator"
    capabilities = [PluginCapability.READ, PluginCapability.REALTIME]

    def __init__(
        self,
        config: PluginConfig,
        providers: list[DataPlugin] | None = None,
        aggregation_config: AggregationConfig | None = None,
    ) -> None:
        """Initialize aggregator with providers."""
        super().__init__(config)
        self._providers: list[DataPlugin] = providers or []
        self._agg_config = aggregation_config or AggregationConfig()
        self._stats: dict[str, ProviderStats] = {}

        # Initialize stats for each provider
        for provider in self._providers:
            self._stats[provider.name] = ProviderStats(provider_name=provider.name)

    @property
    def providers(self) -> list[DataPlugin]:
        """Get registered providers."""
        return self._providers.copy()

    @property
    def provider_stats(self) -> dict[str, ProviderStats]:
        """Get provider statistics."""
        return self._stats.copy()

    def add_provider(self, provider: DataPlugin) -> None:
        """Add a data provider."""
        if provider not in self._providers:
            self._providers.append(provider)
            self._stats[provider.name] = ProviderStats(provider_name=provider.name)
            logger.info("Added provider: %s", provider.name)

    def remove_provider(self, provider: DataPlugin) -> None:
        """Remove a data provider."""
        if provider in self._providers:
            self._providers.remove(provider)
            self._stats.pop(provider.name, None)
            logger.info("Removed provider: %s", provider.name)

    async def initialize(self) -> bool:
        """Initialize all providers."""
        if not self._providers:
            logger.warning("No providers configured for aggregator")
            return False

        results = await asyncio.gather(
            *[p.initialize() for p in self._providers],
            return_exceptions=True,
        )

        success_count = sum(1 for r in results if r is True)
        logger.info(
            "Aggregator initialized: %d/%d providers ready",
            success_count,
            len(self._providers),
        )

        if success_count >= self._agg_config.min_providers:
            await self._set_status(PluginStatus.READY)
            return True

        await self._set_status(PluginStatus.ERROR)
        return False

    async def shutdown(self) -> None:
        """Shutdown all providers."""
        await asyncio.gather(
            *[p.shutdown() for p in self._providers],
            return_exceptions=True,
        )
        await self._set_status(PluginStatus.STOPPED)

    async def health_check(self) -> PluginHealth:
        """Check health of all providers."""
        now = datetime.now(UTC)
        healthy_count = 0

        for provider in self._providers:
            try:
                health = await provider.health_check()
                if health.status == PluginStatus.READY:
                    healthy_count += 1
            except Exception as e:
                logger.debug("Provider %s health check failed: %s", provider.name, e)

        is_healthy = healthy_count >= self._agg_config.min_providers
        self._health = PluginHealth(
            status=PluginStatus.READY if is_healthy else PluginStatus.ERROR,
            last_check=now,
            latency_ms=0.0,
            message=f"{healthy_count}/{len(self._providers)} providers healthy",
        )
        return self._health

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get aggregated quote from all providers."""
        results = await self._fetch_from_all(symbol, "quote")
        aggregated = self._aggregate_quotes(symbol, results)
        return self._quote_to_dict(aggregated)

    async def get_historical(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1d"
    ) -> list[dict[str, Any]]:
        """Get historical data from best available provider."""
        # For historical data, use the first successful provider
        for provider in self._providers:
            try:
                return await provider.get_historical(symbol, start, end, timeframe)
            except Exception as e:
                logger.debug("Provider %s failed historical: %s", provider.name, e)

        raise ValueError(f"No provider could fetch historical data for {symbol}")

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol exists in at least one provider."""
        for provider in self._providers:
            try:
                if await provider.validate_symbol(symbol):
                    return True
            except Exception:
                logger.debug("Provider %s failed symbol validation", provider.name)
        return False

    async def _fetch_from_all(self, symbol: str, data_type: str) -> list[ProviderResult]:
        """Fetch data from all providers in parallel."""
        tasks = []
        for provider in self._providers:
            # Check if provider is enabled in weights
            weight_config = next(
                (w for w in self._agg_config.weights if w.provider_name == provider.name),
                None,
            )
            if weight_config and not weight_config.enabled:
                continue

            tasks.append(self._fetch_from_provider(provider, symbol, data_type))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, ProviderResult):
                valid_results.append(result)

        return valid_results

    async def _fetch_from_provider(
        self, provider: DataPlugin, symbol: str, data_type: str
    ) -> ProviderResult:
        """Fetch data from a single provider."""
        stats = self._stats.get(provider.name)
        if stats:
            stats.request_count += 1

        start_time = datetime.now(UTC)

        try:
            if data_type == "quote":
                data = await asyncio.wait_for(
                    provider.get_quote(symbol),
                    timeout=self._agg_config.timeout_seconds,
                )
            else:
                raise ValueError(f"Unknown data type: {data_type}")

            latency = (datetime.now(UTC) - start_time).total_seconds() * 1000

            if stats:
                stats.success_count += 1
                stats.total_latency_ms += latency
                stats.last_success = datetime.now(UTC)

            return ProviderResult(
                provider_name=provider.name,
                data=data,
                latency_ms=latency,
                timestamp=datetime.now(UTC),
                success=True,
            )

        except Exception as e:
            latency = (datetime.now(UTC) - start_time).total_seconds() * 1000

            if stats:
                stats.error_count += 1
                stats.last_error = datetime.now(UTC)

            return ProviderResult(
                provider_name=provider.name,
                data={},
                latency_ms=latency,
                timestamp=datetime.now(UTC),
                success=False,
                error=str(e),
            )

    def _aggregate_quotes(self, symbol: str, results: list[ProviderResult]) -> AggregatedQuote:
        """Aggregate quotes from multiple providers."""
        # Filter successful results
        successful = [r for r in results if r.success and r.data]

        if not successful:
            raise ValueError(f"No successful quotes for {symbol}")

        # Extract prices
        prices: dict[str, float] = {}
        for result in successful:
            price = self._extract_price(result.data)
            if price is not None:
                prices[result.provider_name] = price

        if not prices:
            raise ValueError(f"No valid prices for {symbol}")

        # Detect outliers
        outliers: set[str] = set()
        if self._agg_config.outlier_detection and len(prices) >= 3:
            outliers = self._detect_outliers(prices)
            for provider in outliers:
                stats = self._stats.get(provider)
                if stats:
                    stats.outlier_count += 1

        # Filter out outliers
        filtered_prices = {k: v for k, v in prices.items() if k not in outliers}

        if not filtered_prices:
            # If all were outliers, use all prices
            filtered_prices = prices
            outliers = set()

        # Aggregate
        aggregated_price = self._aggregate_prices(filtered_prices)

        # Extract additional fields
        bids = [self._extract_field(r.data, "bid") for r in successful]
        asks = [self._extract_field(r.data, "ask") for r in successful]
        volumes = [self._extract_field(r.data, "volume") for r in successful]

        bid = statistics.median([b for b in bids if b is not None]) if any(bids) else None
        ask = statistics.median([a for a in asks if a is not None]) if any(asks) else None
        volume = sum(v for v in volumes if v is not None) if any(volumes) else None

        # Calculate confidence
        confidence = self._calculate_confidence(filtered_prices, outliers)

        # Calculate spread
        spread_pct = None
        if bid and ask and bid > 0:
            spread_pct = ((ask - bid) / bid) * 100

        return AggregatedQuote(
            symbol=symbol,
            price=aggregated_price,
            bid=bid,
            ask=ask,
            volume=int(volume) if volume else None,
            timestamp=datetime.now(UTC),
            method=self._agg_config.method,
            provider_count=len(filtered_prices),
            providers_used=list(filtered_prices.keys()),
            outliers_excluded=list(outliers),
            confidence=confidence,
            spread_pct=spread_pct,
        )

    def _extract_price(self, data: dict[str, Any]) -> float | None:
        """Extract price from quote data."""
        for key in ["price", "last", "lastPrice", "close", "mid"]:
            if key in data and data[key] is not None:
                try:
                    return float(data[key])
                except (ValueError, TypeError):
                    pass
        return None

    def _extract_field(self, data: dict[str, Any], field_name: str) -> float | None:
        """Extract a numeric field from data."""
        value = data.get(field_name)
        if value is not None:
            try:
                return float(value)
            except (ValueError, TypeError):
                pass
        return None

    def _detect_outliers(self, prices: dict[str, float]) -> set[str]:
        """Detect outliers using z-score method."""
        values = list(prices.values())

        if len(values) < 3:
            return set()

        mean = statistics.mean(values)
        stdev = statistics.stdev(values)

        if stdev == 0:
            return set()

        threshold = self._agg_config.outlier_threshold
        outliers: set[str] = set()

        for provider, price in prices.items():
            z_score = abs((price - mean) / stdev)
            if z_score > threshold:
                outliers.add(provider)
                logger.debug(
                    "Outlier detected: %s price=%.4f z-score=%.2f",
                    provider,
                    price,
                    z_score,
                )

        return outliers

    def _aggregate_prices(self, prices: dict[str, float]) -> float:
        """Aggregate prices using configured method."""
        values = list(prices.values())
        method = self._agg_config.method

        if method == AggregationMethod.MEDIAN:
            return statistics.median(values)

        if method == AggregationMethod.MEAN:
            return statistics.mean(values)

        if method == AggregationMethod.WEIGHTED_MEAN:
            return self._weighted_mean(prices)

        if method == AggregationMethod.MIN:
            return min(values)

        if method == AggregationMethod.MAX:
            return max(values)

        if method == AggregationMethod.TRIMMED_MEAN:
            return self._trimmed_mean(values)

        # Default to median
        return statistics.median(values)

    def _weighted_mean(self, prices: dict[str, float]) -> float:
        """Calculate weighted mean based on provider weights."""
        total_weight = 0.0
        weighted_sum = 0.0

        for provider, price in prices.items():
            # Find weight for provider
            weight_config = next(
                (w for w in self._agg_config.weights if w.provider_name == provider),
                None,
            )
            weight = weight_config.weight if weight_config else 1.0

            weighted_sum += price * weight
            total_weight += weight

        if total_weight == 0:
            return statistics.mean(prices.values())

        return weighted_sum / total_weight

    def _trimmed_mean(self, values: list[float]) -> float:
        """Calculate trimmed mean, excluding extreme values."""
        if len(values) < 3:
            return statistics.mean(values)

        sorted_values = sorted(values)
        trim_count = max(1, int(len(values) * self._agg_config.trimmed_mean_pct))

        trimmed = (
            sorted_values[trim_count:-trim_count]
            if trim_count < len(values) // 2
            else sorted_values
        )
        return statistics.mean(trimmed) if trimmed else statistics.mean(values)

    def _calculate_confidence(self, prices: dict[str, float], outliers: set[str]) -> float:
        """
        Calculate confidence score for aggregated price.

        Factors:
        - Number of providers (more = higher confidence)
        - Price agreement (lower spread = higher confidence)
        - Provider reliability (based on historical stats)
        """
        if not prices:
            return 0.0

        # Base confidence from provider count
        provider_factor = min(len(prices) / 5.0, 1.0)  # Max at 5 providers

        # Price agreement factor
        values = list(prices.values())
        if len(values) > 1:
            mean = statistics.mean(values)
            max_deviation = max(abs(v - mean) / mean for v in values) if mean > 0 else 0
            agreement_factor = max(0.0, 1.0 - max_deviation * 10)
        else:
            agreement_factor = 0.5  # Single provider = medium confidence

        # Outlier penalty
        outlier_penalty = len(outliers) * 0.1

        # Provider reliability factor
        reliability_sum = 0.0
        for provider in prices:
            stats = self._stats.get(provider)
            if stats:
                reliability_sum += stats.success_rate

        reliability_factor = reliability_sum / len(prices) if prices else 0.5

        # Combine factors
        confidence = (
            provider_factor * 0.3
            + agreement_factor * 0.4
            + reliability_factor * 0.3
            - outlier_penalty
        )

        return max(0.0, min(1.0, confidence))

    def _quote_to_dict(self, quote: AggregatedQuote) -> dict[str, Any]:
        """Convert aggregated quote to dictionary."""
        return {
            "symbol": quote.symbol,
            "price": quote.price,
            "bid": quote.bid,
            "ask": quote.ask,
            "volume": quote.volume,
            "timestamp": quote.timestamp.isoformat(),
            "aggregation": {
                "method": quote.method.value,
                "provider_count": quote.provider_count,
                "providers_used": quote.providers_used,
                "outliers_excluded": quote.outliers_excluded,
                "confidence": quote.confidence,
            },
            "spread_pct": quote.spread_pct,
        }
