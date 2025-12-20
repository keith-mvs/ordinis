"""
Tests for SignalCore NewsContextHook.

Tests news fetching, sentiment analysis, and signal blocking.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from ordinis.engines.base.hooks import Decision, PreflightContext
from ordinis.engines.signalcore.hooks.news import (
    NewsContextHook,
    NewsItem,
    NewsSentiment,
)


class TestNewsSentiment:
    """Tests for NewsSentiment enum."""

    def test_sentiment_values(self):
        """Test sentiment enum values."""
        assert NewsSentiment.VERY_NEGATIVE.value == -2
        assert NewsSentiment.NEGATIVE.value == -1
        assert NewsSentiment.NEUTRAL.value == 0
        assert NewsSentiment.POSITIVE.value == 1
        assert NewsSentiment.VERY_POSITIVE.value == 2


class TestNewsItem:
    """Tests for NewsItem dataclass."""

    def test_news_item_creation(self):
        """Test creating a news item."""
        item = NewsItem(
            headline="Company reports earnings",
            source="Reuters",
            published_at=datetime.now(),
            sentiment=NewsSentiment.POSITIVE,
            category="earnings",
            relevance_score=0.85,
        )

        assert item.headline == "Company reports earnings"
        assert item.sentiment == NewsSentiment.POSITIVE
        assert item.category == "earnings"
        assert 0.0 <= item.relevance_score <= 1.0


class TestNewsContextHook:
    """Tests for NewsContextHook."""

    @pytest.fixture
    def mock_news_fetcher(self):
        """Create mock news fetcher."""

        async def fetcher(symbol: str, lookback_hours: int):
            return [
                NewsItem(
                    headline=f"News about {symbol}",
                    source="Test",
                    published_at=datetime.now(),
                    sentiment=NewsSentiment.NEUTRAL,
                    category="general",
                    relevance_score=0.7,
                )
            ]

        return fetcher

    @pytest.fixture
    def hook(self, mock_news_fetcher):
        """Create NewsContextHook with mock fetcher."""
        return NewsContextHook(
            news_fetcher=mock_news_fetcher,
            cache_ttl_seconds=300,
            blocking_categories=["bankruptcy", "fraud"],
            min_blocking_relevance=0.8,
        )

    @pytest.mark.asyncio
    async def test_fetches_and_caches_news(self, hook):
        """Test hook fetches and caches news."""
        context = PreflightContext(
            engine="signalcore",
            action="generate_signal",
            inputs={"symbol": "AAPL"},
        )

        result = await hook.preflight(context)
        assert result.decision == Decision.ALLOW

        # Check cache
        assert "AAPL" in hook._cache
        assert len(hook._cache["AAPL"]["news"]) == 1

    @pytest.mark.asyncio
    async def test_uses_cached_news(self, hook):
        """Test hook uses cached news within TTL."""
        # Pre-populate cache
        hook._cache["AAPL"] = {
            "news": [
                NewsItem(
                    headline="Cached news",
                    source="Cache",
                    published_at=datetime.now(),
                    sentiment=NewsSentiment.POSITIVE,
                    category="general",
                    relevance_score=0.5,
                )
            ],
            "timestamp": datetime.now(),
        }

        context = PreflightContext(
            engine="signalcore",
            action="generate_signal",
            inputs={"symbol": "AAPL"},
        )

        # Replace fetcher to track calls
        call_count = 0

        async def counting_fetcher(symbol, lookback):
            nonlocal call_count
            call_count += 1
            return []

        hook._news_fetcher = counting_fetcher

        await hook.preflight(context)
        assert call_count == 0  # Used cache, didn't call fetcher

    @pytest.mark.asyncio
    async def test_blocks_on_very_negative_blocking_category(self, hook):
        """Test hook blocks on very negative news in blocking category."""

        async def negative_fetcher(symbol, lookback):
            return [
                NewsItem(
                    headline="Company files for bankruptcy",
                    source="Reuters",
                    published_at=datetime.now(),
                    sentiment=NewsSentiment.VERY_NEGATIVE,
                    category="bankruptcy",
                    relevance_score=0.95,
                )
            ]

        hook._news_fetcher = negative_fetcher

        context = PreflightContext(
            engine="signalcore",
            action="generate_signal",
            inputs={"symbol": "AAPL"},
        )

        result = await hook.preflight(context)
        assert result.decision == Decision.DENY
        assert "bankruptcy" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_allows_negative_non_blocking_category(self, hook):
        """Test hook allows negative news if not in blocking category."""

        async def negative_fetcher(symbol, lookback):
            return [
                NewsItem(
                    headline="Company misses earnings",
                    source="Reuters",
                    published_at=datetime.now(),
                    sentiment=NewsSentiment.VERY_NEGATIVE,
                    category="earnings",  # Not in blocking list
                    relevance_score=0.95,
                )
            ]

        hook._news_fetcher = negative_fetcher

        context = PreflightContext(
            engine="signalcore",
            action="generate_signal",
            inputs={"symbol": "AAPL"},
        )

        result = await hook.preflight(context)
        assert result.decision == Decision.ALLOW

    @pytest.mark.asyncio
    async def test_adds_news_context_to_result(self, hook):
        """Test hook adds news context to result adjustments."""
        context = PreflightContext(
            engine="signalcore",
            action="generate_signal",
            inputs={"symbol": "AAPL"},
        )

        result = await hook.preflight(context)
        assert "news_context" in result.adjustments
        assert isinstance(result.adjustments["news_context"], list)

    @pytest.mark.asyncio
    async def test_handles_missing_symbol(self, hook):
        """Test hook handles missing symbol gracefully."""
        context = PreflightContext(
            engine="signalcore",
            action="generate_signal",
            inputs={},  # No symbol
        )

        result = await hook.preflight(context)
        assert result.decision == Decision.ALLOW

    @pytest.mark.asyncio
    async def test_handles_fetcher_error(self, hook):
        """Test hook handles news fetcher errors gracefully."""

        async def failing_fetcher(symbol, lookback):
            raise ConnectionError("API unavailable")

        hook._news_fetcher = failing_fetcher

        context = PreflightContext(
            engine="signalcore",
            action="generate_signal",
            inputs={"symbol": "AAPL"},
        )

        result = await hook.preflight(context)
        # Should allow with warning, not crash
        assert result.decision in (Decision.ALLOW, Decision.WARN)

    @pytest.mark.asyncio
    async def test_respects_relevance_threshold(self, hook):
        """Test hook respects minimum relevance for blocking."""

        async def low_relevance_fetcher(symbol, lookback):
            return [
                NewsItem(
                    headline="Rumor about bankruptcy",
                    source="Blog",
                    published_at=datetime.now(),
                    sentiment=NewsSentiment.VERY_NEGATIVE,
                    category="bankruptcy",
                    relevance_score=0.5,  # Below 0.8 threshold
                )
            ]

        hook._news_fetcher = low_relevance_fetcher

        context = PreflightContext(
            engine="signalcore",
            action="generate_signal",
            inputs={"symbol": "AAPL"},
        )

        result = await hook.preflight(context)
        # Low relevance shouldn't block
        assert result.decision == Decision.ALLOW

    def test_clear_cache(self, hook):
        """Test clearing the news cache."""
        hook._cache["AAPL"] = {"news": [], "timestamp": datetime.now()}
        hook._cache["MSFT"] = {"news": [], "timestamp": datetime.now()}

        hook.clear_cache()

        assert len(hook._cache) == 0

    def test_clear_cache_for_symbol(self, hook):
        """Test clearing cache for specific symbol."""
        hook._cache["AAPL"] = {"news": [], "timestamp": datetime.now()}
        hook._cache["MSFT"] = {"news": [], "timestamp": datetime.now()}

        hook.clear_cache("AAPL")

        assert "AAPL" not in hook._cache
        assert "MSFT" in hook._cache
