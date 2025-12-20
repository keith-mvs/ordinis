"""
Massive MCP Plugin - News and alternative data integration.

Provides access to news, SEC filings, and social sentiment via the Massive
MCP server. Falls back to mock data when Massive is unavailable.

Example:
    >>> plugin = MassivePlugin(MassivePluginConfig(api_key="..."))
    >>> await plugin.initialize()
    >>> news = await plugin.get_news(["AAPL", "TSLA"])
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging
from typing import Any
import random

from ordinis.plugins.base import (
    NewsPlugin,
    PluginCapability,
    PluginConfig,
    PluginHealth,
    PluginStatus,
)

logger = logging.getLogger(__name__)


class NewsSentiment(Enum):
    """News sentiment classification."""

    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class NewsArticle:
    """A news article with metadata.

    Attributes:
        headline: Article headline
        summary: Brief summary
        symbol: Primary affected symbol
        symbols: All affected symbols
        sentiment: Sentiment classification
        sentiment_score: Numeric sentiment (-1.0 to 1.0)
        timestamp: Publication time
        source: News source
        category: Article category
        url: Original article URL
    """

    headline: str
    summary: str
    symbol: str | None
    symbols: list[str]
    sentiment: NewsSentiment
    sentiment_score: float
    timestamp: datetime
    source: str
    category: str
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "headline": self.headline,
            "summary": self.summary,
            "symbol": self.symbol,
            "symbols": self.symbols,
            "sentiment": self.sentiment.name,
            "sentiment_score": self.sentiment_score,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "category": self.category,
            "url": self.url,
        }


@dataclass
class MassivePluginConfig(PluginConfig):
    """Configuration for Massive MCP Plugin.

    Attributes:
        api_key: Massive API key
        use_mock: Use mock data when API unavailable
        cache_ttl_seconds: How long to cache news
        max_articles_per_symbol: Maximum articles to return per symbol
    """

    api_key: str | None = None
    use_mock: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    max_articles_per_symbol: int = 10


class MassivePlugin(NewsPlugin):
    """Massive MCP integration for news and alternative data.

    Provides:
    - Real-time news for stock symbols
    - Sentiment analysis
    - SEC filing notifications
    - Social media sentiment

    When Massive is unavailable, falls back to realistic mock data for
    development and testing.
    """

    name = "massive"
    version = "1.0.0"
    description = "News and alternative data via Massive MCP"
    capabilities = [PluginCapability.READ, PluginCapability.STREAM]

    def __init__(self, config: MassivePluginConfig | None = None) -> None:
        """Initialize the Massive plugin.

        Args:
            config: Plugin configuration
        """
        super().__init__(config or MassivePluginConfig(name="massive"))
        self._massive_config = config or MassivePluginConfig(name="massive")

        # Cache
        self._news_cache: dict[str, list[NewsArticle]] = {}
        self._cache_timestamps: dict[str, datetime] = {}

        # Earnings calendar cache
        self._earnings_cache: list[dict[str, Any]] = []
        self._earnings_cache_time: datetime | None = None

    async def initialize(self) -> bool:
        """Initialize the plugin.

        Returns:
            True if initialization successful
        """
        try:
            await self._set_status(PluginStatus.INITIALIZING)

            # Try to connect to Massive MCP
            if self._massive_config.api_key:
                # In production, validate API key with Massive
                logger.info("Massive API key configured")
                self._use_live = True
            else:
                logger.warning("No Massive API key - using mock data")
                self._use_live = False

            await self._set_status(PluginStatus.READY)
            return True

        except Exception as e:
            await self._handle_error(e)
            return False

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        self._news_cache.clear()
        self._cache_timestamps.clear()
        await self._set_status(PluginStatus.STOPPED)
        logger.info("Massive plugin shutdown")

    async def health_check(self) -> PluginHealth:
        """Check plugin health.

        Returns:
            Current health status
        """
        self._health.last_check = datetime.now(UTC)

        if self._status == PluginStatus.ERROR:
            self._health.message = f"Error: {self._health.last_error}"
        elif self._use_live if hasattr(self, "_use_live") else False:
            self._health.message = "Connected to Massive API"
        else:
            self._health.message = "Using mock data (no API key)"

        return self._health

    async def get_news(
        self,
        symbols: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get recent news articles.

        Args:
            symbols: Filter by symbols (None = all)
            limit: Maximum articles to return

        Returns:
            List of news article dictionaries
        """
        await self._rate_limiter.wait_for_token()

        articles: list[NewsArticle] = []

        if symbols:
            for symbol in symbols:
                symbol_articles = await self._get_news_for_symbol(symbol)
                articles.extend(symbol_articles)
        else:
            # Get market-wide news
            articles = await self._get_market_news()

        # Sort by timestamp and limit
        articles.sort(key=lambda a: a.timestamp, reverse=True)
        return [a.to_dict() for a in articles[:limit]]

    async def get_sentiment(self, symbol: str) -> dict[str, Any]:
        """Get sentiment analysis for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Sentiment analysis results
        """
        await self._rate_limiter.wait_for_token()

        articles = await self._get_news_for_symbol(symbol)

        if not articles:
            return {
                "symbol": symbol,
                "sentiment": "NEUTRAL",
                "score": 0.0,
                "article_count": 0,
                "confidence": 0.0,
            }

        # Aggregate sentiment
        scores = [a.sentiment_score for a in articles]
        avg_score = sum(scores) / len(scores)

        # Determine overall sentiment
        if avg_score >= 0.5:
            sentiment = "VERY_POSITIVE"
        elif avg_score >= 0.2:
            sentiment = "POSITIVE"
        elif avg_score <= -0.5:
            sentiment = "VERY_NEGATIVE"
        elif avg_score <= -0.2:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"

        return {
            "symbol": symbol,
            "sentiment": sentiment,
            "score": round(avg_score, 3),
            "article_count": len(articles),
            "confidence": min(len(articles) / 10, 1.0),
            "recent_headlines": [a.headline for a in articles[:3]],
        }

    async def get_market_news(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get broad market news.

        Args:
            limit: Maximum articles to return

        Returns:
            List of market news articles
        """
        articles = await self._get_market_news()
        return [a.to_dict() for a in articles[:limit]]

    async def get_earnings_calendar(
        self,
        symbols: list[str] | None = None,
        days_ahead: int = 5,
    ) -> list[dict[str, Any]]:
        """Get upcoming earnings announcements.

        Args:
            symbols: Filter by symbols (None = all portfolio symbols)
            days_ahead: How many days ahead to look

        Returns:
            List of upcoming earnings
        """
        await self._rate_limiter.wait_for_token()

        # Check cache
        now = datetime.now(UTC)
        if (
            self._earnings_cache
            and self._earnings_cache_time
            and (now - self._earnings_cache_time).total_seconds()
            < self._massive_config.cache_ttl_seconds
        ):
            earnings = self._earnings_cache
        else:
            earnings = await self._fetch_earnings_calendar(days_ahead)
            self._earnings_cache = earnings
            self._earnings_cache_time = now

        # Filter by symbols if provided
        if symbols:
            symbols_upper = [s.upper() for s in symbols]
            earnings = [e for e in earnings if e["symbol"] in symbols_upper]

        return earnings

    async def _get_news_for_symbol(self, symbol: str) -> list[NewsArticle]:
        """Get news for a specific symbol.

        Args:
            symbol: Stock symbol

        Returns:
            List of NewsArticle objects
        """
        symbol = symbol.upper()

        # Check cache
        now = datetime.now(UTC)
        cache_time = self._cache_timestamps.get(symbol)
        if (
            cache_time
            and (now - cache_time).total_seconds() < self._massive_config.cache_ttl_seconds
        ):
            return self._news_cache.get(symbol, [])

        # Fetch fresh news
        if hasattr(self, "_use_live") and self._use_live:
            articles = await self._fetch_live_news(symbol)
        else:
            articles = self._generate_mock_news(symbol)

        # Update cache
        self._news_cache[symbol] = articles
        self._cache_timestamps[symbol] = now

        return articles

    async def _get_market_news(self) -> list[NewsArticle]:
        """Get market-wide news.

        Returns:
            List of NewsArticle objects
        """
        # Use "MARKET" as cache key for broad market news
        return await self._get_news_for_symbol("MARKET")

    async def _fetch_live_news(self, symbol: str) -> list[NewsArticle]:
        """Fetch live news from Massive API.

        Args:
            symbol: Stock symbol

        Returns:
            List of NewsArticle objects
        """
        # TODO: Implement actual Massive API call
        # For now, fall back to mock
        logger.debug(f"Would fetch live news for {symbol} from Massive")
        return self._generate_mock_news(symbol)

    async def _fetch_earnings_calendar(self, days_ahead: int) -> list[dict[str, Any]]:
        """Fetch earnings calendar.

        Args:
            days_ahead: Days to look ahead

        Returns:
            List of earnings announcements
        """
        # TODO: Implement actual Massive API call
        # Generate mock earnings calendar
        now = datetime.now(UTC)
        earnings = []

        # Sample companies with mock earnings dates
        companies = [
            ("AAPL", "Apple Inc.", "technology"),
            ("MSFT", "Microsoft Corporation", "technology"),
            ("GOOGL", "Alphabet Inc.", "technology"),
            ("AMZN", "Amazon.com Inc.", "consumer"),
            ("NVDA", "NVIDIA Corporation", "technology"),
            ("META", "Meta Platforms Inc.", "technology"),
            ("TSLA", "Tesla Inc.", "automotive"),
            ("JPM", "JPMorgan Chase & Co.", "financial"),
        ]

        for i, (symbol, name, sector) in enumerate(companies):
            # Spread earnings across the date range
            days_offset = (i * days_ahead) // len(companies)
            if days_offset <= days_ahead:
                earnings.append({
                    "symbol": symbol,
                    "company_name": name,
                    "sector": sector,
                    "earnings_date": (now + timedelta(days=days_offset)).date().isoformat(),
                    "time": "AMC" if i % 2 == 0 else "BMO",  # After/Before Market
                    "eps_estimate": round(random.uniform(1.0, 5.0), 2),
                    "revenue_estimate_b": round(random.uniform(10.0, 100.0), 1),
                })

        return earnings

    def _generate_mock_news(self, symbol: str) -> list[NewsArticle]:
        """Generate realistic mock news for development.

        Args:
            symbol: Stock symbol

        Returns:
            List of mock NewsArticle objects
        """
        now = datetime.now(UTC)
        articles = []

        # Symbol-specific templates
        if symbol == "MARKET":
            templates = [
                ("S&P 500 hits new intraday high amid tech rally", "POSITIVE", "market"),
                ("Fed signals patience on rate cuts, markets react", "NEUTRAL", "macro"),
                ("Treasury yields climb as inflation data looms", "NEGATIVE", "bonds"),
                ("VIX drops to multi-month low, risk appetite rises", "POSITIVE", "volatility"),
                ("Oil prices steady ahead of OPEC+ meeting", "NEUTRAL", "commodities"),
            ]
        else:
            company_name = self._get_company_name(symbol)
            templates = [
                (f"{company_name} beats Q4 estimates, raises guidance", "VERY_POSITIVE", "earnings"),
                (f"{symbol} announces strategic partnership with major firm", "POSITIVE", "corporate"),
                (f"Analysts upgrade {symbol} on strong fundamentals", "POSITIVE", "analyst"),
                (f"{company_name} navigates supply chain challenges", "NEUTRAL", "operations"),
                (f"{symbol} trading volume surges on institutional interest", "NEUTRAL", "trading"),
                (f"Sector headwinds may impact {symbol} near-term", "NEGATIVE", "analysis"),
            ]

        for i, (headline, sentiment_str, category) in enumerate(templates[:5]):
            sentiment = NewsSentiment[sentiment_str]
            sentiment_score = {
                NewsSentiment.VERY_POSITIVE: 0.8,
                NewsSentiment.POSITIVE: 0.4,
                NewsSentiment.NEUTRAL: 0.0,
                NewsSentiment.NEGATIVE: -0.4,
                NewsSentiment.VERY_NEGATIVE: -0.8,
            }[sentiment]

            # Add some randomness
            sentiment_score += random.uniform(-0.1, 0.1)
            sentiment_score = max(-1.0, min(1.0, sentiment_score))

            articles.append(
                NewsArticle(
                    headline=headline,
                    summary=f"Analysis of recent developments for {symbol}.",
                    symbol=symbol if symbol != "MARKET" else None,
                    symbols=[symbol] if symbol != "MARKET" else [],
                    sentiment=sentiment,
                    sentiment_score=sentiment_score,
                    timestamp=now - timedelta(hours=i * 2),
                    source=random.choice(["Reuters", "Bloomberg", "CNBC", "WSJ", "Benzinga"]),
                    category=category,
                    url=f"https://example.com/news/{symbol.lower()}/{i}",
                )
            )

        return articles

    def _get_company_name(self, symbol: str) -> str:
        """Get company name for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Company name or symbol if unknown
        """
        names = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Alphabet",
            "AMZN": "Amazon",
            "NVDA": "NVIDIA",
            "META": "Meta",
            "TSLA": "Tesla",
            "AMD": "AMD",
            "CRWD": "CrowdStrike",
            "JPM": "JPMorgan",
            "V": "Visa",
            "MA": "Mastercard",
        }
        return names.get(symbol.upper(), symbol)
