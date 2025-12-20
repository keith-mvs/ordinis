"""
News context governance hook for SignalCore.

Injects news context before signal generation and can block signals
based on significant negative news events.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging
from typing import Callable

from ordinis.engines.base.hooks import (
    BaseGovernanceHook,
    Decision,
    PreflightContext,
    PreflightResult,
)
from ordinis.engines.base.models import AuditRecord, EngineError

_logger = logging.getLogger(__name__)


class NewsSentiment(Enum):
    """News sentiment classification."""

    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class NewsItem:
    """A news item with metadata.

    Attributes:
        headline: News headline
        symbol: Affected symbol (or None for market-wide)
        sentiment: Sentiment classification
        timestamp: When the news was published
        source: News source
        category: News category (earnings, fda, lawsuit, etc.)
    """

    headline: str
    symbol: str | None
    sentiment: NewsSentiment
    timestamp: datetime
    source: str = "unknown"
    category: str = "general"


class NewsContextHook(BaseGovernanceHook):
    """Injects news context into signal generation.

    This hook:
    1. Fetches relevant news before signal generation
    2. Can block signals on major negative news (e.g., bad earnings)
    3. Injects sentiment context for signal confidence adjustment

    Example:
        >>> async def fetch_news(symbol: str) -> list[NewsItem]:
        ...     # Fetch from news API
        ...     return [NewsItem(...)]
        >>>
        >>> hook = NewsContextHook(
        ...     news_fetcher=fetch_news,
        ...     block_on_very_negative=True,
        ... )
    """

    def __init__(
        self,
        news_fetcher: Callable[[str], list[NewsItem]] | None = None,
        block_on_very_negative: bool = True,
        news_lookback_hours: int = 24,
        blocking_categories: list[str] | None = None,
    ) -> None:
        """Initialize NewsContextHook.

        Args:
            news_fetcher: Async function to fetch news for a symbol
            block_on_very_negative: Block signals on VERY_NEGATIVE news
            news_lookback_hours: How far back to look for news
            blocking_categories: News categories that can block signals
        """
        super().__init__("signalcore")
        self._news_fetcher = news_fetcher
        self._block_on_very_negative = block_on_very_negative
        self._lookback = timedelta(hours=news_lookback_hours)
        self._blocking_categories = blocking_categories or [
            "earnings_miss",
            "fda_rejection",
            "lawsuit",
            "sec_investigation",
            "bankruptcy",
            "ceo_departure",
        ]
        # Cache recent news
        self._news_cache: dict[str, list[NewsItem]] = {}
        self._cache_expiry: dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=15)
        # Current context for signal generation
        self._current_context: str = ""

    @property
    def current_context(self) -> str:
        """Get the current news context string."""
        return self._current_context

    def set_context(self, context: str) -> None:
        """Manually set market context.

        Args:
            context: Context string to inject
        """
        self._current_context = context
        _logger.info("NewsContextHook: Context set: %s", context[:100])

    def _get_cached_news(self, symbol: str) -> list[NewsItem] | None:
        """Get cached news if still valid.

        Args:
            symbol: Symbol to look up

        Returns:
            Cached news items or None if expired
        """
        if symbol in self._news_cache:
            expiry = self._cache_expiry.get(symbol, datetime.min.replace(tzinfo=UTC))
            if datetime.now(UTC) < expiry:
                return self._news_cache[symbol]
        return None

    def _cache_news(self, symbol: str, news: list[NewsItem]) -> None:
        """Cache news items.

        Args:
            symbol: Symbol
            news: News items to cache
        """
        self._news_cache[symbol] = news
        self._cache_expiry[symbol] = datetime.now(UTC) + self._cache_ttl

    async def _fetch_news(self, symbol: str) -> list[NewsItem]:
        """Fetch news for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            List of recent news items
        """
        # Check cache first
        cached = self._get_cached_news(symbol)
        if cached is not None:
            return cached

        # Fetch fresh news
        if self._news_fetcher:
            try:
                news = self._news_fetcher(symbol)
                self._cache_news(symbol, news)
                return news
            except Exception as e:
                _logger.warning("Failed to fetch news for %s: %s", symbol, e)

        return []

    def _analyze_sentiment(self, news: list[NewsItem]) -> tuple[NewsSentiment, list[str]]:
        """Analyze overall sentiment from news items.

        Args:
            news: List of news items

        Returns:
            Tuple of (overall_sentiment, blocking_reasons)
        """
        if not news:
            return NewsSentiment.NEUTRAL, []

        blocking_reasons: list[str] = []
        sentiment_scores = [n.sentiment.value for n in news]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

        # Check for blocking categories
        for item in news:
            if item.category in self._blocking_categories:
                if item.sentiment in (NewsSentiment.NEGATIVE, NewsSentiment.VERY_NEGATIVE):
                    blocking_reasons.append(f"{item.category}: {item.headline[:50]}...")

        # Determine overall sentiment
        if avg_sentiment <= -1.5:
            return NewsSentiment.VERY_NEGATIVE, blocking_reasons
        if avg_sentiment <= -0.5:
            return NewsSentiment.NEGATIVE, blocking_reasons
        if avg_sentiment >= 1.5:
            return NewsSentiment.VERY_POSITIVE, blocking_reasons
        if avg_sentiment >= 0.5:
            return NewsSentiment.POSITIVE, blocking_reasons
        return NewsSentiment.NEUTRAL, blocking_reasons

    async def preflight(self, context: PreflightContext) -> PreflightResult:
        """Check news before signal generation.

        Args:
            context: Preflight context with symbol and action

        Returns:
            PreflightResult with decision and news context
        """
        # Only check for signal generation actions
        if context.action not in ["generate_signal", "generate_batch", "evaluate_signal"]:
            return PreflightResult(decision=Decision.ALLOW)

        symbol = context.inputs.get("symbol", "")
        direction = context.inputs.get("direction", "long")

        if not symbol:
            return PreflightResult(decision=Decision.ALLOW)

        # Fetch and analyze news
        news = await self._fetch_news(symbol)
        sentiment, blocking_reasons = self._analyze_sentiment(news)

        # Build context string
        if news:
            headlines = [n.headline for n in news[:3]]
            self._current_context = f"Recent news for {symbol}: {'; '.join(headlines)}"
        else:
            self._current_context = f"No recent news for {symbol}"

        # Check for blocking conditions
        if self._block_on_very_negative and sentiment == NewsSentiment.VERY_NEGATIVE:
            if blocking_reasons:
                return PreflightResult(
                    decision=Decision.DENY,
                    reason=f"Blocking signal due to negative news: {blocking_reasons[0]}",
                    policy_id="news_context_hook",
                    policy_version=self.policy_version,
                )

        # For long signals, warn on negative sentiment
        if direction == "long" and sentiment in (
            NewsSentiment.NEGATIVE,
            NewsSentiment.VERY_NEGATIVE,
        ):
            return PreflightResult(
                decision=Decision.WARN,
                reason=f"Negative news sentiment for {symbol}",
                policy_id="news_context_hook",
                policy_version=self.policy_version,
                warnings=[f"News sentiment: {sentiment.name}"],
                adjustments={"news_context": self._current_context},
            )

        # For short signals, warn on positive sentiment
        if direction == "short" and sentiment in (
            NewsSentiment.POSITIVE,
            NewsSentiment.VERY_POSITIVE,
        ):
            return PreflightResult(
                decision=Decision.WARN,
                reason=f"Positive news sentiment for {symbol}",
                policy_id="news_context_hook",
                policy_version=self.policy_version,
                warnings=[f"News sentiment: {sentiment.name}"],
                adjustments={"news_context": self._current_context},
            )

        return PreflightResult(
            decision=Decision.ALLOW,
            reason=f"News sentiment: {sentiment.name}",
            policy_id="news_context_hook",
            policy_version=self.policy_version,
            adjustments={"news_context": self._current_context},
        )

    async def audit(self, record: AuditRecord) -> None:
        """Log news context audit events.

        Args:
            record: Audit record
        """
        if record.action in ["generate_signal", "generate_batch"]:
            _logger.debug(
                "NewsContextHook audit: symbol=%s context=%s",
                record.details.get("symbol", "unknown"),
                self._current_context[:50] if self._current_context else "none",
            )

    async def on_error(self, error: EngineError) -> None:
        """Handle errors.

        Args:
            error: Engine error
        """
        _logger.warning(
            "NewsContextHook error: %s - %s",
            error.code,
            error.message,
        )
