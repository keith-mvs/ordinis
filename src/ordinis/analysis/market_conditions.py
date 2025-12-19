"""
Market conditions analysis using plugin-based real-time data sources.

This module provides comprehensive market analysis including:
- Index snapshots (SPY, QQQ, IWM, DIA)
- Volatility regime classification (VIX)
- Sector rotation analysis
- Market breadth indicators
- Regime classification
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from typing import Any

from ordinis.adapters.market_data.iex import IEXDataPlugin
from ordinis.adapters.market_data.massive import MassiveDataPlugin

logger = logging.getLogger(__name__)


class MarketDataError(Exception):
    """Base exception for market data errors."""


class NoDataSourceError(MarketDataError):
    """Raised when no data sources are configured."""


class AllDataSourcesFailedError(MarketDataError):
    """Raised when all configured data sources have failed."""


class MarketRegime(str, Enum):
    """Market regime classifications."""

    RISK_ON = "Risk-On"
    RISK_OFF = "Risk-Off"
    CHOPPY = "Choppy/Uncertain"
    TRENDING_UP = "Trending Up"
    TRENDING_DOWN = "Trending Down"


class VolatilityRegime(str, Enum):
    """VIX-based volatility classifications."""

    LOW = "Low"  # VIX < 15
    NORMAL = "Normal"  # 15-20
    ELEVATED = "Elevated"  # 20-25
    HIGH = "High"  # 25-35
    EXTREME = "Extreme"  # > 35


@dataclass
class IndexSnapshot:
    """Snapshot of a market index."""

    symbol: str
    price: float
    change: float
    change_pct: float
    volume: int
    high_52w: float
    low_52w: float
    timestamp: datetime


@dataclass
class VolatilityMetrics:
    """Volatility metrics and classification."""

    vix_current: float
    vix_change: float
    vix_change_pct: float
    vix_high_52w: float
    vix_low_52w: float
    regime: VolatilityRegime
    trend: str  # "Rising", "Falling", "Stable"


@dataclass
class SectorPerformance:
    """Sector ETF performance data."""

    symbol: str
    sector_name: str
    price: float
    change_pct_daily: float
    change_pct_weekly: float | None
    change_pct_monthly: float | None


@dataclass
class BreadthIndicators:
    """Market breadth indicators."""

    advance_decline_ratio: float | None
    pct_above_50day_ma: float | None
    pct_above_200day_ma: float | None
    new_highs: int | None
    new_lows: int | None
    breadth_sentiment: str  # "Bullish", "Bearish", "Neutral", "Unknown"


@dataclass
class MarketConditionsReport:
    """Complete market conditions report."""

    timestamp: datetime
    indices: dict[str, IndexSnapshot]
    volatility: VolatilityMetrics
    sectors: list[SectorPerformance]
    breadth: BreadthIndicators
    regime: MarketRegime
    regime_rationale: str
    trading_implications: dict[str, Any]


class MarketConditionsAnalyzer:
    """
    Analyze current market conditions using real-time plugin data.

    Uses Massive as primary data source with IEX Cloud fallback.
    Implements caching to minimize API calls.
    """

    # Major market indices
    INDICES = ["SPY", "QQQ", "IWM", "DIA"]

    # Sector ETFs with names
    SECTORS = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLI": "Industrials",
        "XLB": "Materials",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
    }

    def __init__(
        self,
        massive_plugin: MassiveDataPlugin | None = None,
        iex_plugin: IEXDataPlugin | None = None,
        use_cache: bool = True,
        cache_ttl_seconds: int = 900,  # 15 minutes
    ):
        """
        Initialize the analyzer.

        Args:
            massive_plugin: Massive data plugin (primary source)
            iex_plugin: IEX Cloud data plugin (fallback)
            use_cache: Whether to cache results
            cache_ttl_seconds: Cache time-to-live in seconds
        """
        self.massive = massive_plugin
        self.iex = iex_plugin
        self.cache: dict[str, tuple[datetime, Any]] = {} if use_cache else {}
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)

        if not massive_plugin and not iex_plugin:
            raise ValueError("At least one data plugin must be provided")

    def _get_cached(self, key: str) -> Any | None:
        """Get cached value if not expired."""
        if not self.cache or key not in self.cache:
            return None

        cached_time, cached_value = self.cache[key]
        if datetime.utcnow() - cached_time < self.cache_ttl:
            logger.debug(f"Cache hit for {key}")
            return cached_value

        # Expired
        del self.cache[key]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Set cached value with timestamp."""
        if self.cache is not None:
            self.cache[key] = (datetime.utcnow(), value)

    async def get_market_overview(self) -> dict[str, IndexSnapshot]:
        """
        Get current snapshot of major market indices.

        Returns:
            Dictionary mapping symbol to IndexSnapshot

        Raises:
            Exception: If all data sources fail
        """
        cache_key = "market_overview"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        symbols = self.INDICES
        snapshots = {}

        # Try Massive first
        if self.massive:
            try:
                logger.info(f"Fetching market overview from Massive: {symbols}")
                data = await self._get_snapshots_massive(symbols)
                snapshots = self._parse_index_snapshots(data, "massive")
                self._set_cached(cache_key, snapshots)
                return snapshots
            except Exception as e:
                logger.warning(f"Massive failed for market overview: {e}")

        # Fallback to IEX
        if self.iex:
            try:
                logger.info(f"Fetching market overview from IEX: {symbols}")
                data = await self._get_quotes_iex(symbols)
                snapshots = self._parse_index_snapshots(data, "iex")
                self._set_cached(cache_key, snapshots)
                return snapshots
            except Exception as e:
                logger.error(f"IEX failed for market overview: {e}")
                raise AllDataSourcesFailedError(
                    "All data sources failed for market overview"
                ) from e

        raise NoDataSourceError("No data sources configured")

    async def get_volatility_metrics(self) -> VolatilityMetrics:
        """
        Get VIX volatility metrics and classification.

        Returns:
            VolatilityMetrics with current VIX and regime classification
        """
        cache_key = "volatility_metrics"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # Try Massive first
        if self.massive:
            try:
                logger.info("Fetching VIX data from Massive")
                vix_data = await self._get_vix_massive()
                metrics = self._parse_volatility_metrics(vix_data)
                self._set_cached(cache_key, metrics)
                return metrics
            except Exception as e:
                logger.warning(f"Massive failed for VIX: {e}")

        # Fallback to IEX
        if self.iex:
            try:
                logger.info("Fetching VIX data from IEX")
                vix_data = await self._get_vix_iex()
                metrics = self._parse_volatility_metrics(vix_data)
                self._set_cached(cache_key, metrics)
                return metrics
            except Exception as e:
                logger.error(f"IEX failed for VIX: {e}")
                raise AllDataSourcesFailedError("All data sources failed for VIX") from e

        raise NoDataSourceError("No data sources configured")

    async def get_sector_performance(self) -> list[SectorPerformance]:
        """
        Get sector ETF performance data.

        Returns:
            List of SectorPerformance objects sorted by daily performance
        """
        cache_key = "sector_performance"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        symbols = list(self.SECTORS.keys())
        sector_data = []

        # Try Massive first
        if self.massive:
            try:
                logger.info(f"Fetching sector data from Massive: {symbols}")
                data = await self._get_snapshots_massive(symbols)
                sector_data = self._parse_sector_performance(data, "massive")
                self._set_cached(cache_key, sector_data)
                return sector_data
            except Exception as e:
                logger.warning(f"Massive failed for sectors: {e}")

        # Fallback to IEX
        if self.iex:
            try:
                logger.info(f"Fetching sector data from IEX: {symbols}")
                data = await self._get_quotes_iex(symbols)
                sector_data = self._parse_sector_performance(data, "iex")
                self._set_cached(cache_key, sector_data)
                return sector_data
            except Exception as e:
                logger.error(f"IEX failed for sectors: {e}")
                raise AllDataSourcesFailedError("All data sources failed for sector data") from e

        raise NoDataSourceError("No data sources configured")

    async def get_breadth_indicators(self) -> BreadthIndicators:
        """
        Get market breadth indicators.

        Note: This is a simplified implementation. Full breadth data
        requires additional API endpoints or data sources.

        Returns:
            BreadthIndicators (may have None values if data unavailable)
        """
        # For now, return placeholder with unknown sentiment
        # TODO: Implement full breadth calculation when data source identified
        return BreadthIndicators(
            advance_decline_ratio=None,
            pct_above_50day_ma=None,
            pct_above_200day_ma=None,
            new_highs=None,
            new_lows=None,
            breadth_sentiment="Unknown",
        )

    def classify_regime(
        self,
        volatility: VolatilityMetrics,
        breadth: BreadthIndicators,
        sector_performance: list[SectorPerformance],
    ) -> tuple[MarketRegime, str]:
        """
        Classify market regime based on multiple indicators.

        Args:
            volatility: VIX and volatility metrics
            breadth: Market breadth indicators
            sector_performance: Sector rotation data

        Returns:
            Tuple of (MarketRegime, rationale string)
        """
        # Get top 3 sectors
        leaders = sorted(sector_performance, key=lambda s: s.change_pct_daily, reverse=True)[:3]
        leader_symbols = [s.symbol for s in leaders]

        # Cyclical sectors
        cyclicals = ["XLK", "XLY", "XLF", "XLE", "XLI"]
        # Defensive sectors
        defensives = ["XLV", "XLP", "XLU", "XLRE"]

        cyclical_leadership = sum(1 for s in leader_symbols if s in cyclicals)
        defensive_leadership = sum(1 for s in leader_symbols if s in defensives)

        # Classification logic
        vix = volatility.vix_current
        vix_trend = volatility.trend

        # Risk-On: Low VIX, cyclicals leading
        if (
            vix < 18
            and vix_trend == "Falling"
            and cyclical_leadership >= 2
            and breadth.breadth_sentiment in ("Bullish", "Unknown")
        ):
            return (
                MarketRegime.RISK_ON,
                f"VIX low ({vix:.1f}), falling trend, cyclicals leading ({', '.join(leader_symbols)})",
            )

        # Risk-Off: High VIX, defensives leading
        if (
            vix > 25
            or (vix > 20 and defensive_leadership >= 2)
            or breadth.breadth_sentiment == "Bearish"
        ):
            return (
                MarketRegime.RISK_OFF,
                f"VIX elevated ({vix:.1f}), defensive leadership ({', '.join(leader_symbols)})",
            )

        # Choppy: Mixed signals
        if 18 <= vix <= 25 or (cyclical_leadership == defensive_leadership):
            return (
                MarketRegime.CHOPPY,
                f"VIX in normal range ({vix:.1f}), mixed sector leadership",
            )

        # Default to choppy if unclear
        return (
            MarketRegime.CHOPPY,
            f"Mixed market signals - VIX: {vix:.1f}, unclear sector rotation",
        )

    def generate_trading_implications(
        self, regime: MarketRegime, volatility: VolatilityMetrics
    ) -> dict[str, Any]:
        """
        Generate trading recommendations based on regime.

        Args:
            regime: Classified market regime
            volatility: VIX metrics

        Returns:
            Dictionary with position sizing, strategy bias, and considerations
        """
        implications = {
            "regime": regime,
            "position_sizing": "Normal",
            "strategy_bias": ["Trend Following", "Mean Reversion"],
            "sectors_favor": [],
            "sectors_avoid": [],
            "special_considerations": [],
        }

        if regime == MarketRegime.RISK_ON:
            implications.update(
                {
                    "position_sizing": "Normal to Aggressive",
                    "strategy_bias": ["Trend Following", "Momentum"],
                    "sectors_favor": ["XLK", "XLY", "XLF"],
                    "sectors_avoid": ["XLU", "XLP"],
                    "special_considerations": [
                        "Favor growth and cyclicals",
                        "Use wider stops for trends",
                    ],
                }
            )

        elif regime == MarketRegime.RISK_OFF:
            implications.update(
                {
                    "position_sizing": "Reduced (50-75%)",
                    "strategy_bias": ["Mean Reversion", "Defensive"],
                    "sectors_favor": ["XLV", "XLP", "XLU"],
                    "sectors_avoid": ["XLK", "XLY", "Small Caps"],
                    "special_considerations": [
                        "Tighten stops to 1-1.5%",
                        "Consider hedging with VIX calls",
                        "Reduce max portfolio heat to 3-4%",
                    ],
                }
            )

        elif regime == MarketRegime.CHOPPY:
            implications.update(
                {
                    "position_sizing": "Reduced (50-75%)",
                    "strategy_bias": ["Mean Reversion", "Range Trading"],
                    "sectors_favor": ["Quality names across sectors"],
                    "sectors_avoid": ["Low volume, high beta"],
                    "special_considerations": [
                        "Scale into positions (1/3 at a time)",
                        "Avoid chasing breakouts",
                        "Watch for regime shift signals",
                    ],
                }
            )

        # Add VIX-specific warnings
        special_considerations = list(implications["special_considerations"])
        if volatility.vix_current > 30:
            special_considerations.append(
                f"VIX extreme ({volatility.vix_current:.1f}) - consider defensive positioning"
            )
        elif volatility.vix_current < 15:
            special_considerations.append(
                f"VIX very low ({volatility.vix_current:.1f}) - watch for complacency"
            )
        implications["special_considerations"] = special_considerations

        return implications

    async def analyze(self) -> MarketConditionsReport:
        """
        Perform complete market conditions analysis.

        Returns:
            MarketConditionsReport with all metrics and classifications
        """
        logger.info("Starting market conditions analysis")

        # Gather all data concurrently
        indices_task = asyncio.create_task(self.get_market_overview())
        volatility_task = asyncio.create_task(self.get_volatility_metrics())
        sectors_task = asyncio.create_task(self.get_sector_performance())
        breadth_task = asyncio.create_task(self.get_breadth_indicators())

        indices = await indices_task
        volatility = await volatility_task
        sectors = await sectors_task
        breadth = await breadth_task

        # Classify regime
        regime, rationale = self.classify_regime(volatility, breadth, sectors)

        # Generate implications
        implications = self.generate_trading_implications(regime, volatility)

        report = MarketConditionsReport(
            timestamp=datetime.utcnow(),
            indices=indices,
            volatility=volatility,
            sectors=sorted(sectors, key=lambda s: s.change_pct_daily, reverse=True),
            breadth=breadth,
            regime=regime,
            regime_rationale=rationale,
            trading_implications=implications,
        )

        logger.info(f"Market analysis complete - Regime: {regime}")
        return report

    # -------------------------------------------------------------------------
    # Private helper methods for data fetching
    # -------------------------------------------------------------------------

    async def _get_snapshots_massive(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch snapshot data from Massive."""
        assert self.massive is not None
        # Use existing plugin method
        results = {}
        for symbol in symbols:
            snapshot = await self.massive.get_snapshot(symbol)
            results[symbol] = snapshot
        return results

    async def _get_quotes_iex(self, symbols: list[str]) -> dict[str, Any]:
        """Fetch quotes from IEX."""
        assert self.iex is not None
        # Use existing plugin method
        results = {}
        for symbol in symbols:
            quote = await self.iex.get_quote(symbol)
            results[symbol] = quote
        return results

    async def _get_vix_massive(self) -> dict[str, Any]:
        """Fetch VIX data from Massive."""
        assert self.massive is not None
        snapshot = await self.massive.get_snapshot("VIX")
        prev_close = await self.massive.get_previous_close("VIX")
        return {"snapshot": snapshot, "previous": prev_close}

    async def _get_vix_iex(self) -> dict[str, Any]:
        """Fetch VIX data from IEX."""
        assert self.iex is not None
        quote = await self.iex.get_quote("VIX")
        return {"quote": quote}

    # -------------------------------------------------------------------------
    # Private helper methods for data parsing
    # -------------------------------------------------------------------------

    def _parse_index_snapshots(self, data: dict[str, Any], source: str) -> dict[str, IndexSnapshot]:
        """Parse index data into IndexSnapshot objects."""
        snapshots = {}

        for symbol, raw_data in data.items():
            if source == "massive":
                ticker = raw_data.get("ticker", {})
                snapshots[symbol] = IndexSnapshot(
                    symbol=symbol,
                    price=ticker.get("lastTrade", {}).get("p", 0.0),
                    change=ticker.get("todaysChange", 0.0),
                    change_pct=ticker.get("todaysChangePerc", 0.0),
                    volume=ticker.get("day", {}).get("v", 0),
                    high_52w=ticker.get("prevDay", {}).get("h", 0.0),  # Approximate
                    low_52w=ticker.get("prevDay", {}).get("l", 0.0),  # Approximate
                    timestamp=datetime.utcnow(),
                )
            elif source == "iex":
                snapshots[symbol] = IndexSnapshot(
                    symbol=symbol,
                    price=raw_data.get("latestPrice", 0.0),
                    change=raw_data.get("change", 0.0),
                    change_pct=raw_data.get("changePercent", 0.0) * 100,
                    volume=raw_data.get("latestVolume", 0),
                    high_52w=raw_data.get("week52High", 0.0),
                    low_52w=raw_data.get("week52Low", 0.0),
                    timestamp=datetime.utcnow(),
                )

        return snapshots

    def _parse_volatility_metrics(self, data: dict[str, Any]) -> VolatilityMetrics:
        """Parse VIX data into VolatilityMetrics."""
        # Extract VIX values (structure depends on source)
        if "snapshot" in data:  # Massive
            ticker = data["snapshot"].get("ticker", {})
            current = ticker.get("lastTrade", {}).get("p", 16.0)
            prev = data["previous"].get("c", current)
        else:  # IEX
            quote = data.get("quote", {})
            current = quote.get("latestPrice", 16.0)
            prev = quote.get("previousClose", current)

        change = current - prev
        change_pct = (change / prev * 100) if prev > 0 else 0.0

        # Classify regime
        if current < 15:
            regime = VolatilityRegime.LOW
        elif current < 20:
            regime = VolatilityRegime.NORMAL
        elif current < 25:
            regime = VolatilityRegime.ELEVATED
        elif current < 35:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.EXTREME

        # Determine trend
        if change_pct > 5:
            trend = "Rising"
        elif change_pct < -5:
            trend = "Falling"
        else:
            trend = "Stable"

        return VolatilityMetrics(
            vix_current=current,
            vix_change=change,
            vix_change_pct=change_pct,
            vix_high_52w=60.0,  # TODO: Get from historical data
            vix_low_52w=12.0,  # TODO: Get from historical data
            regime=regime,
            trend=trend,
        )

    def _parse_sector_performance(
        self, data: dict[str, Any], source: str
    ) -> list[SectorPerformance]:
        """Parse sector ETF data into SectorPerformance objects."""
        sectors = []

        for symbol, raw_data in data.items():
            if symbol not in self.SECTORS:
                continue

            if source == "massive":
                ticker = raw_data.get("ticker", {})
                change_pct = ticker.get("todaysChangePerc", 0.0)
                price = ticker.get("lastTrade", {}).get("p", 0.0)
            elif source == "iex":
                change_pct = raw_data.get("changePercent", 0.0) * 100
                price = raw_data.get("latestPrice", 0.0)

            sectors.append(
                SectorPerformance(
                    symbol=symbol,
                    sector_name=self.SECTORS[symbol],
                    price=price,
                    change_pct_daily=change_pct,
                    change_pct_weekly=None,  # TODO: Calculate from historical
                    change_pct_monthly=None,  # TODO: Calculate from historical
                )
            )

        return sectors
