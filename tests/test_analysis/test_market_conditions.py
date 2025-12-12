"""Tests for market conditions analyzer."""

from unittest.mock import AsyncMock

import pytest

from src.analysis.market_conditions import (
    AllDataSourcesFailedError,
    BreadthIndicators,
    IndexSnapshot,
    MarketConditionsAnalyzer,
    MarketRegime,
    SectorPerformance,
    VolatilityMetrics,
    VolatilityRegime,
)


@pytest.fixture
def mock_polygon():
    """Create mock Polygon plugin."""
    plugin = AsyncMock()
    plugin.get_snapshot = AsyncMock()
    plugin.get_previous_close = AsyncMock()
    return plugin


@pytest.fixture
def mock_iex():
    """Create mock IEX plugin."""
    plugin = AsyncMock()
    plugin.get_quote = AsyncMock()
    return plugin


@pytest.fixture
def analyzer(mock_polygon, mock_iex):
    """Create analyzer with mock plugins."""
    return MarketConditionsAnalyzer(
        polygon_plugin=mock_polygon, iex_plugin=mock_iex, use_cache=False
    )


class TestMarketConditionsAnalyzer:
    """Test MarketConditionsAnalyzer class."""

    def test_init_requires_plugin(self):
        """Test that at least one plugin is required."""
        with pytest.raises(ValueError, match="At least one data plugin"):
            MarketConditionsAnalyzer(polygon_plugin=None, iex_plugin=None)

    def test_init_with_polygon_only(self, mock_polygon):
        """Test initialization with only Polygon plugin."""
        analyzer = MarketConditionsAnalyzer(polygon_plugin=mock_polygon)
        assert analyzer.polygon == mock_polygon
        assert analyzer.iex is None

    def test_init_with_iex_only(self, mock_iex):
        """Test initialization with only IEX plugin."""
        analyzer = MarketConditionsAnalyzer(iex_plugin=mock_iex)
        assert analyzer.iex == mock_iex
        assert analyzer.polygon is None

    def test_cache_enabled_by_default(self, mock_polygon):
        """Test that caching is enabled by default."""
        analyzer = MarketConditionsAnalyzer(polygon_plugin=mock_polygon)
        assert analyzer.cache is not None
        assert isinstance(analyzer.cache, dict)

    def test_cache_disabled(self, mock_polygon):
        """Test that caching can be disabled."""
        analyzer = MarketConditionsAnalyzer(polygon_plugin=mock_polygon, use_cache=False)
        assert analyzer.cache == {}

    @pytest.mark.asyncio
    async def test_get_market_overview_polygon(self, analyzer, mock_polygon):
        """Test getting market overview from Polygon."""
        # Mock Polygon response
        mock_polygon.get_snapshot.return_value = {
            "ticker": {
                "lastTrade": {"p": 450.0},
                "todaysChange": 5.0,
                "todaysChangePerc": 1.12,
                "day": {"v": 1000000},
                "prevDay": {"h": 455.0, "l": 445.0},
            }
        }

        result = await analyzer.get_market_overview()

        assert "SPY" in result
        assert isinstance(result["SPY"], IndexSnapshot)
        assert result["SPY"].symbol == "SPY"
        assert result["SPY"].price == 450.0
        assert mock_polygon.get_snapshot.call_count == 4  # SPY, QQQ, IWM, DIA

    @pytest.mark.asyncio
    async def test_get_market_overview_fallback_to_iex(self, analyzer, mock_polygon, mock_iex):
        """Test fallback to IEX when Polygon fails."""
        # Polygon fails
        mock_polygon.get_snapshot.side_effect = Exception("Polygon API error")

        # IEX succeeds
        mock_iex.get_quote.return_value = {
            "symbol": "SPY",
            "latestPrice": 450.0,
            "change": 5.0,
            "changePercent": 0.0112,
            "latestVolume": 1000000,
            "week52High": 460.0,
            "week52Low": 400.0,
        }

        result = await analyzer.get_market_overview()

        assert "SPY" in result
        assert result["SPY"].price == 450.0
        assert mock_iex.get_quote.call_count == 4  # Fallback for all indices

    @pytest.mark.asyncio
    async def test_get_market_overview_all_fail(self, analyzer, mock_polygon, mock_iex):
        """Test exception when all sources fail."""
        mock_polygon.get_snapshot.side_effect = Exception("Polygon failed")
        mock_iex.get_quote.side_effect = Exception("IEX failed")

        with pytest.raises(AllDataSourcesFailedError, match="All data sources failed"):
            await analyzer.get_market_overview()

    @pytest.mark.asyncio
    async def test_get_volatility_metrics_polygon(self, analyzer, mock_polygon):
        """Test getting VIX metrics from Polygon."""
        mock_polygon.get_snapshot.return_value = {"ticker": {"lastTrade": {"p": 18.5}}}
        mock_polygon.get_previous_close.return_value = {"c": 20.0}

        result = await analyzer.get_volatility_metrics()

        assert isinstance(result, VolatilityMetrics)
        assert result.vix_current == 18.5
        assert result.vix_change == -1.5
        assert result.regime == VolatilityRegime.NORMAL
        assert result.trend == "Falling"

    @pytest.mark.asyncio
    async def test_volatility_regime_classification(self):
        """Test VIX regime classification thresholds."""
        test_cases = [
            (12.0, VolatilityRegime.LOW),  # < 15
            (17.5, VolatilityRegime.NORMAL),  # 15-20
            (22.0, VolatilityRegime.ELEVATED),  # 20-25
            (30.0, VolatilityRegime.HIGH),  # 25-35
            (40.0, VolatilityRegime.EXTREME),  # > 35
        ]

        for vix_level, expected_regime in test_cases:
            # Create fresh mock for each iteration to avoid state leakage
            mock_poly = AsyncMock()
            mock_poly.get_snapshot.return_value = {"ticker": {"lastTrade": {"p": vix_level}}}
            mock_poly.get_previous_close.return_value = {"c": vix_level}

            analyzer = MarketConditionsAnalyzer(polygon_plugin=mock_poly, use_cache=False)
            result = await analyzer.get_volatility_metrics()
            assert (
                result.regime == expected_regime
            ), f"VIX {vix_level} classified as {result.regime}, expected {expected_regime}"

    @pytest.mark.asyncio
    async def test_get_sector_performance(self, analyzer, mock_polygon):
        """Test getting sector performance data."""
        mock_polygon.get_snapshot.return_value = {
            "ticker": {
                "lastTrade": {"p": 150.0},
                "todaysChangePerc": 2.5,
            }
        }

        result = await analyzer.get_sector_performance()

        assert len(result) == 10  # All 10 sectors
        assert isinstance(result[0], SectorPerformance)
        assert result[0].symbol in analyzer.SECTORS
        assert result[0].change_pct_daily == 2.5

    @pytest.mark.asyncio
    async def test_get_breadth_indicators(self, analyzer):
        """Test getting breadth indicators."""
        result = await analyzer.get_breadth_indicators()

        assert isinstance(result, BreadthIndicators)
        # Current implementation returns placeholders
        assert result.breadth_sentiment == "Unknown"

    def test_classify_regime_risk_on(self, analyzer):
        """Test regime classification for Risk-On conditions."""
        volatility = VolatilityMetrics(
            vix_current=14.0,
            vix_change=-1.0,
            vix_change_pct=-6.7,
            vix_high_52w=60.0,
            vix_low_52w=12.0,
            regime=VolatilityRegime.LOW,
            trend="Falling",
        )

        breadth = BreadthIndicators(
            advance_decline_ratio=1.5,
            pct_above_50day_ma=65.0,
            pct_above_200day_ma=70.0,
            new_highs=100,
            new_lows=20,
            breadth_sentiment="Bullish",
        )

        sectors = [
            SectorPerformance("XLK", "Technology", 150.0, 2.5, None, None),
            SectorPerformance("XLY", "Consumer Disc", 120.0, 2.0, None, None),
            SectorPerformance("XLF", "Financials", 110.0, 1.8, None, None),
        ]

        regime, rationale = analyzer.classify_regime(volatility, breadth, sectors)

        assert regime == MarketRegime.RISK_ON
        assert "VIX low" in rationale
        assert "falling" in rationale.lower()

    def test_classify_regime_risk_off(self, analyzer):
        """Test regime classification for Risk-Off conditions."""
        volatility = VolatilityMetrics(
            vix_current=28.0,
            vix_change=5.0,
            vix_change_pct=21.7,
            vix_high_52w=60.0,
            vix_low_52w=12.0,
            regime=VolatilityRegime.HIGH,
            trend="Rising",
        )

        breadth = BreadthIndicators(
            advance_decline_ratio=0.7,
            pct_above_50day_ma=35.0,
            pct_above_200day_ma=40.0,
            new_highs=20,
            new_lows=150,
            breadth_sentiment="Bearish",
        )

        sectors = [
            SectorPerformance("XLV", "Healthcare", 130.0, 1.5, None, None),
            SectorPerformance("XLU", "Utilities", 70.0, 1.2, None, None),
            SectorPerformance("XLP", "Consumer Staples", 80.0, 0.8, None, None),
        ]

        regime, rationale = analyzer.classify_regime(volatility, breadth, sectors)

        assert regime == MarketRegime.RISK_OFF
        assert "elevated" in rationale.lower() or "vix" in rationale.lower()

    def test_classify_regime_choppy(self, analyzer):
        """Test regime classification for Choppy conditions."""
        volatility = VolatilityMetrics(
            vix_current=19.0,
            vix_change=0.5,
            vix_change_pct=2.7,
            vix_high_52w=60.0,
            vix_low_52w=12.0,
            regime=VolatilityRegime.NORMAL,
            trend="Stable",
        )

        breadth = BreadthIndicators(
            advance_decline_ratio=1.0,
            pct_above_50day_ma=50.0,
            pct_above_200day_ma=55.0,
            new_highs=75,
            new_lows=75,
            breadth_sentiment="Neutral",
        )

        sectors = [
            SectorPerformance("XLK", "Technology", 150.0, 1.0, None, None),
            SectorPerformance("XLV", "Healthcare", 130.0, 0.8, None, None),
            SectorPerformance("XLF", "Financials", 110.0, 0.5, None, None),
        ]

        regime, rationale = analyzer.classify_regime(volatility, breadth, sectors)

        assert regime == MarketRegime.CHOPPY
        assert "normal" in rationale.lower() or "mixed" in rationale.lower()

    def test_generate_trading_implications_risk_on(self, analyzer):
        """Test trading implications for Risk-On regime."""
        volatility = VolatilityMetrics(
            vix_current=14.0,
            vix_change=-1.0,
            vix_change_pct=-6.7,
            vix_high_52w=60.0,
            vix_low_52w=12.0,
            regime=VolatilityRegime.LOW,
            trend="Falling",
        )

        implications = analyzer.generate_trading_implications(MarketRegime.RISK_ON, volatility)

        assert "Aggressive" in implications["position_sizing"]
        assert "Trend Following" in implications["strategy_bias"]
        assert "XLK" in implications["sectors_favor"]
        assert len(implications["special_considerations"]) > 0

    def test_generate_trading_implications_risk_off(self, analyzer):
        """Test trading implications for Risk-Off regime."""
        volatility = VolatilityMetrics(
            vix_current=32.0,
            vix_change=8.0,
            vix_change_pct=33.3,
            vix_high_52w=60.0,
            vix_low_52w=12.0,
            regime=VolatilityRegime.HIGH,
            trend="Rising",
        )

        implications = analyzer.generate_trading_implications(MarketRegime.RISK_OFF, volatility)

        assert "Reduced" in implications["position_sizing"]
        assert "Mean Reversion" in implications["strategy_bias"]
        assert "XLV" in implications["sectors_favor"] or "XLP" in implications["sectors_favor"]
        # Check for VIX extreme warning
        assert any("VIX extreme" in c for c in implications["special_considerations"])

    def test_generate_trading_implications_choppy(self, analyzer):
        """Test trading implications for Choppy regime."""
        volatility = VolatilityMetrics(
            vix_current=19.0,
            vix_change=0.5,
            vix_change_pct=2.7,
            vix_high_52w=60.0,
            vix_low_52w=12.0,
            regime=VolatilityRegime.NORMAL,
            trend="Stable",
        )

        implications = analyzer.generate_trading_implications(MarketRegime.CHOPPY, volatility)

        assert "Reduced" in implications["position_sizing"]
        assert "Mean Reversion" in implications["strategy_bias"]
        assert any(
            "Scale into" in c or "Avoid chasing" in c
            for c in implications["special_considerations"]
        )

    @pytest.mark.asyncio
    async def test_analyze_end_to_end(self, analyzer, mock_polygon):
        """Test complete analysis workflow."""
        # Mock all data sources
        mock_polygon.get_snapshot.return_value = {
            "ticker": {
                "lastTrade": {"p": 450.0},
                "todaysChange": 5.0,
                "todaysChangePerc": 1.12,
                "day": {"v": 1000000},
                "prevDay": {"h": 455.0, "l": 445.0},
            }
        }
        mock_polygon.get_previous_close.return_value = {"c": 20.0}

        report = await analyzer.analyze()

        assert report.timestamp is not None
        assert len(report.indices) == 4  # SPY, QQQ, IWM, DIA
        assert report.volatility is not None
        assert len(report.sectors) == 10
        assert report.breadth is not None
        assert report.regime in MarketRegime
        assert report.regime_rationale != ""
        assert report.trading_implications is not None

    def test_cache_functionality(self, mock_polygon):
        """Test that caching works correctly."""
        analyzer = MarketConditionsAnalyzer(
            polygon_plugin=mock_polygon, use_cache=True, cache_ttl_seconds=60
        )

        # Set cached value
        test_data = {"test": "data"}
        analyzer._set_cached("test_key", test_data)

        # Should retrieve cached value
        cached = analyzer._get_cached("test_key")
        assert cached == test_data

        # Non-existent key
        assert analyzer._get_cached("nonexistent") is None


class TestDataParsing:
    """Test data parsing helper methods."""

    def test_parse_index_snapshots_polygon(self, analyzer):
        """Test parsing Polygon index data."""
        data = {
            "SPY": {
                "ticker": {
                    "lastTrade": {"p": 450.0},
                    "todaysChange": 5.0,
                    "todaysChangePerc": 1.12,
                    "day": {"v": 1000000},
                    "prevDay": {"h": 455.0, "l": 445.0},
                }
            }
        }

        result = analyzer._parse_index_snapshots(data, "polygon")

        assert "SPY" in result
        assert result["SPY"].price == 450.0
        assert result["SPY"].change == 5.0
        assert result["SPY"].change_pct == 1.12

    def test_parse_index_snapshots_iex(self, analyzer):
        """Test parsing IEX index data."""
        data = {
            "SPY": {
                "symbol": "SPY",
                "latestPrice": 450.0,
                "change": 5.0,
                "changePercent": 0.0112,
                "latestVolume": 1000000,
                "week52High": 460.0,
                "week52Low": 400.0,
            }
        }

        result = analyzer._parse_index_snapshots(data, "iex")

        assert "SPY" in result
        assert result["SPY"].price == 450.0
        assert result["SPY"].change == 5.0
        assert abs(result["SPY"].change_pct - 1.12) < 0.01

    def test_parse_sector_performance(self, analyzer):
        """Test parsing sector performance data."""
        data = {
            "XLK": {
                "ticker": {
                    "lastTrade": {"p": 150.0},
                    "todaysChangePerc": 2.5,
                }
            }
        }

        result = analyzer._parse_sector_performance(data, "polygon")

        assert len(result) == 1
        assert result[0].symbol == "XLK"
        assert result[0].sector_name == "Technology"
        assert result[0].change_pct_daily == 2.5
