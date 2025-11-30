# Market Data Enhancement Plan

## Executive Summary

Replace web search-based market condition analysis with direct API integration using existing Polygon.io and IEX Cloud plugins for real-time, accurate market data.

**Status:** Planning Phase
**Priority:** High
**Estimated Effort:** 2-3 days
**Branch:** user/interface

---

## Problem Statement

### Current Limitations (Web Search Approach)

1. **Data Quality Issues**
   - Inconsistent formatting across sources
   - Stale data from web crawlers (often 15min - 24hr delayed)
   - Parsing errors from HTML/content changes
   - No guarantee of data accuracy

2. **Performance Issues**
   - High latency (2-5 seconds per search)
   - Rate limiting from search engines
   - Multiple searches needed for complete picture
   - Token-heavy responses

3. **Reliability Issues**
   - Source availability varies
   - No SLA guarantees
   - Difficult to validate data integrity
   - Can't handle real-time requirements

### Evidence from Recent Execution

From `/market-conditions` execution (Nov 30, 2025):
- Required 6 web searches for incomplete data
- Missing real-time sector performance
- Incomplete breadth data
- ~30 seconds total execution time
- Data was 1-2 days old in some cases

---

## Proposed Solution

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         /market-conditions Command                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │ Market Data  │───▶│ Analysis     │───▶│ Report   │  │
│  │ Aggregator   │    │ Engine       │    │ Generator│  │
│  └──────────────┘    └──────────────┘    └──────────┘  │
│         │                                                │
│         ├─────────────┬─────────────┬──────────────┐    │
│         ▼             ▼             ▼              ▼    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐│
│  │ Polygon  │  │   IEX    │  │  FRED    │  │ Fallback││
│  │ Plugin   │  │  Plugin  │  │  (Future)│  │Web Search│
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘│
└─────────────────────────────────────────────────────────┘
```

### Component Design

#### 1. Market Data Aggregator

**Location:** `src/analysis/market_conditions.py` (new)

**Responsibilities:**
- Orchestrate data collection from multiple sources
- Implement fallback logic (plugin → web search)
- Cache results (15-minute TTL for real-time data)
- Handle rate limiting across providers

**Key Methods:**
```python
async def get_index_snapshot(symbols: list[str]) -> dict
async def get_volatility_metrics() -> dict
async def get_sector_performance() -> dict
async def get_breadth_indicators() -> dict
async def get_economic_calendar(days_ahead: int) -> list
```

#### 2. Plugin Integration Layer

**Polygon.io Plugin** (Already exists: `src/plugins/market_data/polygon.py`)
- **Use for:**
  - Real-time quotes (SPY, QQQ, IWM, DIA)
  - Sector ETF prices (XLK, XLF, XLE, etc.)
  - Aggregated bars (daily/weekly performance)
  - Market status

**IEX Cloud Plugin** (Already exists: `src/plugins/market_data/iex.py`)
- **Use for:**
  - Backup/fallback for quotes
  - Fundamental data
  - Economic calendar (if available)
  - Company news

**Future: FRED Plugin** (To be created)
- Economic indicators
- VIX term structure
- Treasury yields
- Fed data

#### 3. Fallback Strategy

```python
# Priority cascade
async def get_data(data_type: str):
    try:
        return await polygon_plugin.get(data_type)
    except (PluginError, RateLimitError):
        try:
            return await iex_plugin.get(data_type)
        except (PluginError, RateLimitError):
            # Last resort: web search
            return await web_search_fallback(data_type)
```

---

## Implementation Plan

### Phase 1: Core Integration (Day 1)

#### Task 1.1: Create Market Data Aggregator
**File:** `src/analysis/market_conditions.py`

```python
"""Market conditions analysis using plugin-based data sources."""

from datetime import datetime, timedelta
from typing import Any

from src.plugins.market_data.polygon import PolygonDataPlugin
from src.plugins.market_data.iex import IEXDataPlugin
from src.core.rate_limiter import RateLimiter


class MarketConditionsAnalyzer:
    """Analyze current market conditions using real-time data."""

    def __init__(
        self,
        polygon_plugin: PolygonDataPlugin,
        iex_plugin: IEXDataPlugin,
        use_cache: bool = True,
        cache_ttl_seconds: int = 900,  # 15 minutes
    ):
        self.polygon = polygon_plugin
        self.iex = iex_plugin
        self.cache = {} if use_cache else None
        self.cache_ttl = cache_ttl_seconds

    async def get_market_overview(self) -> dict[str, Any]:
        """Get current state of major indices."""
        symbols = ["SPY", "QQQ", "IWM", "DIA"]

        # Try Polygon first
        try:
            quotes = await self.polygon.get_quotes(symbols)
            return self._format_index_data(quotes)
        except Exception as e:
            # Fallback to IEX
            quotes = await self.iex.get_batch_quotes(symbols)
            return self._format_index_data(quotes)

    async def get_volatility_regime(self) -> dict[str, Any]:
        """Get VIX and volatility metrics."""
        # Implementation here
        pass

    async def get_sector_rotation(self) -> dict[str, Any]:
        """Get sector ETF performance."""
        sector_etfs = [
            "XLK", "XLF", "XLE", "XLV", "XLY",
            "XLP", "XLI", "XLB", "XLU", "XLRE"
        ]
        # Implementation here
        pass

    async def get_breadth_indicators(self) -> dict[str, Any]:
        """Get market breadth metrics."""
        # Implementation here
        pass

    def classify_regime(
        self,
        volatility: dict,
        breadth: dict,
        sector_rotation: dict
    ) -> str:
        """Classify market regime based on indicators."""
        # Logic from knowledge base
        pass
```

**Testing:**
```python
# tests/test_analysis/test_market_conditions.py
pytest tests/test_analysis/test_market_conditions.py -v
```

#### Task 1.2: Update Market Conditions Command
**File:** `.claude/commands/market-conditions.md`

Update to use `MarketConditionsAnalyzer` instead of web search.

### Phase 2: Enhanced Features (Day 2)

#### Task 2.1: Add VIX Analysis
- VIX current level via Polygon
- VIX term structure (VIX, VIX3M, VIX6M)
- Historical percentile ranking

#### Task 2.2: Add Breadth Indicators
- Advance/decline data from Polygon
- New highs/lows
- % stocks above 50-day/200-day MA

#### Task 2.3: Add Sector Rotation
- Real-time sector ETF quotes
- Calculate relative performance
- Identify leaders/laggards

### Phase 3: Economic Calendar Integration (Day 3)

#### Task 3.1: FRED Integration (Optional)
Create new plugin: `src/plugins/economic/fred.py`

**Capabilities:**
- Economic calendar
- Fed funds rate
- CPI/PPI data
- Treasury yields

#### Task 3.2: Trading Economics API (Alternative)
If FRED doesn't provide calendar, use Trading Economics API.

---

## Data Mapping

### Index Snapshot

| Data Point | Polygon API | IEX API | Fallback |
|------------|-------------|---------|----------|
| Current Price | `/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}` | `/stock/{symbol}/quote` | Web search |
| Daily Change | Included in snapshot | `changePercent` field | Web search |
| 52-week High/Low | Previous day aggs | `week52High`/`week52Low` | Web search |
| Volume | Snapshot | `latestVolume` | Web search |

### Volatility

| Data Point | Source | API Endpoint | Notes |
|------------|--------|--------------|-------|
| VIX Current | Polygon | `/v2/snapshot/locale/us/markets/stocks/tickers/VIX` | Primary |
| VIX Previous | Polygon | `/v2/aggs/ticker/VIX/prev` | For change calc |
| VIX 52-week range | Polygon | Historical aggs | Cache daily |

### Sector Performance

| ETF | Sector | Data Source |
|-----|--------|-------------|
| XLK | Technology | Polygon/IEX |
| XLF | Financials | Polygon/IEX |
| XLE | Energy | Polygon/IEX |
| XLV | Healthcare | Polygon/IEX |
| XLY | Consumer Discretionary | Polygon/IEX |
| XLP | Consumer Staples | Polygon/IEX |
| XLI | Industrials | Polygon/IEX |
| XLB | Materials | Polygon/IEX |
| XLU | Utilities | Polygon/IEX |
| XLRE | Real Estate | Polygon/IEX |

---

## Testing Strategy

### Unit Tests
```python
# tests/test_analysis/test_market_conditions.py

class TestMarketConditionsAnalyzer:
    """Test market conditions analysis."""

    @pytest.mark.asyncio
    async def test_get_market_overview(self, mock_polygon):
        """Test index snapshot retrieval."""
        analyzer = MarketConditionsAnalyzer(mock_polygon, None)
        result = await analyzer.get_market_overview()

        assert "SPY" in result
        assert "price" in result["SPY"]
        assert "change_pct" in result["SPY"]

    @pytest.mark.asyncio
    async def test_fallback_to_iex(self, failing_polygon, mock_iex):
        """Test fallback when Polygon fails."""
        analyzer = MarketConditionsAnalyzer(failing_polygon, mock_iex)
        result = await analyzer.get_market_overview()

        # Should get data from IEX
        assert result is not None

    def test_regime_classification_risk_on(self):
        """Test risk-on regime detection."""
        analyzer = MarketConditionsAnalyzer(None, None)

        regime = analyzer.classify_regime(
            volatility={"vix": 14.5, "trend": "falling"},
            breadth={"ad_ratio": 1.5, "pct_above_50ma": 65},
            sector_rotation={"leaders": ["XLK", "XLY", "XLF"]}
        )

        assert regime == "Risk-On"
```

### Integration Tests
```python
# tests/integration/test_market_data_flow.py

@pytest.mark.integration
@pytest.mark.requires_api
async def test_end_to_end_market_conditions():
    """Test complete market conditions workflow."""
    # Requires real API keys
    config = load_config()

    polygon = PolygonDataPlugin(config.polygon)
    await polygon.initialize()

    analyzer = MarketConditionsAnalyzer(polygon, None)
    result = await analyzer.get_market_overview()

    assert result is not None
    assert len(result) == 4  # SPY, QQQ, IWM, DIA
```

---

## Performance Comparison

### Before (Web Search)

| Metric | Value |
|--------|-------|
| Total Execution Time | ~30 seconds |
| API Calls | 6 web searches |
| Data Freshness | 1-24 hours stale |
| Success Rate | ~85% (source availability) |
| Token Usage | ~3,000 tokens |

### After (Plugin Integration)

| Metric | Value | Improvement |
|--------|-------|-------------|
| Total Execution Time | ~2-3 seconds | **10x faster** |
| API Calls | 3-5 plugin calls | **More efficient** |
| Data Freshness | Real-time (15min delayed max) | **24x+ fresher** |
| Success Rate | ~99% (with fallbacks) | **+14%** |
| Token Usage | ~500 tokens | **6x reduction** |

---

## Configuration

### Environment Variables
```bash
# .env
POLYGON_API_KEY=your_polygon_key
IEX_API_KEY=your_iex_key
MARKET_DATA_CACHE_TTL=900  # 15 minutes
MARKET_DATA_PRIMARY_PROVIDER=polygon
```

### Plugin Configuration
```python
# config/market_data.yaml
polygon:
  api_key: ${POLYGON_API_KEY}
  rate_limit:
    requests_per_minute: 5
    burst: 10
  timeout_seconds: 10

iex:
  api_key: ${IEX_API_KEY}
  sandbox: false
  rate_limit:
    requests_per_minute: 100
  timeout_seconds: 5

cache:
  enabled: true
  ttl_seconds: 900  # 15 minutes for real-time data
  max_size_mb: 50
```

---

## Migration Strategy

### Phase 1: Dual-Mode Operation
- Implement plugin-based approach
- Keep web search as fallback
- Compare results for validation
- Gradual rollout

### Phase 2: Plugin-First
- Switch to plugin as primary
- Web search only on failure
- Monitor error rates

### Phase 3: Plugin-Only
- Remove web search code
- Full plugin dependency
- Implement robust error handling

---

## Risk Mitigation

### Risk: API Rate Limits

**Mitigation:**
1. Implement caching (15-minute TTL)
2. Use rate limiter from `src/core/rate_limiter.py`
3. Dual-provider fallback (Polygon → IEX)
4. Batch requests where possible

### Risk: API Costs

**Mitigation:**
1. Cache aggressively
2. Use free tier limits intelligently
3. Monitor usage via plugin health metrics
4. Implement circuit breakers

### Risk: Plugin Failures

**Mitigation:**
1. Health checks before requests
2. Graceful degradation (IEX fallback)
3. Web search last resort
4. Comprehensive error logging

---

## Success Criteria

### Functional Requirements
- ✅ Real-time data (<1 minute delay)
- ✅ <3 second response time
- ✅ 99%+ success rate
- ✅ Accurate regime classification

### Technical Requirements
- ✅ All tests passing (>90% coverage)
- ✅ Proper error handling
- ✅ Rate limiting implemented
- ✅ Caching functional

### User Experience
- ✅ Faster command execution
- ✅ More reliable results
- ✅ Current (not stale) data
- ✅ Clear error messages

---

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1** | 1 day | MarketConditionsAnalyzer, core integration |
| **Phase 2** | 1 day | VIX, breadth, sector rotation |
| **Phase 3** | 1 day | Economic calendar, FRED integration |
| **Testing** | 0.5 days | Unit tests, integration tests |
| **Documentation** | 0.5 days | API docs, user guide |
| **Total** | **3 days** | Full implementation |

---

## Next Steps

1. **Immediate (Today)**
   - ✅ Document current limitations
   - ⬜ Review existing plugin architecture
   - ⬜ Create skeleton for `MarketConditionsAnalyzer`

2. **Day 1**
   - Implement core aggregator
   - Integrate Polygon/IEX plugins
   - Create unit tests

3. **Day 2**
   - Add advanced features (VIX, breadth, sectors)
   - Integration testing
   - Performance benchmarking

4. **Day 3**
   - Economic calendar integration
   - Documentation
   - User testing

---

## References

### Existing Code
- `src/plugins/market_data/polygon.py` - Polygon.io plugin (lines 1-386)
- `src/plugins/market_data/iex.py` - IEX Cloud plugin (lines 1-307)
- `src/plugins/base.py` - Plugin architecture (lines 1-343)
- `src/core/rate_limiter.py` - Rate limiting (lines 1-376)

### External Documentation
- [Polygon.io API Docs](https://polygon.io/docs/stocks/getting-started)
- [IEX Cloud API Docs](https://iexcloud.io/docs/api/)
- [FRED API Docs](https://fred.stlouisfed.org/docs/api/fred/)

### Knowledge Base
- `docs/knowledge-base/07_risk_management/README.md` - Regime classification
- `docs/knowledge-base/04_fundamental_analysis/README.md` - Macro indicators
- `docs/CURRENT_STATUS_AND_NEXT_STEPS.md` - Project status

---

**Document Version:** 1.0
**Last Updated:** 2025-11-30
**Author:** Claude Code
**Status:** Planning → Ready for Implementation
