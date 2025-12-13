"""
OptionsCore Engine Integration Tests

End-to-end tests for options chain fetching, enrichment, and caching.
"""

from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from ordinis.engines.optionscore import OptionsCoreEngine, OptionsEngineConfig
from ordinis.engines.optionscore.data import OptionType


@pytest.fixture
def mock_polygon_plugin():
    """Create mock Polygon plugin."""
    plugin = MagicMock()
    plugin.status = MagicMock()
    plugin.status.value = "ready"

    # Mock get_quote
    async def mock_get_quote(symbol):
        return {"symbol": symbol, "last": 100.0, "timestamp": datetime.utcnow().isoformat()}

    # Mock get_options_chain
    async def mock_get_options_chain(
        symbol, expiration=None, strike_price=None, contract_type=None
    ):
        return {
            "symbol": symbol,
            "contracts": [
                {
                    "ticker": f"O:{symbol}260117C00100000",
                    "underlying": symbol,
                    "contract_type": "call",
                    "strike_price": 100.0,
                    "expiration_date": "2026-01-17",
                    "shares_per_contract": 100,
                    "primary_exchange": "CBOE",
                },
                {
                    "ticker": f"O:{symbol}260117P00100000",
                    "underlying": symbol,
                    "contract_type": "put",
                    "strike_price": 100.0,
                    "expiration_date": "2026-01-17",
                    "shares_per_contract": 100,
                    "primary_exchange": "CBOE",
                },
                {
                    "ticker": f"O:{symbol}260117C00105000",
                    "underlying": symbol,
                    "contract_type": "call",
                    "strike_price": 105.0,
                    "expiration_date": "2026-01-17",
                    "shares_per_contract": 100,
                    "primary_exchange": "CBOE",
                },
            ],
            "count": 3,
        }

    plugin.get_quote = mock_get_quote
    plugin.get_options_chain = mock_get_options_chain

    return plugin


@pytest.fixture
def engine_config():
    """Create engine configuration."""
    return OptionsEngineConfig(
        engine_id="test_engine",
        enabled=True,
        cache_ttl_seconds=60,
        default_risk_free_rate=0.05,
    )


@pytest.fixture
async def engine(engine_config, mock_polygon_plugin):
    """Create and initialize engine."""
    engine = OptionsCoreEngine(engine_config, mock_polygon_plugin)
    await engine.initialize()
    return engine


@pytest.mark.asyncio
async def test_engine_initialization(engine_config, mock_polygon_plugin):
    """Test engine initialization."""
    engine = OptionsCoreEngine(engine_config, mock_polygon_plugin)

    assert not engine.initialized
    result = await engine.initialize()
    assert result is True
    assert engine.initialized


@pytest.mark.asyncio
async def test_full_chain_workflow(engine):
    """Test complete workflow from chain fetch to enrichment."""
    chain = await engine.get_option_chain("AAPL")

    # Verify chain structure
    assert chain.symbol == "AAPL"
    assert chain.underlying_price == 100.0
    assert len(chain.contracts) == 3

    # Verify enrichment
    for enriched_contract in chain.contracts:
        assert enriched_contract.contract is not None
        assert enriched_contract.pricing is not None
        assert enriched_contract.greeks is not None

        # Verify pricing result
        pricing = enriched_contract.pricing
        assert pricing.theoretical_price > 0
        assert pricing.model_used == "black_scholes"
        assert "underlying_price" in pricing.parameters_used

        # Verify Greeks result
        greeks = enriched_contract.greeks
        assert greeks.delta != 0  # Delta should not be zero for ATM options
        assert greeks.gamma >= 0  # Gamma always positive for long options
        assert greeks.vega >= 0  # Vega always positive


@pytest.mark.asyncio
async def test_cache_behavior(engine):
    """Test caching with TTL."""
    # First call - should fetch from Polygon
    chain1 = await engine.get_option_chain("AAPL")
    assert len(chain1.contracts) == 3

    # Second call - should hit cache
    chain2 = await engine.get_option_chain("AAPL")
    assert chain2 is chain1  # Should be same object from cache

    # Clear cache
    engine.clear_cache("AAPL")

    # Third call - should fetch again
    chain3 = await engine.get_option_chain("AAPL")
    assert chain3 is not chain1  # Should be new object


@pytest.mark.asyncio
async def test_cache_stats(engine):
    """Test cache statistics."""
    # Initially empty
    stats = engine.get_cache_stats()
    assert stats["total_items"] == 0

    # Fetch chain
    await engine.get_option_chain("AAPL")

    # Check stats
    stats = engine.get_cache_stats()
    assert stats["total_items"] == 1
    assert stats["active_items"] == 1
    assert stats["expired_items"] == 0


@pytest.mark.asyncio
async def test_greeks_calculation(engine):
    """Test Greeks calculation for calls and puts."""
    chain = await engine.get_option_chain("AAPL")

    # Find call and put at same strike
    calls = [c for c in chain.contracts if c.contract.option_type == OptionType.CALL]
    puts = [c for c in chain.contracts if c.contract.option_type == OptionType.PUT]

    assert len(calls) > 0
    assert len(puts) > 0

    call = calls[0]
    put = puts[0]

    # Call delta should be positive
    assert call.greeks.delta > 0
    assert call.greeks.delta <= 1.0

    # Put delta should be negative
    assert put.greeks.delta < 0
    assert put.greeks.delta >= -1.0

    # Gamma should be same for call and put at same strike
    assert abs(call.greeks.gamma - put.greeks.gamma) < 0.0001

    # Vega should be same for call and put at same strike
    assert abs(call.greeks.vega - put.greeks.vega) < 0.0001


@pytest.mark.asyncio
async def test_summary_statistics(engine):
    """Test chain summary statistics."""
    chain = await engine.get_option_chain("AAPL")

    summary = chain.summary
    assert summary["total_contracts"] == 3
    assert summary["total_calls"] == 2
    assert summary["total_puts"] == 1
    assert summary["unique_expirations"] == 1
    assert summary["unique_strikes"] == 2  # 100 and 105
    assert summary["atm_strike"] == 100.0  # Closest to underlying price


@pytest.mark.asyncio
async def test_invalid_symbol_handling(engine):
    """Test error handling for invalid symbols."""
    with pytest.raises(ValueError, match="Invalid symbol"):
        await engine.get_option_chain("")

    with pytest.raises(ValueError, match="Invalid symbol"):
        await engine.get_option_chain("123")


@pytest.mark.asyncio
async def test_uninitialized_engine_error(engine_config, mock_polygon_plugin):
    """Test that uninitialized engine raises error."""
    engine = OptionsCoreEngine(engine_config, mock_polygon_plugin)

    with pytest.raises(RuntimeError, match="Engine not initialized"):
        await engine.get_option_chain("AAPL")


@pytest.mark.asyncio
async def test_filtered_chain_fetch(engine):
    """Test fetching chain with filters."""
    # Test with expiration filter
    chain = await engine.get_option_chain("AAPL", expiration="2025-01-17")
    assert chain.symbol == "AAPL"
    assert len(chain.contracts) == 3

    # Test with contract type filter
    chain_calls = await engine.get_option_chain("AAPL", contract_type="call")
    assert chain_calls.symbol == "AAPL"


@pytest.mark.asyncio
async def test_price_contract_method(engine):
    """Test pricing individual contract."""
    from ordinis.engines.optionscore.data import OptionContract

    contract = OptionContract(
        symbol="O:AAPL260117C00150000",
        underlying="AAPL",
        option_type=OptionType.CALL,
        strike=150.0,
        expiration=date(2026, 1, 17),
    )

    pricing = await engine.price_contract(contract, underlying_price=145.0)

    assert pricing is not None
    assert pricing.theoretical_price > 0
    assert pricing.model_used == "black_scholes"
    assert pricing.contract == contract


@pytest.mark.asyncio
async def test_calculate_greeks_method(engine):
    """Test calculating Greeks for individual contract."""
    from ordinis.engines.optionscore.data import OptionContract

    contract = OptionContract(
        symbol="O:AAPL260117C00150000",
        underlying="AAPL",
        option_type=OptionType.CALL,
        strike=150.0,
        expiration=date(2026, 1, 17),
    )

    greeks = await engine.calculate_greeks(contract, underlying_price=145.0)

    assert greeks is not None
    assert greeks.delta > 0  # OTM call has positive delta
    assert greeks.gamma > 0
    assert greeks.vega > 0
    assert greeks.underlying_price == 145.0
