"""
Enrichment Engine Unit Tests

Tests for options chain enrichment and contract transformation.
"""

import pytest

from ordinis.engines.optionscore.core.enrichment import ChainEnrichmentEngine
from ordinis.engines.optionscore.data import OptionType
from ordinis.engines.optionscore.pricing.black_scholes import BlackScholesEngine
from ordinis.engines.optionscore.pricing.greeks import GreeksCalculator


@pytest.fixture
def enrichment_engine():
    """Create enrichment engine instance."""
    pricing_engine = BlackScholesEngine()
    greeks_calc = GreeksCalculator(pricing_engine)
    return ChainEnrichmentEngine(pricing_engine, greeks_calc)


def test_enrich_call_contract(enrichment_engine):
    """Test enriching a call option contract."""
    contract_data = {
        "ticker": "O:AAPL260117C00150000",
        "underlying": "AAPL",
        "contract_type": "call",
        "strike_price": 150.0,
        "expiration_date": "2026-01-17",
        "shares_per_contract": 100,
        "primary_exchange": "CBOE",
    }

    enriched = enrichment_engine.enrich_contract(
        contract_data=contract_data,
        underlying_price=145.0,
        risk_free_rate=0.05,
    )

    # Verify contract
    assert enriched.contract.symbol == "O:AAPL260117C00150000"
    assert enriched.contract.underlying == "AAPL"
    assert enriched.contract.option_type == OptionType.CALL
    assert enriched.contract.strike == 150.0

    # Verify pricing
    assert enriched.pricing is not None
    assert enriched.pricing.theoretical_price > 0
    assert enriched.pricing.model_used == "black_scholes"

    # Verify Greeks
    assert enriched.greeks is not None
    assert enriched.greeks.delta > 0  # Call delta is positive
    assert enriched.greeks.gamma > 0
    assert enriched.greeks.vega > 0


def test_enrich_put_contract(enrichment_engine):
    """Test enriching a put option contract."""
    contract_data = {
        "ticker": "O:AAPL260117P00150000",
        "underlying": "AAPL",
        "contract_type": "put",
        "strike_price": 150.0,
        "expiration_date": "2026-01-17",
        "shares_per_contract": 100,
        "primary_exchange": "CBOE",
    }

    enriched = enrichment_engine.enrich_contract(
        contract_data=contract_data,
        underlying_price=145.0,
        risk_free_rate=0.05,
    )

    # Verify contract type
    assert enriched.contract.option_type == OptionType.PUT

    # Verify pricing
    assert enriched.pricing.theoretical_price > 0

    # Verify Greeks
    assert enriched.greeks.delta < 0  # Put delta is negative
    assert enriched.greeks.gamma > 0
    assert enriched.greeks.vega > 0


def test_enrich_chain(enrichment_engine):
    """Test enriching a full options chain."""
    chain_data = {
        "symbol": "AAPL",
        "contracts": [
            {
                "ticker": "O:AAPL260117C00150000",
                "underlying": "AAPL",
                "contract_type": "call",
                "strike_price": 150.0,
                "expiration_date": "2026-01-17",
                "shares_per_contract": 100,
            },
            {
                "ticker": "O:AAPL260117P00150000",
                "underlying": "AAPL",
                "contract_type": "put",
                "strike_price": 150.0,
                "expiration_date": "2026-01-17",
                "shares_per_contract": 100,
            },
            {
                "ticker": "O:AAPL260117C00155000",
                "underlying": "AAPL",
                "contract_type": "call",
                "strike_price": 155.0,
                "expiration_date": "2026-01-17",
                "shares_per_contract": 100,
            },
        ],
    }

    enriched_chain = enrichment_engine.enrich_chain(
        chain_data=chain_data,
        underlying_price=145.0,
        risk_free_rate=0.05,
    )

    # Verify chain structure
    assert enriched_chain.symbol == "AAPL"
    assert enriched_chain.underlying_price == 145.0
    assert len(enriched_chain.contracts) == 3

    # Verify unique expirations and strikes
    assert len(enriched_chain.expirations) == 1
    assert enriched_chain.expirations[0] == "2026-01-17"
    assert len(enriched_chain.strikes) == 2
    assert 150.0 in enriched_chain.strikes
    assert 155.0 in enriched_chain.strikes


def test_calculate_summary_stats(enrichment_engine):
    """Test summary statistics calculation."""
    chain_data = {
        "symbol": "AAPL",
        "contracts": [
            {
                "ticker": "O:AAPL260117C00145000",
                "underlying": "AAPL",
                "contract_type": "call",
                "strike_price": 145.0,
                "expiration_date": "2026-01-17",
                "shares_per_contract": 100,
            },
            {
                "ticker": "O:AAPL260117P00145000",
                "underlying": "AAPL",
                "contract_type": "put",
                "strike_price": 145.0,
                "expiration_date": "2026-01-17",
                "shares_per_contract": 100,
            },
            {
                "ticker": "O:AAPL260117C00150000",
                "underlying": "AAPL",
                "contract_type": "call",
                "strike_price": 150.0,
                "expiration_date": "2026-01-17",
                "shares_per_contract": 100,
            },
        ],
    }

    enriched_chain = enrichment_engine.enrich_chain(
        chain_data=chain_data,
        underlying_price=147.5,
        risk_free_rate=0.05,
    )

    summary = enriched_chain.summary

    # Verify summary statistics
    assert summary["total_contracts"] == 3
    assert summary["total_calls"] == 2
    assert summary["total_puts"] == 1
    assert summary["unique_expirations"] == 1
    assert summary["unique_strikes"] == 2
    assert summary["underlying_price"] == 147.5

    # ATM strike should be closest to underlying price (147.5)
    # Should be 145.0 (diff = 2.5) or 150.0 (diff = 2.5), both equally close
    assert summary["atm_strike"] in [145.0, 150.0]


def test_invalid_expiration_date(enrichment_engine):
    """Test handling of invalid expiration date format."""
    contract_data = {
        "ticker": "O:AAPL250117C00150000",
        "underlying": "AAPL",
        "contract_type": "call",
        "strike_price": 150.0,
        "expiration_date": "invalid-date",
        "shares_per_contract": 100,
    }

    with pytest.raises(ValueError):
        enrichment_engine.enrich_contract(
            contract_data=contract_data,
            underlying_price=145.0,
            risk_free_rate=0.05,
        )


def test_contract_with_dividend_yield(enrichment_engine):
    """Test contract enrichment with dividend yield."""
    contract_data = {
        "ticker": "O:AAPL260117C00150000",
        "underlying": "AAPL",
        "contract_type": "call",
        "strike_price": 150.0,
        "expiration_date": "2026-01-17",
        "shares_per_contract": 100,
    }

    enriched = enrichment_engine.enrich_contract(
        contract_data=contract_data,
        underlying_price=145.0,
        risk_free_rate=0.05,
        dividend_yield=0.02,  # 2% dividend yield
    )

    # Verify dividend yield was used
    assert enriched.pricing.parameters_used["dividend_yield"] == 0.02

    # Price should be lower than without dividend for call
    enriched_no_div = enrichment_engine.enrich_contract(
        contract_data=contract_data,
        underlying_price=145.0,
        risk_free_rate=0.05,
        dividend_yield=0.0,
    )

    assert enriched.pricing.theoretical_price < enriched_no_div.pricing.theoretical_price


def test_contract_with_custom_volatility(enrichment_engine):
    """Test contract enrichment with custom volatility."""
    contract_data = {
        "ticker": "O:AAPL260117C00150000",
        "underlying": "AAPL",
        "contract_type": "call",
        "strike_price": 150.0,
        "expiration_date": "2026-01-17",
        "shares_per_contract": 100,
    }

    # Higher volatility should result in higher option price
    enriched_high_vol = enrichment_engine.enrich_contract(
        contract_data=contract_data,
        underlying_price=145.0,
        risk_free_rate=0.05,
        assumed_volatility=0.40,  # 40% volatility
    )

    enriched_low_vol = enrichment_engine.enrich_contract(
        contract_data=contract_data,
        underlying_price=145.0,
        risk_free_rate=0.05,
        assumed_volatility=0.10,  # 10% volatility
    )

    assert enriched_high_vol.pricing.theoretical_price > enriched_low_vol.pricing.theoretical_price
    assert enriched_high_vol.greeks.vega > enriched_low_vol.greeks.vega
