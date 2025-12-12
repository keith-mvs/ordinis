"""
OptionsCore Engine Demo

Demonstrates basic usage of the OptionsCore pricing and Greeks engine.

Features:
- Fetch and price options chains
- Calculate Greeks for individual contracts
- Display enriched chain data
- Cache demonstration

Usage:
    python examples/optionscore_demo.py

Note:
    Requires Polygon.io API key in environment variable: POLYGON_API_KEY
    If not available, runs with mock data for demonstration.

Author: Ordinis Project
"""

import asyncio
from datetime import datetime
import os
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.engines.optionscore import OptionsCoreEngine, OptionsEngineConfig
from src.plugins.base import PluginConfig
from src.plugins.market_data.polygon import PolygonDataPlugin


async def demo_with_mock_data():
    """Demo with mock data (no API key required)."""
    print("=" * 80)
    print("OptionsCore Engine Demo - Mock Data Mode")
    print("=" * 80)
    print()

    # Mock Polygon plugin for demo
    from unittest.mock import MagicMock

    mock_polygon = MagicMock()
    mock_polygon.status = MagicMock()
    mock_polygon.status.value = "ready"

    async def mock_get_quote(symbol):
        return {"symbol": symbol, "last": 150.0, "timestamp": datetime.utcnow().isoformat()}

    async def mock_get_options_chain(
        symbol, expiration=None, strike_price=None, contract_type=None
    ):
        return {
            "symbol": symbol,
            "contracts": [
                {
                    "ticker": f"O:{symbol}260117C00145000",
                    "underlying": symbol,
                    "contract_type": "call",
                    "strike_price": 145.0,
                    "expiration_date": "2026-01-17",
                    "shares_per_contract": 100,
                },
                {
                    "ticker": f"O:{symbol}260117C00150000",
                    "underlying": symbol,
                    "contract_type": "call",
                    "strike_price": 150.0,
                    "expiration_date": "2026-01-17",
                    "shares_per_contract": 100,
                },
                {
                    "ticker": f"O:{symbol}260117C00155000",
                    "underlying": symbol,
                    "contract_type": "call",
                    "strike_price": 155.0,
                    "expiration_date": "2026-01-17",
                    "shares_per_contract": 100,
                },
                {
                    "ticker": f"O:{symbol}260117P00145000",
                    "underlying": symbol,
                    "contract_type": "put",
                    "strike_price": 145.0,
                    "expiration_date": "2026-01-17",
                    "shares_per_contract": 100,
                },
                {
                    "ticker": f"O:{symbol}260117P00150000",
                    "underlying": symbol,
                    "contract_type": "put",
                    "strike_price": 150.0,
                    "expiration_date": "2026-01-17",
                    "shares_per_contract": 100,
                },
            ],
            "count": 5,
        }

    mock_polygon.get_quote = mock_get_quote
    mock_polygon.get_options_chain = mock_get_options_chain

    # Create engine configuration
    config = OptionsEngineConfig(
        engine_id="demo_engine",
        cache_ttl_seconds=300,
        default_risk_free_rate=0.05,
        default_dividend_yield=0.0,
    )

    print("Configuration:")
    print(f"  Engine ID: {config.engine_id}")
    print(f"  Cache TTL: {config.cache_ttl_seconds}s")
    print(f"  Risk-free rate: {config.default_risk_free_rate*100:.1f}%")
    print()

    # Create and initialize engine
    engine = OptionsCoreEngine(config, mock_polygon)
    await engine.initialize()
    print("[OK] Engine initialized")
    print()

    # Fetch options chain
    symbol = "AAPL"
    print(f"Fetching options chain for {symbol}...")
    chain = await engine.get_option_chain(symbol)

    print(f"[OK] Retrieved {len(chain.contracts)} contracts")
    print(f"  Underlying price: ${chain.underlying_price:.2f}")
    print(f"  ATM strike: ${chain.summary['atm_strike']:.2f}")
    print(f"  Expirations: {', '.join(chain.expirations)}")
    print()

    # Display contract details
    print("Contract Details:")
    print("-" * 80)
    print(f"{'Strike':<10} {'Type':<6} {'Price':<10} {'Delta':<10} {'Gamma':<10} {'Theta':<10}")
    print("-" * 80)

    for contract in chain.contracts:
        strike = contract.contract.strike
        opt_type = contract.contract.option_type.value
        price = contract.pricing.theoretical_price
        delta = contract.greeks.delta
        gamma = contract.greeks.gamma
        theta = contract.greeks.theta

        print(
            f"${strike:<9.2f} {opt_type:<6} ${price:<9.2f} {delta:<9.4f} {gamma:<9.4f} ${theta:<9.4f}"
        )

    print()

    # Cache demonstration
    print("Cache Statistics:")
    stats = engine.get_cache_stats()
    print(f"  Cached items: {stats['active_items']}")
    print(f"  Cache size: {stats['size_bytes']} bytes")
    print()

    # Fetch again (should hit cache)
    print(f"Fetching {symbol} chain again (should hit cache)...")
    chain2 = await engine.get_option_chain(symbol)
    print(f"[OK] Retrieved from cache: {chain2 is chain}")
    print()

    print("Demo complete!")


async def demo_with_live_data():
    """Demo with live Polygon data (requires API key)."""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("POLYGON_API_KEY not found in environment")
        print("Running mock data demo instead...")
        print()
        await demo_with_mock_data()
        return

    print("=" * 80)
    print("OptionsCore Engine Demo - Live Data Mode")
    print("=" * 80)
    print()

    # Create Polygon plugin
    polygon_config = PluginConfig(name="polygon", api_key=api_key)
    polygon = PolygonDataPlugin(polygon_config)
    await polygon.initialize()

    # Create engine
    config = OptionsEngineConfig(
        engine_id="live_demo",
        cache_ttl_seconds=300,
        default_risk_free_rate=0.05,
    )

    engine = OptionsCoreEngine(config, polygon)
    await engine.initialize()
    print("[OK] Engine initialized with live Polygon data")
    print()

    # Fetch real options chain
    symbol = input("Enter symbol (default: AAPL): ").strip().upper() or "AAPL"
    print(f"\nFetching live options chain for {symbol}...")

    try:
        chain = await engine.get_option_chain(symbol)

        print(f"[OK] Retrieved {len(chain.contracts)} contracts")
        print(f"  Underlying price: ${chain.underlying_price:.2f}")
        print(f"  ATM strike: ${chain.summary['atm_strike']:.2f}")
        print(f"  Total calls: {chain.summary['total_calls']}")
        print(f"  Total puts: {chain.summary['total_puts']}")
        print()

        # Show ATM options
        atm_strike = chain.summary["atm_strike"]
        print(f"At-the-money options (strike ${atm_strike:.2f}):")
        print("-" * 80)

        for contract in chain.contracts:
            if abs(contract.contract.strike - atm_strike) < 0.01:
                opt_type = contract.contract.option_type.value
                price = contract.pricing.theoretical_price
                delta = contract.greeks.delta
                gamma = contract.greeks.gamma
                theta = contract.greeks.theta
                vega = contract.greeks.vega

                print(f"\n{opt_type.upper()}:")
                print(f"  Theoretical price: ${price:.2f}")
                print(f"  Delta: {delta:.4f}")
                print(f"  Gamma: {gamma:.4f}")
                print(f"  Theta: ${theta:.4f}/day")
                print(f"  Vega: ${vega:.4f}/1% IV")

    except Exception as e:
        print(f"Error fetching chain: {e}")

    finally:
        await polygon.shutdown()

    print("\nDemo complete!")


async def main():
    """Main demo entry point."""
    # Check for API key
    api_key = os.getenv("POLYGON_API_KEY")

    if api_key:
        print("Polygon API key found - running live data demo")
        await demo_with_live_data()
    else:
        print("No Polygon API key - running mock data demo")
        await demo_with_mock_data()


if __name__ == "__main__":
    asyncio.run(main())
