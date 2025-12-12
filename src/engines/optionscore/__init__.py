"""
OptionsCore Engine

Foundational options pricing and Greeks calculation engine for Ordinis.

Provides:
    - Black-Scholes pricing for European options
    - Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
    - Options chain enrichment with theoretical pricing
    - Caching layer for performance optimization
    - Integration with Polygon.io market data

Main Classes:
    - OptionsCoreEngine: Main orchestration engine
    - OptionsEngineConfig: Engine configuration
    - BlackScholesEngine: Pricing calculations
    - GreeksCalculator: Greeks calculations
    - OptionContract: Contract data model
    - OptionLeg: Strategy leg model
    - OptionPosition: Position tracking model

Usage:
    >>> from src.engines.optionscore import OptionsCoreEngine, OptionsEngineConfig
    >>> from src.plugins.market_data.polygon import PolygonDataPlugin
    >>>
    >>> # Initialize
    >>> config = OptionsEngineConfig(engine_id="main", cache_ttl_seconds=300)
    >>> engine = OptionsCoreEngine(config, polygon_plugin)
    >>> await engine.initialize()
    >>>
    >>> # Fetch enriched chain
    >>> chain = await engine.get_option_chain("AAPL")
    >>> print(f"Found {len(chain.contracts)} contracts")

Author: Ordinis Project
License: MIT
"""

from .core import OptionsCoreEngine, OptionsEngineConfig
from .data import OptionContract, OptionLeg, OptionPosition, OptionType
from .pricing import BlackScholesEngine, GreeksCalculator, PricingParameters

__all__ = [
    "OptionsCoreEngine",
    "OptionsEngineConfig",
    "BlackScholesEngine",
    "GreeksCalculator",
    "PricingParameters",
    "OptionContract",
    "OptionLeg",
    "OptionPosition",
    "OptionType",
]

__version__ = "1.0.0"
