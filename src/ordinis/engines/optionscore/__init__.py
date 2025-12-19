"""
OptionsCore Engine

Foundational options pricing and Greeks calculation engine for Ordinis.

Provides:
    - Black-Scholes pricing for European options
    - Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
    - Options chain enrichment with theoretical pricing
    - Caching layer for performance optimization
    - Integration with market data providers

Main Classes:
    - OptionsCoreEngine: Main orchestration engine
    - OptionsEngineConfig: Engine configuration
    - BlackScholesEngine: Pricing calculations
    - GreeksCalculator: Greeks calculations
    - OptionContract: Contract data model
    - OptionLeg: Strategy leg model
    - OptionPosition: Position tracking model

Usage:
    >>> from ordinis.engines.optionscore import OptionsCoreEngine, OptionsEngineConfig
    >>>
    >>> # Initialize
    >>> config = OptionsEngineConfig(engine_id="main", cache_ttl_seconds=300)
    >>> engine = OptionsCoreEngine(config, market_data_provider)
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
    "BlackScholesEngine",
    "GreeksCalculator",
    "OptionContract",
    "OptionLeg",
    "OptionPosition",
    "OptionType",
    "OptionsCoreEngine",
    "OptionsEngineConfig",
    "PricingParameters",
]

__version__ = "1.0.0"
