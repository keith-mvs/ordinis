"""
OptionsCore Engine Core Module

Core orchestration, configuration, and enrichment for options analytics.

Exports:
    - OptionsCoreEngine: Main engine orchestrator
    - OptionsEngineConfig: Engine configuration
    - ChainEnrichmentEngine: Chain data enrichment
    - PricingResult: Pricing calculation result
    - GreeksResult: Greeks calculation result
    - EnrichedContract: Enriched contract data
    - EnrichedOptionsChain: Enriched chain data

Author: Ordinis Project
License: MIT
"""

from .config import OptionsEngineConfig
from .engine import OptionsCoreEngine
from .enrichment import (
    ChainEnrichmentEngine,
    EnrichedContract,
    EnrichedOptionsChain,
    GreeksResult,
    PricingResult,
)

__all__ = [
    "OptionsEngineConfig",
    "OptionsCoreEngine",
    "ChainEnrichmentEngine",
    "PricingResult",
    "GreeksResult",
    "EnrichedContract",
    "EnrichedOptionsChain",
]
