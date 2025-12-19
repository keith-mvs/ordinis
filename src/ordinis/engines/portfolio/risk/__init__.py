"""
Risk Attribution - Factor-based risk decomposition.

This module provides detailed attribution of portfolio risk to
factors, securities, sectors, and asset classes.
"""

from ordinis.engines.portfolio.risk.attribution_engine import (
    FactorExposure,
    FactorReturns,
    RiskAttributionEngine,
    RiskAttributionResult,
    RiskFactor,
    SectorAttribution,
    SecurityAttribution,
    create_fama_french_engine,
    create_simple_attribution_engine,
)

__all__ = [
    "FactorExposure",
    "FactorReturns",
    "RiskAttributionEngine",
    "RiskAttributionResult",
    "RiskFactor",
    "SectorAttribution",
    "SecurityAttribution",
    "create_fama_french_engine",
    "create_simple_attribution_engine",
]
