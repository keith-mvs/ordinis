"""
Application layer for Ordinis trading system.

Contains:
- services/: Application services (orchestration, coordination)
- strategies/: Trading strategy implementations
"""

# NOTE: Avoid importing services at package import time to prevent heavy optional
# dependencies (e.g., aiosqlite) from being required for light-weight tools.
# Import strategy implementations directly when needed.
from ordinis.application.strategies import (
    ADXFilteredRSIStrategy,
    BaseStrategy,
    BollingerBandsStrategy,
    FibonacciADXStrategy,
    MACDStrategy,
    MomentumBreakoutStrategy,
    MovingAverageCrossoverStrategy,
    ParabolicSARStrategy,
    RSIMeanReversionStrategy,
)

__all__ = [
    # Strategies
    "ADXFilteredRSIStrategy",
    "BaseStrategy",
    "BollingerBandsStrategy",
    "FibonacciADXStrategy",
    "MACDStrategy",
    "MomentumBreakoutStrategy",
    "MovingAverageCrossoverStrategy",
    # Services (import explicitly in components that need them)
    "ParabolicSARStrategy",
    "PositionReconciliation",
    "RSIMeanReversionStrategy",
]

