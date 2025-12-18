"""
Options Trading Strategies

Collection of options-based trading strategies for the Ordinis platform.

Strategies:
    - CoveredCallStrategy: Income generation via selling calls against stock
    - IronCondorStrategy: Neutral premium collection with defined risk
    - MarriedPutStrategy: Protective put for downside protection

Author: Ordinis Project
License: MIT
"""

from .covered_call import CoveredCallStrategy
from .iron_condor import IronCondorAnalysis, IronCondorLegs, IronCondorStrategy
from .married_put import MarriedPutStrategy

__all__ = [
    "CoveredCallStrategy",
    "IronCondorStrategy",
    "IronCondorAnalysis",
    "IronCondorLegs",
    "MarriedPutStrategy",
]
