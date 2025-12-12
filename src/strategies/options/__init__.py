"""
Options Trading Strategies

Collection of options-based trading strategies for the Ordinis platform.

Strategies:
    - CoveredCallStrategy: Income generation via selling calls against stock

Author: Ordinis Project
License: MIT
"""

from .covered_call import CoveredCallStrategy

__all__ = ["CoveredCallStrategy"]
