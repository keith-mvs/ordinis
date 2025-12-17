"""
Pattern Recognition.

Chart and candlestick pattern utilities for discretionary signals
and level-based trading.
"""

from .breakout import BreakoutDetector, BreakoutSignal
from .candlestick import CandlestickPatterns, PatternMatch
from .support_resistance import SupportResistanceLevels, SupportResistanceLocator

__all__ = [
    "BreakoutDetector",
    "BreakoutSignal",
    "CandlestickPatterns",
    "PatternMatch",
    "SupportResistanceLevels",
    "SupportResistanceLocator",
]
