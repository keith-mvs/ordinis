"""
Pattern Recognition.

Chart and candlestick pattern utilities for discretionary signals
and level-based trading.
"""

from .breakout import BreakoutDetector, BreakoutSignal
from .candlestick import CandlestickPatterns, PatternMatch
from .support_resistance import SupportResistanceLevels, SupportResistanceLocator

__all__ = [
    "CandlestickPatterns",
    "PatternMatch",
    "SupportResistanceLocator",
    "SupportResistanceLevels",
    "BreakoutDetector",
    "BreakoutSignal",
]
