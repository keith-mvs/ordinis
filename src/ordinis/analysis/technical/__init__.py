"""
Technical Analysis Framework.

Studies past market data (price/volume) to predict future price movements.
Core belief: All relevant information is reflected in price, and historical
patterns tend to repeat due to market psychology.

Modules:
- indicators/: Mathematical calculations on price/volume data
  - moving_averages: SMA, EMA, WMA, VWAP
  - oscillators: RSI, Stochastic, CCI, Williams %R
  - volatility: ATR, Bollinger Bands, Keltner Channels
  - volume: OBV, Volume Profile, MFI
  - trend: ADX, MACD, Parabolic SAR

- patterns/: Visual formations on charts
  - chart_patterns: Head & Shoulders, Double Top/Bottom, Triangles
  - candlestick: Doji, Hammer, Engulfing, etc.
  - fibonacci: Retracement, Extensions, Time Zones

- analysis/: Higher-level analysis
  - trend_analysis: Direction, strength, support/resistance
"""

from .composite import CompositeIndicator, CompositeResult
from .indicators import (
    MovingAverages,
    Oscillators,
    TechnicalIndicators,
    TrendIndicators,
    VolatilityIndicators,
    VolumeIndicators,
)
from .multi_timeframe import MultiTimeframeAnalyzer, MultiTimeframeResult, TimeframeSignal
from .patterns import (
    BreakoutDetector,
    BreakoutSignal,
    CandlestickPatterns,
    PatternMatch,
    SupportResistanceLevels,
    SupportResistanceLocator,
)

# ChartPatterns and FibonacciAnalysis not yet implemented
# try:
#     from .patterns import ChartPatterns, FibonacciAnalysis
# except ImportError:
#     ChartPatterns = None
#     FibonacciAnalysis = None
try:
    from .trend_analysis import TrendAnalysis
except ImportError:
    TrendAnalysis = None

__all__ = [
    "TechnicalIndicators",
    "MovingAverages",
    "Oscillators",
    "VolumeIndicators",
    "VolatilityIndicators",
    "TrendIndicators",
    "ChartPatterns",
    "CandlestickPatterns",
    "FibonacciAnalysis",
    "TrendAnalysis",
    "SupportResistanceLocator",
    "SupportResistanceLevels",
    "BreakoutDetector",
    "BreakoutSignal",
    "PatternMatch",
    "MultiTimeframeAnalyzer",
    "MultiTimeframeResult",
    "TimeframeSignal",
    "CompositeIndicator",
    "CompositeResult",
]
