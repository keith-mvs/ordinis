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

from .indicators import (
    MovingAverages,
    Oscillators,
    TechnicalIndicators,
    VolatilityIndicators,
    VolumeIndicators,
)
from .patterns import (
    CandlestickPatterns,
    ChartPatterns,
    FibonacciAnalysis,
)
from .trend_analysis import TrendAnalysis

__all__ = [
    "TechnicalIndicators",
    "MovingAverages",
    "Oscillators",
    "VolumeIndicators",
    "VolatilityIndicators",
    "ChartPatterns",
    "CandlestickPatterns",
    "FibonacciAnalysis",
    "TrendAnalysis",
]
