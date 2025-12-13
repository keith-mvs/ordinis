"""
SignalCore ML Engine - Numerical Signal Generation.

Generates quantitative trade signals using explicit, testable numerical models.
All signals are probabilistic assessments, not direct orders.
"""

from .core.model import Model, ModelConfig, ModelRegistry
from .core.signal import Direction, Signal, SignalBatch, SignalType
from .features.technical import TechnicalIndicators
from .models.rsi_mean_reversion import RSIMeanReversionModel
from .models.sma_crossover import SMACrossoverModel

__all__ = [
    # Core types
    "Signal",
    "SignalBatch",
    "SignalType",
    "Direction",
    # Model framework
    "Model",
    "ModelConfig",
    "ModelRegistry",
    # Models
    "SMACrossoverModel",
    "RSIMeanReversionModel",
    # Features
    "TechnicalIndicators",
]
