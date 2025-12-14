"""
SignalCore ML Engine - Numerical Signal Generation.

Generates quantitative trade signals using explicit, testable numerical models.
All signals are probabilistic assessments, not direct orders.

The engine follows the standard Ordinis engine template with:
- core/ - Engine, config, and domain models
- hooks/ - Governance hooks for preflight/audit
- models/ - Signal model implementations
- features/ - Feature engineering
"""

# Core engine components
from ordinis.engines.signalcore.core import (
    Direction,
    Model,
    ModelConfig,
    ModelRegistry,
    Signal,
    SignalBatch,
    SignalCoreEngine,
    SignalCoreEngineConfig,
    SignalType,
)

# Feature engineering
from ordinis.engines.signalcore.features.technical import TechnicalIndicators

# Governance hooks
from ordinis.engines.signalcore.hooks import (
    DataQualityRule,
    ModelValidationRule,
    SignalCoreGovernanceHook,
    SignalThresholdRule,
)

# Model implementations
from ordinis.engines.signalcore.models.rsi_mean_reversion import RSIMeanReversionModel
from ordinis.engines.signalcore.models.sma_crossover import SMACrossoverModel

__all__ = [
    # Governance Hooks
    "DataQualityRule",
    # Core types
    "Direction",
    "Model",
    "ModelConfig",
    "ModelRegistry",
    "ModelValidationRule",
    # Models
    "RSIMeanReversionModel",
    "SMACrossoverModel",
    "Signal",
    "SignalBatch",
    # Core Engine
    "SignalCoreEngine",
    "SignalCoreEngineConfig",
    "SignalCoreGovernanceHook",
    "SignalThresholdRule",
    "SignalType",
    # Features
    "TechnicalIndicators",
]
