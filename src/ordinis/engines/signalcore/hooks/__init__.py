"""
SignalCore engine governance hooks.

Provides governance checks for signal generation operations.
"""

from ordinis.engines.signalcore.hooks.governance import (
    DataQualityRule,
    ModelValidationRule,
    SignalCoreGovernanceHook,
    SignalThresholdRule,
)
from ordinis.engines.signalcore.hooks.news import (
    NewsContextHook,
    NewsItem,
    NewsSentiment,
)
from ordinis.engines.signalcore.hooks.regime import (
    MarketRegime,
    RegimeHook,
    RegimeState,
)

__all__ = [
    "DataQualityRule",
    "MarketRegime",
    "ModelValidationRule",
    "NewsContextHook",
    "NewsItem",
    "NewsSentiment",
    "RegimeHook",
    "RegimeState",
    "SignalCoreGovernanceHook",
    "SignalThresholdRule",
]
