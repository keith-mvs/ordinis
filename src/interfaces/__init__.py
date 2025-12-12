"""
DEPRECATED: Use core.protocols instead.

This module re-exports protocols from core.protocols for backward compatibility.
Will be removed in a future version.
"""

from __future__ import annotations

import warnings as _warnings

# Re-export from new location for backward compatibility
from core.protocols import (
    BrokerAdapter,
    CostModel,
    Event,
    EventBus,
    ExecutionEngine,
    FillModel,
    RiskPolicy,
)

# Emit deprecation warning on import
_warnings.warn(
    "interfaces module is deprecated. Use core.protocols instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BrokerAdapter",
    "CostModel",
    "Event",
    "EventBus",
    "ExecutionEngine",
    "FillModel",
    "RiskPolicy",
]
