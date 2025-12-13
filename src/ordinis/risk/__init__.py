"""
Risk management module for Ordinis trading system.

Provides position sizing, risk metrics, and drawdown management.
"""

from .metrics import PerformanceMetrics, RiskMetrics

# Position sizing submodule - import on demand
# Usage: from ordinis.risk.position_sizing import KellyCriterion, etc.

__all__ = [
    # Metrics
    "RiskMetrics",
    "PerformanceMetrics",
]
