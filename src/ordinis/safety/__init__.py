"""
Safety module for Ordinis live trading.

Provides:
- KillSwitch: Emergency stop mechanism with multiple trigger sources
- CircuitBreaker: API connectivity monitoring with automatic degradation

Safety components are designed to fail-safe: if uncertain, halt trading.
"""

from ordinis.safety.circuit_breaker import CircuitBreaker, CircuitState
from ordinis.safety.kill_switch import KillSwitch, KillSwitchReason

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "KillSwitch",
    "KillSwitchReason",
]
