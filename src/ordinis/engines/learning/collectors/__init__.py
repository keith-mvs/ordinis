"""
Learning Engine Data Collectors.

Collectors capture events and feedback from:
- Development: sensitivity analysis, backtests, parameter sweeps
- Trading: trade outcomes, signal accuracy, regime changes
- Models: predictions, drift, feature importance

Key Components:
- FeedbackCollector: Central hub for collecting and routing feedback
- FeedbackRecord: Data structure for feedback items
- FeedbackType: Types of feedback (sensitivity, trade, regime, etc.)
- CircuitBreakerMonitor: Real-time error rate monitoring and trading halt
- CircuitBreakerState: Circuit breaker states (CLOSED, HALF_OPEN, OPEN)
- ErrorWindow: Sliding window for error rate calculation
"""

from .feedback import (
    CircuitBreakerMonitor,
    CircuitBreakerState,
    ErrorWindow,
    FeedbackCollector,
    FeedbackPriority,
    FeedbackRecord,
    FeedbackType,
)

__all__ = [
    "CircuitBreakerMonitor",
    "CircuitBreakerState",
    "ErrorWindow",
    "FeedbackCollector",
    "FeedbackPriority",
    "FeedbackRecord",
    "FeedbackType",
]
