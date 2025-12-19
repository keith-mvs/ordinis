"""
Portfolio State Management.

Provides transactional state management for PortfolioEngine with:
- Immutable state snapshots
- Atomic updates with commit/rollback
- Optimistic locking for concurrent access
- State validation and integrity checks
"""

from ordinis.engines.portfolio.state.portfolio_state import (
    OptimisticLockError,
    PortfolioStateManager,
    PortfolioStateSnapshot,
    PositionSnapshot,
    StateChange,
    StateValidationError,
    TransactionState,
)

__all__ = [
    "OptimisticLockError",
    "PortfolioStateManager",
    "PortfolioStateSnapshot",
    "PositionSnapshot",
    "StateChange",
    "StateValidationError",
    "TransactionState",
]
