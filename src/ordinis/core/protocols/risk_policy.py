"""
RiskPolicy protocol for risk management checks.

Defines contract for pre-trade risk evaluation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from typing import Any


class RiskPolicy(Protocol):
    """
    Defines risk management rules for trading actions.

    Encapsulates:
    - Exposure limits
    - Leverage constraints
    - Drawdown limits
    - Position size limits
    - Sector concentration

    Sync vs Async:
    - Checks are synchronous (fast computations)
    - No network I/O

    Idempotency:
    - Same inputs produce same results
    - No side effects (read-only evaluation)
    - Does not modify portfolio or order
    """

    def approve_signal(self, signal: Any, portfolio: Any) -> bool:
        """
        Evaluate trading signal against risk criteria.

        Pre-trade check before signal becomes an order.

        Args:
            signal: Trading signal from strategy
            portfolio: Current portfolio state

        Returns:
            True if signal is safe to act on
            False if signal should be ignored due to risk

        Note:
            Called before order sizing.
            Checks overall risk appetite, not specific sizing.
        """
        ...

    def approve_order(self, order: Any, portfolio: Any) -> bool:
        """
        Evaluate proposed order against risk limits.

        Final check with specific size and price.

        Args:
            order: Proposed order with quantity
            portfolio: Current portfolio state

        Returns:
            True if order is within risk limits
            False if order violates policy

        Note:
            Checks exposure, leverage, concentration limits.
            Order may be resized rather than rejected by some implementations.
        """
        ...

    def check_portfolio_risk(self, portfolio: Any) -> tuple[bool, list[str]]:
        """
        Check current portfolio against all risk limits.

        Args:
            portfolio: Current portfolio state

        Returns:
            Tuple of (within_limits, list_of_violations)
        """
        ...

    def get_available_risk_budget(self, symbol: str, portfolio: Any) -> dict[str, float]:
        """
        Calculate remaining risk capacity for a symbol.

        Args:
            symbol: Symbol to check
            portfolio: Current portfolio state

        Returns:
            Dictionary with available capacity metrics
        """
        ...
