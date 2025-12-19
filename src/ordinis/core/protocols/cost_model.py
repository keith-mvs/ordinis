"""
CostModel protocol for transaction cost calculation.

Calculates commissions, fees, and slippage costs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from typing import Any


class CostModel(Protocol):
    """
    Defines how trading costs are calculated.

    Includes:
    - Commissions (per share, per trade, tiered)
    - Exchange fees
    - SEC fees
    - Slippage cost (if not in FillModel)

    Statelessness:
    - Calculations should be pure (no internal state)
    - Depends only on input parameters
    - Returns consistent results for same inputs

    Units:
    - Returns cost in absolute currency terms
    - Same currency as trade P&L
    """

    def calculate_cost(self, order: Any, fill: Any) -> float:
        """
        Compute transaction cost for a fill.

        Args:
            order: Original order
            fill: Fill event with quantity and price

        Returns:
            Cost value in trade currency (positive = cost)

        Note:
            Should not modify any input state.
            Returns 0 if no cost applicable.
        """
        ...

    def estimate_cost(self, order: Any, estimated_price: float) -> float:
        """
        Estimate cost before execution.

        Args:
            order: Proposed order
            estimated_price: Expected fill price

        Returns:
            Estimated cost in trade currency
        """
        ...

    def get_fee_schedule(self) -> dict[str, float]:
        """
        Get current fee schedule.

        Returns:
            Dictionary of fee types and rates
        """
        ...
