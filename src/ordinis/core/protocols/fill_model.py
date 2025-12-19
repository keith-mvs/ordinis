"""
FillModel protocol for order fill simulation.

Encapsulates logic for simulating fills in backtest environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from typing import Any


class FillModel(Protocol):
    """
    Strategy for simulating order fills.

    Different implementations model varying market conditions:
    - Instant fill at market price
    - Partial fills based on volume
    - Slippage impact based on order size
    - Queue simulation with L2 data

    Determinism:
    - Methods should be pure functions (no side effects)
    - Same inputs produce same outputs
    - State about remaining quantity tracked externally

    Mutability:
    - Should not modify Order object
    - Returns fill result as new object
    """

    def simulate_fill(self, order: Any, market_data: Any) -> Any:
        """
        Determine how order would be filled given market conditions.

        Args:
            order: Order to simulate fill for
            market_data: Current market snapshot (price, volume, spread)

        Returns:
            FillEvent with executed quantity, price, and remaining

        Note:
            Does not modify Order; just computes outcome.
            If order not fully filled, remaining quantity stays open.
        """
        ...

    def get_fill_probability(self, order: Any, market_data: Any) -> float:
        """
        Estimate probability of order being filled.

        Args:
            order: Order to evaluate
            market_data: Current market conditions

        Returns:
            Probability 0.0 to 1.0
        """
        ...
