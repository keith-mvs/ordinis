"""
BrokerAdapter protocol for external broker integration.

Abstracts broker-specific operations for order execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    pass


class BrokerAdapter(Protocol):
    """
    Adapter for executing trades via external broker API.

    Abstracts operations like placing/canceling orders.
    Methods are async for non-blocking network I/O.

    Idempotency:
    - cancel_order: Safe to call multiple times
    - submit_order: NOT idempotent (calling twice creates duplicates)
    """

    async def submit_order(self, order: Any) -> dict[str, Any]:
        """
        Place new order with broker.

        Args:
            order: Order to submit

        Returns:
            Response dict with keys:
            - success: bool
            - broker_order_id: str (if success)
            - error: str (if failure)

        Note:
            NOT idempotent - calling twice places duplicate orders.
        """
        ...

    async def cancel_order(self, broker_order_id: str) -> dict[str, Any]:
        """
        Request cancellation of existing order.

        Safe to call multiple times; no effect if already closed.

        Args:
            broker_order_id: Broker order ID to cancel

        Returns:
            Response dict with success/error status
        """
        ...

    async def get_positions(self) -> list[dict[str, Any]]:
        """
        Get current positions from broker.

        Returns:
            List of position dictionaries
        """
        ...

    async def get_account(self) -> dict[str, Any]:
        """
        Get account information.

        Returns:
            Account details (equity, buying power, etc.)
        """
        ...
