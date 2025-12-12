"""
BrokerAdapter protocol for external broker integration.

Abstracts broker-specific operations for order execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from typing import Any


class BrokerAdapter(Protocol):
    """
    Adapter for executing trades via external broker API.

    Abstracts operations like placing/canceling orders.
    Methods are async for non-blocking network I/O.

    Idempotency:
    - cancel_order: Safe to call multiple times
    - submit_order: NOT idempotent (calling twice creates duplicates)
    """

    async def submit_order(self, order: Any) -> str:
        """
        Place new order with broker.

        Args:
            order: Order to submit

        Returns:
            Broker-assigned order ID

        Raises:
            BrokerError: If submission fails

        Note:
            NOT idempotent - calling twice places duplicate orders.
        """
        ...

    async def cancel_order(self, order_id: str) -> None:
        """
        Request cancellation of existing order.

        Safe to call multiple times; no effect if already closed.

        Args:
            order_id: Broker order ID to cancel
        """
        ...

    async def get_order_status(self, order_id: str) -> Any:
        """
        Retrieve current order status.

        Args:
            order_id: Broker order ID

        Returns:
            Order status (pending, filled, cancelled, etc.)
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
