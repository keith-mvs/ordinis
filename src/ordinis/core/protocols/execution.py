"""
ExecutionEngine protocol for order processing.

Abstracts order execution for backtest vs live environments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from typing import Any


class ExecutionEngine(Protocol):
    """
    Engine that processes orders and produces fills.

    In backtesting: simulates fills using FillModel
    In live trading: routes orders to BrokerAdapter

    Thread-safety: Implementations should handle concurrent order submission.

    Idempotency:
    - execute_order: NOT idempotent (each call is a distinct request)
    - cancel_order: Idempotent (no effect if already filled/cancelled)
    """

    def execute_order(self, order: Any) -> None:
        """
        Submit order for execution.

        In backtesting: queues order for simulation
        In live trading: forwards to broker

        Non-blocking: returns immediately.
        Fills reported via events or callbacks.

        Args:
            order: Order to execute
        """
        ...

    def cancel_order(self, order_id: str) -> None:
        """
        Cancel previously submitted order.

        In live trading: forwards to broker
        In simulation: removes from pending queue

        Idempotent: does nothing if already filled/cancelled.

        Args:
            order_id: Order ID to cancel
        """
        ...

    async def execute_order_async(self, order: Any) -> tuple[bool, str]:
        """
        Async order execution with result.

        Args:
            order: Order to execute

        Returns:
            Tuple of (success, message)
        """
        ...

    async def cancel_order_async(self, order_id: str, reason: str = "") -> tuple[bool, str]:
        """
        Async order cancellation with result.

        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason

        Returns:
            Tuple of (success, message)
        """
        ...
