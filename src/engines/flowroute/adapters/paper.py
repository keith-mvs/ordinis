"""
Paper trading broker adapter.

Simulates order execution for testing and paper trading.
"""

from datetime import datetime
from typing import Any
import uuid

from ..core.engine import BrokerAdapter
from ..core.orders import Fill, Order


class PaperBrokerAdapter(BrokerAdapter):
    """
    Paper trading broker adapter.

    Simulates realistic order execution without actual broker connection.
    """

    def __init__(
        self,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
        fill_delay_ms: float = 100.0,
    ):
        """
        Initialize paper broker.

        Args:
            slippage_bps: Simulated slippage in basis points
            commission_per_share: Commission per share
            fill_delay_ms: Simulated fill latency in milliseconds
        """
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.fill_delay_ms = fill_delay_ms

        self._positions: dict[str, dict[str, Any]] = {}
        self._cash: float = 100000.0  # Starting cash
        self._fills: list[Fill] = []

    async def submit_order(self, order: Order) -> dict[str, Any]:
        """
        Simulate order submission.

        Args:
            order: Order to submit

        Returns:
            Response dict with success/error
        """
        # Generate broker order ID
        broker_order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"

        # Simulate order acceptance
        return {
            "success": True,
            "broker_order_id": broker_order_id,
            "status": "accepted",
            "message": "Order accepted for paper trading",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def cancel_order(self, broker_order_id: str) -> dict[str, Any]:
        """
        Simulate order cancellation.

        Args:
            broker_order_id: Broker's order ID

        Returns:
            Response dict with success/error
        """
        return {
            "success": True,
            "broker_order_id": broker_order_id,
            "status": "cancelled",
            "message": "Order cancelled in paper trading",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_positions(self) -> list[dict[str, Any]]:
        """
        Get current paper trading positions.

        Returns:
            List of position dicts
        """
        return [
            {
                "symbol": symbol,
                **position,
            }
            for symbol, position in self._positions.items()
        ]

    async def get_account(self) -> dict[str, Any]:
        """
        Get paper trading account information.

        Returns:
            Account info dict
        """
        total_position_value = sum(
            pos.get("quantity", 0) * pos.get("current_price", 0) for pos in self._positions.values()
        )

        return {
            "cash": self._cash,
            "total_position_value": total_position_value,
            "total_equity": self._cash + total_position_value,
            "positions": len(self._positions),
            "buying_power": self._cash,
        }

    def simulate_fill(
        self, order: Order, fill_price: float, current_price: float | None = None
    ) -> Fill:
        """
        Simulate order fill.

        Args:
            order: Order to fill
            fill_price: Fill price
            current_price: Current market price (for slippage calc)

        Returns:
            Fill object
        """
        if current_price is None:
            current_price = fill_price

        # Calculate slippage
        if order.side == "buy":
            expected_price = current_price * (1 + self.slippage_bps / 10000)
            slippage_bps = ((fill_price - current_price) / current_price) * 10000
        else:  # sell
            expected_price = current_price * (1 - self.slippage_bps / 10000)
            slippage_bps = ((current_price - fill_price) / current_price) * 10000

        # Calculate commission
        commission = order.quantity * self.commission_per_share

        # Create fill
        fill = Fill(
            fill_id=f"FILL-{uuid.uuid4().hex[:8].upper()}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            timestamp=datetime.utcnow(),
            latency_ms=self.fill_delay_ms,
            slippage_bps=slippage_bps,
            vs_arrival_bps=slippage_bps,  # For paper trading, same as slippage
        )

        self._fills.append(fill)

        # Update positions
        self._update_position(order.symbol, order.side, order.quantity, fill_price, commission)

        return fill

    def _update_position(
        self, symbol: str, side: str, quantity: int, price: float, commission: float
    ) -> None:
        """Update position after fill."""
        if symbol not in self._positions:
            self._positions[symbol] = {
                "quantity": 0,
                "avg_price": 0.0,
                "current_price": price,
                "unrealized_pnl": 0.0,
            }

        position = self._positions[symbol]

        if side == "buy":
            # Add to position
            total_cost = (position["quantity"] * position["avg_price"]) + (quantity * price)
            position["quantity"] += quantity
            position["avg_price"] = (
                total_cost / position["quantity"] if position["quantity"] > 0 else 0.0
            )
            self._cash -= (quantity * price) + commission

        else:  # sell
            # Reduce position
            position["quantity"] -= quantity
            self._cash += (quantity * price) - commission

            if position["quantity"] == 0:
                # Position closed
                del self._positions[symbol]

        if symbol in self._positions:
            position["current_price"] = price
            position["unrealized_pnl"] = position["quantity"] * (price - position["avg_price"])

    def reset(self, initial_cash: float = 100000.0) -> None:
        """Reset paper trading state."""
        self._positions = {}
        self._cash = initial_cash
        self._fills = []

    def get_fills(self) -> list[Fill]:
        """Get all fills."""
        return self._fills.copy()
