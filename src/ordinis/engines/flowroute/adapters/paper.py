"""
Paper trading broker adapter.

Simulates order execution for testing and paper trading.
"""

import asyncio
from datetime import datetime
import logging
from types import SimpleNamespace
from typing import Any
import uuid

from ordinis.core.protocols import BrokerAdapter
from ordinis.domain.orders import Fill, Order

logger = logging.getLogger(__name__)


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
        market_data_plugin=None,
        price_cache_seconds: float = 1.0,
    ):
        """
        Initialize paper broker.

        Args:
            slippage_bps: Simulated slippage in basis points
            commission_per_share: Commission per share
            fill_delay_ms: Simulated fill latency in milliseconds
            market_data_plugin: Plugin for fetching real-time prices
            price_cache_seconds: How long to cache prices
        """
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.fill_delay_ms = fill_delay_ms
        self.market_data_plugin = market_data_plugin
        self.price_cache_seconds = price_cache_seconds

        self._positions: dict[str, dict[str, Any]] = {}
        self._cash: float = 100000.0  # Starting cash
        self._fills: list[Fill] = []
        self._pending_orders: dict[str, Order] = {}
        self._price_cache: dict[str, dict[str, Any]] = {}

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

        # Store order as pending
        self._pending_orders[broker_order_id] = order

        # If market data available, auto-fill immediately
        if self.market_data_plugin:
            try:
                price_data = await self._fetch_current_price(order.symbol)
                if price_data:
                    # Use appropriate price based on side
                    fill_price = price_data["ask"] if order.side == "BUY" else price_data["bid"]
                    if fill_price and fill_price > 0:
                        # Simulate fill with delay
                        await asyncio.sleep(self.fill_delay_ms / 1000.0)
                        fill = self.simulate_fill(order, fill_price, price_data["last"])
                        # Remove from pending
                        self._pending_orders.pop(broker_order_id, None)
                        return {
                            "success": True,
                            "broker_order_id": broker_order_id,
                            "status": "filled",
                            "fill": {
                                "fill_id": fill.fill_id,
                                "price": fill.price,
                                "quantity": fill.quantity,
                                "commission": fill.commission,
                            },
                            "message": "Order filled in paper trading",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
            except Exception as e:
                # If price fetch fails, keep order pending
                logger.debug("Failed to fetch price for auto-fill: %s", e)

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
        # Remove from pending if exists
        self._pending_orders.pop(broker_order_id, None)

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

    async def get_account(self) -> SimpleNamespace:
        """
        Get paper trading account information.

        Returns:
            Account info object with equity, cash, buying_power, portfolio_value
        """
        total_position_value = sum(
            pos.get("quantity", 0) * pos.get("current_price", 0) for pos in self._positions.values()
        )
        total_equity = self._cash + total_position_value

        return SimpleNamespace(
            cash=self._cash,
            equity=total_equity,
            total_equity=total_equity,
            buying_power=self._cash,
            portfolio_value=total_position_value,
            total_position_value=total_position_value,
            positions=len(self._positions),
        )

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
        if order.side == "BUY":
            expected_price = current_price * (1 + self.slippage_bps / 10000)
            slippage_bps = ((fill_price - current_price) / current_price) * 10000
        else:  # SELL
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

        if side == "BUY":
            # Add to position
            total_cost = (position["quantity"] * position["avg_price"]) + (quantity * price)
            position["quantity"] += quantity
            position["avg_price"] = (
                total_cost / position["quantity"] if position["quantity"] > 0 else 0.0
            )
            self._cash -= (quantity * price) + commission

        else:  # SELL
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
        self._pending_orders = {}
        self._price_cache = {}

    def get_fills(self) -> list[Fill]:
        """Get all fills."""
        return self._fills.copy()

    def get_pending_orders(self) -> list[dict[str, Any]]:
        """Get all pending orders."""
        return [
            {
                "broker_order_id": broker_id,
                "order": order,
            }
            for broker_id, order in self._pending_orders.items()
        ]

    async def _fetch_current_price(self, symbol: str) -> dict[str, Any] | None:
        """
        Fetch current price from market data plugin with caching.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with bid, ask, last prices or None
        """
        # Check cache
        if symbol in self._price_cache:
            cached = self._price_cache[symbol]
            age = (datetime.utcnow() - cached["timestamp"]).total_seconds()
            if age < self.price_cache_seconds:
                return cached["data"]

        # Fetch new price
        if not self.market_data_plugin:
            return None

        try:
            quote = await self.market_data_plugin.get_quote(symbol)
            price_data = {
                "symbol": symbol,
                "bid": quote.get("bid"),
                "ask": quote.get("ask"),
                "last": quote.get("last"),
            }

            # Update cache
            self._price_cache[symbol] = {
                "data": price_data,
                "timestamp": datetime.utcnow(),
            }

            return price_data

        except Exception:
            return None

    async def process_pending_orders(self) -> list[Fill]:
        """
        Process pending orders and attempt to fill them.

        Returns:
            List of new fills
        """
        fills = []
        orders_to_remove = []

        for broker_order_id, order in self._pending_orders.items():
            try:
                price_data = await self._fetch_current_price(order.symbol)
                if price_data:
                    # Use bid/ask if available, otherwise fallback to last price with slippage
                    fill_price = price_data["ask"] if order.side == "BUY" else price_data["bid"]

                    if not fill_price or fill_price <= 0:
                        # Fallback to last price with slippage simulation
                        last_price = price_data.get("last", 0)
                        if last_price > 0:
                            slippage_factor = self.slippage_bps / 10000.0
                            fill_price = (
                                last_price * (1 + slippage_factor)
                                if order.side == "BUY"
                                else last_price * (1 - slippage_factor)
                            )

                    if fill_price and fill_price > 0:
                        fill = self.simulate_fill(order, fill_price, price_data["last"])
                        fills.append(fill)
                        orders_to_remove.append(broker_order_id)
            except Exception as e:
                logger.debug("Failed to process pending order %s: %s", broker_order_id, e)
                continue

        # Remove filled orders
        for broker_order_id in orders_to_remove:
            self._pending_orders.pop(broker_order_id, None)

        return fills

    async def run_paper_trading_loop(
        self, symbols: list[str], interval_seconds: float = 1.0, max_iterations: int | None = None
    ) -> None:
        """
        Run paper trading loop that continuously processes orders.

        Args:
            symbols: Symbols to monitor
            interval_seconds: Update interval
            max_iterations: Max iterations (None = infinite)
        """
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            # Process pending orders
            await self.process_pending_orders()

            # Update position prices
            for symbol in symbols:
                if symbol in self._positions:
                    try:
                        price_data = await self._fetch_current_price(symbol)
                        if price_data and price_data["last"]:
                            pos = self._positions[symbol]
                            pos["current_price"] = price_data["last"]
                            pos["unrealized_pnl"] = pos["quantity"] * (
                                price_data["last"] - pos["avg_price"]
                            )
                    except Exception as e:
                        logger.debug("Failed to update position price for %s: %s", symbol, e)
                        continue

            await asyncio.sleep(interval_seconds)
            iteration += 1
