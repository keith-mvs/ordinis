"""
Broker Adapters for Paper and Live Trading.

Supports:
- Alpaca (paper and live)
- Simulated broker for backtesting
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status - includes all Alpaca API statuses."""

    # Initial states
    NEW = "new"
    PENDING_NEW = "pending_new"
    ACCEPTED = "accepted"
    ACCEPTED_FOR_BIDDING = "accepted_for_bidding"

    # Fill states
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"

    # Terminal states
    CANCELLED = "cancelled"
    CANCELED = "canceled"  # Alpaca uses American spelling
    REJECTED = "rejected"
    EXPIRED = "expired"
    REPLACED = "replaced"
    DONE_FOR_DAY = "done_for_day"

    # Pending states
    PENDING = "pending"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"

    # Other states
    STOPPED = "stopped"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"


class PositionSide(Enum):
    """Position side."""

    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """Order representation."""

    symbol: str
    side: OrderSide
    quantity: Decimal | int | float
    order_type: OrderType = OrderType.MARKET
    id: str = ""
    status: OrderStatus = OrderStatus.NEW
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    filled_quantity: Decimal = Decimal("0")
    filled_avg_price: Decimal | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Convert quantity to Decimal."""
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))


@dataclass
class Position:
    """Position representation."""

    symbol: str
    side: PositionSide
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal


@dataclass
class AccountInfo:
    """Account information."""

    account_id: str
    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    portfolio_value: Decimal
    is_paper: bool = True


class BrokerAdapter(ABC):
    """Abstract broker adapter interface."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker."""

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Get account information."""

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all open positions."""

    @abstractmethod
    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a specific symbol."""

    @abstractmethod
    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        """Submit an order."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""

    @abstractmethod
    async def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""

    @abstractmethod
    async def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        """Get orders, optionally filtered by status."""


class AlpacaBroker(BrokerAdapter):
    """
    Alpaca broker adapter for paper and live trading.

    Requires environment variables:
    - APCA_API_KEY_ID
    - APCA_API_SECRET_KEY
    - APCA_API_BASE_URL (paper: https://paper-api.alpaca.markets)
    """

    def __init__(self, paper: bool = True):
        """Initialize Alpaca broker."""
        self.paper = paper
        self._api = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            import os

            import alpaca_trade_api as tradeapi

            base_url = os.environ.get(
                "APCA_API_BASE_URL",
                "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets",
            )

            self._api = tradeapi.REST(
                key_id=os.environ.get("APCA_API_KEY_ID"),
                secret_key=os.environ.get("APCA_API_SECRET_KEY"),
                base_url=base_url,
            )

            # Test connection
            account = self._api.get_account()
            logger.info(f"Connected to Alpaca {'paper' if self.paper else 'live'}: {account.id}")
            self._connected = True
            return True

        except ImportError:
            logger.error("alpaca-trade-api not installed. Run: pip install alpaca-trade-api")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        self._api = None
        self._connected = False

    async def get_account(self) -> AccountInfo:
        """Get Alpaca account info."""
        if not self._connected:
            raise RuntimeError("Not connected to Alpaca")

        account = self._api.get_account()

        return AccountInfo(
            account_id=account.id,
            equity=Decimal(account.equity),
            cash=Decimal(account.cash),
            buying_power=Decimal(account.buying_power),
            portfolio_value=Decimal(account.portfolio_value),
            is_paper=self.paper,
        )

    async def get_positions(self) -> list[Position]:
        """Get all Alpaca positions."""
        if not self._connected:
            raise RuntimeError("Not connected to Alpaca")

        positions = self._api.list_positions()

        return [
            Position(
                symbol=p.symbol,
                side=PositionSide.LONG if float(p.qty) > 0 else PositionSide.SHORT,
                quantity=Decimal(p.qty),
                avg_entry_price=Decimal(p.avg_entry_price),
                current_price=Decimal(p.current_price),
                market_value=Decimal(p.market_value),
                unrealized_pnl=Decimal(p.unrealized_pl),
                unrealized_pnl_pct=Decimal(p.unrealized_plpc) * 100,
            )
            for p in positions
        ]

    async def get_position(self, symbol: str) -> Position | None:
        """Get specific position."""
        if not self._connected:
            raise RuntimeError("Not connected to Alpaca")

        try:
            p = self._api.get_position(symbol)
            return Position(
                symbol=p.symbol,
                side=PositionSide.LONG if float(p.qty) > 0 else PositionSide.SHORT,
                quantity=Decimal(p.qty),
                avg_entry_price=Decimal(p.avg_entry_price),
                current_price=Decimal(p.current_price),
                market_value=Decimal(p.market_value),
                unrealized_pnl=Decimal(p.unrealized_pl),
                unrealized_pnl_pct=Decimal(p.unrealized_plpc) * 100,
            )
        except Exception:
            return None

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        """Submit order to Alpaca."""
        if not self._connected:
            raise RuntimeError("Not connected to Alpaca")

        order = self._api.submit_order(
            symbol=symbol,
            qty=str(quantity),
            side=side.value,
            type=order_type.value,
            time_in_force="day",
            limit_price=str(limit_price) if limit_price else None,
            stop_price=str(stop_price) if stop_price else None,
        )

        return Order(
            id=order.id,
            symbol=order.symbol,
            side=OrderSide(order.side),
            order_type=OrderType(order.type),
            quantity=Decimal(order.qty),
            status=OrderStatus(order.status),
            limit_price=Decimal(order.limit_price) if order.limit_price else None,
            stop_price=Decimal(order.stop_price) if order.stop_price else None,
            filled_quantity=Decimal(order.filled_qty) if order.filled_qty else Decimal("0"),
            filled_avg_price=Decimal(order.filled_avg_price) if order.filled_avg_price else None,
            created_at=order.created_at,
            updated_at=order.updated_at,
        )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        if not self._connected:
            raise RuntimeError("Not connected to Alpaca")

        try:
            self._api.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        if not self._connected:
            raise RuntimeError("Not connected to Alpaca")

        try:
            order = self._api.get_order(order_id)
            return Order(
                id=order.id,
                symbol=order.symbol,
                side=OrderSide(order.side),
                order_type=OrderType(order.type),
                quantity=Decimal(order.qty),
                status=OrderStatus(order.status),
                limit_price=Decimal(order.limit_price) if order.limit_price else None,
                stop_price=Decimal(order.stop_price) if order.stop_price else None,
                filled_quantity=Decimal(order.filled_qty) if order.filled_qty else Decimal("0"),
                filled_avg_price=Decimal(order.filled_avg_price)
                if order.filled_avg_price
                else None,
            )
        except Exception:
            return None

    async def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        """Get orders."""
        if not self._connected:
            raise RuntimeError("Not connected to Alpaca")

        orders = self._api.list_orders(status=status.value if status else None)

        return [
            Order(
                id=o.id,
                symbol=o.symbol,
                side=OrderSide(o.side),
                order_type=OrderType(o.type),
                quantity=Decimal(o.qty),
                status=OrderStatus(o.status),
                limit_price=Decimal(o.limit_price) if o.limit_price else None,
                stop_price=Decimal(o.stop_price) if o.stop_price else None,
                filled_quantity=Decimal(o.filled_qty) if o.filled_qty else Decimal("0"),
                filled_avg_price=Decimal(o.filled_avg_price) if o.filled_avg_price else None,
            )
            for o in orders
        ]


class SimulatedBroker(BrokerAdapter):
    """
    Simulated broker for backtesting and paper trading without external API.

    Tracks positions and orders in memory.
    """

    def __init__(self, initial_cash: Decimal = Decimal("100000")):
        """Initialize simulated broker."""
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}
        self._order_counter = 0
        self._connected = False

    async def connect(self) -> bool:
        """Connect (no-op for simulated)."""
        self._connected = True
        logger.info(f"Simulated broker connected with ${self.initial_cash} cash")
        return True

    async def disconnect(self) -> None:
        """Disconnect (no-op for simulated)."""
        self._connected = False

    async def get_account(self) -> AccountInfo:
        """Get simulated account info."""
        portfolio_value = self.cash + sum(p.market_value for p in self.positions.values())

        return AccountInfo(
            account_id="simulated",
            equity=portfolio_value,
            cash=self.cash,
            buying_power=self.cash,
            portfolio_value=portfolio_value,
            is_paper=True,
        )

    async def get_positions(self) -> list[Position]:
        """Get all simulated positions."""
        return list(self.positions.values())

    async def get_position(self, symbol: str) -> Position | None:
        """Get specific position."""
        return self.positions.get(symbol)

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        """Submit simulated order (immediately fills for market orders)."""
        self._order_counter += 1
        order_id = f"sim_{self._order_counter}"

        # For market orders, fill immediately at limit price or use placeholder
        fill_price = limit_price or Decimal("100")  # Would need real price feed

        order = Order(
            id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            status=OrderStatus.FILLED if order_type == OrderType.MARKET else OrderStatus.NEW,
            limit_price=limit_price,
            stop_price=stop_price,
            filled_quantity=quantity if order_type == OrderType.MARKET else Decimal("0"),
            filled_avg_price=fill_price if order_type == OrderType.MARKET else None,
        )

        self.orders[order_id] = order

        # Update position if filled
        if order.status == OrderStatus.FILLED:
            self._update_position(symbol, side, quantity, fill_price)

        return order

    def _update_position(
        self, symbol: str, side: OrderSide, quantity: Decimal, price: Decimal
    ) -> None:
        """Update position after fill."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            if side == OrderSide.BUY:
                # Add to long / reduce short
                new_qty = pos.quantity + quantity
                if new_qty > 0:
                    new_avg = (pos.avg_entry_price * pos.quantity + price * quantity) / new_qty
                    pos.quantity = new_qty
                    pos.avg_entry_price = new_avg
                else:
                    del self.positions[symbol]
            else:
                # Reduce long / add to short
                new_qty = pos.quantity - quantity
                if new_qty > 0:
                    pos.quantity = new_qty
                elif new_qty < 0:
                    pos.quantity = abs(new_qty)
                    pos.side = PositionSide.SHORT
                    pos.avg_entry_price = price
                else:
                    del self.positions[symbol]
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                side=PositionSide.LONG if side == OrderSide.BUY else PositionSide.SHORT,
                quantity=quantity,
                avg_entry_price=price,
                current_price=price,
                market_value=quantity * price,
                unrealized_pnl=Decimal("0"),
                unrealized_pnl_pct=Decimal("0"),
            )

        # Update cash
        if side == OrderSide.BUY:
            self.cash -= quantity * price
        else:
            self.cash += quantity * price

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel simulated order."""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    async def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self.orders.get(order_id)

    async def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        """Get orders."""
        orders = list(self.orders.values())
        if status:
            orders = [o for o in orders if o.status == status]
        return orders
