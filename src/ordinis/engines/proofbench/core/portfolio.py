"""Portfolio management with position tracking and P&L calculation."""

from datetime import datetime

from ordinis.domain.enums import OrderSide, PositionSide
from ordinis.domain.orders import Fill, Order
from ordinis.domain.positions import Position, Trade


class Portfolio:
    """Portfolio manager for backtesting.

    Tracks positions, cash, equity, and generates trade history.
    Provides mark-to-market updates and P&L calculation.
    """

    def __init__(self, initial_capital: float = 100000.0):
        """Initialize portfolio.

        Args:
            initial_capital: Starting cash balance
        """
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")

        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.orders: dict[str, Order] = {}

    @property
    def equity(self) -> float:
        """Get total equity (cash + positions).

        Returns:
            Current total equity
        """
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value

    @property
    def total_pnl(self) -> float:
        """Get total P&L since inception.

        Returns:
            Total profit/loss
        """
        return self.equity - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        """Get total P&L as percentage.

        Returns:
            P&L percentage
        """
        return (self.total_pnl / self.initial_capital) * 100

    @property
    def position_value(self) -> float:
        """Get total value of all positions.

        Returns:
            Sum of all position values
        """
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def num_positions(self) -> int:
        """Get number of open positions.

        Returns:
            Count of non-flat positions
        """
        return sum(1 for pos in self.positions.values() if not pos.is_flat())

    def update_position(self, fill: Fill, timestamp: datetime):
        """Update position based on fill.

        Args:
            fill: Fill to apply
            timestamp: Update timestamp
        """
        symbol = fill.symbol

        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol, entry_time=timestamp, last_update_time=timestamp
            )

        pos = self.positions[symbol]

        if fill.side == OrderSide.BUY:
            self._process_buy(pos, fill, timestamp)
        else:
            self._process_sell(pos, fill, timestamp)

    def _process_buy(self, pos: Position, fill: Fill, timestamp: datetime):
        """Process buy fill.

        Args:
            pos: Position to update
            fill: Buy fill
            timestamp: Fill timestamp
        """
        if pos.side == PositionSide.FLAT or pos.side == PositionSide.LONG:
            # Add to long position or open new long
            total_cost = pos.avg_entry_price * pos.quantity + fill.price * fill.quantity
            pos.quantity += fill.quantity
            pos.avg_entry_price = total_cost / pos.quantity
            pos.side = PositionSide.LONG

            # Update cash (pay for shares + commission)
            self.cash -= fill.total_cost

            if pos.entry_time is None:
                pos.entry_time = timestamp

        elif pos.side == PositionSide.SHORT:
            # Cover short position
            self._cover_short(pos, fill, timestamp)

        pos.current_price = fill.price
        pos.last_update_time = timestamp
        pos.update_price(fill.price, timestamp)

    def _process_sell(self, pos: Position, fill: Fill, timestamp: datetime):
        """Process sell fill.

        Args:
            pos: Position to update
            fill: Sell fill
            timestamp: Fill timestamp
        """
        if pos.side == PositionSide.FLAT or pos.side == PositionSide.SHORT:
            # Add to short position or open new short
            total_cost = pos.avg_entry_price * pos.quantity + fill.price * fill.quantity
            pos.quantity += fill.quantity
            pos.avg_entry_price = total_cost / pos.quantity
            pos.side = PositionSide.SHORT

            # Update cash (receive proceeds - commission)
            self.cash += fill.net_proceeds

            if pos.entry_time is None:
                pos.entry_time = timestamp

        elif pos.side == PositionSide.LONG:
            # Close/reduce long position
            self._close_long(pos, fill, timestamp)

        pos.current_price = fill.price
        pos.last_update_time = timestamp
        pos.update_price(fill.price, timestamp)

    def _close_long(self, pos: Position, fill: Fill, timestamp: datetime):
        """Close or reduce long position.

        Args:
            pos: Position to close/reduce
            fill: Sell fill
            timestamp: Fill timestamp
        """
        # Calculate realized P&L
        realized = (fill.price - pos.avg_entry_price) * fill.quantity - fill.commission
        pos.realized_pnl += realized

        # Update cash
        self.cash += fill.net_proceeds

        # Record trade if fully closing some quantity
        if pos.entry_time and fill.quantity > 0:
            trade = Trade(
                symbol=pos.symbol,
                side=PositionSide.LONG,
                entry_time=pos.entry_time,
                exit_time=timestamp,
                entry_price=pos.avg_entry_price,
                exit_price=fill.price,
                quantity=fill.quantity,
                pnl=realized,
                pnl_pct=((fill.price - pos.avg_entry_price) / pos.avg_entry_price) * 100,
                commission=fill.commission,
                duration=(timestamp - pos.entry_time).total_seconds(),
            )
            self.trades.append(trade)

        # Update quantity
        pos.quantity -= fill.quantity

        if pos.quantity == 0:
            pos.side = PositionSide.FLAT
            pos.avg_entry_price = 0.0
            pos.entry_time = None

    def _cover_short(self, pos: Position, fill: Fill, timestamp: datetime):
        """Cover or reduce short position.

        Args:
            pos: Position to cover/reduce
            fill: Buy fill
            timestamp: Fill timestamp
        """
        # Calculate realized P&L
        realized = (pos.avg_entry_price - fill.price) * fill.quantity - fill.commission
        pos.realized_pnl += realized

        # Update cash
        self.cash -= fill.total_cost

        # Record trade if fully covering some quantity
        if pos.entry_time and fill.quantity > 0:
            trade = Trade(
                symbol=pos.symbol,
                side=PositionSide.SHORT,
                entry_time=pos.entry_time,
                exit_time=timestamp,
                entry_price=pos.avg_entry_price,
                exit_price=fill.price,
                quantity=fill.quantity,
                pnl=realized,
                pnl_pct=((pos.avg_entry_price - fill.price) / pos.avg_entry_price) * 100,
                commission=fill.commission,
                duration=(timestamp - pos.entry_time).total_seconds(),
            )
            self.trades.append(trade)

        # Update quantity
        pos.quantity -= fill.quantity

        if pos.quantity == 0:
            pos.side = PositionSide.FLAT
            pos.avg_entry_price = 0.0
            pos.entry_time = None

    def update_prices(self, prices: dict[str, float], timestamp: datetime):
        """Update all position prices (mark-to-market).

        Args:
            prices: Dictionary of symbol -> price
            timestamp: Update timestamp
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price, timestamp)

    def record_equity(self, timestamp: datetime):
        """Record current equity for equity curve.

        Args:
            timestamp: Timestamp for recording
        """
        self.equity_curve.append((timestamp, self.equity))

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position or None if no position
        """
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has position in symbol.

        Args:
            symbol: Stock symbol

        Returns:
            True if position exists and is not flat
        """
        pos = self.get_position(symbol)
        return pos is not None and not pos.is_flat()

    def can_trade(self, symbol: str, quantity: int, price: float, side: OrderSide) -> bool:
        """Check if portfolio can execute a trade.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            side: Buy or sell

        Returns:
            True if trade is possible
        """
        if side == OrderSide.BUY:
            # Check if enough cash
            required = quantity * price
            return self.cash >= required
        # Check if enough shares to sell
        pos = self.get_position(symbol)
        if pos is None or pos.side != PositionSide.LONG:
            return False
        return pos.quantity >= quantity

    def get_summary(self) -> dict:
        """Get portfolio summary.

        Returns:
            Dictionary with portfolio metrics
        """
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "position_value": self.position_value,
            "equity": self.equity,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "num_positions": self.num_positions,
            "num_trades": len(self.trades),
        }
