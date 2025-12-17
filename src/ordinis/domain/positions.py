"""
Position domain models.
"""

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from ordinis.domain.enums import PositionSide


class Position(BaseModel):
    """
    Trading position for a symbol.
    Tracks current holdings, P&L, and position history.
    """

    symbol: str
    side: PositionSide = PositionSide.FLAT
    quantity: int = Field(default=0, ge=0, description="Absolute quantity")

    # Pricing & PnL
    avg_entry_price: float = Field(default=0.0, ge=0.0)
    current_price: float = Field(default=0.0, ge=0.0)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Metadata
    sector: str | None = None
    entry_time: datetime | None = None
    last_update_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Instrument details
    multiplier: float = 1.0
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    expiry: datetime | None = None

    @property
    def market_value(self) -> float:
        """Get current market value of position."""
        return self.quantity * self.current_price * self.multiplier

    @property
    def cost_basis(self) -> float:
        """Get total cost basis."""
        return self.quantity * self.avg_entry_price * self.multiplier

    @property
    def total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def pnl_pct(self) -> float:
        """Get P&L as percentage of cost basis."""
        if self.cost_basis == 0:
            return 0.0
        return (self.total_pnl / self.cost_basis) * 100

    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.quantity > 0 and self.side != PositionSide.FLAT

    def is_flat(self) -> bool:
        """Check if position is flat (no holdings)."""
        return not self.is_open

    def update_price(self, price: float, timestamp: datetime):
        """Update current price and unrealized P&L."""
        self.current_price = price
        self.last_update_time = timestamp

        if self.quantity > 0:
            diff = price - self.avg_entry_price
            if self.side == PositionSide.SHORT:
                diff = -diff

            self.unrealized_pnl = diff * self.quantity * self.multiplier
        else:
            self.unrealized_pnl = 0.0


class Trade(BaseModel):
    """
    Completed trade with entry and exit.
    """

    trade_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    side: PositionSide
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    commission: float
    duration: float  # In seconds

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0

    @property
    def is_loser(self) -> bool:
        """Check if trade was a loss."""
        return self.pnl < 0
