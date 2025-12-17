"""
Options Data Contract Models

Canonical data structures for options contracts, legs, and positions across
the Ordinis platform. All strategy modules reference these models.

Author: Ordinis Project
License: MIT
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


class OptionType(Enum):
    """Option contract type."""

    CALL = "call"
    PUT = "put"


class PositionSide(Enum):
    """Position direction."""

    LONG = "long"  # Bought / Holding
    SHORT = "short"  # Sold / Written


class OrderStatus(Enum):
    """Order execution status."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELED = "canceled"
    REJECTED = "rejected"


@dataclass
class OptionContract:
    """
    Represents a single options contract.

    This is the atomic unit of options data in the Ordinis platform.
    All strategy legs reference OptionContract instances.

    Attributes:
        symbol: OCC option symbol (e.g., 'SPY250117C00450000')
        underlying: Underlying ticker symbol
        option_type: CALL or PUT
        strike: Strike price
        expiration: Expiration date
        multiplier: Contract multiplier (typically 100)
        exchange: Exchange code
        last_price: Last traded price
        bid: Current bid price
        ask: Current ask price
        implied_volatility: Implied volatility (decimal)
        open_interest: Open interest
        volume: Daily volume
    """

    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiration: date
    multiplier: int = 100
    exchange: str = ""
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    implied_volatility: float = 0.0
    open_interest: int = 0
    volume: int = 0

    def __post_init__(self):
        """Validate contract parameters."""
        if self.strike <= 0:
            raise ValueError("Strike price must be positive")
        if self.multiplier <= 0:
            raise ValueError("Multiplier must be positive")
        if not isinstance(self.expiration, date):
            raise TypeError("Expiration must be a date object")

    @property
    def days_to_expiration(self) -> int:
        """Calculate days until expiration."""
        return (self.expiration - date.today()).days

    @property
    def time_to_expiration(self) -> float:
        """Calculate time to expiration in years."""
        return self.days_to_expiration / 365.0

    @property
    def mid_price(self) -> float:
        """Calculate mid-market price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last_price

    def is_itm(self, underlying_price: float) -> bool:
        """
        Determine if option is in-the-money.

        Args:
            underlying_price: Current price of underlying

        Returns:
            True if ITM, False otherwise
        """
        if self.option_type == OptionType.CALL:
            return underlying_price > self.strike
        return underlying_price < self.strike

    def intrinsic_value(self, underlying_price: float) -> float:
        """
        Calculate intrinsic value.

        Args:
            underlying_price: Current price of underlying

        Returns:
            Intrinsic value (always >= 0)
        """
        if self.option_type == OptionType.CALL:
            return max(underlying_price - self.strike, 0)
        return max(self.strike - underlying_price, 0)

    def time_value(self, underlying_price: float) -> float:
        """
        Calculate time value (extrinsic value).

        Args:
            underlying_price: Current price of underlying

        Returns:
            Time value = Option Price - Intrinsic Value
        """
        return self.mid_price - self.intrinsic_value(underlying_price)


@dataclass
class OptionLeg:
    """
    Represents a single leg in a multi-leg options strategy.

    Used by all strategy modules to define strategy structure.

    Attributes:
        contract: OptionContract instance
        position_side: LONG or SHORT
        quantity: Number of contracts
        premium: Premium paid/received per contract
        entry_date: Date position was entered
        order_id: Associated order ID
    """

    contract: OptionContract
    position_side: PositionSide
    quantity: int = 1
    premium: float = 0.0
    entry_date: datetime | None = None
    order_id: str | None = None

    def __post_init__(self):
        """Validate leg parameters."""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.entry_date is None:
            self.entry_date = datetime.now()

    @property
    def cost_basis(self) -> float:
        """
        Calculate total cost basis for this leg.

        Returns:
            Total cost (negative for long, positive for short)
        """
        multiplier = self.contract.multiplier * self.quantity

        if self.position_side == PositionSide.LONG:
            return -self.premium * multiplier  # Cash outflow
        return self.premium * multiplier  # Cash inflow

    @property
    def current_value(self) -> float:
        """
        Calculate current market value of this leg.

        Returns:
            Current value based on mid-market price
        """
        multiplier = self.contract.multiplier * self.quantity
        return self.contract.mid_price * multiplier

    @property
    def unrealized_pnl(self) -> float:
        """
        Calculate unrealized P&L for this leg.

        Returns:
            Profit/loss (positive = profit)
        """
        if self.position_side == PositionSide.LONG:
            return self.current_value + self.cost_basis
        return self.cost_basis - self.current_value


@dataclass
class OptionPosition:
    """
    Represents a complete options position (single or multi-leg strategy).

    This is the primary container used by strategy modules and portfolio tracking.

    Attributes:
        position_id: Unique position identifier
        strategy_name: Name of strategy (e.g., 'covered_call', 'iron_butterfly')
        underlying: Underlying ticker symbol
        legs: List of OptionLeg instances
        entry_date: Position entry date
        underlying_shares: Number of underlying shares (if applicable)
        metadata: Additional strategy-specific data
    """

    position_id: str
    strategy_name: str
    underlying: str
    legs: list[OptionLeg] = field(default_factory=list)
    entry_date: datetime | None = None
    underlying_shares: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize position."""
        if self.entry_date is None:
            self.entry_date = datetime.now()

    def add_leg(self, leg: OptionLeg) -> None:
        """Add a leg to the position."""
        self.legs.append(leg)

    @property
    def num_legs(self) -> int:
        """Number of option legs in position."""
        return len(self.legs)

    @property
    def total_cost_basis(self) -> float:
        """
        Calculate total cost basis for all legs.

        Returns:
            Net premium paid/received (negative = debit, positive = credit)
        """
        return sum(leg.cost_basis for leg in self.legs)

    @property
    def current_value(self) -> float:
        """Calculate current market value of all legs."""
        return sum(leg.current_value for leg in self.legs)

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L for entire position."""
        return sum(leg.unrealized_pnl for leg in self.legs)

    @property
    def is_credit_strategy(self) -> bool:
        """Determine if position was opened for net credit."""
        return self.total_cost_basis > 0

    @property
    def is_debit_strategy(self) -> bool:
        """Determine if position was opened for net debit."""
        return self.total_cost_basis < 0

    def days_in_trade(self) -> int:
        """Calculate days since position entry."""
        return (datetime.now() - self.entry_date).days

    def summary(self) -> dict:
        """
        Generate position summary.

        Returns:
            Dictionary with key position metrics
        """
        return {
            "position_id": self.position_id,
            "strategy": self.strategy_name,
            "underlying": self.underlying,
            "num_legs": self.num_legs,
            "entry_date": self.entry_date.isoformat(),
            "days_in_trade": self.days_in_trade(),
            "cost_basis": self.total_cost_basis,
            "current_value": self.current_value,
            "unrealized_pnl": self.unrealized_pnl,
            "pnl_percent": (self.unrealized_pnl / abs(self.total_cost_basis) * 100)
            if self.total_cost_basis != 0
            else 0,
            "strategy_type": "CREDIT" if self.is_credit_strategy else "DEBIT",
        }


if __name__ == "__main__":
    print("=== Options Data Contract Models ===\n")

    # Example: Create a covered call position
    from datetime import date, timedelta

    # Create call contract
    call_contract = OptionContract(
        symbol="SPY250117C00450000",
        underlying="SPY",
        option_type=OptionType.CALL,
        strike=450.0,
        expiration=date.today() + timedelta(days=30),
        last_price=5.50,
        bid=5.45,
        ask=5.55,
        implied_volatility=0.18,
    )

    # Create option leg (short call)
    call_leg = OptionLeg(
        contract=call_contract, position_side=PositionSide.SHORT, quantity=1, premium=5.50
    )

    # Create position
    position = OptionPosition(
        position_id="POS-001", strategy_name="covered_call", underlying="SPY", underlying_shares=100
    )
    position.add_leg(call_leg)

    # Display summary
    summary = position.summary()
    print("Covered Call Position:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
