"""
Financial instrument domain models.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class InstrumentType(str, Enum):
    """Types of financial instruments."""

    EQUITY = "equity"
    FUTURE = "future"
    OPTION = "option"
    CRYPTO = "crypto"
    FOREX = "forex"


class BaseInstrument(BaseModel):
    """Base class for all financial instruments."""

    symbol: str
    type: InstrumentType
    exchange: str | None = None
    currency: str = "USD"
    tick_size: float = 0.01
    lot_size: float = 1.0


class FutureContract(BaseInstrument):
    """
    Futures contract definition.
    """

    type: InstrumentType = Field(default=InstrumentType.FUTURE, frozen=True)
    expiry_date: datetime
    multiplier: float = 1.0
    margin_requirement: float = 0.1  # 10% initial margin
    underlying_symbol: str
    settlement_method: str = "cash"  # or "physical"

    @property
    def days_to_expiry(self) -> int:
        """Calculate days remaining until expiry."""
        delta = self.expiry_date - datetime.now()
        return max(0, delta.days)
