"""
Domain enumerations for Ordinis trading system.
"""

from enum import Enum


class OrderSide(str, Enum):
    """Order side (buy/sell)."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"


class TimeInForce(str, Enum):
    """Time in force for orders."""

    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill


class OrderStatus(str, Enum):
    """Order status lifecycle."""

    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    PENDING_SUBMIT = "PENDING_SUBMIT"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    ERROR = "ERROR"


class PositionSide(str, Enum):
    """Position side."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"
