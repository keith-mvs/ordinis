"""
Persistence layer for Ordinis live trading.

Provides SQLite-based state persistence for:
- Portfolio positions
- Order lifecycle
- Trade history
- System state (kill switch, checkpoints)

Uses aiosqlite for async operations with WAL mode for concurrent reads.
"""

from persistence.database import DatabaseManager, get_database
from persistence.models import (
    OrderRow,
    PositionRow,
    SystemStateRow,
    TradeRow,
)
from persistence.repositories.order import OrderRepository
from persistence.repositories.position import PositionRepository
from persistence.repositories.system_state import SystemStateRepository
from persistence.repositories.trade import TradeRepository

__all__ = [
    "DatabaseManager",
    "OrderRepository",
    "OrderRow",
    "PositionRepository",
    "PositionRow",
    "SystemStateRepository",
    "SystemStateRow",
    "TradeRepository",
    "TradeRow",
    "get_database",
]
