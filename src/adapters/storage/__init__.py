"""
Storage adapter for Ordinis live trading.

Provides SQLite-based state persistence for:
- Portfolio positions
- Order lifecycle
- Trade history
- System state (kill switch, checkpoints)

Uses aiosqlite for async operations with WAL mode for concurrent reads.
"""

from adapters.storage.database import DatabaseManager, get_database
from adapters.storage.models import (
    OrderRow,
    PositionRow,
    SystemStateRow,
    TradeRow,
)
from adapters.storage.repositories.order import OrderRepository
from adapters.storage.repositories.position import PositionRepository
from adapters.storage.repositories.system_state import SystemStateRepository
from adapters.storage.repositories.trade import TradeRepository

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
