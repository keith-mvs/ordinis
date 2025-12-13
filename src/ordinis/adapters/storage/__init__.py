"""
Storage adapter for Ordinis live trading.

Provides SQLite-based state persistence for:
- Portfolio positions
- Order lifecycle
- Trade history
- System state (kill switch, checkpoints)

Uses aiosqlite for async operations with WAL mode for concurrent reads.
"""

from ordinis.adapters.storage.database import DatabaseManager, get_database
from ordinis.adapters.storage.models import (
    OrderRow,
    PositionRow,
    SystemStateRow,
    TradeRow,
)
from ordinis.adapters.storage.repositories.order import OrderRepository
from ordinis.adapters.storage.repositories.position import PositionRepository
from ordinis.adapters.storage.repositories.system_state import SystemStateRepository
from ordinis.adapters.storage.repositories.trade import TradeRepository

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
