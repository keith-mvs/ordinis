"""
Repository classes for database access.

Provides typed CRUD operations for each entity type:
- PositionRepository: Portfolio positions
- OrderRepository: Order lifecycle
- TradeRepository: Completed trades
- SystemStateRepository: System state and kill switch
"""

from ordinis.adapters.storage.repositories.order import OrderRepository
from ordinis.adapters.storage.repositories.position import PositionRepository
from ordinis.adapters.storage.repositories.system_state import SystemStateRepository
from ordinis.adapters.storage.repositories.trade import TradeRepository

__all__ = [
    "OrderRepository",
    "PositionRepository",
    "SystemStateRepository",
    "TradeRepository",
]
