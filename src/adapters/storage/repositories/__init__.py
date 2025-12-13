"""
Repository classes for database access.

Provides typed CRUD operations for each entity type:
- PositionRepository: Portfolio positions
- OrderRepository: Order lifecycle
- TradeRepository: Completed trades
- SystemStateRepository: System state and kill switch
"""

from adapters.storage.repositories.order import OrderRepository
from adapters.storage.repositories.position import PositionRepository
from adapters.storage.repositories.system_state import SystemStateRepository
from adapters.storage.repositories.trade import TradeRepository

__all__ = [
    "OrderRepository",
    "PositionRepository",
    "SystemStateRepository",
    "TradeRepository",
]
