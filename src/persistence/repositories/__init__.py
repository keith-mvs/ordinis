"""
Repository classes for database access.

Provides typed CRUD operations for each entity type:
- PositionRepository: Portfolio positions
- OrderRepository: Order lifecycle
- TradeRepository: Completed trades
- SystemStateRepository: System state and kill switch
"""

from persistence.repositories.order import OrderRepository
from persistence.repositories.position import PositionRepository
from persistence.repositories.system_state import SystemStateRepository
from persistence.repositories.trade import TradeRepository

__all__ = [
    "OrderRepository",
    "PositionRepository",
    "SystemStateRepository",
    "TradeRepository",
]
