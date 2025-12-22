"""
Repository classes for database access.

Provides typed CRUD operations for each entity type:
- PositionRepository: Portfolio positions
- OrderRepository: Order lifecycle
- TradeRepository: Completed trades
- SystemStateRepository: System state and kill switch
- CortexRepository: Cortex outputs and strategy hypotheses
"""

from ordinis.adapters.storage.repositories.cortex import (
    CortexOutputRow,
    CortexRepository,
    StrategyHypothesisRow,
)
from ordinis.adapters.storage.repositories.order import OrderRepository
from ordinis.adapters.storage.repositories.position import PositionRepository
from ordinis.adapters.storage.repositories.system_state import SystemStateRepository
from ordinis.adapters.storage.repositories.trade import TradeRepository

__all__ = [
    "CortexOutputRow",
    "CortexRepository",
    "OrderRepository",
    "PositionRepository",
    "StrategyHypothesisRow",
    "SystemStateRepository",
    "TradeRepository",
]
