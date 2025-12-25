"""
Storage adapter for Ordinis live trading.

Provides SQLite-based state persistence for:
- Portfolio positions
- Order lifecycle
- Trade history
- System state (kill switch, checkpoints)

Uses aiosqlite for async operations with WAL mode for concurrent reads.
Includes governance hooks for path validation and audit logging.
"""

from ordinis.adapters.storage.database import DatabaseManager, get_database
from ordinis.adapters.storage.governance import (
    BackupValidationRule,
    ChromaDBPathRule,
    DatabaseIntegrityRule,
    PathValidationRule,
    StorageGovernanceHook,
    StorageRule,
)
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
    # Database
    "DatabaseManager",
    "get_database",
    # Repositories
    "OrderRepository",
    "PositionRepository",
    "SystemStateRepository",
    "TradeRepository",
    # Models
    "OrderRow",
    "PositionRow",
    "SystemStateRow",
    "TradeRow",
    # Governance
    "StorageGovernanceHook",
    "StorageRule",
    "PathValidationRule",
    "DatabaseIntegrityRule",
    "ChromaDBPathRule",
    "BackupValidationRule",
]
