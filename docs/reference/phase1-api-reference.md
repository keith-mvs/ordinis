# Phase 1 API Reference
# Ordinis Trading System
# Version: 1.0.0
# Last Updated: 2025-12-12

---

## Overview

This document provides API reference documentation for Phase 1 production infrastructure components:
- Persistence Layer
- Safety Layer
- Orchestration Layer
- Alerting Layer
- Interface Layer

For comprehensive architecture documentation, see [Production Architecture](../architecture/production-architecture.md).

---

## Persistence Layer

### DatabaseManager

**Location**: `src/persistence/database.py`

Core database connection manager with SQLite and WAL mode support.

#### Initialization

```python
from persistence.database import DatabaseManager
from pathlib import Path

db = DatabaseManager(
    db_path=Path("data/ordinis.db"),
    backup_dir=Path("data/backups")
)

# Initialize database (creates schema, enables WAL)
success = await db.initialize()
```

#### Core Methods

```python
# Execute query (SELECT)
async def execute_query(
    self,
    query: str,
    params: tuple = ()
) -> list[tuple]:
    """Execute read query and return results."""

# Execute update (INSERT, UPDATE, DELETE)
async def execute_update(
    self,
    query: str,
    params: tuple = ()
) -> int:
    """Execute write query and return affected rows."""

# Create backup
async def backup(self) -> bool:
    """Create timestamped database backup."""

# Health check
async def health_check(self) -> bool:
    """Verify database connection is healthy."""

# Close connection
async def close(self) -> None:
    """Graceful shutdown with pending transaction completion."""
```

#### Configuration

```python
# WAL mode settings (automatic)
PRAGMA journal_mode=WAL          # Concurrent reads
PRAGMA synchronous=NORMAL        # Balance safety/performance
PRAGMA foreign_keys=ON           # Referential integrity
PRAGMA busy_timeout=5000         # Lock retry timeout (5s)
```

---

### PositionRepository

**Location**: `src/persistence/repositories/position.py`

Manages trading position persistence.

#### Data Model

```python
from persistence.models import PositionRow

position = PositionRow(
    id=None,                    # Auto-increment
    symbol="AAPL",
    side="LONG",                # LONG | SHORT | FLAT
    quantity=100,
    avg_cost=150.25,
    current_price=152.50,
    realized_pnl=0.0,
    unrealized_pnl=225.0,
    entry_time="2025-12-12T09:30:00",
    last_update="2025-12-12T15:45:00",
    created_at=None,            # Auto-populated
    updated_at=None             # Auto-populated
)
```

#### Methods

```python
# Upsert position (insert or update)
async def upsert(self, position: PositionRow) -> bool

# Get position by symbol
async def get_by_symbol(self, symbol: str) -> PositionRow | None

# Get all positions
async def get_all(self) -> list[PositionRow]

# Get active positions (quantity > 0)
async def get_active(self) -> list[PositionRow]

# Delete position
async def delete(self, symbol: str) -> bool
```

---

### OrderRepository

**Location**: `src/persistence/repositories/order.py`

Tracks order lifecycle from creation to fill.

#### Data Model

```python
from persistence.models import OrderRow

order = OrderRow(
    order_id="ord_123456",              # Internal UUID
    symbol="AAPL",
    side="BUY",                         # BUY | SELL
    quantity=100,
    order_type="MARKET",                # MARKET | LIMIT | STOP
    limit_price=None,
    stop_price=None,
    status="created",                   # created | submitted | filled | rejected
    broker_order_id=None,               # Populated on submission
    broker_response=None,               # JSON response
    avg_fill_price=None,
    filled_quantity=0,
    commission=0.0,
    error_message=None,
    retries=0,
    submitted_at=None,
    filled_at=None,
    created_at="2025-12-12T09:30:00",
    updated_at="2025-12-12T09:30:00"
)
```

#### Methods

```python
# Create new order
async def create(self, order: OrderRow) -> OrderRow

# Update order status
async def update_status(
    self,
    order_id: str,
    status: str,
    broker_order_id: str | None = None,
    error_message: str | None = None
) -> bool

# Get order by ID
async def get_by_id(self, order_id: str) -> OrderRow | None

# Get pending orders
async def get_pending(self) -> list[OrderRow]

# Get orders by symbol
async def get_by_symbol(self, symbol: str) -> list[OrderRow]
```

---

### TradeRepository

**Location**: `src/persistence/repositories/trade.py`

Records completed round-trip trades.

#### Data Model

```python
from persistence.models import TradeRow

trade = TradeRow(
    trade_id="trade_123456",
    symbol="AAPL",
    strategy="momentum_breakout",
    entry_order_id="ord_123",
    exit_order_id="ord_456",
    side="LONG",                    # LONG | SHORT
    quantity=100,
    entry_price=150.25,
    exit_price=152.50,
    entry_time="2025-12-12T09:30:00",
    exit_time="2025-12-12T15:45:00",
    realized_pnl=225.0,
    commission=1.0,
    duration_seconds=22500,
    created_at="2025-12-12T15:45:00"
)
```

#### Methods

```python
# Record trade
async def record_trade(self, trade: TradeRow) -> bool

# Get trades by date range
async def get_by_date_range(
    self,
    start_date: str,
    end_date: str
) -> list[TradeRow]

# Get trades by symbol
async def get_by_symbol(self, symbol: str) -> list[TradeRow]

# Get trades by strategy
async def get_by_strategy(self, strategy: str) -> list[TradeRow]
```

---

### SystemStateRepository

**Location**: `src/persistence/repositories/system_state.py`

Persists system-wide state and configuration.

#### Methods

```python
# Set state value
async def set_state(self, key: str, value: str) -> bool

# Get state value
async def get_state(self, key: str) -> str | None

# Delete state
async def delete_state(self, key: str) -> bool

# Get all state
async def get_all_state(self) -> dict[str, str]
```

#### Common Keys

```python
# Kill switch state
"kill_switch_active": "true" | "false"
"kill_switch_reason": "daily_loss_limit"
"kill_switch_timestamp": "2025-12-12T15:00:00"

# Circuit breaker state
"circuit_breaker_alpaca_state": "CLOSED" | "OPEN" | "HALF_OPEN"

# System configuration
"trading_enabled": "true" | "false"
"session_start_time": "2025-12-12T09:30:00"
```

---

## Safety Layer

### KillSwitch

**Location**: `src/safety/kill_switch.py`

Emergency halt mechanism with multiple trigger sources.

#### Initialization

```python
from safety.kill_switch import KillSwitch, KillSwitchReason
from pathlib import Path

kill_switch = KillSwitch(
    kill_file=Path("data/KILL_SWITCH"),
    db_manager=db,
    daily_loss_limit=1000.0,
    max_drawdown_pct=5.0,
    consecutive_loss_limit=5,
    check_interval_seconds=1.0
)

await kill_switch.initialize()
```

#### Core Methods

```python
# Check if active
@property
def is_active(self) -> bool

# Trigger kill switch
async def trigger(
    self,
    reason: KillSwitchReason,
    message: str,
    triggered_by: str,
    metadata: dict | None = None
) -> None

# Reset kill switch
async def reset(self) -> None

# Get current state
async def get_state(self) -> KillSwitchState

# Auto-trigger check
async def check_auto_triggers(
    self,
    current_pnl: float,
    peak_value: float,
    consecutive_losses: int
) -> bool
```

#### KillSwitchReason Enum

```python
class KillSwitchReason(Enum):
    MANUAL = "manual"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_DRAWDOWN = "max_drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    API_FAILURE = "api_failure"
    RECONCILIATION_FAILURE = "reconciliation_failure"
    CUSTOM = "custom"
```

#### Usage Example

```python
# Manual trigger
await kill_switch.trigger(
    reason=KillSwitchReason.MANUAL,
    message="Manual halt requested by operator",
    triggered_by="admin"
)

# Check before trading
if kill_switch.is_active:
    raise RuntimeError("Trading halted: kill switch active")

# Auto-trigger check (in risk engine)
triggered = await kill_switch.check_auto_triggers(
    current_pnl=-1200.0,
    peak_value=10000.0,
    consecutive_losses=6
)
```

---

### CircuitBreaker

**Location**: `src/safety/circuit_breaker.py`

API resilience with automatic failure detection and recovery.

#### Initialization

```python
from safety.circuit_breaker import CircuitBreaker, CircuitState

breaker = CircuitBreaker(
    name="alpaca_api",
    failure_threshold=5,
    success_threshold=3,
    recovery_timeout_seconds=30.0,
    half_open_max_calls=3
)
```

#### Core Methods

```python
# Context manager for protected calls
async def __aenter__(self) -> CircuitBreaker
async def __aexit__(self, exc_type, exc_val, exc_tb) -> None

# Manual state control
def open(self) -> None
def close(self) -> None
def reset_statistics(self) -> None

# State inspection
@property
def state(self) -> CircuitState

@property
def statistics(self) -> CircuitStats
```

#### CircuitState Enum

```python
class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Blocking calls
    HALF_OPEN = "half_open"  # Testing recovery
```

#### Usage Example

```python
# Protect API calls
async def call_broker_api():
    async with circuit_breaker:
        response = await broker.get_positions()
        return response

# Check state before trading
if circuit_breaker.state == CircuitState.OPEN:
    logger.warning("API circuit breaker open, skipping trade")
    return

# Access statistics
stats = circuit_breaker.statistics
logger.info(f"API success rate: {stats.success_rate:.2%}")
```

---

## Orchestration Layer

### OrdinisOrchestrator

**Location**: `src/orchestration/orchestrator.py`

Central system lifecycle coordinator.

#### Initialization

```python
from orchestration.orchestrator import OrdinisOrchestrator, OrchestratorConfig
from pathlib import Path

config = OrchestratorConfig(
    db_path=Path("data/ordinis.db"),
    backup_dir=Path("data/backups"),
    kill_file=Path("data/KILL_SWITCH"),
    daily_loss_limit=1000.0,
    max_drawdown_pct=5.0,
    consecutive_loss_limit=5,
    reconciliation_on_startup=True,
    cancel_stale_orders=True,
    shutdown_timeout_seconds=30.0,
    health_check_interval_seconds=30.0
)

orchestrator = OrdinisOrchestrator(config)
```

#### Core Methods

```python
# Start system
async def start(self) -> bool
    """
    Startup sequence:
    1. Initialize database
    2. Check kill switch
    3. Position reconciliation
    4. Initialize components
    5. Start health monitoring
    """

# Stop system
async def stop(self) -> None
    """
    Shutdown sequence:
    1. Stop accepting orders
    2. Cancel pending orders
    3. Wait for in-flight operations
    4. Persist final state
    5. Close connections
    """

# Get system state
@property
def state(self) -> SystemState

# Health check
async def health_check(self) -> dict[str, bool]
```

#### SystemState Enum

```python
class SystemState(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
```

---

### PositionReconciliation

**Location**: `src/orchestration/reconciliation.py`

Synchronizes local database positions with broker account.

#### Initialization

```python
from orchestration.reconciliation import (
    PositionReconciliation,
    ReconciliationAction
)

reconciliation = PositionReconciliation(
    position_repo=position_repo,
    broker_adapter=broker,
    default_action=ReconciliationAction.ALERT_ONLY
)
```

#### Core Methods

```python
# Run reconciliation
async def reconcile(self) -> ReconciliationResult
    """
    Compare local and broker positions:
    1. Fetch both position sets
    2. Identify discrepancies
    3. Classify by severity
    4. Take action per policy
    """

# Auto-correct discrepancies
async def auto_correct(
    self,
    discrepancies: list[PositionDiscrepancy]
) -> int
    """Update local DB to match broker positions."""
```

#### ReconciliationResult

```python
@dataclass
class ReconciliationResult:
    success: bool
    timestamp: datetime
    local_positions: int
    broker_positions: int
    discrepancies: list[PositionDiscrepancy]
    corrections_made: int
    errors: list[str]
```

#### DiscrepancyType Enum

```python
class DiscrepancyType(Enum):
    QUANTITY_MISMATCH = "quantity_mismatch"
    SIDE_MISMATCH = "side_mismatch"
    MISSING_LOCAL = "missing_local"     # Broker has, we don't
    MISSING_BROKER = "missing_broker"   # We have, broker doesn't
    PRICE_MISMATCH = "price_mismatch"
```

---

## Alerting Layer

### AlertManager

**Location**: `src/alerting/manager.py`

Multi-channel alert dispatcher with rate limiting and deduplication.

#### Initialization

```python
from alerting.manager import AlertManager, AlertSeverity, AlertType

alert_manager = AlertManager(
    rate_limit_seconds=60,
    deduplication_window_seconds=300,
    max_history=1000
)

# Register channels
await alert_manager.register_channel(
    name="desktop",
    async_send_func=send_desktop_notification,
    min_severity=AlertSeverity.WARNING,
    enabled=True
)
```

#### Core Methods

```python
# Send alert
async def send_alert(
    self,
    alert_type: AlertType,
    severity: AlertSeverity,
    title: str,
    message: str,
    metadata: dict | None = None
) -> bool

# Get alert history
def get_history(
    self,
    limit: int = 100,
    severity: AlertSeverity | None = None
) -> list[Alert]

# Register alert channel
async def register_channel(
    self,
    name: str,
    async_send_func: Callable,
    min_severity: AlertSeverity,
    enabled: bool = True
) -> None
```

#### AlertSeverity Enum

```python
class AlertSeverity(Enum):
    INFO = "info"           # Informational
    WARNING = "warning"     # Attention required
    CRITICAL = "critical"   # Immediate action
    EMERGENCY = "emergency" # System halt
```

#### AlertType Enum

```python
class AlertType(Enum):
    KILL_SWITCH = "kill_switch"
    RISK_BREACH = "risk_breach"
    ORDER_REJECTED = "order_rejected"
    POSITION_RECONCILIATION = "position_reconciliation"
    API_CONNECTIVITY = "api_connectivity"
    SYSTEM_HEALTH = "system_health"
    TRADE_EXECUTED = "trade_executed"
    DAILY_SUMMARY = "daily_summary"
    CUSTOM = "custom"
```

#### Usage Example

```python
# Send critical alert
await alert_manager.send_alert(
    alert_type=AlertType.KILL_SWITCH,
    severity=AlertSeverity.EMERGENCY,
    title="Kill Switch Activated",
    message="Daily loss limit exceeded: -$1,250",
    metadata={"trigger": "daily_loss_limit", "pnl": -1250.0}
)

# Review alert history
recent_alerts = alert_manager.get_history(
    limit=50,
    severity=AlertSeverity.CRITICAL
)
```

---

## Interface Layer

### Protocol Definitions

**Location**: `src/interfaces/`

Clean architecture protocol contracts.

#### EventBus Protocol

**Location**: `src/interfaces/event.py`

```python
from typing import Protocol, Callable, Any
from dataclasses import dataclass

@dataclass
class Event:
    """Base event class."""
    event_type: str
    timestamp: datetime
    data: dict[str, Any]

class EventBus(Protocol):
    """Event publication and subscription contract."""

    async def publish(self, event: Event) -> None:
        """Publish event to all subscribers."""

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None]
    ) -> None:
        """Subscribe handler to event type."""

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None]
    ) -> None:
        """Unsubscribe handler from event type."""
```

#### BrokerAdapter Protocol

**Location**: `src/interfaces/broker.py`

```python
from typing import Protocol
from dataclasses import dataclass

@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    side: str  # LONG | SHORT

class BrokerAdapter(Protocol):
    """Broker API abstraction contract."""

    async def get_positions(self) -> list[Position]:
        """Fetch current positions from broker."""

    async def submit_order(self, order: OrderIntent) -> OrderResult:
        """Submit order to broker."""

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Query order status."""
```

#### ExecutionEngine Protocol

**Location**: `src/interfaces/execution.py`

```python
from typing import Protocol

class ExecutionEngine(Protocol):
    """Order execution contract."""

    async def execute_order(
        self,
        order: OrderIntent,
        broker: BrokerAdapter
    ) -> OrderResult:
        """Execute order via broker."""

    async def track_fill(
        self,
        order_id: str,
        broker: BrokerAdapter
    ) -> Fill | None:
        """Monitor order for fill."""
```

#### RiskPolicy Protocol

**Location**: `src/interfaces/risk.py`

```python
from typing import Protocol
from dataclasses import dataclass

@dataclass
class RiskEvaluation:
    passed: bool
    reason: str
    action: str  # approve | reject | reduce

class RiskPolicy(Protocol):
    """Risk evaluation contract."""

    async def evaluate_order(
        self,
        order: OrderIntent,
        portfolio: Portfolio
    ) -> RiskEvaluation:
        """Evaluate order against risk limits."""
```

---

## Usage Examples

### Complete System Initialization

```python
from pathlib import Path
from persistence.database import DatabaseManager
from persistence.repositories.position import PositionRepository
from persistence.repositories.order import OrderRepository
from safety.kill_switch import KillSwitch
from safety.circuit_breaker import CircuitBreaker
from orchestration.orchestrator import OrdinisOrchestrator, OrchestratorConfig
from alerting.manager import AlertManager

async def initialize_system():
    """Initialize all Phase 1 components."""

    # Database
    db = DatabaseManager(
        db_path=Path("data/ordinis.db"),
        backup_dir=Path("data/backups")
    )
    await db.initialize()

    # Repositories
    position_repo = PositionRepository(db)
    order_repo = OrderRepository(db)

    # Safety
    kill_switch = KillSwitch(
        kill_file=Path("data/KILL_SWITCH"),
        db_manager=db,
        daily_loss_limit=1000.0
    )
    await kill_switch.initialize()

    circuit_breaker = CircuitBreaker(name="alpaca_api")

    # Alerting
    alert_manager = AlertManager()

    # Orchestrator
    config = OrchestratorConfig(
        db_path=Path("data/ordinis.db"),
        backup_dir=Path("data/backups"),
        kill_file=Path("data/KILL_SWITCH")
    )
    orchestrator = OrdinisOrchestrator(config)

    # Start system
    success = await orchestrator.start()
    return orchestrator, db, kill_switch
```

### Order Lifecycle with Persistence

```python
async def submit_order_with_persistence(
    order_intent: OrderIntent,
    order_repo: OrderRepository,
    broker: BrokerAdapter,
    circuit_breaker: CircuitBreaker,
    kill_switch: KillSwitch
):
    """Submit order with full Phase 1 integration."""

    # Check kill switch
    if kill_switch.is_active:
        raise RuntimeError("Kill switch active")

    # Create order record
    order_row = OrderRow(
        order_id=str(uuid.uuid4()),
        symbol=order_intent.symbol,
        side=order_intent.side,
        quantity=order_intent.quantity,
        order_type=order_intent.order_type,
        status="created"
    )
    await order_repo.create(order_row)

    try:
        # Submit via circuit breaker
        async with circuit_breaker:
            broker_response = await broker.submit_order(order_intent)

        # Update with broker ID
        await order_repo.update_status(
            order_id=order_row.order_id,
            status="submitted",
            broker_order_id=broker_response.id
        )

    except Exception as e:
        # Record error
        await order_repo.update_status(
            order_id=order_row.order_id,
            status="rejected",
            error_message=str(e)
        )
        raise
```

---

## Error Handling

All Phase 1 components use consistent error handling:

### Database Errors

```python
from persistence.database import DatabaseError

try:
    await db.execute_update(query, params)
except DatabaseError as e:
    logger.error(f"Database operation failed: {e}")
    # Database automatically rolls back transaction
```

### Kill Switch Errors

```python
from safety.kill_switch import KillSwitchError

try:
    await kill_switch.trigger(...)
except KillSwitchError as e:
    logger.critical(f"Kill switch trigger failed: {e}")
    # Manual intervention required
```

### Circuit Breaker Errors

```python
from safety.circuit_breaker import CircuitBreakerOpen

try:
    async with circuit_breaker:
        await api_call()
except CircuitBreakerOpen:
    logger.warning("Circuit breaker open, API unavailable")
    # Retry later or use fallback
```

---

## Testing Support

### Mock Components

Phase 1 components support dependency injection for testing:

```python
# Mock database for testing
class MockDatabase:
    async def execute_query(self, query, params):
        return []

# Mock broker adapter
class MockBroker:
    async def get_positions(self):
        return []

# Use in tests
async def test_position_reconciliation():
    mock_db = MockDatabase()
    mock_broker = MockBroker()

    position_repo = PositionRepository(mock_db)
    reconciliation = PositionReconciliation(
        position_repo=position_repo,
        broker_adapter=mock_broker
    )

    result = await reconciliation.reconcile()
    assert result.success
```

---

## Configuration

### Environment Variables

```bash
# Database
ORDINIS_DB_PATH=data/ordinis.db
ORDINIS_BACKUP_DIR=data/backups

# Kill Switch
ORDINIS_KILL_FILE=data/KILL_SWITCH
ORDINIS_DAILY_LOSS_LIMIT=1000.0
ORDINIS_MAX_DRAWDOWN_PCT=5.0

# Circuit Breaker
ORDINIS_API_FAILURE_THRESHOLD=5
ORDINIS_API_RECOVERY_TIMEOUT=30

# Alerting
ORDINIS_ALERT_RATE_LIMIT=60
ORDINIS_ALERT_DEDUP_WINDOW=300
```

### Configuration File

```yaml
# config/ordinis.yaml
persistence:
  db_path: data/ordinis.db
  backup_dir: data/backups
  wal_mode: true

safety:
  kill_switch:
    kill_file: data/KILL_SWITCH
    daily_loss_limit: 1000.0
    max_drawdown_pct: 5.0
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 30

orchestration:
  reconciliation_on_startup: true
  cancel_stale_orders: true
  shutdown_timeout: 30

alerting:
  rate_limit_seconds: 60
  deduplication_window: 300
```

---

## See Also

- [Production Architecture](../architecture/production-architecture.md) - Comprehensive Phase 1 architecture
- [Architecture Review Response](../architecture/architecture-review-response.md) - Gap analysis
- [SignalCore System](../architecture/signalcore-system.md) - Trading engine integration

---

## Document Metadata

```yaml
version: "phase-1-baseline"
last_reviewed: "{{ now().strftime('%Y-%m-%d') }}"
status: "published"
```

---

**Document Version**: 1.0.0
**Last Updated**: 2025-12-12
**Status**: Current (Phase 1)
