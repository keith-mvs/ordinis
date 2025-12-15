# Production Architecture - Phase 1
# Ordinis Trading System
# Version: 1.0.0
# Last Updated: 2025-12-12

---

## Executive Summary

This document describes the production-ready architecture implemented in Phase 1, which addresses critical operational requirements for reliable paper and live trading. The implementation focuses on **persistence, safety, orchestration, and observability** while maintaining the core SignalCore engine design.

### Phase 1 Achievements

Phase 1 transforms Ordinis from a research prototype to a production-capable system by adding:

1. **Persistent State Management**: SQLite-based persistence with WAL mode, automatic backups
2. **Safety Controls**: Kill switch with multiple triggers, circuit breaker for API resilience
3. **System Orchestration**: Coordinated startup/shutdown, position reconciliation, component lifecycle
4. **Alerting Infrastructure**: Multi-channel alerts with rate limiting and deduplication

### Architecture Review Gaps Addressed

This implementation directly addresses several P0 gaps from the external architecture review:

| Review Gap | Phase 1 Implementation | Status |
|------------|----------------------|--------|
| OMS abstraction | Order repository with lifecycle tracking | ✅ Addressed |
| Execution model | Order state machine, broker reconciliation | ✅ Addressed |
| Kill switch controls | Kill switch with file/DB/programmatic triggers | ✅ Addressed |
| Backtest/live parity | Position/order/fill persistence for both modes | ✅ Addressed |
| Observability | Alert manager, structured logging foundation | 🟡 Partial |
| Event model | Not addressed (deferred to Phase 2) | ⏸️ Deferred |
| Typed domain objects | Using Pydantic models in persistence layer | 🟡 Partial |
| Config management | Not addressed (deferred) | ⏸️ Deferred |

---

## 1. System Architecture Overview

### 1.1 Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ORDINIS PRODUCTION ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                 ORCHESTRATION LAYER                             │    │
│  │  ┌──────────────────┐        ┌──────────────────────────┐      │    │
│  │  │  Orchestrator    │───────▶│  Position Reconciliation │      │    │
│  │  │  - Startup       │        │  - Broker sync           │      │    │
│  │  │  - Shutdown      │        │  - Discrepancy detect    │      │    │
│  │  │  - Health checks │        │  - Auto-correction       │      │    │
│  │  └──────────────────┘        └──────────────────────────┘      │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                    │                                    │
│                                    ▼                                    │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                     SAFETY LAYER                                │    │
│  │  ┌──────────────────┐        ┌──────────────────────────┐      │    │
│  │  │  Kill Switch     │        │  Circuit Breaker         │      │    │
│  │  │  - File trigger  │        │  - API monitoring        │      │    │
│  │  │  - DB persist    │        │  - Auto-recovery         │      │    │
│  │  │  - Risk triggers │        │  - Failure detection     │      │    │
│  │  └──────────────────┘        └──────────────────────────┘      │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                    │                                    │
│                                    ▼                                    │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                SIGNALCORE ENGINE LAYER                          │    │
│  │  ┌───────────┐    ┌───────────┐    ┌───────────────────┐      │    │
│  │  │SignalCore │───▶│ RiskGuard │───▶│    FlowRoute      │      │    │
│  │  │(Signals)  │    │  (Risk)   │    │   (Execution)     │      │    │
│  │  └───────────┘    └───────────┘    └───────────────────┘      │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                    │                                    │
│                                    ▼                                    │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                  PERSISTENCE LAYER                              │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │    │
│  │  │Position  │  │  Order   │  │  Fill    │  │ System State │   │    │
│  │  │Repository│  │Repository│  │Repository│  │  Repository  │   │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │    │
│  │                           │                                     │    │
│  │                           ▼                                     │    │
│  │                ┌─────────────────────┐                         │    │
│  │                │  DatabaseManager    │                         │    │
│  │                │  - SQLite + WAL     │                         │    │
│  │                │  - Auto backup      │                         │    │
│  │                │  - Transactions     │                         │    │
│  │                └─────────────────────┘                         │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                    │                                    │
│                                    ▼                                    │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                   ALERTING LAYER                                │    │
│  │  ┌──────────────────────────────────────────────────────┐      │    │
│  │  │              AlertManager                             │      │    │
│  │  │  - Desktop notifications  - Email (future)            │      │    │
│  │  │  - Rate limiting         - Deduplication              │      │    │
│  │  │  - Severity routing      - Alert history              │      │    │
│  │  └──────────────────────────────────────────────────────┘      │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Repository Structure

```
ordinis/
├── src/
│   ├── persistence/          # NEW: State persistence
│   │   ├── database.py       # Database connection manager
│   │   ├── models.py         # Pydantic data models
│   │   ├── schema.py         # Database schema DDL
│   │   └── repositories/     # Repository pattern
│   │       ├── position.py
│   │       ├── order.py
│   │       ├── fill.py
│   │       ├── trade.py
│   │       └── system_state.py
│   │
│   ├── safety/               # NEW: Safety controls
│   │   ├── kill_switch.py    # Emergency halt
│   │   └── circuit_breaker.py # API resilience
│   │
│   ├── orchestration/        # NEW: System coordination
│   │   ├── orchestrator.py   # Lifecycle manager
│   │   └── reconciliation.py # Position sync
│   │
│   ├── alerting/             # NEW: Multi-channel alerts
│   │   └── manager.py        # Alert dispatcher
│   │
│   ├── engines/              # Existing: Core engines
│   │   ├── signalcore/
│   │   ├── riskguard/
│   │   ├── flowroute/
│   │   └── proofbench/
│   │
│   └── ...                   # Other existing modules
│
├── data/                     # Persistent data
│   ├── ordinis.db           # SQLite database
│   ├── ordinis.db-wal       # WAL file
│   ├── ordinis.db-shm       # Shared memory
│   ├── backups/             # Automatic backups
│   └── KILL_SWITCH          # Emergency halt trigger
│
└── docs/
    └── architecture/
        ├── production-architecture.md  # This file
        ├── layered-system-architecture.md
        └── signalcore-system.md
```

---

## 2. Persistence Layer

### 2.1 Design Principles

- **Single Source of Truth**: SQLite database as authoritative state store
- **WAL Mode**: Write-Ahead Logging for concurrent read performance
- **Automatic Backups**: Pre-session backups with timestamped archives
- **Transaction Safety**: ACID guarantees with explicit transaction management
- **Schema Versioning**: Tracked schema version for migrations

### 2.2 Database Manager

**Location**: `src/persistence/database.py`

**Responsibilities**:
- Async SQLite connection management via aiosqlite
- WAL mode configuration for concurrent reads
- Automatic backup creation (timestamp-based)
- Schema initialization and versioning
- Connection health monitoring
- Graceful shutdown with pending transaction completion

**Key Features**:
```python
class DatabaseManager:
    - initialize() -> bool              # Connect + schema setup
    - close() -> None                   # Graceful shutdown
    - execute_query() -> list[tuple]    # Read operations
    - execute_update() -> int           # Write operations
    - backup() -> bool                  # Manual backup
    - health_check() -> bool            # Connection status
```

**Configuration**:
```python
PRAGMA journal_mode=WAL          # Concurrent reads
PRAGMA synchronous=NORMAL        # Balance safety/performance
PRAGMA foreign_keys=ON           # Referential integrity
PRAGMA busy_timeout=5000         # Lock retry timeout
```

### 2.3 Repository Pattern

**Location**: `src/persistence/repositories/`

Each repository provides a clean interface to database operations for a specific entity:

| Repository | Entity | Key Operations |
|------------|--------|----------------|
| `PositionRepository` | Trading positions | upsert, get_by_symbol, get_all, get_active |
| `OrderRepository` | Order lifecycle | create, update_status, get_by_id, get_pending |
| `FillRepository` | Order fills | record_fill, get_by_order_id |
| `TradeRepository` | Completed trades | record_trade, get_by_date_range |
| `SystemStateRepository` | System state | set_state, get_state |

**Interface Pattern**:
```python
class PositionRepository:
    def __init__(self, db: DatabaseManager)

    async def upsert(self, position: PositionRow) -> bool
    async def get_by_symbol(self, symbol: str) -> PositionRow | None
    async def get_all(self) -> list[PositionRow]
    async def get_active(self) -> list[PositionRow]
    async def delete(self, symbol: str) -> bool
```

### 2.4 Data Models

**Location**: `src/persistence/models.py`

Pydantic models provide type-safe representations of database rows:

```python
class PositionRow(BaseModel):
    id: int | None = None
    symbol: str
    side: str  # 'LONG' | 'SHORT' | 'FLAT'
    quantity: int
    avg_cost: float
    current_price: float
    realized_pnl: float
    unrealized_pnl: float
    entry_time: str | None
    last_update: str
    created_at: str | None
    updated_at: str | None
```

```python
class OrderRow(BaseModel):
    order_id: str
    symbol: str
    side: str  # 'BUY' | 'SELL'
    quantity: int
    order_type: str  # 'MARKET' | 'LIMIT' | 'STOP'
    status: str  # 'created' | 'submitted' | 'filled' | 'rejected'
    broker_order_id: str | None
    broker_response: str | None  # JSON
    error_message: str | None
    # ... additional fields
```

### 2.5 Database Schema

**Location**: `src/persistence/schema.py`

**Schema Version**: 1

**Tables**:

1. **positions**: Current trading positions
   - Primary key: symbol (unique)
   - Tracks: side, quantity, P&L, entry/update times

2. **orders**: Order lifecycle tracking
   - Primary key: order_id (internal UUID)
   - Foreign key: broker_order_id (broker system)
   - Tracks: status, fills, retries, errors

3. **fills**: Individual order fills
   - Links to orders table
   - Tracks: fill price, quantity, timestamp

4. **trades**: Completed round-trip trades
   - Aggregates entry and exit fills
   - Tracks: realized P&L, duration, strategy

5. **system_state**: System configuration and state
   - Key-value store for system-wide state
   - Used by kill switch, reconciliation

**Backup Strategy**:
- Pre-initialization backup if DB exists
- On-demand backup via `DatabaseManager.backup()`
- Backup format: `ordinis_backup_YYYYMMDD_HHMMSS.db`
- Retention: Manual cleanup (future: automated retention policy)

---

## 3. Safety Layer

### 3.1 Kill Switch

**Location**: `src/safety/kill_switch.py`

**Purpose**: Emergency halt mechanism with multiple trigger sources and persistent state.

**Trigger Sources**:
1. **File-based**: Create `data/KILL_SWITCH` file (manual emergency)
2. **Programmatic**: API call from risk engine or operator
3. **Auto-triggers**:
   - Daily loss limit breach
   - Max drawdown exceeded
   - Consecutive loss limit
   - API connectivity failure
   - Position reconciliation failure

**State Model**:
```python
class KillSwitchState:
    active: bool
    reason: KillSwitchReason
    message: str
    timestamp: datetime
    triggered_by: str
    metadata: dict[str, Any]
```

**Lifecycle**:
```
INACTIVE ──[trigger]──▶ ACTIVE ──[reset]──▶ INACTIVE
              │                      │
              └──▶ [persist to DB]   │
              └──▶ [create file]     │
              └──▶ [notify callbacks]│
                                     │
                          [manual intervention required]
```

**Integration**:
- RiskGuard checks kill switch before order submission
- FlowRoute blocks order routing when active
- Orchestrator monitors kill switch state
- Alert manager notifies on activation/deactivation

**Configuration**:
```python
daily_loss_limit: float = 1000.0        # Auto-trigger threshold
max_drawdown_pct: float = 5.0           # Percent from peak
consecutive_loss_limit: int = 5         # Losing trades
check_interval_seconds: float = 1.0     # File polling
```

### 3.2 Circuit Breaker

**Location**: `src/safety/circuit_breaker.py`

**Purpose**: Protect against cascading API failures using circuit breaker pattern.

**States**:
```
CLOSED (normal) ──[failures >= threshold]──▶ OPEN (blocking)
       ▲                                           │
       │                                           │
       │                                  [recovery timeout]
       │                                           │
       │                                           ▼
       └──[success]─── HALF_OPEN (testing) ◀──────┘
                              │
                     [failure]──▶ OPEN
```

**Statistics Tracked**:
```python
class CircuitStats:
    total_calls: int
    successful_calls: int
    failed_calls: int
    consecutive_failures: int
    consecutive_successes: int
    last_failure_time: datetime | None
    last_success_time: datetime | None
    state_changes: list[tuple[datetime, CircuitState]]
```

**Configuration**:
```python
failure_threshold: int = 5              # Consecutive failures to open
success_threshold: int = 3              # Consecutive successes to close
recovery_timeout_seconds: float = 30.0  # Time before testing recovery
half_open_max_calls: int = 3           # Max calls in half-open state
```

**Usage Example**:
```python
circuit = CircuitBreaker(name="alpaca_api")

async def call_api():
    async with circuit:
        response = await broker_api.get_positions()
        return response
```

**Integration with Kill Switch**:
- Circuit breaker can trigger kill switch on sustained API failure
- Kill switch activation stops all API calls (circuit forced open)

---

## 4. Orchestration Layer

### 4.1 System Orchestrator

**Location**: `src/orchestration/orchestrator.py`

**Purpose**: Central coordinator for system lifecycle, component initialization, and shutdown sequences.

**System States**:
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

**Startup Sequence**:
```
1. Database initialization
   └─▶ Create schema if needed
   └─▶ Verify integrity
   └─▶ Create backup

2. Kill switch check
   └─▶ Load persisted state
   └─▶ Check file trigger
   └─▶ If active: HALT

3. Position reconciliation
   └─▶ Fetch broker positions
   └─▶ Compare with local DB
   └─▶ Log discrepancies
   └─▶ Optional auto-correct

4. Component startup
   └─▶ Initialize alert manager
   └─▶ Connect broker adapter
   └─▶ Start engines (Signal, Risk)
   └─▶ Begin health monitoring

5. Transition to RUNNING
```

**Shutdown Sequence**:
```
1. Stop accepting new orders
2. Cancel pending orders (optional)
3. Wait for in-flight operations
4. Stop health monitoring
5. Close broker connections
6. Persist final state
7. Close database connection
8. Transition to STOPPED
```

**Configuration**:
```python
@dataclass
class OrchestratorConfig:
    db_path: Path
    backup_dir: Path
    kill_file: Path
    daily_loss_limit: float
    max_drawdown_pct: float
    consecutive_loss_limit: int
    reconciliation_on_startup: bool = True
    cancel_stale_orders: bool = True
    shutdown_timeout_seconds: float = 30.0
    health_check_interval_seconds: float = 30.0
```

**Health Monitoring**:
- Periodic health checks for all components
- Database connection liveness
- Broker API connectivity (via circuit breaker)
- Kill switch state monitoring
- Metrics collection (future: Prometheus integration)

### 4.2 Position Reconciliation

**Location**: `src/orchestration/reconciliation.py`

**Purpose**: Ensure consistency between local database positions and broker account positions.

**Discrepancy Types**:
```python
class DiscrepancyType(Enum):
    QUANTITY_MISMATCH = "quantity_mismatch"
    SIDE_MISMATCH = "side_mismatch"
    MISSING_LOCAL = "missing_local"         # Broker has, we don't
    MISSING_BROKER = "missing_broker"       # We have, broker doesn't
    PRICE_MISMATCH = "price_mismatch"
```

**Reconciliation Actions**:
```python
class ReconciliationAction(Enum):
    ALERT_ONLY = "alert_only"           # Log and notify
    AUTO_CORRECT = "auto_correct"       # Update local DB from broker
    HALT_TRADING = "halt_trading"       # Trigger kill switch
```

**Reconciliation Flow**:
```
1. Fetch broker positions via FlowRoute
2. Fetch local positions from PositionRepository
3. Compare symbol by symbol:
   - Check quantity match
   - Check side consistency
   - Flag missing positions
4. Classify discrepancies by severity
5. Take action based on policy:
   - Critical: Trigger kill switch + alert
   - Medium: Alert + optional auto-correct
   - Low: Log only
6. Generate ReconciliationResult
7. Persist to audit trail
```

**Result Model**:
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

**Integration**:
- Orchestrator calls reconciliation on startup
- Periodic reconciliation during runtime (configurable)
- Post-trade reconciliation after significant fills
- Alert manager notified on discrepancies

---

## 5. Alerting Layer

### 5.1 Alert Manager

**Location**: `src/alerting/manager.py`

**Purpose**: Centralized multi-channel alerting with rate limiting, deduplication, and severity-based routing.

**Alert Types**:
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

**Severity Levels**:
```python
class AlertSeverity(Enum):
    INFO = "info"           # Informational
    WARNING = "warning"     # Attention required
    CRITICAL = "critical"   # Immediate action
    EMERGENCY = "emergency" # System halt
```

**Alert Channels**:

| Channel | Implementation | Status | Min Severity |
|---------|----------------|--------|--------------|
| Desktop | plyer notifications | ✅ Implemented | WARNING |
| Email | SMTP (future) | ⏸️ Planned | CRITICAL |
| SMS | Twilio (future) | ⏸️ Planned | EMERGENCY |
| Slack | Webhook (future) | ⏸️ Planned | WARNING |
| Log | Python logging | ✅ Implemented | INFO |

**Rate Limiting**:
- Per alert type cooldown (default: 60 seconds)
- Prevents alert flooding
- Tracks suppressed alerts for reporting

**Deduplication**:
- Content-based hashing (title + message)
- Deduplication window (default: 5 minutes)
- Suppresses identical alerts within window

**Alert Model**:
```python
@dataclass
class Alert:
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    metadata: dict[str, Any]
    acknowledged: bool
    channels_sent: list[str]
```

**Integration**:
- Kill switch triggers EMERGENCY alerts
- Reconciliation failures trigger CRITICAL alerts
- Order rejections trigger WARNING alerts
- System health triggers INFO/WARNING alerts

---

## 6. Integration with SignalCore Engines

### 6.1 Engine Responsibilities

The Phase 1 infrastructure integrates with the existing SignalCore 5-engine architecture:

| Engine | Phase 1 Integration | New Responsibilities |
|--------|-------------------|---------------------|
| **Cortex** | No changes | Advisory layer (unchanged) |
| **SignalCore** | No changes | Signal generation (unchanged) |
| **RiskGuard** | ✅ Enhanced | Kill switch check, circuit breaker monitoring |
| **FlowRoute** | ✅ Enhanced | Order persistence, reconciliation, kill switch enforcement |
| **ProofBench** | 🟡 Partial | Backtest persistence (future) |

### 6.2 RiskGuard Integration

**New Risk Checks**:
```python
async def evaluate_order(
    self,
    order: OrderIntent,
    portfolio: Portfolio
) -> RiskEvaluation:

    # Existing checks
    position_size_check()
    sector_concentration_check()
    daily_loss_check()

    # NEW: Phase 1 checks
    if kill_switch.is_active:
        return RiskEvaluation(
            passed=False,
            reason="kill_switch_active",
            action="reject"
        )

    if circuit_breaker.is_open:
        return RiskEvaluation(
            passed=False,
            reason="circuit_breaker_open",
            action="reject"
        )

    # Existing risk calculations
    ...
```

### 6.3 FlowRoute Integration

**Order Lifecycle with Persistence**:
```
1. Receive OrderIntent from RiskGuard
2. Create OrderRow in database (status='created')
3. Submit to broker API
4. Update OrderRow (status='submitted', broker_order_id)
5. Stream fill updates
6. Record each fill in FillRepository
7. Update OrderRow (status='filled', avg_fill_price)
8. Update PositionRepository
9. If position closed, record in TradeRepository
10. Trigger position reconciliation
```

**Enhanced Error Handling**:
```python
async def submit_order(self, order: OrderIntent) -> OrderResult:
    # Create database record
    order_row = await order_repo.create(order)

    try:
        # Submit via circuit breaker
        async with circuit_breaker:
            broker_response = await broker.submit_order(order)

        # Update with broker order ID
        await order_repo.update_status(
            order_id=order_row.order_id,
            status='submitted',
            broker_order_id=broker_response.id
        )

    except APIError as e:
        # Persist error
        await order_repo.update_status(
            order_id=order_row.order_id,
            status='rejected',
            error_message=str(e)
        )

        # Alert
        await alert_manager.send_alert(
            alert_type=AlertType.ORDER_REJECTED,
            severity=AlertSeverity.WARNING,
            message=f"Order rejected: {e}"
        )

        raise
```

---

## 7. Data Flow Diagrams

### 7.1 System Startup Flow

```
┌──────────────┐
│  User Start  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│ Orchestrator.start()         │
├──────────────────────────────┤
│ 1. Initialize Database       │
│    - Create schema           │
│    - Verify integrity        │
│    - Create backup           │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ 2. Check Kill Switch         │
│    - Load DB state           │
│    - Check file trigger      │
│    - Load configuration      │
└──────┬───────────────────────┘
       │
       ▼ [If active]
┌──────────────────────────────┐
│ HALT: Kill Switch Active     │
│ - Alert EMERGENCY            │
│ - Wait for manual reset      │
└──────────────────────────────┘

       │ [If inactive]
       ▼
┌──────────────────────────────┐
│ 3. Position Reconciliation   │
│    - Fetch broker positions  │
│    - Compare with DB         │
│    - Log discrepancies       │
│    - Auto-correct if enabled │
└──────┬───────────────────────┘
       │
       ▼ [If critical error]
┌──────────────────────────────┐
│ Trigger Kill Switch          │
│ - Alert CRITICAL             │
│ - Stop startup               │
└──────────────────────────────┘

       │ [If pass or corrected]
       ▼
┌──────────────────────────────┐
│ 4. Initialize Components     │
│    - Alert Manager           │
│    - Broker Adapter          │
│    - Circuit Breakers        │
│    - Signal Engine           │
│    - Risk Engine             │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ 5. Start Health Monitoring   │
│    - Component health        │
│    - API connectivity        │
│    - Kill switch state       │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ System State: RUNNING        │
└──────────────────────────────┘
```

### 7.2 Order Execution Flow

```
┌─────────────┐
│ Signal      │
│ Generated   │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────┐
│ RiskGuard.evaluate()         │
├──────────────────────────────┤
│ Check:                       │
│ - Kill switch status    ❌   │
│ - Circuit breaker       ❌   │
│ - Position limits       ❌   │
│ - Risk limits           ❌   │
└──────┬───────────────────────┘
       │
       ▼ [Rejected]
┌──────────────────────────────┐
│ Alert: ORDER_REJECTED        │
│ Persist rejection reason     │
└──────────────────────────────┘

       │ [Approved]
       ▼
┌──────────────────────────────┐
│ FlowRoute.submit_order()     │
├──────────────────────────────┤
│ 1. Create OrderRow in DB     │
│    status='created'          │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ 2. Submit via Circuit        │
│    Breaker to Broker API     │
└──────┬───────────────────────┘
       │
       ▼ [API Error]
┌──────────────────────────────┐
│ Circuit Breaker Triggered    │
│ - Record failure             │
│ - Update OrderRow error      │
│ - Alert: API_CONNECTIVITY    │
└──────────────────────────────┘

       │ [Success]
       ▼
┌──────────────────────────────┐
│ 3. Update OrderRow           │
│    status='submitted'        │
│    broker_order_id=...       │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ 4. Stream Fill Updates       │
│    - Partial fills           │
│    - Full fills              │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ 5. Record Each Fill          │
│    - FillRepository.record() │
│    - Update OrderRow         │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ 6. Update Position           │
│    - PositionRepository      │
│    - Calculate P&L           │
└──────┬───────────────────────┘
       │
       ▼ [Position closed]
┌──────────────────────────────┐
│ 7. Record Trade              │
│    - TradeRepository         │
│    - Realized P&L            │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ 8. Position Reconciliation   │
│    - Verify broker sync      │
│    - Alert if discrepancy    │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│ 9. Alert: TRADE_EXECUTED     │
│    - Summary notification    │
└──────────────────────────────┘
```

### 7.3 Kill Switch Trigger Flow

```
┌─────────────────────────────────────────────┐
│         Kill Switch Triggers                 │
├─────────────────────────────────────────────┤
│                                              │
│  ┌─────────────────┐  ┌─────────────────┐  │
│  │ File Trigger    │  │ Programmatic    │  │
│  │ KILL_SWITCH     │  │ API Call        │  │
│  │ file created    │  │ from code       │  │
│  └────────┬────────┘  └────────┬────────┘  │
│           │                     │           │
│  ┌────────▼─────────────────────▼────────┐ │
│  │        Auto-Triggers                  │ │
│  │  - Daily loss limit                   │ │
│  │  - Max drawdown                       │ │
│  │  - Consecutive losses                 │ │
│  │  - API failure (circuit breaker)      │ │
│  │  - Position reconciliation critical   │ │
│  └────────────────┬──────────────────────┘ │
│                   │                         │
└───────────────────┼─────────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ KillSwitch.trigger() │
         └──────────┬───────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌───────────┐ ┌─────────┐ ┌────────────┐
│ Persist   │ │ Create  │ │ Notify     │
│ to DB     │ │ File    │ │ Callbacks  │
│ state     │ │ Marker  │ │            │
└───────────┘ └─────────┘ └─────┬──────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
            ┌──────────┐  ┌───────────┐ ┌────────────┐
            │RiskGuard │  │FlowRoute  │ │ Alert      │
            │blocks    │  │blocks     │ │ EMERGENCY  │
            │orders    │  │submission │ │            │
            └──────────┘  └───────────┘ └────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ System State: HALTED │
         │                      │
         │ Manual reset required│
         └──────────────────────┘
```

---

## 8. Operational Considerations

### 8.1 Database Management

**Backup Strategy**:
- Automatic backup on system start (if DB exists)
- Manual backup command: `DatabaseManager.backup()`
- Backup location: `data/backups/ordinis_backup_YYYYMMDD_HHMMSS.db`
- Recommended: Automated backup before trading sessions
- Future: Retention policy with automated cleanup

**Recovery Procedures**:
1. Identify latest good backup in `data/backups/`
2. Stop Ordinis system
3. Copy backup to `data/ordinis.db`
4. Restart system
5. Verify position reconciliation

**Schema Migrations** (Future):
- Schema version tracked in database
- Migration scripts in `src/persistence/migrations/`
- Applied on startup if version mismatch

### 8.2 Kill Switch Operations

**Manual Activation**:
```bash
# Method 1: Touch file
touch data/KILL_SWITCH

# Method 2: API call (in code)
await kill_switch.trigger(
    reason=KillSwitchReason.MANUAL,
    message="Manual halt requested",
    triggered_by="operator"
)
```

**Manual Deactivation**:
```bash
# Method 1: Remove file
rm data/KILL_SWITCH

# Method 2: API call (in code)
await kill_switch.reset()
```

**Best Practices**:
- Always review position reconciliation before resetting
- Check broker account state before resuming trading
- Document reason for activation in metadata
- Alert team when kill switch activated

### 8.3 Position Reconciliation

**When to Reconcile**:
- System startup (mandatory)
- After fills (automatic)
- Periodic interval (configurable, e.g., hourly)
- On-demand via API call
- Before/after trading day

**Handling Discrepancies**:

| Severity | Condition | Action |
|----------|-----------|--------|
| CRITICAL | Side mismatch, missing position (broker) | Trigger kill switch, CRITICAL alert |
| MEDIUM | Quantity mismatch > 10% | WARNING alert, optional auto-correct |
| LOW | Price drift, small quantity diff | INFO log only |

**Auto-Correction**:
- Configurable via `reconciliation_on_startup` and `auto_correct_discrepancies`
- Updates local DB to match broker state
- Logs all corrections to audit trail
- Does NOT modify broker positions

### 8.4 Alert Management

**Channel Configuration**:
```python
# Desktop notifications (implemented)
alert_manager.register_channel(
    name="desktop",
    async_send_func=desktop_notify,
    min_severity=AlertSeverity.WARNING,
    enabled=True
)

# Email (future)
alert_manager.register_channel(
    name="email",
    async_send_func=send_email,
    min_severity=AlertSeverity.CRITICAL,
    enabled=False  # Not yet implemented
)
```

**Alert History**:
- In-memory history (max 1000 alerts by default)
- Future: Persist to database for audit trail
- Access via `alert_manager.get_history()`

**Rate Limiting**:
- Per alert type cooldown (default: 60s)
- Deduplication window (default: 5 minutes)
- Suppressed alerts tracked in metrics

---

## 9. Testing Strategy

### 9.1 Unit Tests

**Persistence Layer**:
- DatabaseManager initialization and backup
- Repository CRUD operations
- Model serialization/deserialization
- Transaction rollback on error

**Safety Layer**:
- Kill switch trigger/reset
- Circuit breaker state transitions
- Auto-trigger conditions

**Orchestration**:
- Startup sequence
- Shutdown sequence
- Position reconciliation logic

**Alerting**:
- Rate limiting
- Deduplication
- Channel routing

### 9.2 Integration Tests

**End-to-End Flows**:
- System startup with reconciliation
- Order submission with persistence
- Fill processing with position updates
- Kill switch activation and recovery
- Circuit breaker failure handling

**Database Integration**:
- Concurrent read/write (WAL mode)
- Transaction isolation
- Backup/restore

**Broker Integration**:
- Position reconciliation with paper broker
- Order lifecycle with Alpaca API
- Error handling and retries

### 9.3 Production Testing

**Paper Trading Validation**:
- Run full system with paper broker
- Verify persistence across restarts
- Test kill switch triggers
- Validate position reconciliation
- Monitor alert delivery

**Chaos Testing** (Future):
- Simulate API failures
- Database corruption scenarios
- Network partitions
- Out-of-order fills

---

## 10. Architecture Review Gap Analysis

### 10.1 Gaps Addressed in Phase 1

| Review Item | Status | Implementation |
|-------------|--------|----------------|
| **OMS abstraction** | ✅ Addressed | OrderRepository with full lifecycle tracking |
| **Execution model** | ✅ Addressed | Order state machine, broker reconciliation |
| **Kill switch** | ✅ Addressed | Multi-trigger kill switch with persistence |
| **Backtest/live parity** | ✅ Addressed | Shared persistence layer for both modes |
| **Failure containment** | 🟡 Partial | Circuit breaker for API, async error handling |
| **Broker adapter interface** | 🟡 Partial | BrokerAdapter protocol defined, Alpaca implemented |
| **Position reconciliation** | ✅ Addressed | Full reconciliation with auto-correct option |
| **Observability foundations** | 🟡 Partial | Alert manager, structured logging started |

### 10.2 Deferred to Phase 2+

| Review Item | Reason for Deferral | Target Phase |
|-------------|---------------------|--------------|
| **Event bus contract** | Requires major refactor, backtest engine redesign | Phase 2 |
| **Typed domain objects everywhere** | Partial implementation sufficient for Phase 1 | Phase 2 |
| **Config management** | Current env vars + code config acceptable | Phase 2 |
| **Full observability** | Metrics/tracing infrastructure requires tooling | Phase 3 |
| **Time model (simulated vs real)** | Complex, tied to event bus redesign | Phase 2 |
| **Async/sync boundary clarity** | Works with current hybrid approach | Phase 2 |
| **Provenance + reconciliation (data)** | Market data focus, not trading execution | Phase 4 |
| **LLM guardrails + audit** | Cortex is advisory, not in critical path | Phase 3 |

### 10.3 Architectural Debt

**Known Limitations**:

1. **No Event Bus**: Order flow is still procedural, not event-driven
   - Impact: Harder to add new consumers of order/fill events
   - Mitigation: Repository pattern provides abstraction for future event emission

2. **DataFrame Boundaries**: Still using pandas DataFrames in strategy interface
   - Impact: Not type-safe at strategy boundary
   - Mitigation: Pydantic models in persistence layer, future conversion

3. **Sync/Async Mixing**: Strategies are sync, infrastructure is async
   - Impact: Requires adapter layer, potential blocking
   - Mitigation: asyncio executor for sync strategies

4. **No Formal Service Contracts**: Engine interfaces are Python protocols, not versioned schemas
   - Impact: Breaking changes harder to detect
   - Mitigation: Good test coverage, semantic versioning

5. **No Distributed Tracing**: Logs are structured but not correlated across components
   - Impact: Harder to debug complex flows
   - Mitigation: Correlation IDs in place, future OpenTelemetry integration

---

## 11. Future Enhancements

### 11.1 Phase 2: Event-Driven Refactor

**Goals**:
- Implement event bus (in-memory, then persistent)
- Define event taxonomy (MarketData, Signal, Order, Fill, Risk, Metric)
- Refactor engines to publish/subscribe model
- Add event replay for debugging

**Benefits**:
- Better backtest/live parity
- Easier to add new consumers
- Audit trail by default
- Support for complex event processing

### 11.2 Phase 3: Observability

**Goals**:
- Prometheus metrics export
- Jaeger distributed tracing
- Grafana dashboards
- Log aggregation (ELK stack or Loki)

**Metrics**:
- Order latency (signal to fill)
- Fill rate, rejection rate
- Position P&L by strategy
- API latency percentiles
- Circuit breaker state changes

### 11.3 Phase 4: Data Provenance

**Goals**:
- Track data source for each bar/quote
- Timestamp reconciliation across providers
- Data quality metrics
- Fallback provider logic

### 11.4 Phase 5: Multi-Asset Support

**Goals**:
- Options lifecycle (exercise/assignment)
- Futures (roll logic)
- Forex (24hr trading)
- Crypto (high-frequency updates)

---

## 12. Deployment Architecture

### 12.1 Development Environment

```
Developer Machine
├── SQLite database (local file)
├── Paper broker connection
├── Desktop alerts
└── Local logs
```

### 12.2 Production Environment (Future)

```
┌─────────────────────────────────────────────┐
│            Production Deployment             │
├─────────────────────────────────────────────┤
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │  Ordinis Orchestrator                │   │
│  │  - Kubernetes deployment             │   │
│  │  - Autoscaling disabled (stateful)   │   │
│  └──────────────┬───────────────────────┘   │
│                 │                            │
│  ┌──────────────▼───────────────────────┐   │
│  │  PostgreSQL (replace SQLite)         │   │
│  │  - High availability                 │   │
│  │  - WAL archiving                     │   │
│  │  - Point-in-time recovery            │   │
│  └──────────────────────────────────────┘   │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │  Observability Stack                 │   │
│  │  - Prometheus (metrics)              │   │
│  │  - Jaeger (tracing)                  │   │
│  │  - ELK/Loki (logs)                   │   │
│  └──────────────────────────────────────┘   │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │  Alert Delivery                      │   │
│  │  - Email (SMTP)                      │   │
│  │  - SMS (Twilio)                      │   │
│  │  - Slack (webhook)                   │   │
│  └──────────────────────────────────────┘   │
│                                              │
└─────────────────────────────────────────────┘
```

### 12.3 Database Migration Path

**Phase 1**: SQLite (current)
- Single-file database
- WAL mode for concurrency
- Suitable for single-instance deployment

**Phase 2**: PostgreSQL
- Multi-client support
- Better concurrency
- Native JSON columns
- Triggers and stored procedures
- Replication for HA

**Migration Strategy**:
- Repository pattern abstracts database type
- Create PostgreSQL repositories alongside SQLite
- Feature flag to switch backends
- Data migration script (SQLite → PostgreSQL)

---

## 13. Security Considerations

### 13.1 Data Protection

**At Rest**:
- Database file permissions (0600)
- Future: Database encryption (SQLCipher or PostgreSQL encryption)
- Backup encryption

**In Transit**:
- Broker API: TLS 1.3 (Alpaca enforced)
- Internal: localhost only (no network exposure)

**Secrets Management**:
- Environment variables for API keys
- Future: HashiCorp Vault integration
- Never log secrets

### 13.2 Access Control

**File System**:
- Database: Read/write by Ordinis process only
- Logs: Read by Ordinis + admin
- Backups: Admin only

**API Keys**:
- Paper trading keys (development)
- Live trading keys (production, restricted)
- Separate keys per environment

### 13.3 Audit Trail

**Logged Events**:
- All order submissions (intent, result, broker response)
- All kills switch activations/deactivations
- Position reconciliation results
- Alert deliveries
- Configuration changes

**Log Retention**:
- 30 days for operational logs
- 7 years for trade audit (compliance)
- Immutable logs (append-only)

---

## 14. Compliance & Regulations

### 14.1 Record Keeping

**Requirements**:
- All orders (submitted, rejected, filled)
- Position history with timestamps
- Configuration snapshots per trading session
- Kill switch activations

**Implementation**:
- Database persistence (orders, fills, trades)
- System state repository (config snapshots)
- Future: Immutable audit log table

### 14.2 Broker Terms of Service

**Alpaca**:
- Rate limits: 200 requests/minute
- Market data: Real-time (live), delayed (paper)
- Pattern day trader rules enforced
- Shorting restrictions

**Implementation**:
- Circuit breaker respects rate limits
- Rate limiter in FlowRoute (future)
- Kill switch on broker errors

---

## 15. Metrics & KPIs

### 15.1 System Health Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Database query latency | < 10ms p99 | > 50ms |
| Order submission latency | < 500ms p99 | > 2s |
| Position reconciliation time | < 5s | > 30s |
| Circuit breaker failures | < 5/hr | > 10/hr |
| Kill switch activations | 0/day | > 0 |

### 15.2 Trading Performance Metrics

| Metric | Tracked By | Reported |
|--------|-----------|----------|
| Fills vs rejections | OrderRepository | Daily summary |
| Slippage (expected vs actual) | FillRepository | Per trade |
| Position hold time | TradeRepository | Per trade |
| Realized P&L | TradeRepository | Daily/weekly |
| Discrepancy rate | Reconciliation | Per reconciliation |

### 15.3 Operational Metrics

| Metric | Source | Usage |
|--------|--------|-------|
| System uptime | Orchestrator | SLA tracking |
| Startup time | Orchestrator | Performance |
| Backup success rate | DatabaseManager | Reliability |
| Alert delivery rate | AlertManager | Reliability |

---

## 16. Glossary

| Term | Definition |
|------|------------|
| **Circuit Breaker** | Failure detection pattern that prevents cascading failures |
| **Kill Switch** | Emergency halt mechanism with multiple trigger sources |
| **Orchestrator** | Central component managing system lifecycle |
| **Position Reconciliation** | Process of syncing local and broker position state |
| **Repository** | Data access abstraction over database tables |
| **WAL Mode** | Write-Ahead Logging, SQLite journal mode for concurrency |
| **Alert Deduplication** | Suppression of duplicate alerts within time window |
| **Rate Limiting** | Throttling mechanism to prevent alert flooding |
| **Fill** | Partial or complete execution of an order |
| **Trade** | Complete round-trip (entry + exit) |

---

## 17. References

### 17.1 Internal Documentation

- [SignalCore System Architecture](signalcore-system.md)
- [Layered System Architecture](layered-system-architecture.md)
- [Architecture Review Response](architecture-review-response.md) - Gap analysis addressing external architecture review

### 17.2 External Resources

- [SQLite WAL Mode](https://www.sqlite.org/wal.html)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Repository Pattern](https://martinfowler.com/eaaCatalog/repository.html)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

## 18. Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
status: "production"
schema: "phase-1-production-readiness"
phase: 1
next_review: "2025-12-31"
```

---

## Appendix A: File Locations Quick Reference

```
C:\Users\kjfle\Workspace\ordinis\
│
├── src\
│   ├── persistence\
│   │   ├── database.py              # Database manager
│   │   ├── models.py                # Pydantic models
│   │   ├── schema.py                # DDL schema
│   │   └── repositories\
│   │       ├── position.py
│   │       ├── order.py
│   │       ├── fill.py
│   │       ├── trade.py
│   │       └── system_state.py
│   │
│   ├── safety\
│   │   ├── kill_switch.py           # Emergency halt
│   │   └── circuit_breaker.py       # API resilience
│   │
│   ├── orchestration\
│   │   ├── orchestrator.py          # System coordinator
│   │   └── reconciliation.py        # Position sync
│   │
│   └── alerting\
│       └── manager.py               # Alert dispatcher
│
├── data\
│   ├── ordinis.db                   # SQLite database
│   ├── KILL_SWITCH                  # Emergency trigger file
│   └── backups\
│       └── ordinis_backup_*.db      # Timestamped backups
│
└── docs\
    └── architecture\
        ├── production-architecture.md     # This file
        ├── signalcore-system.md
        └── layered-system-architecture.md
```

---

## Document Metadata

```yaml
version: "phase-1-baseline"
last_reviewed: "{{ now().strftime('%Y-%m-%d') }}"
status: "published"
```

---

**END OF DOCUMENT**
