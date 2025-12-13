"""
SQLite schema definitions for Ordinis persistence layer.

Tables:
- positions: Current portfolio positions
- orders: Order lifecycle tracking
- fills: Individual order fills
- trades: Completed trades (entry + exit)
- system_state: Kill switch, checkpoints, config
- audit_log: Persistence-level audit events

All tables use TEXT for IDs (UUIDs) and ISO-8601 for timestamps.
"""

# Schema version for migrations
SCHEMA_VERSION = 1

# DDL statements for creating tables
SCHEMA_DDL = """
-- Positions table: current portfolio state
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT', 'FLAT')),
    quantity INTEGER NOT NULL DEFAULT 0,
    avg_cost REAL NOT NULL DEFAULT 0.0,
    current_price REAL NOT NULL DEFAULT 0.0,
    realized_pnl REAL NOT NULL DEFAULT 0.0,
    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
    entry_time TEXT,
    last_update TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_side ON positions(side);

-- Orders table: order lifecycle tracking
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity INTEGER NOT NULL,
    order_type TEXT NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    limit_price REAL,
    stop_price REAL,
    time_in_force TEXT NOT NULL DEFAULT 'day',
    status TEXT NOT NULL DEFAULT 'created',
    filled_quantity INTEGER NOT NULL DEFAULT 0,
    remaining_quantity INTEGER NOT NULL,
    avg_fill_price REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    submitted_at TEXT,
    filled_at TEXT,
    intent_id TEXT,
    signal_id TEXT,
    strategy_id TEXT,
    broker_order_id TEXT,
    broker_response TEXT,  -- JSON
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    metadata TEXT,  -- JSON
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
CREATE INDEX IF NOT EXISTS idx_orders_broker_order_id ON orders(broker_order_id);

-- Fills table: individual order fills
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fill_id TEXT NOT NULL UNIQUE,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    commission REAL NOT NULL DEFAULT 0.0,
    timestamp TEXT NOT NULL,
    latency_ms REAL NOT NULL DEFAULT 0.0,
    slippage_bps REAL NOT NULL DEFAULT 0.0,
    vs_arrival_bps REAL NOT NULL DEFAULT 0.0,
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_fills_fill_id ON fills(fill_id);
CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id);
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);
CREATE INDEX IF NOT EXISTS idx_fills_timestamp ON fills(timestamp);

-- Trades table: completed trades (entry + exit)
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    entry_time TEXT NOT NULL,
    exit_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    pnl REAL NOT NULL,
    pnl_pct REAL NOT NULL,
    commission REAL NOT NULL DEFAULT 0.0,
    duration_seconds REAL NOT NULL,
    entry_order_id TEXT,
    exit_order_id TEXT,
    strategy_id TEXT,
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trades_trade_id ON trades(trade_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);

-- System state table: kill switch, checkpoints, runtime state
CREATE TABLE IF NOT EXISTS system_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,  -- JSON or simple value
    value_type TEXT NOT NULL DEFAULT 'string',  -- string, json, int, float, bool
    description TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_system_state_key ON system_state(key);

-- Audit log: persistence-level events
CREATE TABLE IF NOT EXISTS persistence_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,  -- position, order, fill, trade, system_state
    entity_id TEXT NOT NULL,
    action TEXT NOT NULL,  -- create, update, delete
    old_value TEXT,  -- JSON
    new_value TEXT,  -- JSON
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    session_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_persistence_audit_entity ON persistence_audit(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_persistence_audit_timestamp ON persistence_audit(timestamp);

-- Portfolio snapshots: daily state capture for recovery
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    cash REAL NOT NULL,
    total_equity REAL NOT NULL,
    total_position_value REAL NOT NULL,
    positions_json TEXT NOT NULL,  -- JSON array of positions
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_date ON portfolio_snapshots(snapshot_date);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# Initial system state values
INITIAL_SYSTEM_STATE = [
    ("kill_switch_active", "false", "bool", "Emergency kill switch state"),
    ("kill_switch_reason", "", "string", "Reason for kill switch activation"),
    ("kill_switch_timestamp", "", "string", "When kill switch was activated"),
    ("last_startup", "", "string", "Last successful startup timestamp"),
    ("last_shutdown", "", "string", "Last graceful shutdown timestamp"),
    ("last_checkpoint", "", "string", "Last state checkpoint timestamp"),
    ("trading_enabled", "true", "bool", "Whether trading is enabled"),
    ("daily_loss_limit", "1000.0", "float", "Maximum daily loss before kill switch"),
    ("max_drawdown_pct", "5.0", "float", "Maximum drawdown percentage"),
    ("schema_version", str(SCHEMA_VERSION), "int", "Current schema version"),
]


def get_create_schema_sql() -> str:
    """Get full schema creation SQL."""
    return SCHEMA_DDL


def get_initial_state_sql() -> list[tuple[str, str, str, str]]:
    """Get initial system state values."""
    return INITIAL_SYSTEM_STATE
