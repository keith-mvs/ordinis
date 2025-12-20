"""
SQLite schema definitions for Ordinis persistence layer.

Tables:
- positions: Current portfolio positions
- orders: Order lifecycle tracking
- fills: Individual order fills
- trades: Completed trades (entry + exit)
- sessions: Session tracking and state
- messages: Conversation messages for context
- session_summaries: Rolling summaries for long sessions
- system_state: Kill switch, checkpoints, config
- audit_log: Persistence-level audit events

All tables use TEXT for IDs (UUIDs) and ISO-8601 for timestamps.
"""

# Schema version for migrations
SCHEMA_VERSION = 2  # Bumped for session/memory integration

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
    session_id TEXT,  -- R1: Link to session
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_side ON positions(side);
CREATE INDEX IF NOT EXISTS idx_positions_session_id ON positions(session_id);

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
    session_id TEXT,  -- R1: Link to session
    broker_order_id TEXT,
    broker_response TEXT,  -- JSON
    error_message TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    chroma_synced INTEGER NOT NULL DEFAULT 0,  -- R3: Dual-write tracking
    metadata TEXT,  -- JSON
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_orders_order_id ON orders(order_id);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
CREATE INDEX IF NOT EXISTS idx_orders_broker_order_id ON orders(broker_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_session_id ON orders(session_id);
CREATE INDEX IF NOT EXISTS idx_orders_strategy_id ON orders(strategy_id) WHERE strategy_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_orders_unsynced ON orders(chroma_synced) WHERE chroma_synced = 0;

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
    session_id TEXT,  -- R1: Link to session
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_fills_fill_id ON fills(fill_id);
CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id);
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);
CREATE INDEX IF NOT EXISTS idx_fills_timestamp ON fills(timestamp);
CREATE INDEX IF NOT EXISTS idx_fills_session_id ON fills(session_id);

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
    session_id TEXT,  -- R1: Link to session
    chroma_synced INTEGER NOT NULL DEFAULT 0,  -- R3: Dual-write tracking
    chroma_id TEXT,  -- R3: Vector ID for linkage
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trades_trade_id ON trades(trade_id);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);
CREATE INDEX IF NOT EXISTS idx_trades_session_id ON trades(session_id);
CREATE INDEX IF NOT EXISTS idx_trades_strategy_id ON trades(strategy_id) WHERE strategy_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trades_unsynced ON trades(chroma_synced) WHERE chroma_synced = 0;

-- Sessions table: R7 session tracking and state
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    start_time TEXT NOT NULL,
    end_time TEXT,
    last_activity TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'closed', 'error')),
    market_regime TEXT DEFAULT 'unknown',
    risk_status TEXT DEFAULT 'normal',
    trades_executed INTEGER NOT NULL DEFAULT 0,
    signals_generated INTEGER NOT NULL DEFAULT 0,
    messages_count INTEGER NOT NULL DEFAULT 0,
    total_pnl REAL NOT NULL DEFAULT 0.0,
    metadata TEXT,  -- JSON
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time);

-- Messages table: Appendix A.3.2 conversation messages for context
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL UNIQUE,  -- {session_id}:{sequence}
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tokens INTEGER DEFAULT 0,
    active_positions TEXT,  -- JSON array
    market_regime TEXT,
    tool_calls TEXT,  -- JSON array
    citations TEXT,  -- JSON array
    chroma_synced INTEGER DEFAULT 0,
    summary_included INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_seq ON messages(session_id, sequence);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_messages_unsynced ON messages(chroma_synced) WHERE chroma_synced = 0;

-- Session summaries table: Appendix A.3.2 rolling summaries
CREATE TABLE IF NOT EXISTS session_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    summary_type TEXT NOT NULL CHECK (summary_type IN ('rolling', 'final', 'key_facts')),
    content TEXT NOT NULL,
    start_sequence INTEGER NOT NULL,
    end_sequence INTEGER NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    tokens INTEGER DEFAULT 0,
    model_used TEXT,
    chroma_synced INTEGER DEFAULT 0,
    chroma_id TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_summaries_session ON session_summaries(session_id);
CREATE INDEX IF NOT EXISTS idx_summaries_type ON session_summaries(session_id, summary_type);
CREATE UNIQUE INDEX IF NOT EXISTS idx_summaries_range ON session_summaries(session_id, summary_type, start_sequence, end_sequence);

-- Chroma sync queue: R3 dual-write reconciliation
CREATE TABLE IF NOT EXISTS chroma_sync_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,  -- trade, order, message, summary
    entity_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('upsert', 'delete')),
    collection_name TEXT NOT NULL,
    payload TEXT,  -- JSON
    retry_count INTEGER DEFAULT 0,
    last_error TEXT,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    created_at TEXT DEFAULT (datetime('now')),
    processed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_sync_queue_status ON chroma_sync_queue(status);
CREATE INDEX IF NOT EXISTS idx_sync_queue_entity ON chroma_sync_queue(entity_type, entity_id);

-- Retention audit: Appendix A.6.1 deletion tracking
CREATE TABLE IF NOT EXISTS retention_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT,
    deleted_data TEXT,  -- JSON snapshot
    timestamp TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_retention_audit_entity ON retention_audit(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_retention_audit_timestamp ON retention_audit(timestamp);

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
CREATE INDEX IF NOT EXISTS idx_persistence_audit_session ON persistence_audit(session_id);

-- Portfolio snapshots: daily state capture for recovery
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    cash REAL NOT NULL,
    total_equity REAL NOT NULL,
    total_position_value REAL NOT NULL,
    positions_json TEXT NOT NULL,  -- JSON array of positions
    session_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_date ON portfolio_snapshots(snapshot_date);

-- FTS5 virtual table for hybrid search: R8
CREATE VIRTUAL TABLE IF NOT EXISTS trades_fts USING fts5(
    trade_id,
    symbol,
    strategy_id,
    metadata,
    content='trades',
    content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS trades_ai AFTER INSERT ON trades BEGIN
    INSERT INTO trades_fts(rowid, trade_id, symbol, strategy_id, metadata)
    VALUES (NEW.id, NEW.trade_id, NEW.symbol, NEW.strategy_id, NEW.metadata);
END;

CREATE TRIGGER IF NOT EXISTS trades_ad AFTER DELETE ON trades BEGIN
    INSERT INTO trades_fts(trades_fts, rowid, trade_id, symbol, strategy_id, metadata)
    VALUES ('delete', OLD.id, OLD.trade_id, OLD.symbol, OLD.strategy_id, OLD.metadata);
END;

CREATE TRIGGER IF NOT EXISTS trades_au AFTER UPDATE ON trades BEGIN
    INSERT INTO trades_fts(trades_fts, rowid, trade_id, symbol, strategy_id, metadata)
    VALUES ('delete', OLD.id, OLD.trade_id, OLD.symbol, OLD.strategy_id, OLD.metadata);
    INSERT INTO trades_fts(rowid, trade_id, symbol, strategy_id, metadata)
    VALUES (NEW.id, NEW.trade_id, NEW.symbol, NEW.strategy_id, NEW.metadata);
END;

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# Migration SQL for upgrading from schema v1 to v2
MIGRATION_V1_TO_V2 = """
-- Add session_id to trades
ALTER TABLE trades ADD COLUMN session_id TEXT;
ALTER TABLE trades ADD COLUMN chroma_synced INTEGER NOT NULL DEFAULT 0;
ALTER TABLE trades ADD COLUMN chroma_id TEXT;
CREATE INDEX IF NOT EXISTS idx_trades_session_id ON trades(session_id);
CREATE INDEX IF NOT EXISTS idx_trades_unsynced ON trades(chroma_synced) WHERE chroma_synced = 0;

-- Add session_id to orders
ALTER TABLE orders ADD COLUMN session_id TEXT;
ALTER TABLE orders ADD COLUMN chroma_synced INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_orders_session_id ON orders(session_id);
CREATE INDEX IF NOT EXISTS idx_orders_unsynced ON orders(chroma_synced) WHERE chroma_synced = 0;

-- Add session_id to fills
ALTER TABLE fills ADD COLUMN session_id TEXT;
CREATE INDEX IF NOT EXISTS idx_fills_session_id ON fills(session_id);

-- Add session_id to positions
ALTER TABLE positions ADD COLUMN session_id TEXT;
CREATE INDEX IF NOT EXISTS idx_positions_session_id ON positions(session_id);

-- Update schema version
INSERT INTO schema_version (version) VALUES (2);
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
    ("current_session_id", "", "string", "Active session identifier"),
    ("chroma_sync_enabled", "true", "bool", "Whether Chroma dual-write is enabled"),
]


def get_create_schema_sql() -> str:
    """Get full schema creation SQL."""
    return SCHEMA_DDL


def get_initial_state_sql() -> list[tuple[str, str, str, str]]:
    """Get initial system state values."""
    return INITIAL_SYSTEM_STATE


def get_migration_sql(from_version: int, to_version: int) -> str | None:
    """Get migration SQL for version upgrade.
    
    Args:
        from_version: Current schema version
        to_version: Target schema version
        
    Returns:
        Migration SQL string or None if no migration needed
    """
    if from_version == 1 and to_version == 2:
        return MIGRATION_V1_TO_V2
    return None
