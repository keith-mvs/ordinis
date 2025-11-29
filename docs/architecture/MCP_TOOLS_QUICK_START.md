# MCP Tools - Quick Start Guide

**Last Updated:** 2025-01-28

---

## TL;DR

**Start with 3 essential MCP servers:**

1. ✅ **Filesystem** - Read/write configs, logs, results
2. ✅ **PostgreSQL/TimescaleDB** - Store market data, signals, trades
3. ✅ **Git** - Version control and audit trail

**Skip connector tools (Daloopa, MT Newswires, etc.) until Phase 2-3**

---

## Essential MCP Servers (Phase 1)

### 1. Filesystem Server ⭐ CRITICAL

**Install:**

```bash
npm install -g @modelcontextprotocol/server-filesystem
```

**Configure:**

```json
// .claude/mcp_servers.json
{
  "filesystem": {
    "command": "npx",
    "args": [
      "-y",
      "@modelcontextprotocol/server-filesystem",
      "/path/to/intelligent-investor"
    ]
  }
}
```

**Tools Exposed:**

- `read_file(path)` - Read file contents
- `write_file(path, content)` - Write to file
- `list_directory(path)` - List files
- `search_files(pattern)` - Find files

**Use Cases:**

```python
# Read strategy config
strategy = await mcp.call_tool("read_file", {
    "path": "config/strategies/ma_crossover.yaml"
})

# Write backtest results
await mcp.call_tool("write_file", {
    "path": "results/backtests/2025-01-28_ma_crossover.json",
    "content": json.dumps(results)
})

# List signals
files = await mcp.call_tool("list_directory", {
    "path": "logs/signals/2025-01"
})
```

---

### 2. PostgreSQL/TimescaleDB Server ⭐ CRITICAL

**Install:**

```bash
npm install -g @modelcontextprotocol/server-postgres
```

**Configure:**

```json
{
  "postgres": {
    "command": "npx",
    "args": [
      "-y",
      "@modelcontextprotocol/server-postgres",
      "postgresql://localhost/intelligent_investor"
    ]
  }
}
```

**Tools Exposed:**

- `query(sql, params)` - Execute SQL
- `get_schema(table)` - Get table schema

**Use Cases:**

```python
# Query historical bars (cached from Polygon)
bars = await mcp.call_tool("query", {
    "sql": """
        SELECT time, open, high, low, close, volume
        FROM ohlcv_bars
        WHERE symbol = $1 AND timeframe = $2
        AND time >= $3
        ORDER BY time
    """,
    "params": ["AAPL", "1D", "2020-01-01"]
})

# Store signal
await mcp.call_tool("query", {
    "sql": """
        INSERT INTO signals (timestamp, symbol, signal_type, score)
        VALUES ($1, $2, $3, $4)
    """,
    "params": [datetime.now(), "AAPL", "LONG", 0.75]
})
```

**Database Schema:**

```sql
-- Market data (cached)
CREATE TABLE ohlcv_bars (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    PRIMARY KEY (time, symbol, timeframe)
);

-- Convert to hypertable (TimescaleDB)
SELECT create_hypertable('ohlcv_bars', 'time');

-- Signals
CREATE TABLE signals (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    signal_type TEXT,
    direction TEXT,
    score NUMERIC,
    model_id TEXT,
    approved BOOLEAN
);

-- Trades
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    entry_time TIMESTAMPTZ,
    exit_time TIMESTAMPTZ,
    symbol TEXT,
    direction TEXT,
    entry_price NUMERIC,
    exit_price NUMERIC,
    quantity INTEGER,
    pnl NUMERIC
);

-- Portfolio state
CREATE TABLE portfolio_snapshots (
    timestamp TIMESTAMPTZ PRIMARY KEY,
    equity NUMERIC,
    cash NUMERIC,
    positions JSONB,
    daily_pnl NUMERIC,
    drawdown NUMERIC
);
```

---

### 3. Git Server 🟢 RECOMMENDED

**Install:**

```bash
npm install -g @modelcontextprotocol/server-git
```

**Configure:**

```json
{
  "git": {
    "command": "npx",
    "args": [
      "-y",
      "@modelcontextprotocol/server-git",
      "/path/to/intelligent-investor"
    ]
  }
}
```

**Tools Exposed:**

- `git_status()` - Current status
- `git_log(max_count)` - Commit history
- `git_diff(path)` - Show changes

**Use Cases:**

```python
# Check what changed before backtesting
status = await mcp.call_tool("git_status", {})
if status["has_changes"]:
    logger.warning("Uncommitted changes in repo!")

# Log strategy changes
log = await mcp.call_tool("git_log", {
    "max_count": 5,
    "path": "config/strategies/"
})
```

---

## Phase 2 Servers (Weeks 3-4)

### 4. Custom Logging Server

**Why:** Centralized logging for all components

**Tools:**

- `log_event(level, message, context)`
- `query_logs(level, since, limit)`

**Implementation:**

```python
# src/mcp_servers/logging_server.py
from mcp import Server, Tool

server = Server("logging")

@server.tool()
async def log_event(level: str, message: str, context: dict = None):
    """Log an event to centralized system."""
    logger = get_logger()
    getattr(logger, level.lower())(message, extra=context)
    return {"status": "logged", "timestamp": datetime.now().isoformat()}

@server.tool()
async def query_logs(
    level: str = None,
    since: str = None,
    limit: int = 100
):
    """Query historical logs."""
    # Query Elasticsearch/Loki/etc.
    pass
```

---

### 5. Custom Notification Server

**Why:** Alerts for kill switches, signals, errors

**Tools:**

- `send_alert(channel, message, priority)`
- `send_report(type, data)`

**Implementation:**

```python
@server.tool()
async def send_alert(
    channel: str,  # "email", "sms", "slack"
    message: str,
    priority: str = "normal"  # "low", "normal", "high", "critical"
):
    """Send alert via specified channel."""
    if channel == "email":
        send_email(to=ALERT_EMAIL, subject=f"[{priority.upper()}] Alert", body=message)
    elif channel == "slack":
        post_to_slack(SLACK_WEBHOOK, message, priority)

    return {"status": "sent", "channel": channel}
```

**Use Cases:**

```python
# Kill switch alert
await mcp.call_tool("send_alert", {
    "channel": "email",
    "message": "KILL SWITCH ACTIVATED: Daily loss limit exceeded (-3.2%)",
    "priority": "critical"
})

# Daily report
await mcp.call_tool("send_report", {
    "type": "daily_pnl",
    "data": {
        "pnl": 1250.00,
        "trades": 5,
        "win_rate": 0.60
    }
})
```

---

### 6. Scholar Gateway Server (KB Quality)

**Why:** Automate academic paper ingestion for Knowledge Base

**Tools:**

- `search_papers(query, source, max_results)`
- `get_paper_details(doi)`
- `get_citations(doi)`

**Use Case:**

```python
# Auto-update KB with latest research
papers = await mcp.call_tool("search_papers", {
    "query": "quantitative trading machine learning",
    "source": "arxiv",
    "year_from": 2023,
    "max_results": 10
})

for paper in papers["papers"]:
    if paper["citations"] > 50:  # High-quality filter
        # Add to KB publications
        add_to_kb(paper)
```

---

## Phase 3+ Servers (Conditional)

### 7. Daloopa Server (IF fundamental strategies)

**Cost:** $500-2,000/month
**Tools:** `get_financials`, `get_transcript`, `search_companies`
**Condition:** Only if fundamental/value strategies proven

### 8. MT Newswires Server (IF event-driven)

**Cost:** $1,000-5,000/month
**Tools:** `get_news`, `subscribe_alerts`, `search_news`
**Condition:** Only if news is primary alpha source

### 9. S&P Aiera Server (IF sentiment strategies)

**Cost:** $2,000-10,000/month
**Tools:** `get_transcript`, `analyze_audio`, `search_transcripts`
**Condition:** Only if sentiment analysis proven valuable

---

## MCP Server Configuration

**Complete `.claude/mcp_servers.json`:**

```json
{
  "mcpServers": {
    // Phase 1: Essential (Always enabled)
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/intelligent-investor"],
      "enabled": true
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/intelligent_investor"],
      "enabled": true
    },
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git", "/path/to/intelligent-investor"],
      "enabled": true
    },

    // Phase 2: Recommended (Enable when ready)
    "logging": {
      "command": "python",
      "args": ["-m", "src.mcp_servers.logging_server"],
      "enabled": false
    },
    "notifications": {
      "command": "python",
      "args": ["-m", "src.mcp_servers.notification_server"],
      "env": {
        "SMTP_HOST": "smtp.gmail.com",
        "SLACK_WEBHOOK": "https://hooks.slack.com/..."
      },
      "enabled": false
    },
    "scholar": {
      "command": "python",
      "args": ["-m", "src.mcp_servers.scholar_server"],
      "enabled": false
    },

    // Phase 3+: Conditional (Enable only if needed)
    "daloopa": {
      "command": "python",
      "args": ["-m", "src.mcp_servers.daloopa_server"],
      "env": {"DALOOPA_API_KEY": "xxx"},
      "enabled": false
    },
    "mt-newswires": {
      "command": "python",
      "args": ["-m", "src.mcp_servers.mt_newswires_server"],
      "env": {"MT_API_KEY": "xxx"},
      "enabled": false
    }
  }
}
```

---

## Tool Security

**Filesystem Restrictions:**

```json
{
  "filesystem": {
    "allowed_paths": [
      "config/",
      "results/",
      "logs/",
      "docs/"
    ],
    "forbidden_paths": [
      ".env",
      "venv/",
      ".git/",
      "__pycache__/"
    ]
  }
}
```

**Database Restrictions:**

```json
{
  "postgres": {
    "allowed_operations": ["SELECT", "INSERT", "UPDATE"],
    "forbidden_operations": ["DROP", "TRUNCATE", "DELETE"],
    "allowed_tables": [
      "ohlcv_bars",
      "signals",
      "trades",
      "portfolio_snapshots"
    ]
  }
}
```

---

## Tool Usage Patterns

### Pattern 1: Read Config, Query Data, Call Engine

```python
# 1. Read strategy config (Filesystem tool)
strategy_yaml = await mcp.call_tool("read_file", {
    "path": "config/strategies/ma_crossover.yaml"
})
strategy = yaml.safe_load(strategy_yaml)

# 2. Query historical data (PostgreSQL tool)
bars = await mcp.call_tool("query", {
    "sql": "SELECT * FROM ohlcv_bars WHERE symbol = $1 AND time >= $2",
    "params": ["AAPL", "2024-01-01"]
})

# 3. Call SignalCore (NOT an MCP tool - direct engine call)
signals = signalcore.generate_signals(strategy, bars)

# 4. Store results (PostgreSQL tool)
await mcp.call_tool("query", {
    "sql": "INSERT INTO signals (...) VALUES (...)",
    "params": [...signals data...]
})
```

### Pattern 2: Alert on Kill Switch

```python
# RiskGuard detects kill switch condition
if daily_pnl < -0.03:  # -3% daily loss
    # Send critical alert (Notification tool)
    await mcp.call_tool("send_alert", {
        "channel": "email",
        "message": f"KILL SWITCH: Daily loss {daily_pnl:.2%}",
        "priority": "critical"
    })

    # Log event (Logging tool)
    await mcp.call_tool("log_event", {
        "level": "critical",
        "message": "Kill switch activated",
        "context": {"daily_pnl": daily_pnl, "equity": equity}
    })
```

### Pattern 3: Version-Controlled Strategy Updates

```python
# Check git status before modifying strategy (Git tool)
status = await mcp.call_tool("git_status", {})
if status["has_uncommitted_changes"]:
    raise ValueError("Uncommitted changes - commit before modifying strategy")

# Update strategy config (Filesystem tool)
await mcp.call_tool("write_file", {
    "path": "config/strategies/ma_crossover.yaml",
    "content": updated_strategy_yaml
})

# Git operations (NOT via MCP - use bash for commits)
# Commits should be done by user or CI/CD, not automated
```

---

## Implementation Checklist

### Phase 1 (This Week)

- [ ] Install `@modelcontextprotocol/server-filesystem`
- [ ] Install `@modelcontextprotocol/server-postgres`
- [ ] Install `@modelcontextprotocol/server-git`
- [ ] Set up TimescaleDB database
- [ ] Create `ohlcv_bars`, `signals`, `trades` tables
- [ ] Configure `.claude/mcp_servers.json`
- [ ] Test filesystem tool (read config file)
- [ ] Test postgres tool (query test data)
- [ ] Test git tool (check status)

### Phase 2 (Next 2-4 Weeks)

- [ ] Build custom logging MCP server
- [ ] Build custom notification MCP server
- [ ] Evaluate Scholar Gateway integration
- [ ] Document all MCP tool usage
- [ ] Set up centralized logging backend

### Phase 3+ (Conditional)

- [ ] **IF fundamental strategies:** Evaluate Daloopa MCP server
- [ ] **IF event-driven:** Evaluate MT Newswires MCP server
- [ ] **IF sentiment:** Evaluate S&P Aiera MCP server

---

## Key Takeaways

1. ✅ **Start with 3 servers** - Filesystem, PostgreSQL, Git
2. ✅ **Tools for data access** - NOT for calculations (use engines)
3. ✅ **Restrict access** - Whitelist paths, tables, operations
4. ✅ **Log everything** - Audit trail for all tool calls
5. ❌ **Skip connector tools** - Until specific alpha proven
6. ⚡ **Add one at a time** - Don't overwhelm the system

---

## Next Steps

1. Install Phase 1 MCP servers (1-2 hours)
2. Set up TimescaleDB (2-3 hours)
3. Test all essential tools (1 hour)
4. Build logging server (4-6 hours)
5. Build notification server (4-6 hours)

**Total Phase 1 effort:** ~12-18 hours

---

## Resources

- **Full Analysis:** `docs/architecture/MCP_TOOLS_EVALUATION.md`
- **MCP Documentation:** <https://modelcontextprotocol.io>
- **Official Servers:** <https://github.com/modelcontextprotocol/servers>

---

**Document Version:** v1.0.0
**Last Updated:** 2025-01-28
**Owner:** Architecture Team
