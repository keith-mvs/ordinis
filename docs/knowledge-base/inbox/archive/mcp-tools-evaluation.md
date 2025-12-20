# MCP Tools - Comprehensive Evaluation

**Date:** 2025-01-28
**Project:** Ordinis Trading System
**Status:** Research & Planning

---

## Overview

This document evaluates MCP tools (functions exposed by MCP servers) for the Ordinis system. We assess both **connector-specific tools** (from data providers) and **general-purpose tools** (filesystem, database, etc.) to determine integration priorities.

---

## Understanding MCP Tools

### What is an MCP Tool?

An MCP tool is a **function** exposed by an MCP server that Claude can call. Tools have:

- **Name** - Unique identifier (e.g., `get_quote`, `read_file`)
- **Description** - What the tool does
- **Input Schema** - Required and optional parameters
- **Output Schema** - What the tool returns

### Tool vs Connector vs Server

- **MCP Server** - The running process that Claude connects to
- **Connector** - A specific MCP server implementation (e.g., Daloopa connector)
- **Tool** - A function exposed by that server (e.g., `get_financials` tool from Daloopa)

**Example:**

```
Daloopa MCP Server (Connector)
├── Tool: get_financials(ticker, period)
├── Tool: get_transcript(ticker, date)
└── Tool: search_companies(query)
```

---

## Part 1: Connector-Specific Tools

### 1. Daloopa Tools (Financial Data)

**MCP Server:** `mcp-daloopa`

**Exposed Tools:**

#### `get_financials`

```json
{
  "name": "get_financials",
  "description": "Get standardized financial statements for a company",
  "inputSchema": {
    "ticker": {"type": "string", "required": true},
    "statement": {"enum": ["income", "balance_sheet", "cash_flow"], "required": true},
    "period": {"enum": ["annual", "quarterly"], "default": "annual"},
    "years": {"type": "integer", "default": 5}
  },
  "outputSchema": {
    "data": {"type": "array"},
    "currency": {"type": "string"},
    "fiscal_year_end": {"type": "string"}
  }
}
```

**Use Case:**

```python
# Cortex calls Daloopa tool
result = await mcp.call_tool("get_financials", {
    "ticker": "AAPL",
    "statement": "income",
    "period": "quarterly",
    "years": 2
})

# Returns: Revenue, COGS, gross profit, operating income, etc.
```

#### `get_transcript`

```json
{
  "name": "get_transcript",
  "description": "Get earnings call transcript with Q&A",
  "inputSchema": {
    "ticker": {"type": "string", "required": true},
    "date": {"type": "string", "format": "date"},
    "event_type": {"enum": ["earnings", "investor_day", "conference"]}
  }
}
```

#### `search_companies`

```json
{
  "name": "search_companies",
  "description": "Search for companies by name, sector, or criteria",
  "inputSchema": {
    "query": {"type": "string"},
    "sector": {"type": "string"},
    "min_market_cap": {"type": "number"}
  }
}
```

**Integration Priority:**  Medium (Phase 2, if fundamental strategies)

---

### 2. Crypto.com Tools (Cryptocurrency)

**MCP Server:** `mcp-crypto-com`

**Exposed Tools:**

#### `get_ticker`

```json
{
  "name": "get_ticker",
  "description": "Get current price and 24h stats for a crypto pair",
  "inputSchema": {
    "pair": {"type": "string", "required": true, "example": "BTC_USDT"}
  },
  "outputSchema": {
    "last_price": {"type": "number"},
    "volume_24h": {"type": "number"},
    "high_24h": {"type": "number"},
    "low_24h": {"type": "number"}
  }
}
```

#### `get_orderbook`

```json
{
  "name": "get_orderbook",
  "description": "Get order book depth for a trading pair",
  "inputSchema": {
    "pair": {"type": "string", "required": true},
    "depth": {"type": "integer", "default": 50}
  }
}
```

#### `get_trades`

```json
{
  "name": "get_trades",
  "description": "Get recent trades for a pair",
  "inputSchema": {
    "pair": {"type": "string", "required": true},
    "limit": {"type": "integer", "default": 100}
  }
}
```

**Integration Priority:**  Low (crypto out of scope)

---

### 3. Scholar Gateway Tools (Academic Research)

**MCP Server:** `mcp-scholar-gateway`

**Exposed Tools:**

#### `search_papers`

```json
{
  "name": "search_papers",
  "description": "Search academic papers by keywords, author, or topic",
  "inputSchema": {
    "query": {"type": "string", "required": true},
    "source": {"enum": ["google_scholar", "arxiv", "ssrn"], "default": "google_scholar"},
    "year_from": {"type": "integer"},
    "max_results": {"type": "integer", "default": 10}
  },
  "outputSchema": {
    "papers": [{
      "title": "string",
      "authors": ["string"],
      "year": "integer",
      "citations": "integer",
      "abstract": "string",
      "url": "string",
      "doi": "string"
    }]
  }
}
```

#### `get_paper_details`

```json
{
  "name": "get_paper_details",
  "description": "Get full metadata for a specific paper",
  "inputSchema": {
    "doi": {"type": "string"},
    "arxiv_id": {"type": "string"}
  }
}
```

#### `get_citations`

```json
{
  "name": "get_citations",
  "description": "Get papers that cite a given paper",
  "inputSchema": {
    "doi": {"type": "string", "required": true},
    "max_results": {"type": "integer", "default": 20}
  }
}
```

**Integration Priority:**  Medium-High (Phase 2, for KB quality)

**Workflow Example:**

```
User: "Add López de Prado to KB"
→ Cortex calls search_papers("López de Prado advances financial machine learning")
→ Returns: DOI, citations, abstract
→ Auto-populate KB publication metadata
→ Validate against publication.schema.json
```

---

### 4. MT Newswires Tools (Market News)

**MCP Server:** `mcp-mt-newswires`

**Exposed Tools:**

#### `get_news`

```json
{
  "name": "get_news",
  "description": "Get latest news for a symbol or topic",
  "inputSchema": {
    "ticker": {"type": "string"},
    "category": {"enum": ["earnings", "m&a", "fda", "general"]},
    "since": {"type": "string", "format": "date-time"},
    "limit": {"type": "integer", "default": 50}
  },
  "outputSchema": {
    "articles": [{
      "headline": "string",
      "summary": "string",
      "timestamp": "string",
      "category": "string",
      "tickers": ["string"],
      "sentiment": "number"
    }]
  }
}
```

#### `subscribe_alerts`

```json
{
  "name": "subscribe_alerts",
  "description": "Subscribe to real-time alerts for specific events",
  "inputSchema": {
    "tickers": {"type": "array"},
    "event_types": {"type": "array", "items": {"enum": ["earnings", "m&a", "fda"]}},
    "webhook_url": {"type": "string", "format": "uri"}
  }
}
```

#### `search_news`

```json
{
  "name": "search_news",
  "description": "Search historical news by keyword",
  "inputSchema": {
    "query": {"type": "string", "required": true},
    "date_from": {"type": "string", "format": "date"},
    "date_to": {"type": "string", "format": "date"}
  }
}
```

**Integration Priority:**  Medium (Phase 2-3, if event-driven)

---

### 5. Moody's Analytics Tools (Credit Risk)

**MCP Server:** `mcp-moodys-analytics`

**Exposed Tools:**

#### `get_credit_rating`

```json
{
  "name": "get_credit_rating",
  "description": "Get Moody's credit rating for an issuer",
  "inputSchema": {
    "ticker": {"type": "string"},
    "issuer_id": {"type": "string"}
  },
  "outputSchema": {
    "rating": "string",
    "outlook": {"enum": ["positive", "stable", "negative"]},
    "rating_date": "string",
    "default_probability": "number"
  }
}
```

#### `get_default_probability`

```json
{
  "name": "get_default_probability",
  "description": "Get probability of default over various horizons",
  "inputSchema": {
    "ticker": {"type": "string", "required": true},
    "horizon_years": {"type": "integer", "default": 1}
  }
}
```

#### `run_stress_test`

```json
{
  "name": "run_stress_test",
  "description": "Run portfolio stress test under Moody's scenarios",
  "inputSchema": {
    "portfolio": {"type": "array"},
    "scenario": {"enum": ["recession", "stagflation", "crisis"]}
  }
}
```

**Integration Priority:**  Low (too expensive, wrong scope)

---

### 6. S&P Aiera Tools (Events & Transcripts)

**MCP Server:** `mcp-aiera`

**Exposed Tools:**

#### `get_transcript`

```json
{
  "name": "get_transcript",
  "description": "Get earnings call transcript with AI-extracted insights",
  "inputSchema": {
    "ticker": {"type": "string", "required": true},
    "event_type": {"enum": ["earnings", "investor_day"]},
    "date": {"type": "string", "format": "date"}
  },
  "outputSchema": {
    "transcript": "string",
    "key_points": ["string"],
    "sentiment_score": "number",
    "topics": [{
      "topic": "string",
      "mentions": "integer",
      "sentiment": "number"
    }]
  }
}
```

#### `analyze_audio`

```json
{
  "name": "analyze_audio",
  "description": "Analyze audio from earnings call for tone/sentiment",
  "inputSchema": {
    "event_id": {"type": "string", "required": true}
  },
  "outputSchema": {
    "ceo_sentiment": "number",
    "cfo_sentiment": "number",
    "tone": {"enum": ["confident", "cautious", "defensive"]},
    "stress_indicators": ["string"]
  }
}
```

#### `search_transcripts`

```json
{
  "name": "search_transcripts",
  "description": "Search across all transcripts for specific phrases",
  "inputSchema": {
    "query": {"type": "string", "required": true},
    "tickers": {"type": "array"},
    "date_from": {"type": "string", "format": "date"}
  }
}
```

**Integration Priority:**  Medium (Phase 3, if sentiment strategies)

---

## Part 2: Essential General-Purpose MCP Tools

These are **fundamental tools** that every Claude-based system should have.

### 7. Filesystem Tools ⭐ CRITICAL

**MCP Server:** `@modelcontextprotocol/server-filesystem`

**Exposed Tools:**

#### `read_file`

```json
{
  "name": "read_file",
  "description": "Read contents of a file",
  "inputSchema": {
    "path": {"type": "string", "required": true}
  }
}
```

#### `write_file`

```json
{
  "name": "write_file",
  "description": "Write content to a file",
  "inputSchema": {
    "path": {"type": "string", "required": true},
    "content": {"type": "string", "required": true}
  }
}
```

#### `list_directory`

```json
{
  "name": "list_directory",
  "description": "List files in a directory",
  "inputSchema": {
    "path": {"type": "string", "required": true}
  }
}
```

#### `search_files`

```json
{
  "name": "search_files",
  "description": "Search for files matching a pattern",
  "inputSchema": {
    "pattern": {"type": "string", "required": true},
    "path": {"type": "string", "default": "."}
  }
}
```

**Integration Priority:**  **CRITICAL** (Phase 1)

**Use Cases:**

- Read strategy specs from `config/strategies/`
- Write backtest results to `results/backtests/`
- Load risk rules from `config/risk_rules.yaml`
- Save signal logs to `logs/signals/`

---

### 8. Database Tools ⭐ HIGH PRIORITY

**MCP Server:** `@modelcontextprotocol/server-sqlite` (or custom for PostgreSQL/TimescaleDB)

**Exposed Tools:**

#### `query`

```json
{
  "name": "query",
  "description": "Execute a SQL query",
  "inputSchema": {
    "sql": {"type": "string", "required": true},
    "params": {"type": "array"}
  }
}
```

#### `get_schema`

```json
{
  "name": "get_schema",
  "description": "Get database schema",
  "inputSchema": {
    "table": {"type": "string"}
  }
}
```

**Integration Priority:**  **HIGH** (Phase 1-2)

**Use Cases:**

- Query historical price data
- Store signal history
- Track trade history
- Portfolio state management

---

### 9. Git Tools  RECOMMENDED

**MCP Server:** `@modelcontextprotocol/server-git`

**Exposed Tools:**

#### `git_status`

```json
{
  "name": "git_status",
  "description": "Get current git status",
  "inputSchema": {}
}
```

#### `git_log`

```json
{
  "name": "git_log",
  "description": "Get commit history",
  "inputSchema": {
    "max_count": {"type": "integer", "default": 10}
  }
}
```

#### `git_diff`

```json
{
  "name": "git_diff",
  "description": "Show changes in working directory",
  "inputSchema": {
    "path": {"type": "string"}
  }
}
```

**Integration Priority:**  **RECOMMENDED** (Phase 1)

**Use Cases:**

- Version control for strategies
- Track KB changes
- Code review workflows
- Audit trail for configuration changes

---

### 10. Time Series Database Tools ⭐ CRITICAL

**MCP Server:** `custom-timescaledb-server`

**Exposed Tools:**

#### `get_ohlcv`

```json
{
  "name": "get_ohlcv",
  "description": "Get OHLCV bars for a symbol",
  "inputSchema": {
    "symbol": {"type": "string", "required": true},
    "timeframe": {"enum": ["1m", "5m", "15m", "1h", "1d"], "required": true},
    "start": {"type": "string", "format": "date-time"},
    "end": {"type": "string", "format": "date-time"},
    "limit": {"type": "integer", "default": 1000}
  }
}
```

#### `store_bars`

```json
{
  "name": "store_bars",
  "description": "Store OHLCV bars to database",
  "inputSchema": {
    "symbol": {"type": "string", "required": true},
    "timeframe": {"type": "string", "required": true},
    "bars": {"type": "array", "required": true}
  }
}
```

**Integration Priority:**  **CRITICAL** (Phase 1)

**Use Cases:**

- Cache market data (reduce API calls)
- Backtesting historical data
- Real-time bar construction
- Performance optimization

---

### 11. Calculation/Math Tools  USEFUL

**MCP Server:** `custom-math-server`

**Exposed Tools:**

#### `calculate_indicators`

```json
{
  "name": "calculate_indicators",
  "description": "Calculate technical indicators (delegated to SignalCore)",
  "inputSchema": {
    "indicator": {"enum": ["sma", "ema", "rsi", "macd", "bbands"]},
    "data": {"type": "array", "required": true},
    "params": {"type": "object"}
  }
}
```

**Integration Priority:**  Medium (Phase 2)

**Note:** Most calculations should be in SignalCore, not MCP tools. This tool is for Cortex to request calculations without implementing logic.

---

### 12. Logging/Observability Tools  RECOMMENDED

**MCP Server:** `custom-logging-server`

**Exposed Tools:**

#### `log_event`

```json
{
  "name": "log_event",
  "description": "Log an event to centralized logging",
  "inputSchema": {
    "level": {"enum": ["debug", "info", "warning", "error", "critical"]},
    "message": {"type": "string", "required": true},
    "context": {"type": "object"}
  }
}
```

#### `query_logs`

```json
{
  "name": "query_logs",
  "description": "Query historical logs",
  "inputSchema": {
    "level": {"type": "string"},
    "since": {"type": "string", "format": "date-time"},
    "limit": {"type": "integer", "default": 100}
  }
}
```

**Integration Priority:**  **RECOMMENDED** (Phase 1-2)

---

### 13. Web Scraping Tools  USEFUL

**MCP Server:** `@modelcontextprotocol/server-puppeteer`

**Exposed Tools:**

#### `navigate`

```json
{
  "name": "navigate",
  "description": "Navigate to a URL and extract content",
  "inputSchema": {
    "url": {"type": "string", "format": "uri", "required": true}
  }
}
```

#### `screenshot`

```json
{
  "name": "screenshot",
  "description": "Take screenshot of a page",
  "inputSchema": {
    "url": {"type": "string", "format": "uri", "required": true},
    "selector": {"type": "string"}
  }
}
```

**Integration Priority:**  Medium (Phase 2-3)

**Use Cases:**

- Scrape earnings calendars
- Extract data from company IR pages
- Monitor trading forum sentiment (with caution)

---

### 14. Notification Tools  RECOMMENDED

**MCP Server:** `custom-notification-server`

**Exposed Tools:**

#### `send_alert`

```json
{
  "name": "send_alert",
  "description": "Send alert via email/SMS/webhook",
  "inputSchema": {
    "channel": {"enum": ["email", "sms", "webhook", "slack"]},
    "message": {"type": "string", "required": true},
    "priority": {"enum": ["low", "normal", "high", "critical"]}
  }
}
```

**Integration Priority:**  **RECOMMENDED** (Phase 2)

**Use Cases:**

- Kill switch activation alerts
- Daily P&L reports
- Signal generation notifications
- Error alerts

---

## Part 3: MCP Marketplace Tools

**Official MCP Marketplace:** [modelcontextprotocol.io/tools](https://modelcontextprotocol.io)

### Available Official Tools

| Tool | Description | Priority | Use Case |
|------|-------------|----------|----------|
| `@mcp/filesystem` | File operations |  Critical | Read/write configs, logs, results |
| `@mcp/git` | Git version control |  High | Strategy versioning, KB changes |
| `@mcp/sqlite` | SQLite database |  High | Local data storage |
| `@mcp/postgres` | PostgreSQL database |  Critical | Production database |
| `@mcp/fetch` | HTTP requests |  High | API calls (if not using plugins) |
| `@mcp/memory` | Conversation memory |  Medium | Session state |
| `@mcp/brave-search` | Web search |  Low | Research tasks |
| `@mcp/puppeteer` | Web automation |  Medium | Scraping |
| `@mcp/google-drive` | Drive access |  Low | Not needed |
| `@mcp/slack` | Slack integration |  Medium | Notifications |

---

## Tool Integration Architecture

### Recommended MCP Server Setup

```yaml
# .claude/mcp_servers.json
{
  "mcpServers": {
    # Essential (Phase 1)
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/project"],
      "priority": "critical"
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/intelligent_investor"],
      "priority": "critical"
    },
    "git": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git", "/path/to/project"],
      "priority": "high"
    },

    # Data Sources (Phase 2+)
    "polygon": {
      "command": "python",
      "args": ["-m", "src.mcp_servers.polygon_server"],
      "env": {"POLYGON_API_KEY": "xxx"},
      "priority": "high"
    },

    # Knowledge Base (Phase 2)
    "knowledge-base": {
      "command": "python",
      "args": ["-m", "src.mcp_servers.kb_server"],
      "priority": "medium"
    },
    "scholar-gateway": {
      "command": "python",
      "args": ["-m", "src.mcp_servers.scholar_server"],
      "priority": "medium"
    },

    # Optional (Phase 3+)
    "daloopa": {
      "command": "python",
      "args": ["-m", "src.mcp_servers.daloopa_server"],
      "env": {"DALOOPA_API_KEY": "xxx"},
      "priority": "low",
      "enabled": false
    }
  }
}
```

### Tool Call Flow

```
┌─────────────────────────────────────────────────────────────┐
│ USER REQUEST                                                 │
│ "Backtest MA crossover on AAPL, last 5 years"               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ CORTEX (LLM)                                                 │
│                                                               │
│ 1. Parse intent                                             │
│ 2. Plan tool calls                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ TOOL 1      │ │ TOOL 2      │ │ TOOL 3      │
│ filesystem  │ │ postgres    │ │ polygon     │
│             │ │             │ │             │
│ read_file(  │ │ query(      │ │ get_historical( │
│  "config/   │ │  "SELECT    │ │  "AAPL",    │
│   strat.yml"│ │   bars..."  │ │  "2020-...")│
│ )           │ │ )           │ │ )           │
└─────┬───────┘ └─────┬───────┘ └─────┬───────┘
      │               │               │
      └───────────────┼───────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ CORTEX (Synthesis)                                           │
│                                                               │
│ - Load strategy spec from filesystem                        │
│ - Query cached data from postgres                           │
│ - Fill gaps from Polygon API                                │
│ - Call ProofBench engine (NOT via MCP tool)                 │
│ - Return results to user                                    │
└─────────────────────────────────────────────────────────────┘
```

**Key Principle:** MCP tools for **data access**, NOT calculations. SignalCore/RiskGuard/ProofBench engines do calculations.

---

## Priority Matrix

###  Critical Priority (Implement Phase 1)

| Tool | Server | Rationale |
|------|--------|-----------|
| `read_file`, `write_file` | `@mcp/filesystem` | Essential for all file operations |
| `query`, `get_schema` | `@mcp/postgres` | Production database access |
| `get_ohlcv`, `store_bars` | Custom TimescaleDB | Market data caching |

**Total: 3 servers, ~10 tools**

###  High Priority (Implement Phase 1-2)

| Tool | Server | Rationale |
|------|--------|-----------|
| `git_status`, `git_log` | `@mcp/git` | Version control, audit trail |
| `log_event`, `query_logs` | Custom logging | Observability |
| `send_alert` | Custom notification | Alerts and reporting |

**Total: 3 servers, ~8 tools**

###  Medium Priority (Phase 2-3)

| Tool | Server | Rationale |
|------|--------|-----------|
| `search_papers` | Scholar Gateway | KB quality (low cost) |
| `get_news` | MT Newswires | IF event-driven strategies |
| `get_financials` | Daloopa | IF fundamental strategies |
| `navigate`, `screenshot` | `@mcp/puppeteer` | Scraping (use sparingly) |

**Total: 4 servers, ~12 tools**

###  Low Priority (Phase 3+)

| Tool | Server | Rationale |
|------|--------|-----------|
| `get_ticker` (crypto) | Crypto.com | Out of scope |
| `get_credit_rating` | Moody's | Too expensive |
| `get_transcript` (audio) | S&P Aiera | Expensive, conditional |

---

## Tool Security & Guardrails

### Access Control

```json
{
  "tool_permissions": {
    "filesystem": {
      "allowed_paths": [
        "/path/to/intelligent-investor/config/",
        "/path/to/intelligent-investor/results/",
        "/path/to/intelligent-investor/logs/"
      ],
      "forbidden_paths": [
        "/.env",
        "/venv/",
        "**/.git/",
        "**/__pycache__/"
      ],
      "read_only": false
    },
    "postgres": {
      "allowed_tables": [
        "ohlcv_bars",
        "signals",
        "trades",
        "portfolio_state"
      ],
      "forbidden_operations": ["DROP", "TRUNCATE"],
      "read_only": false
    },
    "polygon": {
      "rate_limit": "100 calls/minute",
      "read_only": true
    }
  }
}
```

### Validation

**Before executing ANY tool:**

1.  Validate input against schema
2.  Check permissions
3.  Log tool call (who, what, when)
4.  Check rate limits
5.  Validate output before returning

### Error Handling

```python
try:
    result = await mcp.call_tool("read_file", {"path": "config/strategy.yaml"})
except FileNotFoundError:
    # Handle gracefully, don't crash
    return default_strategy()
except PermissionError:
    # Log security issue
    logger.error(f"Unauthorized file access attempt: {path}")
    raise
```

---

## Implementation Roadmap

### Phase 1 (Weeks 1-2): Essential Tools

**Implement:**

1. **Filesystem MCP Server**
   - Tools: `read_file`, `write_file`, `list_directory`
   - Config: Restrict to project directory
   - Priority: CRITICAL

2. **PostgreSQL MCP Server**
   - Tools: `query`, `get_schema`, `execute`
   - Setup: TimescaleDB for time-series
   - Priority: CRITICAL

3. **Git MCP Server**
   - Tools: `git_status`, `git_log`, `git_diff`
   - Config: Read-only initially
   - Priority: HIGH

**Deliverable:** Cortex can read configs, query database, check git status

---

### Phase 2 (Weeks 3-4): Data & Observability

**Implement:**
4. **Custom Logging MCP Server**

- Tools: `log_event`, `query_logs`
- Integration: Centralized logging system
- Priority: HIGH

5. **Custom Notification MCP Server**
   - Tools: `send_alert`, `send_report`
   - Channels: Email, Slack (initially)
   - Priority: RECOMMENDED

6. **Scholar Gateway MCP Server**
   - Tools: `search_papers`, `get_paper_details`
   - Config: API keys for Scholar, SSRN
   - Priority: MEDIUM (KB quality)

**Deliverable:** Observability + KB automation

---

### Phase 3 (Weeks 5-8): Conditional Data Sources

**Evaluate & Implement (IF needed):**
7. **Daloopa MCP Server** (if fundamental strategies)

- Tools: `get_financials`, `get_transcript`
- Cost: $500-2,000/month
- Condition: Fundamental alpha proven

8. **MT Newswires MCP Server** (if event-driven)
   - Tools: `get_news`, `subscribe_alerts`
   - Cost: $1,000-5,000/month
   - Condition: Event-driven alpha proven

**Deliverable:** Data sources aligned with proven strategies

---

## Recommendations

### DO This

 **Start with official MCP tools** (`@mcp/filesystem`, `@mcp/git`, `@mcp/postgres`)
 **Build custom servers** for domain-specific needs (logging, time-series DB)
 **Restrict access** via path/table whitelists
 **Log all tool calls** for audit trail
 **Use tools for data access** NOT calculations (engines do that)

### DON'T Do This

 **Don't add data source tools** until strategies proven
 **Don't expose dangerous operations** (DROP TABLE, rm -rf)
 **Don't let LLM do calculations** via tools (use engines instead)
 **Don't skip validation** on tool inputs/outputs
 **Don't add tools "just in case"** (only when needed)

---

## Summary

**MCP Tools Recommendation:**

1. **Phase 1 (NOW):** Essential tools only
   - Filesystem, Git, PostgreSQL/TimescaleDB
   - ~6 servers, ~20 tools total

2. **Phase 2 (3-6 months):** Observability + KB
   - Logging, Notifications, Scholar Gateway
   - +3 servers, +10 tools

3. **Phase 3+ (conditional):** Data source tools
   - Daloopa, MT Newswires, S&P Aiera
   - ONLY if specific alpha proven
   - +0-3 servers based on need

**Total Recommended Tools:** 20-40 tools across 6-12 servers

**Cost:** $0-50/month (Phase 1-2), $500-10K/month (Phase 3 if data sources added)

---

## Next Steps

1.  **Set up filesystem server** (use official `@mcp/filesystem`)
2.  **Set up database server** (PostgreSQL + TimescaleDB)
3.  **Set up git server** (use official `@mcp/git`)
4.  **Build custom logging server** (Python MCP server)
5.  **Build custom notification server** (Email + Slack)
6.  **Evaluate Scholar Gateway** for KB automation

**Focus:** Get essential tools working before adding data connectors.

---

**Document Version:** v1.0.0
**Last Updated:** 2025-01-28
**Owner:** Architecture Team
