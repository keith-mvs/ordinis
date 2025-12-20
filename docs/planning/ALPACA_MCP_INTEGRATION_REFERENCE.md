# Alpaca MCP Server Integration Reference

**Status:** Future consideration - LLM approach preferred for current implementation
**Date:** December 15, 2025
**Repository:** https://github.com/alpacahq/alpaca-mcp-server

## Overview

Alpaca provides an official Model Context Protocol (MCP) server that enables native trading operations through tool calls instead of LLM token-based execution. This could significantly reduce agent operational costs in future iterations.

## Key Capabilities

The Alpaca MCP server provides **43+ native tools** across:

### Account & Portfolio Management
- `get_account_info` - Account balances, buying power, equity
- `get_all_positions` - All current portfolio positions
- `get_open_position` - Specific position details
- `get_portfolio_history` - Historical equity and P/L tracking

### Trading Operations
- **Stocks:** `place_stock_order` (market, limit, stop, stop_limit, trailing_stop)
- **Options:** `place_option_market_order` (single-leg and multi-leg strategies up to 4 legs)
- **Crypto:** `place_crypto_order` (market, limit, stop_limit)

### Market Data
- **Stocks:** `get_stock_bars`, `get_stock_quotes`, `get_stock_trades`, `get_stock_snapshot`
- **Options:** `get_option_chain`, `get_option_latest_quote`, option greeks and IV
- **Crypto:** `get_crypto_bars`, `get_crypto_quotes`, `get_crypto_trades`, `get_crypto_snapshot`

### Order Management
- `get_orders` - Filter by status, symbols, date range
- `cancel_order_by_id` - Cancel specific order
- `cancel_all_orders` - Cancel all pending orders
- `close_position` - Close by quantity or percentage
- `close_all_positions` - Liquidate entire portfolio

### Market Information
- `get_clock` - Market open/closed status, next open/close times
- `get_calendar` - Trading days and sessions
- `get_corporate_actions` - Earnings, dividends, splits

### Watchlists & Assets
- `get_watchlists`, `create_watchlist`, `update_watchlist`
- `get_asset`, `get_all_assets` - Search stocks, ETFs, crypto, options

## Cost Impact Analysis

### Current LLM Approach (Ordinis)
- **Pattern:** LLM generates queries → parses responses → formats output → reasons about next action
- **Token consumption:** 3,000-5,500 tokens per typical multi-step workflow
- **Cost driver:** Every data retrieval, formatting, and decision incurs token costs

### MCP Tool Approach (Alternative)
- **Pattern:** Deterministic tool calls → structured JSON responses → LLM reasons only on decisions
- **Token consumption:** ~300-500 tokens per workflow (60-85% reduction)
- **Cost driver:** Only strategic reasoning, not data handling

**Example Workflow Comparison:**

| Task | LLM Approach | MCP Approach | Token Savings |
|------|--------------|--------------|---------------|
| Check account balance | 500-1000 tokens | 50 tokens | 90-95% |
| Get all positions | 1000-2000 tokens | 100 tokens | 90-95% |
| Place order & verify | 1500-2500 tokens | 150 tokens | 90-94% |
| Retrieve option chain | 2000-3000 tokens | 200 tokens | 90-93% |
| **Multi-step workflow** | **5,000-6,500 tokens** | **500-700 tokens** | **85-92%** |

### When MCP Makes Sense
1. **High-frequency operations:** Daily rebalancing, frequent position checks
2. **Multi-account management:** Scaling beyond 1-person lean setup
3. **Production deployment:** Consistent, deterministic execution paths
4. **Cost sensitivity:** Token-metered API usage at scale

### When LLM Approach Makes Sense (Current Preference)
1. **Flexibility:** Custom data transformations, complex reasoning chains
2. **Integration:** Existing Python codebase, custom strategies
3. **Development velocity:** Rapid iteration without MCP infrastructure
4. **Small scale:** 1-person lean setup, 10-60 trades/month

## Installation & Setup (Future Reference)

### Quick Start
```bash
# Install via uvx (recommended)
uvx alpaca-mcp-server init

# Configure in Claude Desktop settings
# Add to mcpServers in config.json:
{
  "alpaca": {
    "type": "stdio",
    "command": "uvx",
    "args": ["alpaca-mcp-server", "serve"],
    "env": {
      "ALPACA_API_KEY": "your_key",
      "ALPACA_SECRET_KEY": "your_secret",
      "ALPACA_PAPER_TRADE": "True"
    }
  }
}
```

### VS Code Integration
Create `.vscode/mcp.json`:
```json
{
  "mcp": {
    "servers": {
      "alpaca": {
        "type": "stdio",
        "command": "uvx",
        "args": ["alpaca-mcp-server", "serve"],
        "env": {
          "ALPACA_API_KEY": "${env:ALPACA_API_KEY}",
          "ALPACA_SECRET_KEY": "${env:ALPACA_SECRET_KEY}"
        }
      }
    }
  }
}
```

### Docker Deployment (Cloud)
```bash
docker run -d \
  -e ALPACA_API_KEY=your_key \
  -e ALPACA_SECRET_KEY=your_secret \
  -e ALPACA_PAPER_TRADE=True \
  mcp/alpaca:latest
```

## Integration Scenarios

### Scenario 1: Hybrid Approach
- **MCP tools:** Order execution, position queries, market data retrieval
- **LLM reasoning:** Signal generation, portfolio optimization, risk assessment
- **Benefit:** Reduced tokens on execution, flexible reasoning for strategy

### Scenario 2: MCP for Production, LLM for Development
- **Development:** Full LLM approach for rapid strategy iteration
- **Production:** MCP tools for deterministic execution
- **Benefit:** Best of both worlds—flexibility during R&D, reliability in production

### Scenario 3: Gradual Migration
- **Phase 1:** Pure LLM (current)
- **Phase 2:** Add MCP for high-frequency ops (daily checks, rebalancing)
- **Phase 3:** Full MCP for all broker interactions
- **Benefit:** Controlled transition with measurable cost reduction

## Cost Projection Updates (If MCP Adopted)

### Current Lean LLM Estimate (from llm_inference_costs.csv)
- **Scenario:** Lean-API or Lean-GPU
- **Monthly tokens:** User to configure (est. 1-5M tokens/month for daily operations)
- **Estimated cost:** $10-50/month (API) or $50-150/month (GPU hours)

### Potential MCP Savings
- **Token reduction:** 60-85% on execution tasks
- **Revised estimate:** $2-15/month (API) or $10-50/month (GPU hours)
- **Breakeven:** MCP infrastructure overhead vs. token savings

**Note:** MCP requires additional setup/maintenance time initially, but pays off at scale.

## Documentation & Resources

- **GitHub Repo:** https://github.com/alpacahq/alpaca-mcp-server
- **Features:** 43+ tools (stocks, options, crypto, portfolio, market data)
- **Installation:** `uvx alpaca-mcp-server init`
- **Configuration:** `.env` file or environment variables
- **Supported clients:** Claude Desktop, Cursor, VS Code (with MCP extension)

## Decision Log

**December 15, 2025:**
- Discovered official Alpaca MCP server during cost modeling phase
- **Decision:** Defer MCP integration, continue with LLM approach
- **Rationale:**
  - Current 1-person lean setup benefits from LLM flexibility
  - Development velocity prioritized over token cost optimization
  - MCP adds infrastructure complexity for marginal savings at low volumes
  - Keep MCP as future option if scaling beyond lean operations

**Reassessment triggers:**
1. Trading volume exceeds 200 trades/month (4x current lean estimate)
2. Multi-account management needed
3. Token costs exceed $100/month
4. Deterministic execution becomes critical requirement

## Action Items (Future)

- [ ] Benchmark actual LLM token consumption in Phase 1 production
- [ ] Monitor monthly AI costs vs. MCP savings threshold
- [ ] Evaluate MCP integration if scaling to Standard scenario (60+ trades/month)
- [ ] Consider hybrid approach: MCP for execution, LLM for strategy reasoning

---

**Related Documents:**
- [BROKER_FEES_AND_API_COSTS.md](../artifacts/BROKER_FEES_AND_API_COSTS.md) - Current cost framework
- [llm_inference_costs.csv](../artifacts/llm_inference_costs.csv) - LLM cost model
- [BUDGET_BREAKDOWN.md](../artifacts/BUDGET_BREAKDOWN.md) - Complete OPEX breakdown
