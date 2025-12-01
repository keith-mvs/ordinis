# Context Reference - Intelligent Investor Project
**Last Updated:** 2025-12-01
**Purpose:** Compact reference to reduce token usage in future sessions

---

## Project Identity
**Name:** Intelligent Investor
**Type:** Algorithmic trading system
**Language:** Python 3.11
**Status:** Phase 3 (Paper Trading) - 85% complete

---

## Quick Status
- **Branch:** master (primary development)
- **Commit:** a1cee2a9
- **Tests:** 413 passing, 67% coverage
- **Components:** ProofBench ✅, Paper Broker ✅, 5 Strategies ✅, RAG System ✅

---

## Architecture (3-Layer)
```
1. SignalCore → Generate trading signals
2. RiskGuard → Validate/approve signals
3. FlowRoute → Execute orders (Paper/Live)
```

**Data Flow:** Market Data → Strategy → Signal → Risk Check → Order → Fill → Portfolio Update

---

## Key Components

### ProofBench (Backtesting)
- **Path:** `src/engines/proofbench/`
- **Status:** Working, tested
- **Example:** `tests/backtest/run_first_backtest.py`

### Paper Broker (Paper Trading)
- **Path:** `src/engines/flowroute/adapters/paper.py`
- **Status:** Complete - market data integration, auto-fill, trading loop
- **Test:** `scripts/test_paper_broker.py`

### Strategies
- **Path:** `src/strategies/`
- **Available:** MA Cross, RSI, Momentum, Bollinger, MACD
- **Type:** Event-driven, bar-by-bar execution

### Market Data Plugins
- **IEX:** `src/plugins/market_data/iex.py`
- **Polygon:** `src/plugins/market_data/polygon.py`
- **Status:** Implemented, need API keys

### RAG System
- **Path:** `src/engines/cortex/rag/`
- **Status:** Integrated
- **KB:** 10 domains, 15 publications

---

## Data Files
**Sample Data (Generated):**
1. `data/sample_spy_trending_up.csv` (500 bars, +34%)
2. `data/sample_qqq_volatile.csv` (240 bars, 37% DD)
3. `data/sample_xyz_sideways.csv` (240 bars, range-bound)
4. `data/sample_abc_trending_down.csv` (240 bars, -7%)

**Script:** `scripts/generate_sample_data.py`

---

## Key Files Reference

### Documentation
- `READY_TO_RUN.md` - Quick start guide
- `PAPER_BROKER_PLAN.md` - Phase 3 implementation
- `.claude/CURRENT_SESSION_SUMMARY.md` - Latest session details

### Configuration
- `.claude/SESSION_STATE.json` - Phase tracking, component status
- `.claude/settings.local.json` - Claude Code settings
- `pyproject.toml` - Project config, dependencies

### Critical Code
- `src/engines/proofbench/core/simulator.py` - Backtesting engine
- `src/engines/flowroute/adapters/paper.py` - Paper trading (377 lines)
- `src/engines/flowroute/core/orders.py` - Order types
- `src/strategies/moving_average_crossover.py` - Example strategy

---

## Phase Status

| Phase | Name | Completion | Status |
|-------|------|------------|--------|
| 1 | Knowledge Base | 100% | ✅ Complete |
| 2 | Backtesting | 90% | 🟡 In Progress |
| 3 | Paper Trading | 85% | 🟡 In Progress |
| 4 | Risk Management | 15% | 🔴 Started |
| 5 | System Integration | 5% | 🔴 Planned |
| 6 | Production Prep | 0% | 🔴 Planned |

---

## Next Priorities (In Order)

### 1. Paper Broker → Strategy Integration
**File:** Create `scripts/run_paper_trading_with_strategy.py`
**Action:** Connect PaperBrokerAdapter to ProofBench
**Goal:** Full workflow: market data → strategy → orders → fills → P&L

### 2. Real Market Data Testing
**Action:** Configure IEX API key in `.env`
**Test:** Live quote fetching, price caching
**Validate:** Auto-fill with real prices

### 3. Monitoring Dashboard
**Action:** Build real-time position/P&L display
**Tech:** StreamLit or Terminal UI
**Metrics:** Positions, fills, equity curve

### 4. Phase 4: RiskGuard
**Path:** `src/engines/riskguard/`
**Features:** Position limits, drawdown monitoring, kill switches

---

## Branches

### master
- **Purpose:** Primary development (phases 2-6)
- **Status:** Clean, up to date with origin
- **Latest:** Paper broker complete

### tests/backtest
- **Purpose:** Strategy experimentation
- **Status:** Clean, first backtest implemented
- **Use:** Test new strategies without affecting master

---

## Git Workflow
1. All changes auto-committed (per user preference)
2. Pre-commit hooks enforce quality
3. Push to origin after each session
4. No feature branches (direct to master or tests/backtest)

---

## User Preferences
- **Output:** Concise, professional
- **Git:** Auto-approve all operations
- **Focus:** First build to ground, then enhancements
- **Quality:** Professional-grade, no shortcuts

---

## Common Commands

### Test
```bash
python -m pytest tests/ -v
python scripts/test_paper_broker.py
```

### Generate Data
```bash
python scripts/generate_sample_data.py
```

### Backtest
```bash
git switch tests/backtest
python run_first_backtest.py
```

### Check Status
```bash
git status
git log --oneline -5
cat .claude/SESSION_STATE.json
```

---

## Important Notes
1. **API keys not required** - can develop with sample data
2. **Paper broker is production-ready** for testing
3. **All code passes pre-commit hooks** - enforced automatically
4. **Context optimization** - Use this file instead of full summaries
5. **Session state** - Always check `.claude/SESSION_STATE.json` first

---

## File Counts
- **Python files:** 86 in src/, 49 in tests/
- **Documentation:** 15+ markdown files
- **Sample data:** 4 CSV files (1,220 total bars)
- **Strategies:** 5 implemented
- **Test coverage:** 67% (413 tests)

---

## Token Optimization
**Reference this file instead of:**
- Full project descriptions
- Detailed architecture explanations
- Component inventories
- File listings

**For details, see:**
- `CURRENT_SESSION_SUMMARY.md` - Last session details
- `SESSION_STATE.json` - Current state
- `READY_TO_RUN.md` - Quick start
- `PAPER_BROKER_PLAN.md` - Phase 3 specifics

---

**End of context reference. Use for quick project orientation.**
