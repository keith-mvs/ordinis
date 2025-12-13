# Ordinis

**Version:** 0.2.0-dev (Development Build - Phase 1 Complete)
**Status:** ✅ Production-Ready Infrastructure

An AI-driven quantitative trading system with production-grade persistence, safety controls, and multi-source market data integration.

## Project Overview

Production-ready trading system featuring:
- **Persistence Layer** - SQLite with WAL mode, auto-backup, repository pattern
- **Safety Controls** - Kill switch with multi-trigger, circuit breaker for API resilience
- **Orchestration** - System lifecycle management, position reconciliation
- **Alerting** - Multi-channel notifications with rate limiting and deduplication
- **4 Market Data APIs** - Alpha Vantage, Finnhub, Polygon/Massive, Twelve Data
- **Paper Trading Engine** - Realistic fill simulation with order lifecycle tracking
- **Backtesting Framework** - ProofBench with performance analytics
- **5 Trading Strategies** - Ready to deploy
- **RiskGuard Framework** - Risk management with hard control gates

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | **Production Readiness** (Persistence, Safety, Orchestration) | ✅ **100% Complete** |
| 2 | Knowledge Base & Strategy Design | ✅ 100% Complete |
| 3 | Code Implementation & Backtesting | ✅ 95% Complete |
| 4 | Paper Trading & Simulation | ✅ 90% Complete |
| 5 | Risk Management Integration | ✅ 85% Complete |
| 6 | Event-Driven Refactor | ⏸️ Planned (Phase 2) |

## Installation

### Requirements
- Python 3.11+
- Anaconda3 or conda environment recommended

### Core Dependencies
```bash
# Install base dependencies
pip install -e .

# Install development tools
pip install -e ".[dev]"

# Install Phase 1 production dependencies
pip install -e ".[live-trading]"  # aiosqlite, plyer

# Install all optional dependencies
pip install -e ".[all]"
```

### Phase 1 Dependencies (NEW)
- `aiosqlite>=0.19.0` - Async SQLite for persistence layer
- `plyer>=2.1.0` - Desktop notifications for alerting

See `pyproject.toml` for complete dependency list.

## Quick Start

### Run Full System Demo
```bash
# Set up environment
cp .env.example .env
# Add your API keys to .env

# Run end-to-end demo
python scripts/demo_full_system.py
```

**Expected output:**
- 3 data sources initialized
- Live market data fetched (AAPL, MSFT, GOOGL)
- 2 trading signals generated
- Orders filled with realistic slippage
- Final P&L displayed

### Test Market Data APIs
```bash
python scripts/test_market_data_apis.py
```

### Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

## Documentation

### Architecture Documentation (Phase 1)
- [Production Architecture](docs/architecture/PRODUCTION_ARCHITECTURE.md) - Complete Phase 1 implementation details
- [Architecture Review Response](docs/architecture/ARCHITECTURE_REVIEW_RESPONSE.md) - Gap analysis and design decisions
- [Layered System Architecture](docs/architecture/LAYERED_SYSTEM_ARCHITECTURE.md) - System integration overview
- [SignalCore System](docs/architecture/SIGNALCORE_SYSTEM.md) - Signal generation engine

### Additional Documentation
- [Knowledge Base](docs/knowledge-base/) - Trading knowledge and research
- [Strategy Guides](docs/strategies/) - Strategy implementation templates
- [User Guides](docs/guides/) - CLI usage and best practices

## Repository Structure

```
ordinis/
├── README.md                    # This file
├── CHANGELOG.md                 # Version history
├── pyproject.toml               # Project dependencies and configuration
├── docs/                        # Documentation and knowledge base
│   ├── architecture/            # System design documents
│   │   ├── PRODUCTION_ARCHITECTURE.md    # Phase 1 implementation
│   │   └── ARCHITECTURE_REVIEW_RESPONSE.md # Gap analysis
│   ├── knowledge-base/          # Trading knowledge base (10 domains)
│   ├── strategies/              # Strategy specifications
│   └── guides/                  # User guides
├── src/                         # Source code
│   ├── persistence/             # NEW: Database and repositories
│   │   ├── database.py          # SQLite manager with WAL mode
│   │   ├── models.py            # Pydantic data models
│   │   ├── schema.py            # Database DDL
│   │   └── repositories/        # Position, Order, Fill, Trade repos
│   ├── safety/                  # NEW: Safety controls
│   │   ├── kill_switch.py       # Emergency halt mechanism
│   │   └── circuit_breaker.py   # API failure protection
│   ├── orchestration/           # NEW: System coordination
│   │   ├── orchestrator.py      # Lifecycle manager
│   │   └── reconciliation.py    # Position sync
│   ├── alerting/                # NEW: Multi-channel notifications
│   │   └── manager.py           # Alert dispatcher
│   ├── interfaces/              # NEW: Protocol definitions
│   ├── engines/                 # Core engines
│   │   ├── signalcore/          # Signal generation
│   │   ├── riskguard/           # Risk management
│   │   ├── flowroute/           # Order execution (enhanced)
│   │   ├── proofbench/          # Backtesting
│   │   └── cortex/              # AI analysis
│   ├── plugins/market_data/     # 4 API integrations
│   ├── strategies/              # 5 implemented strategies
│   └── dashboard/               # Streamlit monitoring dashboard
├── tests/                       # Test suites
├── data/                        # Persistent data
│   ├── ordinis.db               # SQLite database
│   ├── KILL_SWITCH              # Emergency trigger file
│   └── backups/                 # Automatic database backups
└── scripts/                     # Executable scripts and demos
```

## Features

### Phase 1: Production Infrastructure (NEW)

**Persistence Layer**:
- SQLite database with WAL mode for concurrent reads
- Automatic backups before trading sessions
- Repository pattern for clean data access
- Pydantic models for type safety
- Tracks: positions, orders, fills, trades, system state

**Safety Controls**:
- Kill switch with multiple triggers (file, programmatic, auto-triggers)
- Circuit breaker for API resilience (CLOSED/OPEN/HALF_OPEN states)
- Auto-triggers: daily loss limit, max drawdown, consecutive losses
- Persistent state survives system restarts

**Orchestration**:
- System lifecycle management (startup/shutdown sequences)
- Position reconciliation with broker (auto-correct option)
- Health monitoring for all components
- Graceful shutdown with pending operation handling

**Alerting**:
- Multi-channel alerts (desktop, email planned, SMS planned)
- Severity-based routing (INFO, WARNING, CRITICAL, EMERGENCY)
- Rate limiting and deduplication
- Alert history tracking

### Market Data Integration
- **Alpha Vantage**: 25 calls/day, comprehensive fundamentals
- **Finnhub**: 60 calls/min, real-time quotes + news
- **Polygon/Massive**: Market data and status
- **Twelve Data**: 800 calls/day, technical indicators

### Paper Trading (Enhanced)
- Realistic order fill simulation with slippage (5 bps)
- Commission calculation ($0.005/share)
- Position tracking with real-time P&L
- **NEW**: Order lifecycle persistence (created → submitted → filled)
- **NEW**: Fill tracking with broker reconciliation
- **NEW**: Integration with kill switch and circuit breaker

### Backtesting
- Event-driven simulation engine (ProofBench)
- Performance metrics: Sharpe, Sortino, Max Drawdown
- Equity curve tracking
- Walk-forward validation ready
- **Phase 1.5**: Shared persistence layer integration planned

### Execution Engines

**FlowRoute (Enhanced)**:
- Order submission with broker API
- Fill streaming and processing
- **NEW**: Circuit breaker integration
- **NEW**: Order persistence and state tracking
- **NEW**: Kill switch enforcement

**RiskGuard (Enhanced)**:
- Position size limits
- Sector concentration checks
- Daily loss monitoring
- **NEW**: Kill switch integration
- **NEW**: Circuit breaker status checks

### Strategies
1. Moving Average Crossover (50/200 SMA)
2. RSI Mean Reversion
3. Momentum Breakout
4. Bollinger Bands
5. MACD

### Dashboard
- Real-time position monitoring
- P&L visualization
- Trade history
- Performance metrics
- Multi-timeframe analysis

## Disclaimer

**IMPORTANT:**
- All trading involves risk of loss. There are no guarantees of profit.
- Past performance in backtests does not assure future results.
- This system is a research and engineering project, not personalized financial advice.
- The authors and contributors are not licensed financial advisors.

## License

[To be determined]
