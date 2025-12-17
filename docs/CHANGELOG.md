# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-dev] - 2025-12-12

### Added - Phase 1: Production Readiness

**Persistence Layer** (`src/persistence/`)
- `DatabaseManager` - SQLite connection manager with WAL mode and automatic backups
- `schema.py` - Database DDL for positions, orders, fills, trades, system_state tables
- `models.py` - Pydantic data models: PositionRow, OrderRow, FillRow, TradeRow, SystemStateRow
- Repository pattern for clean data access:
  - `PositionRepository` - Trading position management
  - `OrderRepository` - Order lifecycle tracking
  - `FillRepository` - Fill recording and retrieval
  - `TradeRepository` - Completed trade history
  - `SystemStateRepository` - System state persistence

**Safety Layer** (`src/safety/`)
- `KillSwitch` - Emergency halt mechanism with multi-trigger support:
  - File-based trigger (KILL_SWITCH file)
  - Programmatic API trigger
  - Auto-triggers: daily loss limit, max drawdown, consecutive losses
  - Persistent state across restarts
- `CircuitBreaker` - API resilience with state machine (CLOSED/OPEN/HALF_OPEN)
  - Automatic failure detection and recovery
  - Configurable thresholds and timeout
  - Statistics tracking

**Orchestration Layer** (`src/orchestration/`)
- `OrdinisOrchestrator` - System lifecycle coordinator:
  - Ordered startup sequence
  - Graceful shutdown with pending operation handling
  - Component health monitoring
  - State transitions: UNINITIALIZED → INITIALIZING → STARTING → RUNNING → STOPPING → STOPPED
- `PositionReconciliation` - Broker position sync:
  - Discrepancy detection (quantity, side, missing positions)
  - Severity classification and action routing
  - Optional auto-correction
  - Audit trail logging

**Alerting Layer** (`src/alerting/`)
- `AlertManager` - Multi-channel alert dispatcher:
  - Desktop notifications via plyer (Windows toast)
  - Severity-based routing (INFO, WARNING, CRITICAL, EMERGENCY)
  - Rate limiting per alert type
  - Content-based deduplication
  - Alert history tracking
- Alert channels:
  - Desktop (implemented)
  - Email (planned)
  - SMS (planned)

**Interface Layer** (`src/interfaces/`)
- Protocol definitions for clean architecture:
  - `EventBus` - Event publishing and subscription
  - `BrokerAdapter` - Broker API abstraction
  - `ExecutionEngine` - Order execution interface
  - `FillModel` - Fill simulation model
  - `CostModel` - Transaction cost calculation
  - `RiskPolicy` - Risk evaluation contract

**Documentation**
- `docs/architecture/PRODUCTION_ARCHITECTURE.md` - Comprehensive Phase 1 architecture documentation
- `docs/architecture/ARCHITECTURE_REVIEW_RESPONSE.md` - Gap analysis and design decisions

### Changed

**Engine Updates**
- `FlowRoute` execution engine:
  - Added order persistence via OrderRepository
  - Integrated circuit breaker for API calls
  - Kill switch enforcement before order submission
  - Fill tracking with position updates
- `RiskGuard` risk engine:
  - Added kill switch status checks
  - Circuit breaker state monitoring
  - Enhanced risk evaluation with safety layer integration

**Dependencies**
- Added `aiosqlite>=0.19.0` for async SQLite operations
- Added `plyer>=2.1.0` for desktop notifications
- New dependency group: `[live-trading]` for production components

### Technical

**Database**
- SQLite with WAL mode for concurrent reads
- Automatic backups with timestamp naming
- Transaction safety with ACID guarantees
- Schema versioning for future migrations

**Safety**
- Kill switch persists state to database and filesystem
- Circuit breaker tracks success/failure statistics
- Integration between safety components and engines

**Architecture**
- Repository pattern for data access abstraction
- Pydantic models for type safety
- Protocol-based interfaces for component contracts
- Async-first design with aiosqlite

### Previous Releases

## [0.1.0] - 2024-12-08

### Added
- **Governance Engines** - Complete governance framework implementation
  - `AuditEngine` - Immutable audit trails with SHA-256 hash chaining
  - `PPIEngine` - Personal data detection and masking
  - `EthicsEngine` - OECD AI Principles (2024) implementation
  - `GovernanceEngine` - Policy enforcement orchestration
  - `BrokerComplianceEngine` - Alpaca/IB terms of service compliance
- **Documentation System** - MkDocs Material with plugins
  - PDF export capability
  - Section numbering
  - Git revision dates
  - Mermaid diagram support
- **Knowledge Base Updates**
  - Governance engines documentation
  - Advanced risk methods
  - Strategy formulation framework
  - NVIDIA integration guide

### Changed
- Updated KB index with implementation status markers
- Enhanced governance engines documentation with usage examples

### Technical
- 85+ governance engine tests
- OECD AI Principles enumeration with all 5 pillars
- Broker compliance for PDT rules, rate limits, data usage

## [0.1.0] - 2024-11-30

### Added
- **Core Trading Infrastructure**
  - `SignalCore` - Signal generation engine
  - `RiskGuard` - Risk management with kill switches
  - `FlowRoute` - Order routing and execution
- **Knowledge Base** - 90+ markdown files organized by trading workflow
- **Paper Trading** - Alpaca Markets integration
- **AI Integration**
  - NVIDIA NIM model integration
  - RAG system for knowledge retrieval
  - Cortex analysis engine
- **Strategies**
  - SMA Crossover strategy
  - Momentum strategy template

### Technical
- Python 3.11+ codebase
- pytest testing framework
- Streamlit dashboard
- Configuration via YAML

---

## Version Naming Convention

- **Major (X.0.0)**: Breaking API changes
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, documentation updates
- **-dev**: Development version, not for production

## Links

- [Project Documentation](docs/index.md)
- [Knowledge Base](docs/knowledge-base/index.md)
- [Architecture](docs/architecture/index.md)
