# Clean Architecture Migration Plan
# Ordinis Trading System
# Version: 1.0.0
# Date: 2025-12-12

---

## Executive Summary

This document outlines the migration plan from the current flat module structure to a Clean Architecture pattern as specified in the repo architecture conventions document.

---

## Current State Analysis

### Current Directory Structure

```
src/
├── core/                    # Partial - needs expansion
├── engines/                 # Business components (correct placement)
│   ├── cortex/
│   ├── flowroute/
│   ├── governance/
│   ├── optionscore/
│   ├── proofbench/
│   ├── riskguard/
│   └── signalcore/
├── plugins/                 # Should move to adapters/
│   └── market_data/
├── data/                    # Should move to adapters/storage/
├── analysis/                # Could be core/ or application/
├── dashboard/               # Should move to interface/
├── rag/                     # Should move to adapters/ or engines/cortexrag/
├── strategies/              # Application layer use-cases
├── monitoring/              # Should move to adapters/telemetry/
├── visualization/           # Should move to interface/
├── persistence/             # Should move to adapters/storage/
├── safety/                  # Core domain or cross-cutting
├── orchestration/           # Application layer
├── alerting/                # Should move to adapters/
└── interfaces/              # Protocol definitions - should be core/protocols/
```

### Target Directory Structure

```
src/
└── ordinis/
    ├── __init__.py
    │
    ├── core/                 # Pure domain + contracts (NO I/O)
    │   ├── events.py
    │   ├── types.py          # Money, Qty, Price, Timeframe, etc.
    │   ├── instruments.py
    │   ├── orders.py
    │   ├── fills.py
    │   ├── portfolio.py
    │   ├── risk.py
    │   └── protocols/        # Ports (Protocols)
    │       ├── event_bus.py
    │       ├── broker.py
    │       ├── execution.py
    │       ├── fill_model.py
    │       ├── cost_model.py
    │       └── risk_policy.py
    │
    ├── application/          # Use-cases / orchestration (no external I/O)
    │   ├── backtest/
    │   │   └── run_backtest.py
    │   ├── paper/
    │   │   └── run_paper.py
    │   ├── research/
    │   │   └── run_research.py
    │   └── services/
    │       ├── signal_service.py
    │       ├── risk_service.py
    │       └── execution_service.py
    │
    ├── engines/              # Business components wired via ports
    │   ├── signalcore/
    │   ├── proofbench/
    │   ├── riskguard/
    │   ├── optionscore/
    │   └── cortexrag/
    │
    ├── adapters/             # I/O implementations (outbound + inbound)
    │   ├── market_data/
    │   │   ├── alpha_vantage.py
    │   │   ├── finnhub.py
    │   │   ├── polygon.py
    │   │   └── twelve_data.py
    │   ├── broker/
    │   │   └── alpaca.py
    │   ├── storage/
    │   │   ├── parquet_store.py
    │   │   ├── duckdb_store.py
    │   │   └── sqlite_store.py
    │   ├── event_bus/
    │   │   ├── in_memory.py
    │   │   └── asyncio_queue.py
    │   └── telemetry/
    │       ├── prometheus.py
    │       └── logging_structured.py
    │
    ├── interface/            # Delivery mechanisms
    │   ├── cli/
    │   │   ├── __main__.py
    │   │   └── commands/
    │   └── api/              # Future REST
    │       └── app.py
    │
    └── runtime/              # Glue + DI/wiring
        ├── config.py
        ├── container.py      # Dependency injection / registry
        ├── bootstrap.py
        └── logging.py
```

---

## Dependency Rules

These rules are **non-negotiable** for maintaining clean architecture:

1. **`core/`** imports **nothing** outside `core/`
2. **`application/`** imports `core/` and `core/protocols/` only
3. **`engines/`** imports `core/` + `core/protocols/` (and optionally `application/` services)
4. **`adapters/`** can import anything it must, but **nothing in core/application depends on adapters**
5. **`interface/`** (CLI/API) calls **application use-cases**, not engines directly

---

## Migration Phases

### Phase M1: Foundation (Non-Breaking)
**Status**: Complete
**Completed**: 2025-12-10

**Tasks**:
- [x] Create `artifacts/` directory structure
- [x] Create `configs/` with default.yaml and environment files
- [x] Update `.gitignore` with artifact patterns
- [x] Add `.pre-commit-config.yaml`
- [ ] Add `Makefile` with standard targets (deferred)

**Files Created**:
- `configs/default.yaml`
- `configs/environments/dev.yaml`
- `configs/environments/prod.yaml`
- `artifacts/` directory structure

---

### Phase M2: Protocol Consolidation (Minor Breaking)
**Status**: Complete
**Completed**: 2025-12-12

**Tasks**:
- [x] Rename `src/interfaces/` to `src/core/protocols/`
- [x] Update all imports referencing interfaces
- [x] Add missing protocol definitions from conventions doc
- [x] Validate no I/O in protocol definitions

**Result**: All protocol definitions now in `src/core/protocols/`. Imports updated across codebase.

---

### Phase M3: Adapter Extraction (Medium Breaking)
**Status**: Complete
**Completed**: 2025-12-12

**Tasks**:
- [x] Create `src/adapters/` structure
- [x] Move `src/plugins/market_data/` -> `src/adapters/market_data/`
- [x] Move `src/persistence/` -> `src/adapters/storage/`
- [x] Move `src/alerting/` -> `src/adapters/alerting/`
- [x] Move `src/monitoring/` -> `src/adapters/telemetry/`
- [ ] Extract broker adapter from `src/engines/flowroute/adapters/` (deferred - tightly coupled)

**Result**: Adapters layer created with storage/, market_data/, alerting/, telemetry/ modules.

---

### Phase M4: Application Layer Creation (Medium Breaking)
**Status**: Complete
**Completed**: 2025-12-12

**Tasks**:
- [x] Create `src/application/` structure
- [x] Move `src/orchestration/` -> `src/application/services/`
- [x] Move `src/strategies/` -> `src/application/strategies/`
- [ ] Create use-case modules (run_backtest, run_paper, run_research) (deferred)
- [x] Ensure application layer has no direct I/O

**Result**: Application layer created with services/ and strategies/ modules.

---

### Phase M5: Interface Layer Creation (Minor Breaking)
**Status**: Complete
**Completed**: 2025-12-12

**Tasks**:
- [x] Create `src/interface/cli/` structure
- [x] Move dashboard to `src/interface/dashboard/`
- [x] Migrate `src/cli.py` to `src/interface/cli/__main__.py`
- [ ] Add admin command for cleanup (`ordinis admin clean`) (deferred)

**Result**: Interface layer created with cli/ and dashboard/ modules.

---

### Phase M6: Runtime/DI Setup (Non-Breaking)
**Status**: Complete
**Completed**: 2025-12-12

**Tasks**:
- [x] Create `src/runtime/` structure
- [x] Implement `config.py` with Pydantic BaseSettings
- [x] Implement `container.py` for dependency injection (in `src/core/`)
- [x] Implement `bootstrap.py` for application startup
- [x] Consolidate logging configuration (`runtime/logging.py`)

**Result**: Runtime layer created with config, bootstrap, and logging modules. DI container remains in `src/core/container.py` and is re-used by bootstrap.

---

### Phase M7: Package Restructure (Major Breaking)
**Status**: Planned
**Timeline**: Phase 3+

**Tasks**:
- [ ] Create nested `src/ordinis/` package structure
- [ ] Update `pyproject.toml` package configuration
- [ ] Update all imports to use `ordinis.` prefix
- [ ] Update test imports
- [ ] Update documentation

**Note**: This is the most disruptive change and should be done when the codebase is stable.

---

## Naming Convention Audit

### File Naming (snake_case)
Current violations to fix:
- None identified (files follow snake_case)

### Class Naming (PascalCase)
Current patterns are correct:
- `EventBus`, `BrokerAdapter`, `RiskPolicy` - correct
- Protocol files in `core/protocols/` - implementation in `adapters/` or `engines/`

### Event Naming
Target convention:
- Event classes: `SomethingEvent`
- Event topic string: `domain.action` (e.g., `market.bar`, `signal.generated`, `order.submitted`)

---

## Config Management

### Implemented
- `configs/default.yaml` - Base configuration
- `configs/environments/dev.yaml` - Development overrides
- `configs/environments/prod.yaml` - Production overrides

### TODO
- [ ] Create `src/runtime/config.py` with Pydantic BaseSettings
- [ ] Implement config precedence: env vars > environment file > default
- [ ] Add config validation on startup
- [ ] Add config snapshot per run

---

## Admin Cleanup Discipline

### Implemented
- `artifacts/` directory structure created
- `.gitignore` updated to exclude artifacts

### TODO
- [ ] Implement CLI admin command: `ordinis admin clean --older-than 14d`
- [ ] Add retention policy enforcement (by age + max size)
- [ ] Add run folder naming convention: `YYYYMMDD/HHMMSSZ_<task>_<gitsha>/`
- [ ] Add config.snapshot.yaml per run

---

## Risk Assessment

| Phase | Risk Level | Mitigation |
|-------|------------|------------|
| M1: Foundation | Low | Non-breaking, additive only |
| M2: Protocol Consolidation | Low | Simple rename with import updates |
| M3: Adapter Extraction | Medium | Incremental migration with aliases |
| M4: Application Layer | Medium | Careful import analysis required |
| M5: Interface Layer | Low | CLI is separate concern |
| M6: Runtime/DI | Low | Additive, can coexist with current |
| M7: Package Restructure | High | Do last, requires full import sweep |

---

## Testing Strategy

### Before Each Phase
1. Run full test suite
2. Document current import graph
3. Create rollback branch

### After Each Phase
1. Run full test suite
2. Verify no circular imports
3. Run import linter (e.g., `importlinter`)
4. Update documentation

---

## Decision Required

**Question from conventions doc**: Is Ordinis primarily:
- **A)** A reusable Python library others import, or
- **B)** A deployable service/app (CLI first, API later)?

**Recommended Answer**: **B** - Deployable service/app

**Rationale**:
- Primary use is personal trading system
- CLI interface for operation
- Future REST API for monitoring
- Not intended as a pip-installable library for others

**Impact on Structure**:
- Focus on `interface/cli/` and `interface/api/`
- Less emphasis on public API surface
- Can use relative imports within package
- Entry point via `__main__.py`

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
source: "C:\\Users\\kjfle\\Wiki\\ordinis\\Repo architecture and naming cleanup conventions.md"
status: "in_progress"
next_review: "2025-12-31"
```

---

**END OF DOCUMENT**
