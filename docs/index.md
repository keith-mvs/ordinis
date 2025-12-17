# Ordinis Trading System Documentation

**Status:** Phase 1 Complete + Validated
**Last Updated:** {{ git_revision_date_localized }}
**Build Date:** {{ now().strftime('%Y-%m-%d') }}

[:material-lightning-bolt-circle: Getting Started](getting-started/index.md){ .md-button .md-button--primary } [:material-chart-line: View Results](performance/index.md){ .md-button } [:material-file-pdf-box: PDF](print_page/){ .md-button }

---

## Welcome to Ordinis

**Ordinis** is a professional algorithmic trading system that combines AI-powered signal generation, intelligent risk management, and rigorous backtesting to identify high-probability trading opportunities.

**Proven Performance:** 59.7% win rate on real market data (2019-2024), with Sharpe ratio of 2.56
**Conservative:** Only takes high-confidence trades (96% reduction from baseline)
**Transparent:** Full audit trails, documented decisions, open architecture

---

## 🚀 New to Ordinis? Start Here

| Goal | Path |
|------|------|
| **5-minute overview** | [Getting Started](getting-started/index.md) |
| **Run your first backtest** | [Getting Started → Run Demo](getting-started/index.md#option-a-run-a-backtest-fastest---10-minutes) |
| **See the performance data** | [Performance Results](performance/index.md) |
| **Understand how it works** | [Fundamentals](fundamentals/index.md) |

---

## Key Sections

| Section | Description |
|---------|-------------|
| **[Getting Started](getting-started/index.md)** | 5-minute overview, demo backtest, system setup |
| **[Performance Results](performance/index.md)** | Real backtest data (2019-2024), calibration metrics, benchmarks |
| **[Fundamentals](fundamentals/index.md)** | How signals, risk, and execution work together |
| **[Architecture](architecture/index.md)** | Technical system design and component interactions |
| **[User Guides](guides/index.md)** | CLI usage, backtesting, interpreting results |
| **[Reference](reference/index.md)** | API documentation, connectors, specifications |
| **[Decisions](decisions/index.md)** | Architecture Decision Records (ADRs) |
| **[Internal](internal/index.md)** | Developer documentation and contribution guides |

## System Components

```
Ordinis Trading System (Phase 1 Complete)
│
├── Orchestration Layer      # System lifecycle coordination
│   ├── OrdinisOrchestrator  # Startup/shutdown sequences
│   └── PositionReconciliation # Broker position sync
│
├── Safety Layer             # Emergency controls
│   ├── KillSwitch           # Multi-trigger emergency halt
│   └── CircuitBreaker       # API resilience
│
├── Trading Engines
│   ├── SignalCore           # Signal generation engine
│   ├── RiskGuard            # Risk management (integrated with safety)
│   ├── FlowRoute            # Order execution (integrated with persistence)
│   └── Cortex               # AI-powered analysis (NVIDIA integration)
│
├── Persistence Layer        # State management
│   ├── DatabaseManager      # SQLite with WAL mode
│   └── Repositories         # Position, Order, Fill, Trade, SystemState
│
├── Alerting Layer           # Multi-channel notifications
│   └── AlertManager         # Desktop, Email, SMS routing
│
└── Governance               # Compliance and ethics (future enhancement)
    ├── AuditEngine          # Immutable audit trails
    ├── PPIEngine            # Personal data protection
    ├── EthicsEngine         # OECD AI Principles
    └── BrokerCompliance     # Terms of service compliance
```

## Key Features

### Phase 1: Production Infrastructure (Complete)
- **Persistent State Management**: SQLite with WAL mode, automatic backups, repository pattern
- **Safety Controls**: Kill switch with multi-trigger, circuit breaker for API resilience
- **System Orchestration**: Coordinated startup/shutdown, position reconciliation, health monitoring
- **Multi-Channel Alerting**: Desktop notifications, rate limiting, deduplication, severity routing
- **Protocol Interfaces**: Clean architecture contracts for EventBus, BrokerAdapter, ExecutionEngine

### Trading Infrastructure
- **Multi-strategy support**: Technical, fundamental, and quantitative strategies
- **Real-time risk management**: Position limits, drawdown controls, kill switches (Phase 1 enhanced)
- **Paper and live trading**: Alpaca Markets integration (Phase 1 enhanced with persistence)
- **Order lifecycle tracking**: Created → Submitted → Filled with database persistence

### AI Integration
- **NVIDIA NIM Models**: Llama 3.1 for hypothesis generation
- **RAG System**: Knowledge base retrieval for contextual analysis
- **Regime Detection**: ML-based market state classification

### Governance Framework (Future Enhancement)
- **OECD AI Principles**: Ethical AI implementation
- **Audit Trail**: Hash-chained immutable logging
- **Broker Compliance**: Alpaca/IB terms of service enforcement
- **PPI Protection**: Personal data detection and masking

## Getting Started

1. **Review [Architecture](architecture/index.md)** for system design and technical details
2. **Follow [Guides](guides/index.md)** to use CLI and configure the system
3. **Check [Reference](reference/index.md)** for API documentation
4. **Explore the [Knowledge Base (Preview)](kb/index.md)** for domain context
5. **See the [Roadmap](planning/roadmap-phases-2-4.md)** for Phase 2–4 milestones

## Documentation Standards

This documentation follows these conventions:

- **Section Numbering**: All major sections are numbered (1.1, 1.2, etc.)
- **Version Control**: Git-based revision tracking
- **Cross-References**: Internal links use relative paths
- **Code Examples**: Syntax-highlighted with copy buttons

## Version History

| Version | Date | Changes |
|---------|------|---------|
| Phase 1 Baseline | 2025-12-12 | **Phase 1 Complete**: Persistence, Safety, Orchestration, Alerting, Interface protocols |
| 0.1.0 | 2024-12-08 | Added governance engines, OECD principles, broker compliance |
| 0.0.1 | 2024-11-30 | Initial release with core trading infrastructure |

---

*Documentation generated with MkDocs Material*

---

## Document Metadata

```yaml
version: "phase-1-baseline"
created: "2024-11-30"
last_updated: "{{ now().strftime('%Y-%m-%d') }}"
status: "published"
```

---

**END OF DOCUMENT**
