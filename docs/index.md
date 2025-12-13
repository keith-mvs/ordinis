# Ordinis Trading System Documentation

**Version:** 0.2.0-dev
**Last Updated:** {{ git_revision_date_localized }}
**Build Date:** {{ now().strftime('%Y-%m-%d') }}

[:material-file-pdf-box: Download as PDF](print_page/){ .md-button .md-button--primary } [:material-printer: Print Version](print_page/){ .md-button }

---

## Overview

Ordinis is a professional algorithmic trading system built with transparency, ethics, and responsible AI principles at its core. The system implements OECD AI Principles (2024) and provides comprehensive governance for automated trading decisions.

## Quick Links

| Section | Description |
|---------|-------------|
| [Architecture](architecture/index.md) | System design and technical architecture |
| [Knowledge Base](knowledge-base/index.md) | Trading knowledge and domain expertise |
| [Guides](guides/index.md) | User guides and tutorials |
| [Reference](reference/) | API documentation and quick references |
| [Decisions](decisions/) | Architecture Decision Records (ADRs) |
| [Internal](internal/) | Developer documentation (not user-facing) |

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
2. **Explore the [Knowledge Base](knowledge-base/index.md)** for trading domain knowledge
3. **Follow [Guides](guides/index.md)** to use CLI and configure the system
4. **Check [Reference](reference/)** for API documentation

## Documentation Standards

This documentation follows these conventions:

- **Section Numbering**: All major sections are numbered (1.1, 1.2, etc.)
- **Version Control**: Git-based revision tracking
- **Cross-References**: Internal links use relative paths
- **Code Examples**: Syntax-highlighted with copy buttons

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.2.0-dev | 2025-12-12 | **Phase 1 Complete**: Persistence, Safety, Orchestration, Alerting, Interface protocols |
| 0.1.0 | 2024-12-08 | Added governance engines, OECD principles, broker compliance |
| 0.0.1 | 2024-11-30 | Initial release with core trading infrastructure |

---

*Documentation generated with MkDocs Material*
