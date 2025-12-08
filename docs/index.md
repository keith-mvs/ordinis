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
| [Project](project/index.md) | Project scope, status, and roadmap |
| [Architecture](architecture/index.md) | System design and technical architecture |
| [Knowledge Base](knowledge-base/index.md) | Trading knowledge and research |
| [Guides](guides/index.md) | User guides and tutorials |
| [Analysis](analysis/index.md) | Market analysis and backtest results |
| [Testing](testing/index.md) | Testing procedures and benchmarks |

## System Components

```
Ordinis Trading System
├── SignalCore         # Signal generation engine
├── RiskGuard          # Risk management and limits
├── FlowRoute          # Order routing and execution
├── Cortex             # AI-powered analysis (NVIDIA integration)
└── Governance         # Compliance and ethics engines
    ├── AuditEngine    # Immutable audit trails
    ├── PPIEngine      # Personal data protection
    ├── EthicsEngine   # OECD AI Principles
    └── BrokerCompliance # Terms of service compliance
```

## Key Features

### Trading Infrastructure
- **Multi-strategy support**: Technical, fundamental, and quantitative strategies
- **Real-time risk management**: Position limits, drawdown controls, kill switches
- **Paper and live trading**: Alpaca Markets integration

### AI Integration
- **NVIDIA NIM Models**: Llama 3.1 for hypothesis generation
- **RAG System**: Knowledge base retrieval for contextual analysis
- **Regime Detection**: ML-based market state classification

### Governance Framework
- **OECD AI Principles**: Ethical AI implementation
- **Audit Trail**: Hash-chained immutable logging
- **Broker Compliance**: Alpaca/IB terms of service enforcement
- **PPI Protection**: Personal data detection and masking

## Getting Started

1. **Read the [Project Scope](project/PROJECT_SCOPE.md)** to understand system objectives
2. **Review [Architecture](architecture/index.md)** for technical details
3. **Explore the [Knowledge Base](knowledge-base/index.md)** for trading fundamentals
4. **Follow [Testing Guides](testing/index.md)** to run the system

## Documentation Standards

This documentation follows these conventions:

- **Section Numbering**: All major sections are numbered (1.1, 1.2, etc.)
- **Version Control**: Git-based revision tracking
- **Cross-References**: Internal links use relative paths
- **Code Examples**: Syntax-highlighted with copy buttons

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.2.0-dev | 2024-12-08 | Added governance engines, OECD principles, broker compliance |
| 0.1.0 | 2024-11-30 | Initial release with core trading infrastructure |

---

*Documentation generated with MkDocs Material*
