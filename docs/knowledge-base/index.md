# 3. Knowledge Base

**Last Updated:** {{ git_revision_date_localized }}

---

## 3.1 Overview

The Intelligent Investor Knowledge Base provides foundational trading knowledge for automated strategy development. Content is organized by strategy formulation workflow for efficient retrieval during strategy design, backtesting, and deployment.

## 3.2 Navigation

| Section | Content | Status |
|---------|---------|--------|
| [1. Foundations](01_foundations/README.md) | Market structure, microstructure | Complete |
| [2. Signals](02_signals/technical/README.md) | Signal generation methods | Complete |
| [3. Risk](03_risk/README.md) | Risk management | Complete |
| [4. Strategy](04_strategy/BACKTESTING_REQUIREMENTS.md) | Strategy design | Complete |
| [5. Execution](05_execution/README.md) | System architecture | Complete |
| [6. Options](06_options/README.md) | Derivatives | Partial |
| [7. References](07_references/README.md) | Academic sources | Complete |

## 3.3 Knowledge Structure

```
knowledge-base/
├── 00_KB_INDEX.md              # Master index
├── 01_foundations/             # Market structure, microstructure
├── 02_signals/                 # Signal generation methods
│   ├── technical/              # Technical analysis indicators
│   ├── fundamental/            # Fundamental analysis
│   ├── volume/                 # Volume & liquidity
│   ├── sentiment/              # News & social sentiment
│   ├── events/                 # Event-driven strategies
│   └── quantitative/           # Quant strategies & ML
├── 03_risk/                    # Risk management & position sizing
├── 04_strategy/                # Strategy design & evaluation
├── 05_execution/               # System architecture & execution
├── 06_options/                 # Options & derivatives
└── 07_references/              # Academic sources & citations
```

## 3.4 Source Philosophy

- **Primary Sources**: Academic/scholarly sources and peer-reviewed research for core logic
- **Secondary Sources**: Credible industry publications and regulatory documents
- **Validation**: All claimed edges must be academically validated

## 3.5 Key Documents

### Recently Added
- [Governance Engines](05_execution/governance_engines.md) - OECD AI Principles, broker compliance
- [Advanced Risk Methods](03_risk/advanced_risk_methods.md) - VaR, stress testing
- [Strategy Framework](04_strategy/strategy_formulation_framework.md) - Complete development guide
- [NVIDIA Integration](04_strategy/nvidia_integration.md) - AI-enhanced strategies

See the full [Knowledge Base Index](00_KB_INDEX.md) for complete navigation.
