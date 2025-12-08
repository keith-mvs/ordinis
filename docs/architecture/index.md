# 2. System Architecture

**Last Updated:** {{ git_revision_date_localized }}

---

## 2.1 Overview

The Ordinis trading system is built on a modular architecture with clear separation of concerns. Each component handles a specific aspect of the trading lifecycle.

## 2.2 Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ORDINIS ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  SignalCore  │───▶│  RiskGuard   │───▶│  FlowRoute   │   │
│  │   (Signals)  │    │    (Risk)    │    │ (Execution)  │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │                   │            │
│         ▼                   ▼                   ▼            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Governance Layer                     │   │
│  │  Audit │ Ethics │ PPI │ Compliance │ Broker ToS      │   │
│  └──────────────────────────────────────────────────────┘   │
│         │                   │                   │            │
│         ▼                   ▼                   ▼            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Cortex (AI)                        │   │
│  │    NVIDIA NIM │ RAG │ Regime Detection │ Analysis    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 2.3 Document Index

### 2.3.1 System Overview
| Document | Description |
|----------|-------------|
| [SignalCore System](SIGNALCORE_SYSTEM.md) | Signal generation engine |
| [Execution Path](EXECUTION_PATH.md) | Order flow and execution |
| [Simulation Engine](SIMULATION_ENGINE.md) | Backtesting infrastructure |
| [Monitoring](MONITORING.md) | System observability |

### 2.3.2 AI Integration
| Document | Description |
|----------|-------------|
| [NVIDIA Integration](NVIDIA_INTEGRATION.md) | NVIDIA NIM model integration |
| [RAG System](RAG_SYSTEM.md) | Knowledge retrieval architecture |
| [RAG Implementation](RAG_IMPLEMENTATION.md) | Implementation details |

### 2.3.3 Tools & Connectors
| Document | Description |
|----------|-------------|
| [MCP Tools Evaluation](MCP_TOOLS_EVALUATION.md) | Model Context Protocol tools |
| [MCP Quick Start](MCP_TOOLS_QUICK_START.md) | Getting started with MCP |
| [Claude Connectors](CLAUDE_CONNECTORS_EVALUATION.md) | Claude API integration |
| [Connectors Reference](CONNECTORS_QUICK_REFERENCE.md) | Quick reference guide |

### 2.3.4 Development
| Document | Description |
|----------|-------------|
| [System Capabilities](SYSTEM_CAPABILITIES_ASSESSMENT.md) | Feature assessment |
| [Development TODO](DEVELOPMENT_TODO.md) | Development backlog |

## 2.4 Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Data | Pandas, NumPy, Polars |
| AI | NVIDIA NIM, LangChain |
| Broker | Alpaca Markets API |
| Testing | pytest, ProofBench |
| Documentation | MkDocs Material |
