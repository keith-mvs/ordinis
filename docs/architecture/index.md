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
| [SignalCore System](signalcore-system.md) | Signal generation engine |
| [Execution Path](execution-path.md) | Order flow and execution |
| [Simulation Engine](simulation-engine.md) | Backtesting infrastructure |
| [Monitoring](monitoring.md) | System observability |

### 2.3.2 AI Integration
| Document | Description |
|----------|-------------|
| [NVIDIA Integration](nvidia-integration.md) | NVIDIA NIM model integration |
| [RAG System](rag-system.md) | Knowledge retrieval architecture and implementation |

### 2.3.3 Tools & Connectors
| Document | Description |
|----------|-------------|
| [MCP Tools Evaluation](mcp-tools-evaluation.md) | Model Context Protocol tools |
| [MCP Quick Start](mcp-tools-quick-start.md) | Getting started with MCP |
| [Claude Connectors](claude-connectors-evaluation.md) | Claude API integration |
| [Connectors Reference](connectors-quick-reference.md) | Quick reference guide |

### 2.3.4 Production Architecture (Phase 1)
| Document | Description | Status |
|----------|-------------|--------|
| [Production Architecture](production-architecture.md) | **Phase 1 Complete** - Comprehensive architecture documentation | ✅ Current |
| [Phase 1 API Reference](phase1-api-reference.md) | **NEW** - Complete API documentation for persistence, safety, orchestration, alerting | ✅ Current |
| [Architecture Review Response](architecture-review-response.md) | Phase 1 gap analysis - Maps external review feedback to implementation | ✅ Current |
| [Layered System Architecture](layered-system-architecture.md) | **Master spec** - Orchestration and component integration | 🟡 Pre-Phase 1 |
| [Model Alternatives Framework](model-alternatives-framework.md) | Multi-model selection and fallback strategy | 🟡 Planning |
| [NVIDIA Blueprint Integration](nvidia-blueprint-integration.md) | PortOpt and Distillery infrastructure | 🟡 Planning |
| [TensorTrade-Alpaca Deployment](tensortrade-alpaca-deployment.md) | Production deployment specification | 🟡 Planning |

### 2.3.5 Development & Analysis
| Document | Description |
|----------|-------------|
| [System Capabilities](system-capabilities-assessment.md) | Feature assessment |
| [Development TODO](development-todo.md) | Development backlog |
| [Additional Plugins](additional-plugins-analysis.md) | Extended plugin analysis |

## 2.4 Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Data | Pandas, NumPy, Polars |
| AI | NVIDIA NIM, LangChain |
| Broker | Alpaca Markets API |
| Testing | pytest, ProofBench |
| Documentation | MkDocs Material |
