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
| [Monitoring](../knowledge-base/engines/monitoring.md) | System observability (in Knowledge Base) |

### 2.3.2 AI Integration
| Document | Description |
|----------|-------------|
| [NVIDIA Integration](nvidia-integration.md) | NVIDIA NIM model integration |
| [RAG System](rag-system.md) | Knowledge retrieval architecture and implementation |

### 2.3.3 Tools & Connectors
| Document | Description |
|----------|-------------|
| [Connectors Reference](../reference/connectors-quick-reference.md) | Quick reference for API connectors |

*Note: Internal evaluation docs moved to [internal/](../internal/) section.*

### 2.3.4 Production Architecture (Phase 1)
| Document | Description | Status |
|----------|-------------|--------|
| [Production Architecture](production-architecture.md) | **Phase 1 Complete** - Comprehensive architecture documentation | ✅ Current |
| [Phase 1 API Reference](../reference/phase1-api-reference.md) | Complete API documentation for persistence, safety, orchestration, alerting | ✅ Current |
| [Architecture Review Response](architecture-review-response.md) | Phase 1 gap analysis - Maps external review feedback to implementation | ✅ Current |
| [Layered System Architecture](layered-system-architecture.md) | **Master spec** - Orchestration and component integration | 🟡 Pre-Phase 1 |
| [Model Alternatives Framework](model-alternatives-framework.md) | Multi-model selection and fallback strategy | 🟡 Planning |
| [NVIDIA Blueprint Integration](nvidia-blueprint-integration.md) | PortOpt and Distillery infrastructure | 🟡 Planning |
| [TensorTrade-Alpaca Deployment](tensortrade-alpaca-deployment.md) | Production deployment specification | 🟡 Planning |

### 2.3.5 Supplementary
| Document | Description |
|----------|-------------|
| [Additional Plugins](additional-plugins-analysis.md) | Extended plugin analysis |
| [Clean Architecture ADR](../decisions/adr-clean-architecture-migration.md) | Architecture decision record |

## 2.4 Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Data | Pandas, NumPy, Polars |
| AI | NVIDIA NIM, LangChain |
| Broker | Alpaca Markets API |
| Testing | pytest, ProofBench |
| Documentation | MkDocs Material |
