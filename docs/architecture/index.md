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
| [Production Architecture](production-architecture.md) | Phase 1 baseline architecture (orchestration, safety, persistence) |
| [Execution Path](execution-path.md) | End-to-end trade flow across engines |
| [SignalCore System](signalcore-system.md) | Signal generation engine design |
| [Simulation Engine](simulation-engine.md) | Backtesting and ProofBench flow |
| [Layered System Architecture](layered-system-architecture.md) | Pre-Phase 1 layered model (historical reference) |

### 2.3.2 AI Integration
| Document | Description |
|----------|-------------|
| [NVIDIA Integration](nvidia-integration.md) | Nemotron/NIM integration and AI usage constraints |
| [RAG System](rag-system.md) | Retrieval-augmented pipeline for Cortex and Synapse |

### 2.3.3 Tools & Connectors
| Document | Description |
|----------|-------------|
| [Connectors Reference](../reference/connectors-quick-reference.md) | API connector quick reference |
| [Additional Plugins](additional-plugins-analysis.md) | Extended plugin analysis |

*Note: Internal evaluation docs live under [internal](../internal/index.md).*

### 2.3.4 Phase 1 (Current) and Roadmap
| Document | Description | Status |
|----------|-------------|--------|
| [Production Architecture](production-architecture.md) | Phase 1 architecture baseline | ✅ Current |
| [Phase 1 API Reference](../reference/phase1-api-reference.md) | Persistence, safety, orchestration, alerting APIs | ✅ Current |
| [Architecture Review Response](architecture-review-response.md) | Gap analysis and remediation mapping | ✅ Current |
| [Layered System Architecture](layered-system-architecture.md) | Early orchestration/clean architecture blueprint | 🟡 Historical |
| [Model Alternatives Framework](model-alternatives-framework.md) | Multi-model selection and fallback | 🟡 Planning |
| [NVIDIA Blueprint Integration](nvidia-blueprint-integration.md) | GPU portfolio optimization and distillation | 🟡 Planning |
| [TensorTrade-Alpaca Deployment](tensortrade-alpaca-deployment.md) | Deployment blueprint | 🟡 Planning |
| [Updated Specification and Roadmap](../planning/roadmap-phases-2-4.md) | Phase 2-4 roadmap and documentation reorg | ✅ Published |

### 2.3.5 Related Records
| Document | Description |
|----------|-------------|
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
