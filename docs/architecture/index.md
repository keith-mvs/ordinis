<!-- ARCHIVED: This file has been merged into overview.md as of 2025-12-15. All future updates should be made in overview.md. -->

### 📊 **Start Here**

**New to the system?** Start with the [System Overview](overview.md) (with Mermaid diagrams)

### 🎯 **Core Architecture**

| Document | Description | Read Time |
|----------|-------------|-----------|
| [System Overview](overview.md) | Complete architecture with interactive diagrams | 15 min |
| [Production Architecture](production-architecture.md) | Phase 1 baseline (orchestration, safety, persistence) | 20 min |
| [Execution Path](execution-path.md) | End-to-end trade flow across all engines | 15 min |

### 🧠 **Specialized Components**

| Document | Description | Read Time |
|----------|-------------|-----------|
| [SignalCore System](signalcore-system.md) | Signal generation and multi-model voting | 20 min |
| [RAG System](rag-system.md) | Knowledge base retrieval for analysis | 15 min |
| [NVIDIA Integration](nvidia-integration.md) | AI-powered analysis with Llama 3.1 | 15 min |

---

## Key Design Principles

### ✅ Modularity
- Independent, testable components
- Clear interfaces between systems
- Easy to evolve/replace parts

### ✅ Safety
- Multi-trigger kill switch
- Database persistence for recovery
- Comprehensive audit logging

### ✅ Transparency
- All decisions logged
- Confidence scores explained
- Per-trade P&L tracking

### ✅ Extensibility
- New signal models can be plugged in
- Custom risk rules supported
- Multiple execution venues possible
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
