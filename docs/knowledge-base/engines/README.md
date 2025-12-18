# Engines Knowledge Base

This folder documents system engines. For canonical naming and engine roles, treat
`docs/knowledge-base/inbox/documents/system-specification.md` as the source of truth.

## Terminology (canonical)

Use the System Specification engine names consistently:

- `OrchestrationEngine`: coordinates the end-to-end trading cycle
- `StreamingBus`: event transport for market/system events
- `SignalEngine (SignalCore)`: generates trading signals
- `RiskEngine (RiskGuard)`: deterministic risk policy and gating
- `ExecutionEngine (FlowRoute)`: routes/executes orders (simulated or live)
- `PortfolioEngine`: maintains positions, cash, exposures
- `AnalyticsEngine (ProofBench)`: backtesting, evaluation, reporting
- `LearningEngine`: model lifecycle, drift detection, rollout
- `GovernanceEngine`: policy, compliance, audit, preflight
- Supporting services: `Cortex` (LLM reasoning), `Synapse` (retrieval), `Helix` (LLM provider layer)

Rule of thumb: on first mention, write `RoleName (Codename)`; afterwards use `RoleName`.

## Start Here

- System context: `docs/knowledge-base/engines/system-architecture.md`
- Engine definitions and interfaces: `docs/knowledge-base/inbox/documents/system-specification.md`

## Documents In This Folder

### Architecture & Workflow

- `docs/knowledge-base/engines/system-architecture.md`: Phase 1 production architecture (persistence, safety, observability)
- `docs/knowledge-base/engines/execution-path.md`: path from research → backtesting → paper → live

### Engine Implementations

- `docs/knowledge-base/engines/signalcore-engine.md`: `SignalEngine (SignalCore)` architecture and responsibilities
- `docs/knowledge-base/engines/proofbench.md`: `AnalyticsEngine (ProofBench)` simulation/backtesting architecture
- `docs/knowledge-base/engines/proofbench-guide.md`: `AnalyticsEngine (ProofBench)` usage guide and examples
- `docs/knowledge-base/engines/learning_engine.md`: `LearningEngine` API overview

### AI / Retrieval Layer

- `docs/knowledge-base/engines/helix.md`: `Helix` LLM provider layer (routing, caching, fallback)
- `docs/knowledge-base/engines/nvidia-integration.md`: NVIDIA model integration across engines
- `docs/knowledge-base/engines/rag-engine.md`: `Synapse` retrieval (RAG) architecture and integration points

### Observability

- `docs/knowledge-base/engines/monitoring.md`: logging, metrics, health checks
