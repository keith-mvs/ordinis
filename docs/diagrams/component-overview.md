---
title: component-overview.md
date: 2025-12-14
version: 1.0
type: system design diagram (sdd)
description: >
  Overview of the system's architecture and components, including the core data bus, engines, LLM stack, and supporting services.
---

graph TD
    %% Core data bus
    StreamingBus[StreamingBus<br/>(Kafka / NATS)]

    %% Engines
    Orchestration[OrchestrationEngine]
    Signal[SignalEngine]
    Risk[RiskEngine]
    Execution[ExecutionEngine]
    Portfolio[PortfolioEngine]
    Analytics[AnalyticsEngine]
    Opt[PortfolioOptEngine]

    %% LLM stack
    Cortex[Cortex (LLM Reasoning)]
    Synapse[Synapse (RAG Retrieval)]
    Helix[Helix (LLM Provider)]
    CodeGen[CodeGenService]

    %% Supporting services
    Governance[GovernanceEngine]
    Learning[LearningEngine]

    %% External data sources
    MarketData[Market & Alternative Data Sources]

    %% Connections
    MarketData --> StreamingBus
    StreamingBus --> Orchestration
    Orchestration --> Signal
    Orchestration --> Risk
    Orchestration --> Execution
    Orchestration --> Portfolio
    Orchestration --> Analytics
    Orchestration --> Opt
    Orchestration --> Governance
    Orchestration --> Learning

    Signal --> Governance
    Risk --> Governance
    Execution --> Governance
    Portfolio --> Governance
    Analytics --> Governance
    Opt --> Governance
    Learning --> Governance

    %% LLM interactions
    Cortex --> Helix
    Synapse --> Helix
    CodeGen --> Helix
    Cortex --> Synapse
    CodeGen --> Synapse
    Cortex --> Governance
    CodeGen --> Governance

---
    What it shows

All engines communicate through the StreamingBus.
The OrchestrationEngine is the central coordinator.
The LLM stack (Cortex, Synapse, Helix, CodeGen) is isolated from the core trading loop but can be called by any engine that needs reasoning or code assistance.
The GovernanceEngine intercepts every critical action for policy enforcement and audit logging.
The LearningEngine consumes events from the whole system to drive model retraining and knowledge‑base updates.

---
