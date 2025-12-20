---
title: data-flow.md
date: 2025-12-14
version: 1.0
type: system design diagram (sdd)
description: >
  Overview of the system's architecture and components, including the core data bus, engines, LLM stack, and supporting services.
---

flowchart LR
    subgraph Ingestion
        MD[Market Data] -->|publish| SB[StreamingBus]
        AD[Alternative Data] -->|publish| SB
    end

    subgraph CorePipeline
        SB -->|price_tick| Sig[SignalEngine]
        Sig -->|signal| Ris[RiskEngine]
        Ris -->|approved_signal| Exec[ExecutionEngine]
        Exec -->|execution_report| Port[PortfolioEngine]
        Port -->|position_update| Ana[AnalyticsEngine]
        Ana -->|performance_metrics| SB
    end

    subgraph LLMAssist
        Port -->|request_explanation| Cort[Cortex]
        Cort -->|prompt| Hel[Helix]
        Hel -->|response| Cort
        Cort -->|explanation| Ana

        Sig -->|request_context| Syn[Synapse]
        Syn -->|retrieved_snippets| Sig
    end

    subgraph Governance
        Sig --> Gov[GovernanceEngine]
        Ris --> Gov
        Exec --> Gov
        Port --> Gov
        Ana --> Gov
        Cort --> Gov
        Syn --> Gov
        CodeGen --> Gov
    end

---
    What it shows

Market and alternative data are ingested, validated, and published on the bus.
The core pipeline processes a tick through Signal → Risk → Execution → Portfolio → Analytics, with each step emitting events back onto the bus.
LLM assistance (Cortex, Synapse) can be invoked at any point for explanations or context.
Every engine’s outbound request passes through the GovernanceEngine for policy checks and audit logging.
