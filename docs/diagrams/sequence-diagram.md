---
title: sequence-diagram.md
date: 2025-12-14
version: 1.0
type: sequence diagram
description: >
   Sequence diagram illustrating the flow of events and interactions between the system's components.

---

sequenceDiagram
    participant Market as Market Data Source
    participant Bus as StreamingBus
    participant Orchestrator as OrchestrationEngine
    participant Signal as SignalEngine
    participant Risk as RiskEngine
    participant Exec as ExecutionEngine
    participant Port as PortfolioEngine
    participant Anal as AnalyticsEngine
    participant Gov as GovernanceEngine
    participant Cortex as Cortex (LLM)
    participant Helix as Helix
    participant Synapse as Synapse

    Market->>Bus: publish tick event
    Bus->>Orchestrator: deliver event
    Orchestrator->>Gov: preflight (event ingest)
    Gov-->>Orchestrator: allow

    Orchestrator->>Signal: generate_signals(tick)
    Signal->>Gov: preflight (signal generation)
    Gov-->>Signal: allow
    Signal->>Synapse: retrieve(context)   Note right: optional RAG
    Synapse-->>Signal: snippets
    Signal-->>Orchestrator: signal list

    Orchestrator->>Risk: evaluate(signal)
    Risk->>Gov: preflight (risk check)
    Gov-->>Risk: allow
    Risk-->>Orchestrator: approved/adjusted signal

    Orchestrator->>Exec: execute(order)
    Exec->>Gov: preflight (execution)
    Gov-->>Exec: allow
    Exec-->>Orchestrator: execution_report

    Orchestrator->>Port: update_portfolio(report)
    Port->>Gov: preflight (portfolio update)
    Gov-->>Port: allow
    Port-->>Orchestrator: new portfolio state

    Orchestrator->>Anal: analyze(portfolio_state)
    Anal->>Cortex: request_explanation(metrics)
    Cortex->>Helix: generate(prompt)
    Helix-->>Cortex: text
    Cortex-->>Anal: narrative
    Anal-->>Orchestrator: analytics_report

    Orchestrator->>Gov: audit(event_summary)
    Gov-->>Orchestrator: logged

---
    What it shows

The step‑by‑step flow for a single market tick, including governance checks at each stage.
Optional retrieval‑augmented assistance for the SignalEngine via Synapse.
Narrative generation for the analytics report using Cortex and Helix.
Final audit logging by the GovernanceEngine.
