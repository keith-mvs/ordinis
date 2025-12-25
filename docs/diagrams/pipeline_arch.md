# Pipeline architecture (Mermaid)

Annotated system diagram for the Ordinis pipeline. Declare `AnalyticsEngine` before the main pipeline so it renders at the top.

```mermaid
flowchart TB
  %% Top: Analytics
  AnalyticsEngine["AnalyticsEngine"]
  Gov["GovernanceEngine\n(pre‑flight checks & audit)"]

  subgraph pipeline
    direction LR
    Market["Market / Alt‑Data"] --> StreamingBus["StreamingBus"] --> SignalEngine["SignalEngine"] --> RiskEngine["RiskEngine"] --> ExecutionEngine["ExecutionEngine"] --> PortfolioEngine["PortfolioEngine"]
  end

  %% Data flow into analytics
  PortfolioEngine --> AnalyticsEngine
  ExecutionEngine --> AnalyticsEngine

  %% Governance preflight/audit links (annotated)
  SignalEngine -.->|preflight:\n• whitelist/blacklist\n• signal validity (prob≥min)\n• ADX>25 & slope gating\n• fib level match (±tol)\n• volume confirmation| Gov
  RiskEngine -.->|preflight:\n• position size & sector caps\n• daily loss / drawdown\n• leverage limits / adjust| Gov
  ExecutionEngine -.->|preflight:\n• market hours & tradable\n• liquidity / avg vol check\n• order size vs min tick\n• slippage estimate| Gov
  PortfolioEngine -.->|preflight:\n• buying power & margin\n• concentration / HHI caps\n• cross-product exposure| Gov
  AnalyticsEngine -.->|audit:\n• metrics & model_version\n• event snapshot & provenance| Gov

  classDef topEngine fill:#eef,stroke:#333,stroke-width:1px;
  class AnalyticsEngine topEngine
```
