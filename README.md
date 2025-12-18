# Ordinis

**Version:** 0.2.0‑dev (Development Build – Clean Architecture Complete)
**Status:** ✅ Production‑Ready Infrastructure | Clean Architecture Migration Complete

An AI‑driven quantitative trading system with production‑grade persistence, safety controls, multi‑source market data integration, and a fully documented modular architecture.

---

## Project Overview

- **Persistence Layer** – SQLite with WAL mode, automatic backups, repository pattern, and Pydantic models.
- **Safety Controls** – Kill‑switch with multiple triggers, circuit‑breaker for API resilience, daily loss‑limit and draw‑down monitoring.
- **Orchestration** – System lifecycle management, position reconciliation, health monitoring, graceful shutdown.
- **Alerting** – Multi‑channel notifications, severity‑based routing, rate limiting, deduplication, alert history.
- **Market Data** – Integration with Alpha Vantage, Finnhub, Massive, Twelve Data.
- **Paper‑Trading Engine** – Realistic fill simulation, order‑lifecycle tracking, slippage and commission modeling.
- **Back‑testing Framework** – Event‑driven ProofBench with performance analytics (Sharpe, Sortino, max draw‑down, etc.).
- **RiskGuard Framework** – Hard control gates for exposure, sector caps, daily loss, and integration with kill‑switch/circuit‑breaker.
- **Advanced Technical Analysis** – Ichimoku Cloud, candlestick/breakout detection, composite and multi‑timeframe analysis.
- **Five Ready‑to‑Deploy Strategies** – Moving‑Average Crossover, RSI Mean Reversion, Momentum Breakout, Bollinger Bands, MACD.

The system follows a **clean‑architecture** layout (see `src/ordinis/`), with all business logic isolated in engines and external concerns handled by adapters.

---

## Architecture Overview

The platform is composed of independent, event‑driven engines that communicate through a unified **StreamingBus** (Kafka or NATS/Redis Streams).  Governance, learning, and LLM services are layered on top of the core trading loop.

```
Market / Alt‑Data → StreamingBus → SignalEngine → RiskEngine → ExecutionEngine → PortfolioEngine → AnalyticsEngine
                                   ↑                ↑                ↑                ↑
                                   │                │                │                │
                                   └─ GovernanceEngine (pre‑flight checks & audit) ──┘
```

Supporting services:

| Service | Role |
|---------|------|
| **Cortex** | LLM reasoning and code‑analysis (deep code review, research synthesis). |
| **Synapse** | Retrieval‑augmented generation – indexes documentation, code, research papers and provides relevant snippets to Cortex. |
| **Helix** | Unified LLM provider façade – dispatches requests to the appropriate model, handles authentication, retries, rate‑limits, and safety filtering. |
| **LearningEngine** | Continuous improvement – collects events, retrains signal models, fine‑tunes LLM prompts, re‑indexes Synapse, and runs the benchmark suite before promotion. |
| **CodeGenService** | AI‑assisted code generation and patching, using Cortex for reasoning and Synapse for context. |

All engines expose a small, well‑defined interface, making it straightforward to add new asset classes, models, or data sources.

---

## Core Engines – Responsibilities and Interfaces

| Engine | Primary Responsibility | Key Interface |
|--------|------------------------|----------------|
| **OrchestrationEngine** | Coordinates the full trading cycle (live or back‑test), propagates context, emits tracing events, invokes GovernanceEngine pre‑flight checks. | `run_cycle(event)`, `run_backtest(config)` |
| **StreamingBus** | Schema‑validated event bus, publishes and subscribes to topics, supports Kafka or NATS/Redis Streams. | `publish(event)`, `subscribe(topic, handler)` |
| **SignalEngine** | Generates trading signals from enriched market data using GBM, XGBoost, LSTM, or Transformer models; optional ensemble; basic sanity checks. | `generate_signals(data_frame) → List[Signal]` |
| **RiskEngine** | Enforces deterministic risk policies (exposure, leverage, sector caps, stop‑loss, volatility, liquidity, contract‑specific rules); returns allow/adjust/deny decisions. | `evaluate(signal, portfolio_state) → (bool, Signal, List[Reason])` |
| **ExecutionEngine** | Creates orders, routes them to paper‑trader, broker API, FIX, or internal matching engine; applies fill‑model plug‑ins; publishes execution reports. | `execute(order, market_state) → ExecutionReport` |
| **PortfolioEngine** | Maintains positions, cash, margin; performs rebalancing; enforces portfolio‑level constraints; provides snapshots to other engines. | `rebalance(target_allocations, constraints)`, `get_portfolio_state()` |
| **AnalyticsEngine** | Computes performance metrics (CAGR, Sharpe, Sortino, draw‑down, profit factor, etc.); generates narrative reports via Cortex; emits analytics events. | `analyze(results_dataset) → Report` |
| **PortfolioOptEngine** | GPU‑accelerated mean‑CVaR or mean‑variance optimisation using NVIDIA cuOpt (GPU) with CPU fallback; returns optimal weights and risk metrics. | `optimize(returns_data, constraints) → OptResult` |
| **Cortex** | LLM reasoning and code‑analysis assistant (no live‑trading decisions). | `analyze_code(code, type)`, `synthesize_research(query, sources)` |
| **Synapse** | Retrieval‑augmented generation – indexes documentation, embeds with NVIDIA EmbedLM‑300M, provides top‑k snippets with metadata. | `retrieve(query, context)` |
| **Helix** | Unified LLM provider – dispatches `generate(messages, model_id=None, **options)` calls; handles auth, retries, safety filtering. | `generate(...)` |
| **GovernanceEngine** | Cross‑cutting policy enforcement and immutable audit logging; pre‑flight decisions and real‑time alerts. | `preflight(context)`, `audit(event)` |
| **LearningEngine** | Continuous improvement pipeline – records events, retrains models, runs benchmark suite, supports shadow‑mode rollout. | `record_event(event)`, `train(models)`, `evaluate(new_model, benchmark)` |
| **CodeGenService** | AI‑assisted code generation and patching; uses Cortex for reasoning and Synapse for context; governance filters for safety. | `propose_change(prompt, files_context)` |

---

## Model Mapping – Consolidated Table

| Engine / Service | Default Model | Helix Model ID | Primary Use‑Case |
|------------------|---------------|----------------|------------------|
| Cortex (code review, reasoning) | Nemotron‑49B | `nemotron-super-49b-v1.5` | Deep code analysis, strategy reasoning |
| Cortex (fallback) | Nemotron‑8B | `nemotron-8b-v3.1` | Quick explanations, low‑latency tasks |
| Helix (generic) | Nemotron‑49B | `nemotron-super-49b-v1.5` | Default high‑quality LLM |
| Helix (lightweight) | Nemotron‑8B | `nemotron-8b-v3.1` | Fast, inexpensive inference |
| AnalyticsEngine (narrative) | Nemotron‑8B | `nemotron-8b-v3.1` | Report generation |
| SignalEngine (ML) | Gradient‑Boosting, XGBoost, LSTM, Transformer | N/A (traditional ML) | Predictive signal generation |
| RiskEngine (aux.) | Optional volatility forecaster | N/A | Dynamic risk limits |
| Synapse – Embedding | NVIDIA EmbedLM‑300M | `nvidia/llama-3.2-nemoretriever-300m-embed-v2` | Document and query vectorisation |
| Synapse – Rerank | NVIDIA Rerank‑500M | `nvidia/llama-3.2-nemoretriever-500m-rerank-v2` | Re‑ranking of retrieved snippets |
| PortfolioOptEngine | cuOpt (GPU LP/QP) | N/A | Mean‑CVaR / mean‑variance optimisation |
| CodeGenService | Nemotron‑49B | `nemotron-super-49b-v1.5` | Complex code synthesis |
| RiskGuard (LLM safety) | Nemoguard‑8B | `nvidia/llama-3.1-nemoguard-8b-content-safety` | Content safety guardrails |
| SignalCore (LLM‑assisted suggestions) | Meta Llama‑3.3‑70B | `meta/llama-3.3-70b-instruct` | Feature‑level explanations |
| RiskGuard (LLM explanations) | Meta Llama‑3.3‑70B | `meta/llama-3.3-70b-instruct` | Risk‑decision narrative |
| AnalyticsEngine (LLM narration) | Meta Llama‑3.3‑70B | `meta/llama-3.3-70b-instruct` | Back‑test narration |
| Helix – Mistral Large (optional) | Mistral‑Large‑3‑675B‑Instruct‑2512 | `mistralai/mistral-large-3-675b-instruct-2512` | Developer assistant, code generation (optional) |

All model IDs are the exact strings accepted by `Helix.generate`. Model selection can be overridden at runtime via environment variables.

---

## Extensibility – Adding Futures Trading (Illustrative)

| Step | Engine | Change Required |
|------|--------|-----------------|
| Data Ingestion | StreamingBus | Add a **FuturesAdapter** (CME, Binance Futures, etc.) that publishes events with fields `symbol`, `bid`, `ask`, `last`, `openInterest`. |
| Signal Generation | SignalEngine | Register a **FuturesModel** (e.g., LSTM on roll‑adjusted prices) and extend the feature pipeline with `contango`, `openInterest`, `daysToExpiry`. |
| Risk Management | RiskEngine | Add a **MarginCalculator** plug‑in, an **ExpiryRule** (no new positions within 5 days of expiry), and update sector‑cap limits for commodity groups. |
| Execution | ExecutionEngine | Implement a **FuturesExecutionAdapter** that handles contract size, tick size, and daily settlement. Use an order‑book replay fill model for realistic simulation. |
| Portfolio Accounting | PortfolioEngine | Track daily mark‑to‑market P&L, lock required margin, and handle contract roll‑overs (auto‑close old contract, open next). |
| Analytics | AnalyticsEngine | Add futures‑specific metrics (leverage utilisation, contract‑level P&L, roll‑cost). Narrative templates now include “Futures contributed X % of total profit”. |
| Governance | GovernanceEngine | Maintain a **FuturesWhitelist** (allowed exchanges, commodity classes) and enforce regulatory position limits. |
| Orchestration | – | No code change; the orchestrator processes the new event types automatically. |

Because each engine respects a stable interface, adding a new asset class is a matter of plugging in adapters and updating configuration files—no architectural rewrite is required.

---

## Training & Continuous Learning – Workflow

1. **Data Collection** – `LearningEngine.record_event` ingests market features, signal outcomes, execution slippage, and LLM interaction logs.
2. **Dataset Versioning** – Snapshots stored as Parquet with accompanying metadata (model version, feature schema).
3. **Model Retraining**
   - **SignalEngine** – weekly or monthly retraining on a rolling two‑year window using XGBoost, LightGBM, or LSTM.
   - **Risk Aux‑Models** – optional volatility or transaction‑cost regressors.
   - **LLM Prompt Tuning** – aggregate `analyze_code` feedback to refine prompts or fine‑tune a smaller Nemotron‑8B on domain‑specific Q&A.
4. **Synapse Index Refresh** – nightly re‑embedding of new or updated documents; on‑push re‑index for CI changes.
5. **Benchmark Evaluation** – run the full benchmark suite (see Section *Historical Back‑Testing & Benchmarking*) for every new model. Acceptance thresholds must be met before promotion.
6. **Controlled Roll‑out** – deploy to shadow mode first (signals logged but not executed), compare live vs. production KPIs, then gradually ramp up via feature flags (`MODEL_VERSION`).
7. **Audit & Versioning** – every artifact receives a semantic version (`SignalModel_v2.3`, `CortexPrompt_v1.0`). Governance logs store `model_version` for each trade or LLM call, ensuring full traceability.

---

## Historical Back‑Testing & Benchmarking

| Benchmark Pack | Horizon | Asset Class | Notable Regime |
|----------------|---------|-------------|----------------|
| Pack‑01 | 3 months | US Equities (S&P 500 constituents) | Bull market (2021‑Q1) |
| Pack‑02 | 6 months | Futures (E‑mini, commodities) | High volatility (2022‑Q3) |
| Pack‑03 | 9 months | Mixed (equities + futures) | Sideways market (2020‑Q2) |
| Pack‑04 | 12 months | Global Equities (MSCI World) | Bear market (2008‑Q4) |
| Pack‑05 | 12 months | Crypto & alternative data | Crypto crash (2021) |

All packs are stored as Parquet files that conform to the StreamingBus schema and include a manifest JSON describing date range, symbols, and key events.

**Back‑test harness** (`run_backtest.py`):

```bash
python run_backtest.py \
    --packs pack_01 pack_04 \
    --initial-capital 1000000 \
    --config config/backtest.yaml
```

The harness replays events through the StreamingBus, executes the full pipeline, and persists `AnalyticsReport` and raw trade logs.

**Pre‑deployment acceptance criteria**

| KPI | Minimum Target |
|-----|----------------|
| Sharpe (average across packs) | ≥ 1.5 |
| Max Drawdown (any pack) | ≤ 20 % |
| Profit Factor | ≥ 1.8 |
| End‑to‑End Latency (p95) | ≤ 200 ms |
| Governance Pass Rate | 100 % (no policy violations) |
| Simulation Uptime | ≥ 99.9 % |

Only when **all** thresholds are satisfied may a new model be promoted to live trading.

---

## Key Performance Indicators (KPIs)

### System‑Performance KPIs

| KPI | Definition | Target |
|-----|------------|--------|
| End‑to‑End Latency (p50 / p95) | Tick → order‑fill time | p50 ≤ 100 ms, p95 ≤ 200 ms |
| Throughput | Events processed per second | ≥ 5 000 events s⁻¹ (peak) |
| Uptime | Percentage of market‑hour operation | ≥ 99.9 % |
| Critical Error Rate | Errors per 10 k events | ≤ 1 |
| Data Freshness | Source‑timestamp lag | ≤ 1 s |
| Schema‑Validation Pass | Percentage of events passing validation | 100 % |
| Audit Coverage | Percentage of actions logged | 100 % |
| Governance Pre‑flight Coverage | Percentage of trades checked | 100 % |
| GPU Utilisation (LLM / cuOpt) | Average utilisation during active periods | ≤ 80 % (headroom) |
| LLM Call Success | Successful / total calls | ≥ 99 % |
| LLM p95 Response | Time to receive completion | ≤ 5 s |
| Content‑Safety Incidents | Disallowed output occurrences | 0 |

### Trading‑Performance KPIs

| KPI | Formula / Note | Example Target |
|-----|----------------|----------------|
| CAGR | (Ending/Beginning)^(1/Years) – 1 | ≥ 20 % (back‑test) |
| Sharpe | (Mean – risk‑free) / σ | ≥ 2.0 |
| Sortino | (Mean – risk‑free) / σ_down | ≥ 3.0 |
| Calmar | CAGR / Max‑DD | ≥ 3.0 |
| Max Drawdown | Largest peak‑to‑trough loss | ≤ 15 % |
| Profit Factor | Gross profit / gross loss | ≥ 2.0 |
| Win Rate | Winning trades / total | 55 % – 60 % |
| Avg Win / Avg Loss | Mean win ÷ mean loss | ≥ 1.5 |
| Slippage | Avg. execution‑price deviation (bps) | ≤ 5 bps |
| Turnover (annual) | % of portfolio traded per year | ≤ 150 % |
| Transaction‑Cost Ratio | Cost / gross profit | ≤ 10 % |
| VaR (95 %) | One‑day 95 % value‑at‑risk | ≤ 2 % of equity |
| CVaR (95 %) | Expected shortfall beyond VaR | ≤ 3 % of equity |
| Leverage | Gross exposure / equity | ≤ 5 × |
| Herfindahl Index | Σ w_i² (diversification) | ≤ 0.15 |
| Alpha vs. Benchmark | Excess return over S&P 500 | Positive and statistically significant |
| Beta | Regression slope vs. market | Near 0 for market‑neutral strategies |
| Information Ratio | (Strategy – Benchmark) / tracking‑error | ≥ 0.5 |

KPIs are visualised on a real‑time dashboard (Grafana/Prometheus) and stored in a time‑series database for post‑mortem analysis.

---

## Governance & Auditing

Every critical action passes through `GovernanceEngine.preflight` and is recorded via `GovernanceEngine.audit`.

| Context | Example Policy | Action |
|---------|----------------|--------|
| Trade Execution | No trades on black‑listed securities; max sector exposure 5 % | Decision `allow=False`; order rejected; audit entry created |
| LLM Prompt | Strip PII, enforce model whitelist | Prompt sanitized; model ID validated |
| Data Publish | Tag PII‑containing events, restrict to EU region | Event dropped or re‑routed; alert emitted |
| Model Deployment | New model must pass benchmark suite before promotion | Automated gate in CI/CD pipeline |

All audit logs are immutable JSON‑lines stored in a secure object store (e.g., S3 with bucket policy). Each entry includes timestamp, engine, action, model version, decision, reasons, and policy version.

---

## Installation

### Requirements
- Python 3.11+
- Conda or virtualenv recommended

### Core Dependencies
```bash
# Install base package
pip install -e .

# Development tools
pip install -e ".[dev]"

# Production (live‑trading) dependencies
pip install -e ".[live-trading]"   # aiosqlite, plyer

# All optional dependencies
pip install -e ".[all]"
```

See `pyproject.toml` for the complete dependency list.

---

## Quick Start

### Run the Full System Demo
```bash
# Copy example environment file and add your API keys
cp .env.example .env

# Execute end‑to‑end demo
python scripts/demo_full_system.py
```
Expected output:
- Initialization of three market data sources (e.g., Alpha Vantage, Finnhub, Massive).
- Live market data fetched for a few symbols.
- Generation of trading signals, order creation, realistic fill simulation, and final P&L display.

### Technical Analysis (CLI)
```bash
python -m ordinis.interface.cli analyze --data data/AAPL_historical.csv
```

### Test Market Data APIs
```bash
python scripts/test_market_data_apis.py
```

### Launch Dashboard
```bash
streamlit run src/ordinis/interface/dashboard/app.py
```

---

## Documentation

- **Architecture Documentation** – `docs/architecture/PRODUCTION_ARCHITECTURE.md` (complete Phase 1 implementation details).
- **Architecture Review Response** – `docs/architecture/ARCHITECTURE_REVIEW_RESPONSE.md`.
- **Layered System Architecture** – `docs/architecture/LAYERED_SYSTEM_ARCHITECTURE.md`.
- **SignalCore System** – `docs/architecture/SIGNALCORE_SYSTEM.md`.
- **Position Sizing Logic** – `docs/POSITION_SIZING_LOGIC.md` (comprehensive guide to portfolio allocation and optimization).
  - **Quick Reference** – `docs/POSITION_SIZING_QUICK_REF.md` (examples and common patterns).
  - **Flow Diagrams** – `docs/diagrams/position_sizing_flow.md` (visual architecture and decision flows).
- **Knowledge Base** – `docs/knowledge‑base/` (trading research, strategy notes, incident post‑mortems).
- **Strategy Guides** – `docs/strategies/` (implementation templates and performance notes).
- **User Guides** – `docs/guides/` (CLI usage, dashboard operation, deployment).

All documentation follows the clean‑architecture principles and is kept up‑to‑date with the code base.

---

## Repository Structure

```
ordinis/
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── docs/
│   ├── architecture/
│   ├── decisions/
│   ├── knowledge-base/
│   ├── strategies/
│   └── guides/
├── src/ordinis/
│   ├── core/
│   ├── application/
│   ├── adapters/
│   │   ├── storage/
│   │   ├── market_data/
│   │   ├── alerting/
│   │   └── telemetry/
│   ├── engines/
│   │   ├── cortex/
│   │   ├── flowroute/
│   │   ├── proofbench/
│   │   ├── riskguard/
│   │   └── signalcore/
│   ├── interface/
│   │   ├── cli/
│   │   └── dashboard/
│   ├── runtime/
│   ├── safety/
│   ├── plugins/
│   ├── analysis/
│   ├── visualization/
│   └── rag/
├── tests/
├── data/            # .gitignore – runtime data, SQLite DB, logs
└── configs/
```

---

## Disclaimer

- Trading involves risk of loss; there are no guarantees of profit.
- Past performance in back‑tests does not assure future results.
- This system is a research and engineering project, not personalized financial advice.
- The authors and contributors are not licensed financial advisors.

---

## License

*To be determined.*

---
