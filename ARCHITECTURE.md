---
title: System Architecture
filename: ARCHTIECTURE.md
date: 2025-12-15
version: 1.1
type: system design specification (sdr)
description: >
Comprehensive architecture and design specification for a modular, AI-driven trading system focused on reliability, extensibility, and robust risk management.
---
**System Specification – Architecture (SDR) – Ordinis**
**Date:** 2025‑12‑14 **Version:** 1.1 **Type:** System Design Specification (SDR)
**Description:** A modular, AI‑driven trading platform built for reliability, extensibility, and robust risk‑management. All components are versioned, auditable, and governed by a central GovernanceEngine.

---

## 1. High‑Level Overview

The platform consists of a set of independent engines that communicate through a unified event bus. The main flow is:

1. Market and alternative data are ingested and published onto the **StreamingBus**.
2. The **SignalEngine** consumes the data and produces trading signals.
3. The **RiskEngine** evaluates each signal against risk policies and either approves, modifies, or rejects it.
4. Approved signals are turned into orders by the **ExecutionEngine**.
5. The **PortfolioEngine** updates positions, cash, and margin based on execution results.
6. The **AnalyticsEngine** calculates performance metrics and, when needed, generates narrative reports using the LLM stack.
7. The **LearningEngine** collects outcomes, retrains models, and updates the knowledge base.
8. Supporting services (**Cortex**, **Synapse**, **Helix**) provide reasoning, retrieval‑augmented generation, and model dispatch.
9. The **GovernanceEngine** enforces policy checks and records an immutable audit trail for every critical action.

All engines are loosely coupled; they can be scaled independently or replaced as long as they respect the defined interfaces.

---

## 2. Core Engines – Responsibilities and Interfaces

| Engine | Role | Key Responsibilities | Primary Interface(s) |
|--------|------|----------------------|----------------------|
| **OrchestrationEngine** | Coordinates the complete trading cycle (live or back‑test). | • Initiates cycles on new data or scheduled intervals.<br>• Calls each engine in the correct order.<br>• Propagates context (portfolio state, market snapshot).<br>• Emits tracing and timing events for analytics.<br>• Invokes GovernanceEngine pre‑flight checks before each sub‑step. | `run_cycle(event)`, `run_backtest(config)` |
| **StreamingBus** | Unified, schema‑validated event bus. | • Publishes events after schema validation and governance tagging.<br>• Subscribes handlers to topics and manages back‑pressure.<br>• Supports durable Kafka or low‑latency NATS/Redis Streams. | `publish(event)`, `subscribe(topic, handler)` |
| **SignalEngine** | Generates trading signals from enriched market data. | • Aggregates features (technical indicators, sentiment, etc.).<br>• Runs predictive models (GBM, XGBoost, LSTM, Transformer).<br>• Optionally ensembles multiple models.<br>• Performs basic sanity checks before emitting signals.<br>• Sends `Signal` events on the bus. | `generate_signals(data_frame) → List[Signal]` |
| **RiskEngine** | Enforces deterministic risk policies. | • Checks exposure, leverage, sector caps, stop‑loss, volatility, liquidity, and contract‑specific rules.<br>• Adjusts or rejects signals and returns reasons for audit.<br>• Logs every decision to GovernanceEngine. | `evaluate(signal, portfolio_state) → (bool, Signal, List[Reason])` |
| **ExecutionEngine** | Turns approved signals into orders and simulates/fills them. | • Constructs order objects (market, limit, IOC, etc.).<br>• Routes orders to paper‑trader, broker API, FIX, or internal matching engine.<br>• Applies fill‑model plug‑ins (immediate, order‑book replay, stochastic).<br>• Publishes `ExecutionReport` events.<br>• Performs governance double‑check before sending to venue. | `execute(order, market_state) → ExecutionReport` |
| **PortfolioEngine** | Maintains positions, cash, margin and rebalancing logic. | • Updates MTM P&L, cash balance, and margin requirements.<br>• Calculates target allocations and generates rebalancing orders.<br>• Enforces portfolio‑level constraints (turnover, cash reserve, sector caps).<br>• Provides snapshots to RiskEngine and SignalEngine. | `rebalance(target_allocations, constraints)`, `get_portfolio_state()` |
| **AnalyticsEngine** | Computes performance metrics and generates narratives. | • Calculates CAGR, Sharpe, Sortino, max drawdown, profit factor, etc.<br>• Analyzes trade‑level statistics (win rate, average win/loss, slippage).<br>• Builds human‑readable reports; optionally calls Cortex for narrative text.<br>• Emits `AnalyticsReport` events for dashboards. | `analyze(results_dataset) → Report` |
| **PortfolioOptEngine** | GPU‑accelerated portfolio optimisation (mean‑CVaR or mean‑variance). | • Generates return scenarios (historical bootstrap or Monte‑Carlo).<br>• Formulates optimisation problem with constraints (weights sum to 1, bounds, sector limits).<br>• Solves using NVIDIA cuOpt (GPU) with CPU fallback.<br>• Returns optimal weights and risk metrics. | `optimize(returns_data, constraints) → OptResult` |
| **Cortex** | LLM reasoning and code‑analysis assistant (does not drive live trading). | • `analyze_code(code, type)` – returns structured review (issues, complexity, suggestions).<br>• `synthesize_research(query, sources)` – returns a summary with citations.<br>• Uses Helix for model dispatch. | `analyze_code(...)`, `synthesize_research(...)` |
| **Synapse** | Retrieval‑augmented generation for Cortex and CodeGen. | • Indexes documentation, runbooks, code, and research papers.<br>• Embeds documents with a dedicated embedding model and stores vectors in FAISS/ElasticSearch.<br>• `retrieve(query, context)` – returns top‑k snippets with metadata. | `retrieve(query, context)` |
| **Helix** | Unified LLM provider layer. | • `generate(messages, model_id=None, **options)` – returns text, model identifier, usage stats.<br>• Handles authentication, retries, rate‑limits, and safety filtering.<br>• Selects default large model (Nemotron‑49B) or smaller fallback (Nemotron‑8B) based on configuration. | `generate(...)` |
| **GovernanceEngine** | Cross‑cutting policy enforcement and audit logging. | • `preflight(context)` – returns allow/deny decision with optional modifications.<br>• `audit(event)` – writes immutable JSON‑line log entry (timestamp, actor, decision, policy version).<br>• Generates real‑time alerts on violations. | `preflight(context)`, `audit(event)` |
| **LearningEngine** | Continuous improvement pipeline. | • `record_event(event)` – ingests data from all engines.<br>• Retrains signal models, fine‑tunes LLM prompts, re‑indexes Synapse.<br>• Runs benchmark suite before promotion.<br>• Supports controlled rollout (shadow mode → gradual ramp). | `record_event(event)`, `train(models)`, `evaluate(new_model, benchmark)` |
| **CodeGenService** | AI‑assisted code generation and patching. | • `propose_change(prompt, files_context)` – returns a code diff and optional test results.<br>• Uses Cortex for reasoning and Synapse for context.<br>• Governance filters for secrets, licensing, and unsafe patterns. | `propose_change(...)` |

---

## 3. Model Mapping – Consolidated Table

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

## 4. Extensibility – Adding Futures Trading (Illustrative)

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

## 5. Adapters, Plugins & Hooks – Implementation Summary

| Layer | Technology | Example Plugins |
|-------|------------|-----------------|
| StreamingBus | Kafka (durable) or NATS/Redis Streams (low‑latency) | `MarketDataAdapter`, `NewsAdapter`, `HistoricalLoader` |
| SignalEngine | Scikit‑learn, XGBoost, PyTorch, TensorFlow | `EquityGBM`, `FuturesLSTM`, `EnsembleVoting` |
| RiskEngine | YAML/JSON rule files + Python functions | `ExposureCap`, `MarginCheck`, `VaRPlugin` |
| ExecutionEngine | REST/WebSocket broker wrappers, FIX client, internal matching engine | `PaperTrader`, `BrokerAPIAdapter`, `OrderBookSimulator` |
| PortfolioEngine | Custom position ledger + optional `PnLCalculator` plug‑in | `MarginTracker`, `Rebalancer` |
| AnalyticsEngine | Pandas, NumPy, PyFolio/Alphalens, optional LLM narration | `MetricCalculator`, `NarrativePromptBuilder` |
| PortfolioOptEngine | NVIDIA cuOpt (GPU) – fallback to PuLP/CBC | `MeanCVaROptimizer`, `ScenarioGenerator` |
| Cortex / CodeGen | Helix → Nemotron, Synapse retrieval | `CodeReviewPrompt`, `ResearchPrompt` |
| Synapse | FAISS + NVIDIA EmbedLM‑300M, optional BM25 fallback | `EmbeddingIndexer`, `SimilaritySearcher` |
| GovernanceEngine | Simple Python rule functions or OPA‑style DSL | `TradePolicy`, `LLMContentFilter`, `AuditLogger` |
| LearningEngine | Airflow / Prefect orchestration, PyTorch‑Lightning for training | `SignalModelTrainer`, `LLMFineTuner`, `EmbeddingUpdater` |
| CodeGenService | Cortex + Synapse for context, Helix for generation | `ProposeChange`, `RunTests` |

All plugins register via a decorator‑based registry (e.g., `@register("signal_model")`) enabling dynamic discovery at runtime.

---

## 6. Training & Continuous Learning – Workflow

1. **Data Collection** – `LearningEngine.record_event` ingests market features, signal outcomes, execution slippage, and LLM interaction logs.
2. **Dataset Versioning** – Snapshots stored as Parquet with accompanying metadata (model version, feature schema).
3. **Model Retraining**
   * **SignalEngine** – weekly or monthly retraining on a rolling two‑year window using XGBoost, LightGBM, or LSTM.
   * **Risk Aux‑Models** – optional volatility or transaction‑cost regressors.
   * **LLM Prompt Tuning** – aggregate `analyze_code` feedback to refine prompts or fine‑tune a smaller Nemotron‑8B on domain‑specific Q&A.
4. **Synapse Index Refresh** – nightly re‑embedding of new or updated documents; on‑push re‑index for CI changes.
5. **Benchmark Evaluation** – run the full benchmark suite (see Section 7) for every new model. Acceptance thresholds must be met before promotion.
6. **Controlled Roll‑out** – deploy to shadow mode first (signals logged but not executed), compare live vs. production KPIs, then gradually ramp up via feature flags (`MODEL_VERSION`).
7. **Audit & Versioning** – every artifact receives a semantic version (`SignalModel_v2.3`, `CortexPrompt_v1.0`). Governance logs store `model_version` for each trade or LLM call, ensuring full traceability.

---

## 7. Historical Back‑Testing & Benchmarking

| Benchmark Pack | Horizon | Asset Class | Notable Regime |
|----------------|---------|-------------|----------------|
| Pack‑01 | 3 months | US Equities (S&P 500 constituents) | Bull market (2021‑Q1) |
| Pack‑02 | 6 months | Futures (E‑mini, commodities) | High volatility (2022‑Q3) |
| Pack‑03 | 9 months | Mixed (equities + futures) | Sideways market (2020‑Q2) |
| Pack‑04 | 12 months | Global Equities (MSCI World) | Bear market (2008‑Q4) |
| Pack‑05 | 12 months | Crypto & alternative data | Crypto crash (2021) |

All packs are stored as Parquet files that conform to the StreamingBus schema and include a manifest JSON describing date range, symbols, and key events.

**Back‑test Harness** (`run_backtest.py`):

Market / Alt‑Data → StreamingBus → SignalEngine → RiskEngine → ExecutionEngine → PortfolioEngine → AnalyticsEngine
                                   ↑                ↑                ↑                ↑
                                   │                │                │                │
                                   └─ GovernanceEngine (pre‑flight checks & audit) ──┘```bash
python run_backtest.py \
    --packs pack_01 pack_04 \
    --initial-capital 1000000 \
    --config config/backtest.yaml
```

The harness:

1. Replays events through StreamingBus in chronological order.
2. Executes the full pipeline (Orchestration → Signal → Risk → Execution → Portfolio → Analytics).
3. Persists `AnalyticsReport` and raw trade logs for each run.

**Pre‑deployment Acceptance Criteria**

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

## 8. Key Performance Indicators (KPIs)

### 8.1 System‑Performance KPIs

| KPI | Definition | Target |
|-----|------------|--------|
| End‑to‑End Latency (p50 / p95) | Tick to order‑fill time | p50 ≤ 100 ms, p95 ≤ 200 ms |
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

### 8.2 Trading‑Performance KPIs

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

KPIs are visualised on a real‑time operations dashboard (Grafana/Prometheus) and stored in a time‑series database for post‑mortem analysis.

---

## 9. Governance & Auditing

Every critical action passes through `GovernanceEngine.preflight` and is recorded via `GovernanceEngine.audit`.

| Context | Example Policy | Action |
|---------|----------------|--------|
| Trade Execution | No trades on black‑listed securities; max sector exposure 5 % | Decision `allow=False`; order rejected; audit entry created |
| LLM Prompt | Strip PII, enforce model whitelist | Prompt sanitized; model ID validated |
| Data Publish | Tag PII‑containing events, restrict to EU region | Event dropped or re‑routed; alert emitted |
| Model Deployment | New model must pass benchmark suite before promotion | Automated gate in CI/CD pipeline |

All audit logs are immutable JSON‑lines stored in a secure object store (e.g., S3 with bucket policy). Each entry includes timestamp, engine, action, model version, decision, reasons, and policy version.

---

## 10. Strategic Model Selection & Optimization

### 10.1 Engine-Specific Model Strategy

| Engine | Model Strategy | Rationale | Optimization Path |
|--------|----------------|-----------|-------------------|
| **Cortex** | Nemotron‑49B (Fallback: 8B) | High capacity needed for deep code analysis and reasoning. | **Quantisation (INT4)** to reduce memory; **Fine-tuned 13B** models for specific linting tasks. |
| **Helix** | Facade over 49B/8B | Centralized governance and rate-limiting. | **Model Aliasing** to support external providers (OpenAI, Anthropic) for fallback or comparison. |
| **Analytics** | Nemotron‑8B | Sufficient for converting structured metrics to narrative. | **Fine-tuned Llama-3-Finance** for better domain terminology. |
| **SignalEngine** | Traditional ML (XGBoost/LSTM) | Superior speed and interpretability for tabular data. | **Tabular Transformers** for potential accuracy gains; Small LLMs for news sentiment features. |
| **RiskEngine** | Deterministic Rules | Compliance requires absolute determinism. | **Lightweight Volatility Models** (GARCH/LSTM) as auxiliary inputs. |
| **Synapse** | EmbedLM-300M | Low latency and storage cost. | **1B Parameter Models** or **Multilingual Embeddings** if knowledge base scales significantly. |
| **CodeGen** | Nemotron‑49B | Complex synthesis requires max parameter count. | **"Simple Mode"** using 7B models for trivial tasks (docstrings, boilerplate). |

### 10.2 Cost & Performance Optimization

*   **GPU Memory Pressure**: Deploy **8-bit/4-bit quantised** checkpoints (e.g., `bitsandbytes` INT4) to reduce memory usage by ~2-3x while retaining ~90% of quality. Use model-sharding only for high-priority full-precision requests.
*   **High-Frequency Latency**: Maintain the strict separation of LLMs from the critical trading path (Signal → Risk → Execution). LLMs should only be invoked for asynchronous post-trade analysis.
*   **Inference Cost**: Shift bulk workloads (e.g., nightly code reviews) to batch windows using quantised models. Enable **autoscaling** on Helix to dynamically provision 49B instances only when queue depth demands it.

### 10.3 Future-Proofing Recommendations

1.  **Model-Agnostic Configuration**: Expose a `helix_models.yaml` to map logical aliases (`default`, `fast`, `safety`) to concrete model IDs, facilitating easy swaps (e.g., to Llama-3.3-70B or Claude-3.5).
2.  **Hybrid Retrieval**: Combine semantic search (FAISS) with BM25 keyword search to improve recall for specific entity names and financial terms.
3.  **Domain-Specific Fine-Tuning**: Allocate resources to fine-tune 8B/13B models on internal documentation and reports, potentially replacing larger models for specific tasks.
4.  **Continuous Safety Updates**: Schedule quarterly retraining of the Nemoguard safety model to address evolving compliance requirements without disrupting the core model stack.

---

## 11. References

1. **Metrics for Trustworthy AI – OECD.AI**
   <https://oecd.ai/en/catalogue/metrics>

2. **Accelerating Real‑Time Financial Decisions with Quantitative Portfolio Optimization** – NVIDIA Technical Blog
   <https://developer.nvidia.com/blog/accelerating-real-time-financial-decisions-with-quantitative-portfolio-optimization/>

3. **NVIDIA Build Platform – API & Model Catalog**
   <https://build.nvidia.com>

---

**Summary**

The refined architecture now:

* Explicitly maps every engine to its concrete model(s).
* Includes all missing components (OrchestrationEngine, StreamingBus, GovernanceEngine, LearningEngine, CodeGenService).
* Provides a clear extensibility path (futures example).
* Standardises adapters and plug‑in mechanisms for future growth.
* Aligns KPI definitions with both system‑level and trading‑level health checks.
* Integrates a mandatory benchmark suite as a gate before any production change.
* **Incorporates a strategic roadmap for model optimization and future-proofing.**

With these enhancements the system is ready for implementation, continuous improvement, and regulated deployment.
