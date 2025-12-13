Architecture Review: what’s solid vs. what’s missing
----------------------------------------------------

You’ve got a sensible high-level decomposition (presentation/app/engine/data/external) for a modular, event-driven research/backtest/paper-trade stack . The problem is: **the architecture is currently a set of named boxes without the contracts and control-plane that make those boxes behave predictably at scale**.

Below is the technical feedback I’d give if this were heading toward “institutional-grade” reliability.

* * *

1) The biggest architectural gap: “event-driven” is claimed, but the event system is undefined

----------------------------------------------------------------------------------------------

You state “modular, event-driven” and “event-driven backtesting” , but there’s no explicit:

* **Event bus contract** (types, schemas, ordering guarantees, replayability)

* **Event taxonomy** (MarketDataEvent, SignalEvent, OrderEvent, FillEvent, RiskEvent, MetricEvent, etc.)

* **Delivery semantics** (at-most-once vs at-least-once, idempotency strategy)

* **Time model** (wall-clock vs simulated clock; how time moves in backtest vs live)

* **Failure containment** (what happens if a strategy crashes mid-bar?)

### What to do

Define a “core event protocol” as a first-class artifact:

* `Event` = `{id, ts_event, ts_ingest, type, source, payload, correlation_id, causation_id, version}`

* Strongly typed payloads (dataclasses/Pydantic)

* A single event loop interface that supports:

  * **Simulated time** for ProofBench

  * **Real time** for paper/live trading

This is how you make “event-driven” real instead of aspirational.

* * *

2) Boundary / layer confusion: you’ve got duplicate concepts and unclear ownership

----------------------------------------------------------------------------------

Your diagram shows “Dashboard” in the **Presentation Layer** and also `dashboard.py` in the **Visualization Layer** . Monitoring and visualization appear both as app-level boxes and as dedicated modules .

That’s survivable early, but it becomes tech debt fast because no one knows where a concern “belongs,” and you get sideways dependencies.

### What to do

Pick a sharper architecture style (this repo is a good candidate for “Clean Architecture” / ports-and-adapters):

* **Domain/Core** (no I/O): instruments, bars, orders, fills, portfolio, risk state, event definitions

* **Application/Use-cases**: “run_backtest”, “run_paper_session”, “rebalance_portfolio”

* **Engines**: SignalCore, RiskGuard, ProofBench become use-case services wired behind interfaces

* **Infrastructure/Adapters**: market data providers, broker adapters, storage, dashboards

* **Interfaces**: CLI / REST / UI call into application layer only

Net: fewer circular dependencies, cleaner test surfaces, easier refactors.

* * *

3) Contract mismatch: async market data providers vs synchronous strategies

---------------------------------------------------------------------------

Your `MarketDataProvider` is **async** , but `Strategy.generate_signals()` is **sync** and takes a `pd.DataFrame` .

That creates ambiguity:

* Is the system fundamentally async (live) with sync islands?

* Or sync (backtest) with async adapters?

Either choice is fine, but it needs to be explicit or you’ll end up with deadlocks, blocking calls, and inconsistent performance.

### What to do

Make one of these decisions and enforce it:

**Option A (recommended): “Async at the edges, sync in the core.”**

* Data providers are async; they feed a queue/event bus

* Core engines run sync off an internal loop (or a controlled async loop)

* Strategies remain sync but operate on **typed bar objects** not raw DataFrames

**Option B: “Fully async pipeline.”**

* `generate_signals()` becomes async

* Everything is awaitable end-to-end

* You enforce strict non-blocking behavior

Either way: define it as a non-negotiable system invariant.

* * *

4) Backtest/live parity is broken by design right now

-----------------------------------------------------

Your backtesting flow does not show risk gating, but your paper flow does . That means:

* Backtest results will be systematically optimistic vs paper/live

* You can’t trust your “research → live” transfer

### What to do

Force parity:

* RiskGuard must run **in both** ProofBench and live flows.

* Same order sizing logic, same constraints, same limits.

* Same transaction cost model (slippage/fees/partials already listed for ProofBench , but apply consistently).

If you don’t do this, you’ll spend months chasing “why didn’t it behave live like it did in backtest?”

* * *

5) Missing core component: an OMS / Execution abstraction

---------------------------------------------------------

Your flow is “Signals → Strategy → Orders → fills” . What’s missing is the layer that makes orders real:

* **Order Management System (OMS)**: lifecycle, cancels/replaces, state transitions

* **Execution model**: order routing, fills, partials, rejections

* **Broker adapter interface**: Alpaca is one broker now, but the system should not be Alpaca-shaped

### What to do

Define explicit interfaces:

* `BrokerAdapter` (paper/live): `submit(order)`, `cancel(order_id)`, `replace(order_id, order)`, `stream_fills()`

* `ExecutionEngine`: maps signals to executable orders (incl. throttles, batching, smart sizing)

* `OMS`: authoritative state machine for order lifecycle

This also becomes the natural place for:

* idempotency keys

* retry logic

* reconciliation loops (broker vs internal state)

* * *

6) “Type safety” claim is undermined by heavy DataFrame boundaries

------------------------------------------------------------------

You claim “Type Safety: comprehensive type hints and runtime validation” , but your key data pipes are `pd.DataFrame` .

A DataFrame is not type-safe. It’s dynamically typed, column-name fragile, and silently coerces types.

### What to do

Use typed domain objects at boundaries:

* `Quote`, `Bar`, `Bars`, `OptionChain`, `Fill`, `Order`, `Position`, `PortfolioSnapshot`

* Keep DataFrames as an internal implementation detail for feature engineering, not as your system contract.

If you want runtime validation, enforce schemas at ingress:

* Validate provider output (columns, dtypes, missingness thresholds, timezone handling) in your data validators .

* Convert to canonical typed objects immediately.

* * *

7) Data layer: multi-provider reality needs “provenance + reconciliation”, not just connectors

----------------------------------------------------------------------------------------------

You list multiple external providers and a unified interface . In practice, multi-provider introduces structural problems:

* timestamp drift / timezone differences

* adjusted vs unadjusted prices

* corporate actions disagreements

* gaps / rate limits

* vendor-specific symbol mapping

### What to do

Add two capabilities that usually separate “toy system” from “real system”:

1. **Provenance**
* Every datum gets `source`, `ingested_at`, `vendor_ts`, `quality_flags`

* Store this alongside bars/quotes (not just in logs)
2. **Reconciliation and fallback**
* Define a priority list by field (e.g., Polygon for intraday bars, Finnhub for fundamentals, etc.)

* Implement “quorum” or “compare and flag” rules for critical fields

Also: your “compressed CSV” storage target is a scalability ceiling. Use columnar formats (Parquet/Arrow) and/or DuckDB for research/backtests as soon as you care about speed + reproducibility.

* * *

8) RiskGuard: needs separation between “risk analytics” and “risk controls”

---------------------------------------------------------------------------

RiskGuard lists VaR, Sharpe/Sortino, drawdown monitoring, limits . Sharpe/Sortino are performance metrics; they’re not control-plane risk checks. This is a conceptual mix.

### What to do

Split into two parts:

* **Risk Analytics** (reporting):

  * VaR / CVaR, drawdown, volatility, beta, factor exposures, option Greeks exposure

* **Risk Controls** (hard gates):

  * max gross/net exposure

  * max position size per symbol

  * max leverage / margin utilization

  * max daily loss / max drawdown

  * max order rate, max notional per order

  * kill switch

In both backtest and paper/live.

* * *

9) ProofBench: “realistic fill simulation” needs explicit assumptions and a cost model contract

-----------------------------------------------------------------------------------------------

You mention slippage, partial fills, multiple timeframes . Great — but the key is: **what market microstructure assumption are you simulating?**

Without explicit fill rules, the engine becomes a black box and results won’t be defendable.

### What to do

Define fill model types (pluggable):

* **Bar-based** (fast): fill at next open, mid, VWAP, etc.

* **Spread-based**: apply bid/ask spread + slippage model

* **Volume participation**: cap fills by bar volume * participation_rate

* **Queue simulation** (advanced): if you get L2 later

Also define a transaction cost model contract:

* commissions, fees, borrow, dividends, assignment/exercise for options

* * *

10) CortexRAG / LLMEnhancedModel: keep the LLM out of the control-plane unless you can audit it

-----------------------------------------------------------------------------------------------

You have `CortexRAG` for trading knowledge and `LLMEnhancedModel` in SignalCore . That’s fine for research, but it’s a major operational risk if it’s in the critical path of order placement.

### What to do

Treat LLM features as **advisory** by default:

* LLM generates _explanations_, _hypotheses_, _feature suggestions_

* A deterministic model generates the signal that can place orders

If you do allow LLM-in-the-loop for signals:

* You need **prompt/version pinning**, **output schema validation**, **latency budgets**, **fallback behavior**, and **full audit logs** (inputs + retrieved context hashes + outputs).

* Reproducibility matters: backtests must be replayable. That’s hard if the model is nondeterministic.

* * *

11) Configuration management: needs a single validated config object + snapshotting

-----------------------------------------------------------------------------------

You list env vars, YAML, pyproject, runtime overrides . Good list, but in practice this becomes “config spaghetti” unless you force:

* precedence rules

* schema validation

* immutable run configuration for backtest reproducibility

### What to do

* Load all sources into **one canonical config object** (typed + validated).

* On every run, write a config snapshot into the output artifacts (reports/ results folder), so you can reproduce “exactly what ran.”

* * *

12) Observability: “Monitoring layer” exists, but you need a unified telemetry model

------------------------------------------------------------------------------------

Monitoring is described as KPIs/metrics/alerts (alerts future) . For trading systems, observability isn’t optional.

### What to do

Define a unified telemetry contract:

* **Metrics**: strategy PnL, drawdown, exposure, event loop lag, data latency, order rejection rates

* **Structured logs**: every decision with correlation IDs

* **Traces**: “market tick → signal → risk → order → fill”

This is how you debug live behavior without guessing.

* * *

Concrete corrections to make to the architecture doc itself
===========================================================

If you want the doc to be engineering-actionable (not just conceptual), add these sections:

1. **Core domain model**: Order, Fill, Position, Instrument, Bar, Signal (schemas)

2. **Event model**: event types + sequence diagrams

3. **Time model**: simulated clock vs wall clock

4. **Execution/OMS**: lifecycle states + broker reconciliation

5. **Backtest/live parity rules**: explicit guarantees

6. **Determinism policy**: seeds, data snapshots, model versioning, LLM policy

Right now, those omissions are the main reason the architecture can drift during implementation.

* * *

Priority roadmap (what I would do next)
=======================================

### P0 — “Make it trustworthy”

* Define event bus + event types (system backbone)

* Add OMS + BrokerAdapter abstraction

* Enforce backtest/live parity including RiskGuard

* Canonical typed domain objects (stop using DataFrame as a contract)

### P1 — “Make it scalable”

* Replace compressed CSV storage with columnar storage + query layer

* Add provenance + reconciliation for multi-provider data

* Standardize transaction cost + fill model interfaces

### P2 — “Make it extensible and safe”

* Formal plugin contracts and versioning (strategies/models/providers)

* LLM/RAG guardrails + auditability

* Full observability: metrics/logs/traces

* * *

If you want, I can also propose a **clean interface spec** (Python protocols) for:

* `EventBus`

* `BrokerAdapter`

* `ExecutionEngine`

* `FillModel`

* `CostModel`

* `RiskPolicy`


  …and map each existing named box (SignalCore, ProofBench, RiskGuard, OptionsCore, CortexRAG ) to concrete “ports” so it’s obvious what talks to what and under what contract.
