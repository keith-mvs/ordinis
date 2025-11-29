# Model Integration Patterns

> **How LLMs (Cortex) orchestrate numerical AI/ML engines without performing calculations**

---

## Overview

The Intelligent Investor system uses a **hybrid architecture** where:
- **Cortex (LLM)** orchestrates, documents, and explains
- **Numerical Engines** (SignalCore, RiskGuard, ProofBench) perform all calculations
- **Strict separation** prevents LLM hallucination in trading decisions

This document defines **integration patterns** that ensure the LLM stays in an orchestration/documentation role while underlying engines perform calculations and signal generation.

---

## Core Principle: Orchestration vs Calculation

| Component | Role | Does | Does NOT |
|-----------|------|------|----------|
| **Cortex (LLM)** | Orchestration & Documentation | - Parse user intent<br>- Generate strategy specs<br>- Explain results<br>- Recommend publications<br>- Natural language I/O | - Calculate signals<br>- Compute metrics<br>- Make trading decisions<br>- Invent data<br>- Override risk rules |
| **SignalCore** | Signal Generation | - Train ML models<br>- Generate predictions<br>- Compute features<br>- Calculate indicators | - Interpret signals<br>- Make risk decisions<br>- Execute orders |
| **RiskGuard** | Risk Enforcement | - Evaluate rules<br>- Enforce limits<br>- Validate positions<br>- Kill switches | - Generate signals<br>- Modify strategies<br>- Execute orders |
| **ProofBench** | Validation | - Run backtests<br>- Compute metrics<br>- Validate strategies<br>- Simulate scenarios | - Modify strategies<br>- Make decisions<br>- Generate signals |
| **FlowRoute** | Execution | - Submit orders<br>- Track fills<br>- Manage state<br>- Route orders | - Generate signals<br>- Override risk<br>- Modify prices |

---

## Pattern 1: Strategy Research Workflow

**User Request:** "Research AAPL and suggest a trading strategy"

### Dataflow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│ USER INPUT                                                    │
│ "Research AAPL and suggest a trading strategy"               │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ CORTEX (LLM) - Intent Parsing                                │
│                                                                │
│ Extracts:                                                     │
│ - Ticker: AAPL                                                │
│ - Action: Research + Strategy Suggestion                      │
│ - Data Needed: Price, News, Fundamentals                      │
│                                                                │
│ Constructs Call Sequence:                                    │
│  1. /research-ticker AAPL                                     │
│  2. /kb-search "AAPL sector strategies"                       │
│  3. /design-strategy [from findings]                          │
└────────────────────┬──────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│ DATA PLUGIN      │    │ KNOWLEDGE ENGINE │
│ (Polygon/IEX)    │    │                  │
│                  │    │ Semantic Search: │
│ get_quote(AAPL)  │    │ "tech momentum"  │
│ → {price, vol}   │    │ → Publications   │
│                  │    │                  │
│ get_news(AAPL)   │    │                  │
│ → [{...}]        │    │                  │
└─────────┬────────┘    └─────────┬────────┘
          │                       │
          └───────────┬───────────┘
                      ▼
┌──────────────────────────────────────────────────────────────┐
│ CORTEX - Synthesis (NO CALCULATIONS)                         │
│                                                                │
│ Synthesizes from DATA (not invented):                        │
│ - Price: $150.00 (from Polygon)                              │
│ - Trend: Uptrend (from data, not LLM opinion)                │
│ - Sentiment: Positive (from news plugin)                     │
│ - Sector: Tech momentum (from KB publications)               │
│                                                                │
│ Generates Strategy Spec (YAML):                              │
│   type: trend_following                                      │
│   entry: MA crossover + volume                               │
│   exit: bearish crossover OR 5% stop                         │
│                                                                │
│ References Publications:                                     │
│ - Kirkpatrick (Domain 2: Technical Analysis)                 │
│ - Grant (Domain 7: Risk Management)                          │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ VALIDATION                                                    │
│                                                                │
│ a) Schema Validation:                                        │
│    jsonschema.validate(spec, strategy.schema.json) ✓         │
│                                                                │
│ b) RiskGuard Pre-Validation:                                 │
│    RiskGuard.pre_validate_strategy_spec(spec)                │
│    - Check: position limits defined ✓                         │
│    - Check: stop loss defined ✓                               │
│    - Check: max concurrent positions ✓                        │
│    Result: VALID                                             │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ OUTPUT (Cortex → User)                                       │
│                                                                │
│ - Strategy Spec (YAML)                                       │
│ - Rationale (cites publications, not LLM opinion)            │
│ - Risk Considerations (from RiskGuard)                       │
│ - Next Steps: "Run /analyze-backtest to validate"           │
└──────────────────────────────────────────────────────────────┘
```

### Guardrails

1. **No Hallucinated Data**
   - ✅ Use: `price = polygon.get_quote("AAPL")["price"]`
   - ❌ Avoid: LLM generating "AAPL is trading at ~$150" without API call

2. **No Direct Calculations**
   - ✅ Use: `sharpe = proofbench.calculate_sharpe(returns)`
   - ❌ Avoid: LLM computing "Sharpe ratio is approximately 1.2"

3. **Source Attribution**
   - ✅ "Based on Kirkpatrick Ch 12, MA crossover is..."
   - ❌ "I think MA crossover works because..."

4. **Validation Required**
   - All specs validated against JSON Schema
   - RiskGuard pre-validates before user sees output

---

## Pattern 2: Backtest Execution Workflow

**User Request:** "Backtest the MA crossover strategy on AAPL"

### Dataflow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│ CORTEX - Parse & Plan                                        │
│                                                                │
│ Identifies:                                                   │
│ - Strategy: ma_crossover (from registry or spec file)        │
│ - Symbol: AAPL                                                │
│ - Task: Backtest                                             │
│                                                                │
│ Plan:                                                         │
│ a) Load strategy spec                                        │
│ b) Fetch historical data                                      │
│ c) Execute backtest (ProofBench)                              │
│ d) Generate report                                           │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ DATA ACQUISITION (Polygon)                                   │
│                                                                │
│ polygon.get_historical(                                      │
│   symbol="AAPL",                                              │
│   timeframe="1D",                                             │
│   start="2020-01-01",                                         │
│   end="2024-12-31"                                            │
│ )                                                             │
│                                                                │
│ Returns: DataFrame[date, OHLCV]                              │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ SIGNALCORE - Generate Signals (NO LLM INVOLVEMENT)           │
│                                                                │
│ model = MAcrossoverModel(fast=50, slow=200)                  │
│                                                                │
│ for bar in historical_data:                                  │
│   signal = model.generate_signal(bar)                        │
│   # Signal(direction=LONG, probability=0.72, score=0.65)     │
│                                                                │
│ Output: List[Signal] (201 signals)                           │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ RISKGUARD - Validate Signals (NO LLM INVOLVEMENT)            │
│                                                                │
│ for signal in signals:                                       │
│   result = RiskGuard.evaluate_signal(                        │
│     signal, portfolio_state, market_state                    │
│   )                                                           │
│                                                                │
│   if result.passed:                                          │
│     approved_signals.append(signal)                          │
│   else:                                                       │
│     rejected.append((signal, result.reason))                 │
│                                                                │
│ Output:                                                       │
│ - Approved: 24 signals (75%)                                 │
│ - Rejected: 8 signals (25%)                                  │
│   └─ Reasons: 5x low score, 3x position limit                │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ PROOFBENCH - Execute Backtest (NO LLM INVOLVEMENT)           │
│                                                                │
│ backtester = SimpleBacktester(                               │
│   initial_capital=100000,                                    │
│   commission=0.005,                                           │
│   slippage_bps=5.0                                           │
│ )                                                             │
│                                                                │
│ for signal in approved_signals:                              │
│   # Event-driven simulation                                  │
│   - Calculate position size (RiskGuard formula)              │
│   - Apply transaction costs                                  │
│   - Update portfolio P&L                                     │
│   - Track drawdown                                           │
│                                                                │
│ result = backtester.finalize()                               │
│ # Computes: Sharpe, Sortino, drawdown, win rate, etc.       │
│                                                                │
│ Output: BacktestResult {                                     │
│   total_return: 0.452,                                       │
│   sharpe_ratio: 1.23,                                        │
│   max_drawdown: -0.125,                                      │
│   trades: 24,                                                │
│   win_rate: 0.583,                                           │
│   profit_factor: 1.85                                        │
│ }                                                             │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ CORTEX - Generate Report (NO RECALCULATION)                  │
│                                                                │
│ Receives:                                                     │
│ - BacktestResult (from ProofBench) ← SINGLE SOURCE OF TRUTH  │
│ - RejectionLog (from RiskGuard)                              │
│                                                                │
│ Generates Natural Language Report:                           │
│                                                                │
│ "Backtest Result: AAPL MA Crossover (50/200)                 │
│                                                                │
│  Performance (from ProofBench):                              │
│  - Total Return: +45.2% [direct from result.total_return]    │
│  - Sharpe Ratio: 1.23 [direct from result.sharpe_ratio]      │
│  - Max Drawdown: -12.5% [direct from result.max_drawdown]    │
│                                                                │
│  Risk Validation (from RiskGuard):                           │
│  - Signals Approved: 24/32 (75%)                             │
│  - Rejections: 8 (5x low score, 3x position limit)           │
│                                                                │
│  Assessment (from López de Prado 2018):                      │
│  Sharpe 1.23 is solid for single-stock trend following.     │
│  However:                                                    │
│  - Sample size modest (24 trades)                            │
│  - Recommend: Extend to 10+ years                            │
│  - Apply Deflated Sharpe adjustment (Ch 14)                  │
│                                                                │
│  Next: /stress-test to evaluate regime sensitivity"          │
│                                                                │
│ Does NOT:                                                     │
│ - Recalculate any metrics                                   │
│ - Invent numbers                                             │
│ - Override ProofBench results                                │
└──────────────────────────────────────────────────────────────┘
```

### Guardrails

1. **Single Source of Truth**
   - All metrics come from ProofBench
   - No "approximately" or "around" - exact values only
   - LLM formats and explains, never calculates

2. **Publication-Backed Interpretation**
   - ✅ "According to López de Prado (2018, Ch 14)..."
   - ❌ "In my opinion, this Sharpe ratio is..."

3. **Complete Audit Trail**
   - Log every engine call
   - Track all data sources
   - Record all decisions

---

## Pattern 3: Live Signal Validation Workflow

**Scenario:** Production signal generation and order submission

### Dataflow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│ TRIGGER (NO LLM)                                             │
│ - Scheduled (e.g., market open, bar close)                   │
│ - Event-driven (e.g., price alert)                           │
│ - User command                                               │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ SIGNALCORE - Generate Signal (DETERMINISTIC)                 │
│                                                                │
│ strategy = load_strategy("ma_crossover_v1")                  │
│ market_data = polygon.get_quote("SPY")                       │
│                                                                │
│ signal = strategy.generate_signal(                           │
│   symbol="SPY",                                               │
│   market_data=market_data                                    │
│ )                                                             │
│                                                                │
│ Output: Signal(                                              │
│   symbol="SPY",                                               │
│   signal_type=ENTRY,                                         │
│   direction=LONG,                                            │
│   probability=0.72,                                          │
│   score=0.65,                                                │
│   reference_price=450.00,                                    │
│   stop_price=445.00                                          │
│ )                                                             │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ RISKGUARD - Multi-Layer Validation (DETERMINISTIC)           │
│                                                                │
│ Layer 1: PRE_TRADE Rules                                     │
│ ├─ RT001: Position size ≤ 10% equity         ✓               │
│ ├─ RT002: Risk per trade ≤ 1% equity         ✓               │
│ ├─ RT004: Signal score ≥ 0.3                 ✓               │
│ └─ Result: PASS                                              │
│                                                                │
│ Layer 2: PORTFOLIO_LIMIT Rules                               │
│ ├─ RP001: Max leverage ≤ 2.0x                ✓               │
│ ├─ RP002: Concurrent positions ≤ 10          ✓               │
│ └─ Result: PASS                                              │
│                                                                │
│ Layer 3: KILL_SWITCH Rules                                   │
│ ├─ RK001: Daily loss < -3%                   ✓               │
│ ├─ RK002: Drawdown < -15%                    ✓               │
│ └─ Result: PASS                                              │
│                                                                │
│ Layer 4: SANITY_CHECK Rules                                  │
│ ├─ RS002: Market hours                       ✓               │
│ ├─ RS003: Broker connected                   ✓               │
│ └─ Result: PASS                                              │
│                                                                │
│ FINAL RESULT: APPROVED                                       │
│ Audit Log: [timestamp, signal_id, all evaluations]          │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ FLOWROUTE - Order Construction (DETERMINISTIC)               │
│                                                                │
│ position_size = RiskGuard.calculate_position_size(           │
│   entry=450.00, stop=445.00,                                 │
│   account_equity=100000, risk_pct=1.0                        │
│ )                                                             │
│ # Returns: 200 shares                                        │
│                                                                │
│ order = Order(                                               │
│   symbol="SPY", side=BUY, quantity=200,                      │
│   order_type=LIMIT, limit_price=450.05,                      │
│   time_in_force=DAY, stop_loss=445.00                        │
│ )                                                             │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ RISKGUARD - Final ORDER_VALIDATION (DETERMINISTIC)           │
│                                                                │
│ Validates:                                                    │
│ ├─ Order params match signal                 ✓               │
│ ├─ Position size within limits               ✓               │
│ ├─ Stop loss properly set                    ✓               │
│ └─ No manipulation risk                      ✓               │
│                                                                │
│ Result: APPROVED ✓                                           │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ FLOWROUTE - Submit Order (NO LLM)                            │
│                                                                │
│ broker_plugin.submit_order(order)                            │
│                                                                │
│ Response: Order(order_id="ABC123", status=PENDING_NEW)       │
└────────────────────┬──────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ CORTEX - Optional Notification (AFTER EXECUTION)             │
│                                                                │
│ IF notifications_enabled:                                    │
│                                                                │
│   "✓ Signal Executed: SPY LONG                               │
│    Entry: $450.00 | Stop: $445.00                            │
│    Position: 200 shares ($90,000)                            │
│    Risk: $1,000 (1.0% of equity)                             │
│    Rationale: MA crossover (score 0.65)                      │
│    Order ID: ABC123"                                         │
│                                                                │
│ Does NOT:                                                     │
│ - Modify order                                               │
│ - Override risk decisions                                    │
│ - Participate in execution path                              │
└──────────────────────────────────────────────────────────────┘
```

### Critical Guardrails

1. **LLM Not in Critical Path**
   - Order execution is fully deterministic
   - LLM only for post-hoc notifications
   - LLM failure doesn't stop trading

2. **Deterministic Validation**
   - RiskGuard uses mathematical rules, not LLM judgment
   - No "I think this trade looks good"
   - All rules are testable and auditable

3. **Complete Audit Trail**
   - Every signal logged
   - Every rule evaluation recorded
   - Every order tracked
   - Full replay capability

4. **Fail-Safe Defaults**
   - Any error → reject signal
   - Uncertain data → skip trade
   - Communication loss → halt trading

5. **Kill Switch Activation**
   - Automatic on daily loss > 3%
   - Automatic on drawdown > 15%
   - No LLM override possible

---

## Integration Checklist

For every Cortex ↔ Engine integration:

### ✅ Data Flow
- [ ] LLM receives data from engine (not invents)
- [ ] All calculations delegated to engines
- [ ] Results pass through without modification

### ✅ Validation
- [ ] Schema validation on all specs
- [ ] Engine pre-validates before LLM sees results
- [ ] User-facing outputs cite sources

### ✅ Auditability
- [ ] All engine calls logged
- [ ] Input/output recorded
- [ ] Execution time tracked
- [ ] Errors captured

### ✅ Error Handling
- [ ] Engine errors don't crash LLM
- [ ] LLM errors don't affect engine execution
- [ ] Graceful degradation defined

### ✅ Testing
- [ ] Unit tests for each engine call
- [ ] Integration tests for full workflows
- [ ] Mocking for engine responses
- [ ] Performance benchmarks

---

## Anti-Patterns (DO NOT DO)

### ❌ LLM Performing Calculations

**Wrong:**
```python
# Cortex generating code
def backtest_strategy():
    sharpe = returns.mean() / returns.std()  # LLM doing math
    return sharpe
```

**Right:**
```python
# Cortex delegates to ProofBench
sharpe = proofbench.calculate_sharpe(returns)
```

---

### ❌ LLM Inventing Data

**Wrong:**
```
"AAPL is currently trading around $150 with a PE of about 28..."
```

**Right:**
```python
quote = polygon.get_quote("AAPL")
fundamentals = fmp.get_metrics("AAPL")

f"AAPL is trading at ${quote['price']} with a PE of {fundamentals['pe']}"
```

---

### ❌ LLM Overriding Risk Rules

**Wrong:**
```
"While the risk rule says max 1%, I think 2% is okay here..."
```

**Right:**
```
"RiskGuard rejected this signal (rule RT002: risk exceeds 1% limit).
 To proceed, you must modify the risk rule configuration."
```

---

### ❌ LLM Making Trading Decisions

**Wrong:**
```
"I think this is a good trade, let's enter long SPY at $450"
```

**Right:**
```
"SignalCore generated LONG signal for SPY (score 0.65, prob 0.72).
 RiskGuard approved (all 14 rules passed).
 FlowRoute submitted order ABC123."
```

---

## Version History

**v1.0.0** (2025-01-28)
- Initial integration patterns documented
- Three primary workflows defined
- Guardrails and anti-patterns specified

---

## Navigation

- [TAXONOMY](TAXONOMY.md) - Domain definitions
- [GOVERNANCE](GOVERNANCE.md) - Change management
- [Knowledge Base Home](../README.md)

**Document Version:** v1.0.0
**Last Updated:** 2025-01-28
**Status:** Active
