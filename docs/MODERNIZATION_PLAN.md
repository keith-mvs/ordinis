---
title: MCP Integration & Agent Orchestration Roadmap
filename: MODERNIZATION_PLAN.md
date: 2025-12-19
version: 1.1
type: technical roadmap
description: >
  Phased implementation plan for integrating Model Context Protocol (MCP) capabilities
  to enable agent-driven strategy optimization, market analysis, and capital preservation.
---

# MCP Integration & Agent Orchestration Roadmap

**Date:** 2025-12-19 **Version:** 1.1 **Type:** Technical Roadmap

---

## 1. Objective

Enable agents to actively improve trading performance by:

1. **Optimizing strategies** – Dynamically tune parameters based on market regime and recent performance
2. **Incorporating alternative data** – Inject news, sentiment, and qualitative context into signal generation
3. **Preserving capital** – Automatically reduce exposure or halt trading when drawdowns exceed thresholds

The implementation connects the existing MCP server to live trading components, enabling real-time control.

---

## 2. Current State Assessment

### 2.1 Existing Infrastructure

| Component | Location | Trading Relevance |
|-----------|----------|-------------------|
| **SignalCore** | `src/ordinis/engines/signalcore/` | Generates trading signals from market data and strategy configs |
| **RiskGuard** | `src/ordinis/engines/riskguard/` | Position sizing, exposure limits, stop-loss enforcement |
| **KillSwitch** | `src/ordinis/safety/kill_switch.py` | Emergency halt on daily loss limit, max drawdown, consecutive losses |
| **CircuitBreaker** | `src/ordinis/safety/circuit_breaker.py` | API resilience; prevents cascading failures from data feed issues |
| **StreamingBus** | `src/ordinis/bus/engine.py` | Event distribution for market data, signals, and execution reports |
| **Plugin System** | `src/ordinis/plugins/` | Extensible adapters for external data sources |
| **MCP Server** | `src/ordinis/mcp/server.py` | Basic tools (`get_signal`, `evaluate_risk`); uses mock state |

### 2.2 Key Gaps

| Gap | Impact on P&L |
|-----|---------------|
| MCP disconnected from live state | Agents cannot tune strategies or respond to market conditions in real-time |
| No strategy config tools | Cannot adjust parameters (RSI thresholds, position sizes) without restart |
| No market context injection | Signals ignore qualitative factors (news, earnings, Fed announcements) |
| No performance metrics exposed | Agents cannot evaluate what's working and adapt |

---

## 3. Phase 1: Strategy Control & Market Context

**Objective:** Enable agents to tune strategies and inject market intelligence.

### Task 1.1: Strategy Configuration Tools

| Subtask | Description | Expected Outcome |
|---------|-------------|------------------|
| 1.1.1 | Implement `update_strategy_config(strategy, key, value)` tool | Agents can adjust RSI periods, ATR multipliers, entry thresholds |
| 1.1.2 | Implement `get_strategy_config(strategy)` tool | Agents can read current parameters before deciding changes |
| 1.1.3 | Implement `inject_market_context(context: str)` tool | Feed news/sentiment into signal generation |
| 1.1.4 | Add bounds validation (position size ≤ 10%, leverage ≤ 5x) | Prevent catastrophic parameter mistakes |

### Task 1.2: Performance Visibility

| Subtask | Description | Expected Outcome |
|---------|-------------|------------------|
| 1.2.1 | Expose `ordinis://metrics/pnl` resource | Daily P&L, win rate, average win/loss, Sharpe estimate |
| 1.2.2 | Expose `ordinis://metrics/positions` resource | Current positions, unrealized P&L, time in trade |
| 1.2.3 | Expose `ordinis://metrics/signals` resource | Recent signal history with outcomes (hit rate by strategy) |

### Task 1.3: Capital Preservation Controls

| Subtask | Description | Expected Outcome |
|---------|-------------|------------------|
| 1.3.1 | Connect MCP to live `KillSwitch` instance | Agents can halt trading when drawdown exceeds threshold |
| 1.3.2 | Implement `get_drawdown_status()` tool | Returns current drawdown %, daily loss, consecutive losses |
| 1.3.3 | Implement `reduce_exposure(factor: float)` tool | Scale down position sizes without full halt (e.g., 50% reduction) |

### Phase 1 Verification

| Test | Pass Criteria |
|------|---------------|
| Config change | `update_strategy_config("rsi_mean_reversion", "rsi_oversold", 25)` persists and affects next signal |
| Context injection | `inject_market_context("Fed announced rate hold")` appears in signal generation context |
| Drawdown halt | Kill switch activates when daily loss exceeds configured limit |

**Milestone 1:** Agents can tune strategies, inject context, and protect capital.

---

## 4. Phase 2: Adaptive Strategy Agent

**Objective:** Deploy an agent that continuously optimizes strategy parameters.

### Task 2.1: Strategy Optimizer Agent

| Subtask | Description | Expected Outcome |
|---------|-------------|------------------|
| 2.1.1 | Define agent prompt: analyze recent performance, identify underperforming strategies | Agent focuses on improving win rate and risk-adjusted returns |
| 2.1.2 | Configure toolset: `get_strategy_config`, `update_strategy_config`, `ordinis://metrics/signals` | Agent can read performance and adjust parameters |
| 2.1.3 | Implement daily optimization workflow | Agent reviews yesterday's trades, proposes parameter adjustments |

### Task 2.2: Market Regime Agent

| Subtask | Description | Expected Outcome |
|---------|-------------|------------------|
| 2.2.1 | Define agent prompt: classify market regime (trending, ranging, volatile) | Agent determines appropriate strategy mode |
| 2.2.2 | Configure toolset: `get_market_news`, `inject_market_context`, `update_strategy_config` | Agent can fetch news and adjust strategy accordingly |
| 2.2.3 | Implement pre-market briefing workflow | Agent runs before market open, sets regime context |

### Phase 2 Verification

| Test | Pass Criteria |
|------|---------------|
| Regime detection | Agent correctly identifies "high volatility" regime after VIX spike news |
| Parameter adjustment | Agent increases ATR multiplier for stops during volatile regime |
| Performance tracking | Agent logs reasoning for each adjustment; changes are auditable |

**Milestone 2:** Agents actively optimize strategies based on market conditions and performance.

---

## 5. Phase 3: External Data Integration

**Objective:** Connect to external data sources for enhanced signal generation.

### Task 3.1: Massive MCP Integration (News & Alternative Data)

| Subtask | Description | Expected Outcome |
|---------|-------------|------------------|
| 3.1.1 | Create `MassiveMCPPlugin` extending `Plugin` | Access to news, SEC filings, social sentiment |
| 3.1.2 | Implement `get_market_news(symbol)` method | Returns recent headlines affecting the symbol |
| 3.1.3 | Implement `get_earnings_calendar()` method | Returns upcoming earnings for portfolio symbols |

### Task 3.2: News-Aware Signal Enhancement

| Subtask | Description | Expected Outcome |
|---------|-------------|------------------|
| 3.2.1 | Create `NewsContextHook` that fetches news before signal generation | Signals consider recent news (e.g., skip long on bad earnings) |
| 3.2.2 | Implement news sentiment scoring | Positive/negative/neutral classification for context |

### Phase 3 Verification

| Test | Pass Criteria |
|------|---------------|
| News fetch | `get_market_news("AAPL")` returns recent headlines |
| Signal adjustment | Negative earnings news reduces signal confidence or blocks entry |

**Milestone 3:** Trading decisions incorporate real-time news and alternative data.

---

## 6. Phase 4: Drawdown Recovery & Position Management

**Objective:** Automated responses to adverse market conditions.

### Task 4.1: Drawdown Response Agent

| Subtask | Description | Expected Outcome |
|---------|-------------|------------------|
| 4.1.1 | Define agent prompt: monitor drawdown, decide between reduce exposure / halt / continue | Agent makes practical capital preservation decisions |
| 4.1.2 | Configure toolset: `get_drawdown_status`, `reduce_exposure`, `activate_kill_switch` | Agent has graduated response options |
| 4.1.3 | Implement threshold-based workflow | 5% drawdown → reduce 25%; 10% → reduce 50%; 15% → halt |

### Task 4.2: Position Exit Optimization

| Subtask | Description | Expected Outcome |
|---------|-------------|------------------|
| 4.2.1 | Implement `analyze_open_positions()` tool | Returns positions with unrealized P&L, time held, distance to stop |
| 4.2.2 | Define exit optimization prompt | Agent evaluates whether to hold, tighten stops, or exit |

### Phase 4 Verification

| Test | Pass Criteria |
|------|---------------|
| Graduated response | 7% drawdown triggers 25% exposure reduction, not full halt |
| Position analysis | Agent correctly identifies underwater positions needing attention |

**Milestone 4:** System automatically scales risk based on performance, preserving capital during drawdowns.

---

## 7. Hook Architecture

Hooks are the integration points where agents inject logic into the trading pipeline. They run **inside** engine operations and can block, modify, or log actions.

### 7.1 Hook Interfaces

The existing `GovernanceHook` protocol (in `src/ordinis/engines/base/hooks.py`) defines three methods:

```python
class GovernanceHook(Protocol):
    async def preflight(self, context: PreflightContext) -> PreflightResult:
        """Called BEFORE an operation. Can block (DENY) or allow (ALLOW/WARN)."""
        ...

    async def audit(self, record: AuditRecord) -> None:
        """Called AFTER an operation. Records outcome for analysis."""
        ...

    async def on_error(self, error: EngineError) -> None:
        """Called when an error occurs. Used for alerting."""
        ...
```

**Decision outcomes:**

| Decision | Effect | Use Case |
|----------|--------|----------|
| `ALLOW` | Proceed normally | Standard operations |
| `DENY` | Block the operation | Position size exceeds limit; drawdown too high |
| `WARN` | Proceed but log warning | Unusual but acceptable parameters |
| `DEFER` | Escalate for human review | Large parameter changes; novel market conditions |

### 7.2 Trading-Relevant Hooks

| Hook | Location | Trading Function |
|------|----------|------------------|
| **NewsContextHook** | `src/ordinis/engines/signalcore/hooks/news.py` | Injects news context before signal generation; can block signals on major negative news |
| **DrawdownHook** | `src/ordinis/engines/riskguard/hooks/drawdown.py` | Checks current drawdown before position sizing; reduces exposure or blocks new entries |
| **RegimeHook** | `src/ordinis/engines/signalcore/hooks/regime.py` | Adjusts strategy parameters based on detected market regime |
| **PositionLimitHook** | `src/ordinis/engines/riskguard/hooks/limits.py` | Validates position size against max allocation; blocks oversized positions |

### 7.3 Hook Implementation Example

**DrawdownHook** – Reduces exposure as drawdown increases:

```python
class DrawdownHook(BaseGovernanceHook):
    """Graduated exposure reduction based on drawdown."""

    def __init__(self, kill_switch: KillSwitch):
        super().__init__("riskguard")
        self._kill_switch = kill_switch
        self._thresholds = [(0.05, 0.75), (0.10, 0.50), (0.15, 0.0)]  # (drawdown, exposure_factor)

    async def preflight(self, context: PreflightContext) -> PreflightResult:
        if context.action != "calculate_position_size":
            return PreflightResult(decision=Decision.ALLOW)

        status = self._kill_switch.get_status()
        drawdown = status.current_drawdown

        for threshold, factor in self._thresholds:
            if drawdown >= threshold:
                if factor == 0.0:
                    return PreflightResult(
                        decision=Decision.DENY,
                        reason=f"Drawdown {drawdown:.1%} exceeds halt threshold"
                    )
                return PreflightResult(
                    decision=Decision.WARN,
                    reason=f"Reducing exposure to {factor:.0%} due to {drawdown:.1%} drawdown",
                    adjustments={"exposure_factor": factor}
                )

        return PreflightResult(decision=Decision.ALLOW)
```

### 7.4 Hook Composition

Multiple hooks can be composed using `CompositeGovernanceHook`:

```python
# In RiskGuard engine initialization
hooks = CompositeGovernanceHook("riskguard", [
    PositionLimitHook(max_position_pct=0.10),
    DrawdownHook(kill_switch),
    NewsContextHook(news_client),
])
self._governance = hooks
```

**Execution order:** Most restrictive decision wins. If any hook returns `DENY`, the operation is blocked.

---

## 8. Agent Prompts

Prompts define the agent's reasoning framework and decision-making criteria. Each agent has a system prompt that establishes its role and constraints.

### 8.1 Strategy Optimizer Prompt

```yaml
# configs/agents/strategy_optimizer.yaml
name: strategy_optimizer
schedule: "0 6 * * 1-5"  # 6 AM EST, weekdays
model: nemotron-super-49b-v1.5

system_prompt: |
  You are a quantitative strategy optimizer for Ordinis, an automated trading system.

  YOUR OBJECTIVE: Improve risk-adjusted returns by tuning strategy parameters.

  AVAILABLE DATA:
  - Recent signal performance (hit rate, average win/loss, Sharpe)
  - Current strategy configurations (RSI thresholds, ATR multipliers, position sizing)
  - Market regime context (trending, ranging, volatile)

  DECISION FRAMEWORK:
  1. Review signals from the past 5 trading days
  2. Identify strategies with hit rate below 50% or negative Sharpe
  3. Propose parameter adjustments based on market regime:
     - Trending: Widen stops (increase ATR multiplier), reduce RSI thresholds
     - Ranging: Tighten stops, use mean-reversion parameters
     - Volatile: Reduce position sizes, increase confirmation requirements

  CONSTRAINTS:
  - Never set position_size > 10% of capital per trade
  - Never set leverage > 5x
  - Never adjust more than 2 parameters per strategy per day
  - Log reasoning for every change

  OUTPUT FORMAT:
  For each adjustment, specify:
  - strategy_name
  - parameter_name
  - old_value
  - new_value
  - reasoning (1-2 sentences)

tools:
  - get_strategy_config
  - update_strategy_config
  - ordinis://metrics/signals
  - ordinis://metrics/pnl
```

### 8.2 Market Regime Prompt

```yaml
# configs/agents/market_regime.yaml
name: market_regime
schedule: "0 8 * * 1-5"  # 8 AM EST, weekdays (pre-market)
model: nemotron-super-49b-v1.5

system_prompt: |
  You are a market analyst for Ordinis. Your job is to classify the current market regime.

  YOUR OBJECTIVE: Determine market conditions and inject appropriate context for trading.

  REGIME CLASSIFICATIONS:
  - TRENDING_UP: Clear upward momentum, breakouts succeeding, low volatility
  - TRENDING_DOWN: Sustained selling, breakdowns succeeding, increasing fear
  - RANGING: Sideways price action, mean-reversion opportunities
  - HIGH_VOLATILITY: VIX elevated, wide price swings, reduced position sizing appropriate
  - RISK_OFF: Major negative catalyst, defensive positioning recommended

  DATA SOURCES:
  - Recent news headlines for portfolio symbols
  - Earnings calendar (next 5 days)
  - Fed/FOMC announcements
  - VIX level and trend

  DECISION PROCESS:
  1. Fetch news for major indices (SPY, QQQ) and portfolio symbols
  2. Check earnings calendar for imminent announcements
  3. Classify regime based on aggregate sentiment and volatility
  4. Inject context into SignalCore via inject_market_context

  OUTPUT FORMAT:
  {
    "regime": "TRENDING_UP | RANGING | HIGH_VOLATILITY | ...",
    "confidence": 0.0-1.0,
    "key_factors": ["factor1", "factor2"],
    "position_size_modifier": 0.5-1.5,
    "context_message": "Brief description for signal generation"
  }

tools:
  - get_market_news
  - get_earnings_calendar
  - inject_market_context
  - update_strategy_config
```

### 8.3 Drawdown Response Prompt

```yaml
# configs/agents/drawdown_response.yaml
name: drawdown_response
schedule: "*/5 * * * 1-5"  # Every 5 minutes during market hours
model: nemotron-8b-v3.1  # Faster model for frequent checks

system_prompt: |
  You are the capital preservation agent for Ordinis.

  YOUR OBJECTIVE: Protect capital during drawdowns without over-halting.

  DRAWDOWN THRESHOLDS:
  - < 5%: Normal operations, no intervention
  - 5-10%: Reduce new position sizes by 25%
  - 10-15%: Reduce new position sizes by 50%, tighten stops on existing
  - > 15%: Activate kill switch, halt all new entries

  AVAILABLE ACTIONS:
  - reduce_exposure(factor): Scale down position sizing (e.g., 0.75 for 25% reduction)
  - activate_kill_switch(): Full halt—use only when drawdown exceeds 15%
  - analyze_open_positions(): Review positions for potential exits

  DECISION CRITERIA:
  - Prioritize capital preservation over missed opportunities
  - Graduated response: reduce before halt
  - Consider recovery potential: if underwater positions are near stops, let them play out

  LOGGING:
  For every intervention, log:
  - current_drawdown_pct
  - action_taken
  - reasoning

tools:
  - get_drawdown_status
  - reduce_exposure
  - activate_kill_switch
  - analyze_open_positions
```

### 8.4 Prompt Best Practices

| Practice | Rationale |
|----------|-----------|
| **Explicit constraints** | Prevents agent from making catastrophic changes (e.g., max position size) |
| **Quantitative thresholds** | Removes ambiguity; agent knows exactly when to act |
| **Required output format** | Ensures structured, parseable responses for automation |
| **Reasoning requirement** | Creates audit trail; enables human review of agent decisions |
| **Toolset restriction** | Agents only see tools relevant to their role; reduces confusion |

---

## 9. Deliverables Summary

| Deliverable | Location | Purpose |
|-------------|----------|---------|
| Strategy Control Tools | `src/ordinis/mcp/server.py` | Tune parameters, inject context |
| Performance Resources | `src/ordinis/mcp/server.py` | Expose P&L, positions, signal metrics |
| DrawdownHook | `src/ordinis/engines/riskguard/hooks/drawdown.py` | Graduated exposure reduction |
| NewsContextHook | `src/ordinis/engines/signalcore/hooks/news.py` | News injection before signals |
| RegimeHook | `src/ordinis/engines/signalcore/hooks/regime.py` | Market regime adjustments |
| Strategy Optimizer Agent | `configs/agents/strategy_optimizer.yaml` | Daily parameter optimization |
| Market Regime Agent | `configs/agents/market_regime.yaml` | Pre-market regime classification |
| Drawdown Response Agent | `configs/agents/drawdown_response.yaml` | Graduated capital preservation |
| Massive Plugin | `src/ordinis/plugins/massive.py` | News and alternative data |

---

## 10. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Signal hit rate improvement | +5% after 30 days | Compare pre/post agent optimization |
| Drawdown reduction | Max drawdown ≤ 15% | Agent intervention prevents extended losses |
| Strategy adaptability | Config changes within 1 hour of regime shift | News → context injection → parameter update |
| System uptime | ≥ 99.5% during market hours | No unplanned halts from agent errors |

---

## 9. Risk Considerations

| Risk | Mitigation |
|------|------------|
| Agent makes bad parameter change | Bounds validation on all config updates; changes logged and reversible |
| Over-optimization (curve fitting) | Agent reasoning logged; human review of significant changes |
| News misinterpretation | Sentiment scoring as input, not sole decision driver |
| Excessive halts reduce profit potential | Graduated response (reduce before halt); halt only on severe drawdown || Hook blocks valid trades | Hooks use WARN for borderline cases; DENY only for clear violations |
| Prompt drift over time | Version prompts in configs/agents/; review quarterly |
