# Post-Mortem: Trading Day 2025-12-17

**Date:** 2025-12-17
**Status:** CRITICAL FAILURE
**Impact:** 6,239+ failed executions, 0 successful trades, $0 P&L despite 21 open positions

---

## Executive Summary

On December 17, 2025, the Ordinis trading system experienced a catastrophic failure where:
- **6,123** "insufficient buying power" errors occurred
- **68** "pending_new OrderStatus not in enum" errors occurred
- **48** "calculated quantity is 0" errors occurred
- **21** positions were open (consuming $195k of margin on a $100k account)
- **0** signals successfully executed
- **No adaptation** occurred despite 100+ errors/minute

The system continued generating and attempting to execute signals for the entire trading session despite having **zero buying power** after the first ~100 orders.

---

## Root Cause Analysis (Per System Specification)

### 1. RiskEngine: Failed to Block Signals

**SDR Section 3 Requirement:**
> "Exposure Limits: Caps on position size per asset or sector"
> "Leverage/Margin Checks: For futures or leveraged instruments, ensure margin requirements are met"

**What Should Have Happened:**
- RiskEngine should check buying power BEFORE approving signals
- RiskEngine should block signals when margin is exhausted
- RiskEngine should provide feedback to LearningEngine

**What Actually Happened:**
- RiskEngine approved all 6,000+ signals
- No buying power check in pre-flight validation
- No feedback loop to throttle signal generation

**Missing Hook:** `record_insufficient_capital()` → LearningEngine

---

### 2. ExecutionEngine: Failed to Feed Back Rejections

**SDR Section 4 Requirement:**
> "Feedback Loop: Publish execution results (fills, trade confirmations) back to the system"
> "Order State Management: Track orders until completion – handle acknowledgments, partial fills, cancellations, and rejections"

**What Should Have Happened:**
- ExecutionEngine should record each rejection
- Error rate should be monitored
- Circuit breaker should trip after threshold exceeded
- Signal generation should be throttled

**What Actually Happened:**
- Rejections logged but not fed back
- No error rate monitoring
- No circuit breaker
- SignalEngine continued at full speed

**Missing Hook:** `record_execution_failure()` → CircuitBreakerMonitor → SignalEngine throttle

---

### 3. PortfolioEngine: Position Tracking Diverged

**SDR Section 5 Requirement:**
> "Position Tracking: Maintain up-to-date positions for each asset (quantities held, average cost, unrealized P&L)"
> "As execution reports come in, update positions accordingly"

**What Should Have Happened:**
- PortfolioEngine should sync with broker on startup
- Position mismatches should trigger alerts
- Reconciliation should occur periodically

**What Actually Happened:**
- Internal tracking showed 0 positions
- Broker had 21 positions ($195k)
- No reconciliation occurred
- Buying power calculations were wrong

**Missing Hook:** `record_position_mismatch()` → force reconciliation

---

### 4. GovernanceEngine: No Pre-Flight Checks

**SDR Section 9 Requirement:**
> "Policy Checks (Pre-flight): Before any critical action is taken by another engine, the GovernanceEngine can perform checks to ensure:
> - Trading is allowed under current market conditions
> - Risk limits haven't been exceeded
> - Required approvals are in place"

**What Should Have Happened:**
- GovernanceEngine should check buying power before EACH signal
- GovernanceEngine should monitor error rates
- GovernanceEngine should trigger circuit breaker

**What Actually Happened:**
- No pre-flight checks on buying power
- No error rate monitoring
- No circuit breaker
- 6,123 signals attempted despite $0 buying power

**Missing Hook:** `should_allow_signals()` → CircuitBreakerMonitor check

---

### 5. OrchestrationEngine: No Cycle-Level Feedback

**SDR Section 1 Requirement:**
> "run_cycle(event): Processes one market event or tick through the pipeline, returns a composite result"
> "Implement tracing and timing for each step to feed into performance analytics"

**What Should Have Happened:**
- Each cycle should record success/failure counts
- Error patterns should be detected within cycles
- Cycle results should feed into LearningEngine

**What Actually Happened:**
- Cycles completed "successfully" from orchestrator's view
- Execution failures not bubbled up
- No cycle-level error aggregation

**Missing Hook:** `record_trading_cycle()` → aggregate error counts

---

## Timeline Reconstruction

| Time | Event | Errors | Should Have |
|------|-------|--------|-------------|
| 09:30 | Market open | 0 | N/A |
| 09:31 | First signals generated | 0 | N/A |
| 09:32 | First "insufficient buying power" | 1 | Log, continue |
| 09:33 | Error rate at 10/min | 10 | **Monitor** |
| 09:35 | Error rate at 50/min | 100 | **Alert** |
| 09:40 | Error rate at 100/min | 500 | **CIRCUIT BREAKER** |
| 10:00 | Continued failures | 1000+ | **Trading halted** |
| 16:00 | Market close | 6,123 | N/A |

With the new `CircuitBreakerMonitor`:
- **Trip after 3 "insufficient_capital" errors** (threshold set to 3)
- Trading would have halted at ~09:33
- 6,120 unnecessary errors prevented

---

## New Feedback System Architecture

```
ExecutionEngine
    ├── record_execution_failure() → FeedbackCollector
    │                                    ├── SQLite (audit trail)
    │                                    ├── ChromaDB (semantic search)
    │                                    ├── LearningEngine (adaptation)
    │                                    └── CircuitBreakerMonitor
    │                                             ├── is_threshold_exceeded()
    │                                             └── trip_breaker()
    │
    └── should_allow_execution() ← CircuitBreakerMonitor.should_allow_signal()
           │
           └── Returns (False, "circuit_breaker_open") if tripped
```

### Circuit Breaker Thresholds

| Error Type | Window | Threshold | Rationale |
|------------|--------|-----------|-----------|
| `insufficient_capital` | 60s | 3 | Very sensitive - capital exhaustion is critical |
| `order_rejected` | 60s | 5 | Moderate - some rejections are normal |
| `execution_failure` | 60s | 10 | General failures |
| `position_mismatch` | 300s | 1 | Any mismatch is critical |
| `broker_sync_failure` | 60s | 2 | Connection issues |

---

## Implementation Summary

### Files Added/Modified

1. **`src/ordinis/engines/learning/collectors/feedback.py`**
   - Added `CircuitBreakerMonitor` class
   - Added `CircuitBreakerState` enum (CLOSED, HALF_OPEN, OPEN)
   - Added `ErrorWindow` dataclass for sliding window error tracking
   - Added 12 engine-specific recording hooks per SDR

2. **Engine-Specific Hooks Added:**

   | Engine | Hook | Purpose |
   |--------|------|---------|
   | ExecutionEngine | `record_execution_failure()` | Feed back order rejections |
   | ExecutionEngine | `record_order_rejected()` | Track broker rejections |
   | RiskEngine | `record_risk_breach()` | Track exposure limit violations |
   | RiskEngine | `record_insufficient_capital()` | Track capital exhaustion |
   | PortfolioEngine | `record_position_mismatch()` | Track broker sync issues |
   | PortfolioEngine | `record_portfolio_state_snapshot()` | Periodic state capture |
   | GovernanceEngine | `record_circuit_breaker_triggered()` | Track circuit breaker events |
   | GovernanceEngine | `record_signal_throttle()` | Track throttling |
   | SignalEngine | `record_signal_batch()` | Track signal generation patterns |
   | OrchestrationEngine | `record_trading_cycle()` | Track cycle-level metrics |
   | OrchestrationEngine | `record_error_rate_spike()` | Track error rate patterns |

3. **Pre-Flight Check Methods:**
   - `should_allow_signals()` → Check before SignalEngine generates
   - `should_allow_execution()` → Check before ExecutionEngine submits

---

## Verification

Tested circuit breaker with simulated 12/17 pattern:

```python
from ordinis.engines.learning import CircuitBreakerMonitor

cb = CircuitBreakerMonitor()

# Simulate first 2 errors - no trip
cb.record_error('insufficient_capital', 'execution_engine')
cb.record_error('insufficient_capital', 'execution_engine')

# 3rd error - TRIPS
tripped, reason = cb.record_error('insufficient_capital', 'execution_engine')
# Output: tripped=True, "Error rate exceeded for insufficient_capital: 3 errors in 60s"

# Check if execution allowed
allowed, msg = cb.should_allow_signal('execution_engine')
# Output: allowed=False, msg="circuit_breaker_open"
```

**Result:** Trading would have halted after error #3, preventing 6,120 additional failures.

---

## Next Steps

1. **[ ] Integrate FeedbackCollector into ExecutionEngine**
   - Call `record_execution_failure()` on every rejection
   - Check `should_allow_execution()` before submitting orders

2. **[ ] Integrate with RiskEngine**
   - Call `record_risk_breach()` when exposure limits hit
   - Call `record_insufficient_capital()` when buying power insufficient
   - Check buying power in pre-flight validation

3. **[ ] Integrate with PortfolioEngine**
   - Add broker position sync on startup
   - Call `record_position_mismatch()` on reconciliation failures
   - Periodic `record_portfolio_state_snapshot()` every minute

4. **[ ] Integrate with OrchestrationEngine**
   - Call `record_trading_cycle()` after each cycle
   - Aggregate error counts per cycle
   - Feed cycle metrics to LearningEngine

5. **[ ] Integrate with GovernanceEngine**
   - Add pre-flight buying power check
   - Monitor error rates
   - Trigger circuit breaker alerts

---

## Lessons Learned

1. **Feedback loops are essential** - Without them, failures cascade silently
2. **Pre-flight checks prevent waste** - 6,000+ signals should never have been generated
3. **Circuit breakers protect capital** - Fast failure is better than prolonged failure
4. **Position reconciliation is critical** - Divergence from broker state is a serious bug
5. **Error rate monitoring is basic safety** - 100+ errors/minute should halt trading

---

*Document Version: 1.0*
*Created: 2025-12-17*
*Author: Ordinis AI System*
