# Feature Implementation Plan: Broker State Synchronization

**Source**: Trading Day Analysis 2025-12-17
**Priority**: Critical
**Status**: Planning

---

## Executive Summary

Paper trading session on 2025-12-17 revealed critical issues with broker state synchronization:
- 6,123 `insufficient buying power` errors (98% of failures)
- 68 `pending_new` OrderStatus enum mismatches
- 48 position sizing failures due to stale buying power data
- Position tracking diverged from actual broker state (21 positions, $0 buying power)

---

## Implementation Phases

### Phase 1: Critical Fixes (Immediate)

#### 1.1 OrderStatus Enum Extension
**File**: `src/ordinis/adapters/broker/broker.py`

Add missing Alpaca order statuses to enum:
- `pending_new` - Order submitted, awaiting acceptance
- `pending_cancel` - Cancel requested, awaiting confirmation
- `pending_replace` - Replace requested, awaiting confirmation
- `stopped` - Order stopped at exchange
- `suspended` - Order suspended

**Acceptance Criteria**:
- [ ] All Alpaca API order statuses mapped
- [ ] Unit tests for status transitions
- [ ] No enum parsing errors in logs

#### 1.2 Pre-Trade Buying Power Check
**File**: `src/ordinis/engines/flowroute/core/engine.py`

Before every order submission:
1. Query Alpaca account endpoint
2. Validate `buying_power >= order_value`
3. Reject order locally if insufficient funds
4. Log rejection with available/required amounts

**Acceptance Criteria**:
- [ ] Zero `insufficient buying power` errors from Alpaca
- [ ] Local validation catches all cases
- [ ] Latency impact < 50ms per order

#### 1.3 Position Sync on Startup
**File**: `src/ordinis/engines/flowroute/core/engine.py`

Startup sequence enhancement:
1. Query Alpaca positions endpoint
2. Populate internal position tracker
3. Query Alpaca account for buying power
4. Log sync summary (positions, equity, buying power)

**Acceptance Criteria**:
- [ ] Internal state matches broker on startup
- [ ] Handles empty account (no positions)
- [ ] Handles max margin utilization

---

### Phase 2: Position Management (Short-term)

#### 2.1 Position Limit Enforcement
**File**: `src/ordinis/engines/riskguard/core/rules.py`

Enforce limits from Alpaca account, not internal tracking:
- Max positions per account (configurable, default: 20)
- Max position size as % of equity (configurable, default: 5%)
- Max total exposure (configurable, default: 150% equity)

**Acceptance Criteria**:
- [ ] Limits enforced using broker-queried values
- [ ] Config-driven thresholds
- [ ] Clear rejection reasons logged

#### 2.2 Position Reconciliation Loop
**File**: New: `src/ordinis/engines/flowroute/reconciliation.py`

Periodic broker sync (every 30 seconds during market hours):
1. Fetch current positions from Alpaca
2. Compare to internal position tracker
3. Log discrepancies
4. Auto-correct internal state
5. Emit reconciliation metrics

**Acceptance Criteria**:
- [ ] Discrepancies detected and logged
- [ ] Internal state auto-corrected
- [ ] Metrics available for monitoring

#### 2.3 Sector Exposure Limits
**File**: `src/ordinis/engines/riskguard/core/rules.py`

Add sector-based risk rules:
- Max 30% exposure to single sector
- Sector classification via symbol metadata
- Block new positions exceeding sector limit

**Acceptance Criteria**:
- [ ] Sector classification for watchlist symbols
- [ ] Sector exposure tracked per position
- [ ] Violations blocked with clear messaging

---

### Phase 3: Signal Quality (Medium-term)

#### 3.1 RSI Threshold Tuning
**File**: `configs/strategies/atr_optimized_rsi.yaml`

Current: RSI < 35 (oversold trigger)
Proposed: RSI < 30 (more selective)

Also add RSI overbought exit: RSI > 70

**Acceptance Criteria**:
- [ ] Reduced signal count by ~40%
- [ ] Higher quality entry points
- [ ] Exit signals on RSI recovery

#### 3.2 Confirmation Filters
**File**: `src/ordinis/engines/signalcore/models/rsi_mean_reversion.py`

Add secondary confirmation requirements:
- Volume spike: Current volume > 1.5x 20-period average
- Price below VWAP (confirming weakness)
- At least 2 consecutive RSI readings below threshold

**Acceptance Criteria**:
- [ ] Multi-factor signal generation
- [ ] Configurable confirmation requirements
- [ ] Backtested against historical data

---

### Phase 4: Infrastructure Hardening (Medium-term)

#### 4.1 Circuit Breaker Implementation
**File**: New: `src/ordinis/engines/flowroute/circuit_breaker.py`

Automatic trading halt conditions:
- Error rate > 10% over 5-minute window
- 5+ consecutive order failures
- Account equity drop > 5% in session
- Buying power < $1,000

**Acceptance Criteria**:
- [ ] Circuit breaker triggers halt correctly
- [ ] Manual override available
- [ ] Alert notifications sent
- [ ] Auto-recovery after cooldown period

#### 4.2 Enhanced Error Logging
**File**: `src/ordinis/engines/flowroute/core/engine.py`

Structured error logging with:
- Order details (symbol, qty, side, type)
- Account state snapshot (equity, buying power, positions)
- Error classification (retriable vs fatal)
- Correlation ID for tracing

**Acceptance Criteria**:
- [ ] All order errors captured with context
- [ ] Searchable log format
- [ ] Error rate metrics available

#### 4.3 Order State Machine
**File**: `src/ordinis/engines/flowroute/core/orders.py`

Explicit state machine for order lifecycle:
```
CREATED -> PENDING_NEW -> ACCEPTED -> FILLED
                      \-> REJECTED
                      \-> PARTIALLY_FILLED -> FILLED/CANCELLED
```

**Acceptance Criteria**:
- [ ] All state transitions validated
- [ ] Invalid transitions raise exceptions
- [ ] State history tracked per order

---

## Dependencies

| Phase | Depends On | Blocking |
|-------|------------|----------|
| 1.1 | None | 1.2, 1.3 |
| 1.2 | 1.1 | 2.1 |
| 1.3 | 1.1 | 2.2 |
| 2.1 | 1.2 | None |
| 2.2 | 1.3 | None |
| 2.3 | 2.1 | None |
| 3.1 | None | 3.2 |
| 3.2 | 3.1 | None |
| 4.1 | 1.2 | None |
| 4.2 | None | None |
| 4.3 | 1.1 | None |

---

## Testing Strategy

### Unit Tests
- OrderStatus enum completeness
- Buying power validation logic
- Position limit calculations
- Circuit breaker trigger conditions

### Integration Tests
- Alpaca API mock responses
- Position sync accuracy
- Order lifecycle end-to-end

### Paper Trading Validation
- Run full session with Phase 1 fixes
- Monitor error rates
- Verify position sync accuracy
- Measure signal quality improvements

---

## Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| Order error rate | 100% | < 5% |
| Position sync accuracy | Unknown | 100% |
| Signal-to-execution ratio | 0% | > 80% |
| Avg positions held | 21 | 10-15 |
| Max sector exposure | Unlimited | 30% |

---

## Rollout Plan

1. Deploy Phase 1 fixes to paper trading
2. Monitor 2-3 sessions for stability
3. Deploy Phase 2 with position management
4. Tune signal quality in Phase 3
5. Add infrastructure hardening in Phase 4
6. Graduate to live trading only after metrics meet targets

---

*Generated: 2025-12-17*
*Source: trading-day-analysis-2025-12-17.md*
