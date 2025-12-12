# TensorTrade-to-Alpaca Production Deployment Specification

## System Orchestration Directive

This specification defines the operational requirements for transitioning TensorTrade reinforcement learning execution agents from simulation environments to live production trading via the Alpaca brokerage API. The system must maintain deterministic risk control at all integration points.

---

## 1. Broker Integration (Alpaca API)

### 1.1 Data Stream Replacement

| Component | Simulation Source | Production Source |
|-----------|-------------------|-------------------|
| Market Data | TensorTrade synthetic feed | Alpaca WebSocket `wss://stream.data.alpaca.markets` |
| Quote Data | Simulated bid/ask | Alpaca REST `GET /v2/stocks/{symbol}/quotes` |
| Bar Data | Historical replay | Alpaca `GET /v2/stocks/bars` (real-time) |
| Account State | Internal simulation | Alpaca `GET /v2/account` |

**Integration Requirements:**
- Replace `TradingEnvironment.data_feed` with Alpaca WebSocket listener
- Implement reconnection logic with exponential backoff (max 5 retries, 30s ceiling)
- Maintain local order book mirror synchronized to WebSocket updates
- Handle market data gaps with last-known-value interpolation

### 1.2 Order Event Mapping

| TensorTrade Action | Alpaca Endpoint | Method |
|-------------------|-----------------|--------|
| `ExecutionAction.MARKET` | `/v2/orders` | POST (type: market) |
| `ExecutionAction.LIMIT_AGGRESSIVE` | `/v2/orders` | POST (type: limit, time_in_force: day) |
| `ExecutionAction.LIMIT_PASSIVE` | `/v2/orders` | POST (type: limit, time_in_force: gtc) |
| `ExecutionAction.CANCEL` | `/v2/orders/{order_id}` | DELETE |

**Order Submission Protocol:**
1. Validate signal against RiskGuard before submission
2. Generate idempotency key per order attempt
3. Submit via Alpaca REST with timeout (5s default)
4. Store order_id mapping to internal execution_id
5. Subscribe to order updates via WebSocket `trade_updates` stream

### 1.3 Execution Update Handling

**WebSocket Subscription:**
```
wss://paper-api.alpaca.markets/stream (paper)
wss://api.alpaca.markets/stream (live)
```

**Required Event Handlers:**
- `new` - Order accepted, update internal state
- `fill` / `partial_fill` - Execute position update, recalculate P&L
- `canceled` - Clear pending order reference
- `rejected` - Log rejection reason, trigger fallback logic
- `expired` - Handle GTD/GTC expirations

**State Consistency Rules:**
- Reconcile positions via `GET /v2/positions` every 60 seconds
- Compare WebSocket-derived state against REST snapshot
- Alert on discrepancies exceeding 0.1% of position value

### 1.4 Resilience Requirements

| Failure Mode | Handling Strategy |
|--------------|-------------------|
| Rate limit (429) | Exponential backoff, queue retry |
| Network timeout | Retry with idempotency key (max 3) |
| WebSocket disconnect | Auto-reconnect, resync from REST |
| Message loss detection | Sequence number validation, gap fill |
| API degradation | Switch to polling mode (5s interval) |

---

## 2. Risk Controls

### 2.1 Pre-Trade Risk Checks

Every TensorTrade signal MUST pass through RiskGuard before Alpaca submission:

```
Signal -> RiskGuard.evaluate_signal() -> [PASS/REJECT/RESIZE] -> Alpaca
```

**Mandatory Rule Categories:**
- Position limits (max 5% equity per position)
- Sector exposure (max 25% per sector)
- Correlation limits (max 40% correlated exposure)
- Daily loss limits (halt at 3% daily drawdown)

### 2.2 Broker-Level Safeguards

Configure via Alpaca Dashboard and API:

| Control | Configuration |
|---------|---------------|
| Position limit | `max_position_size` per symbol |
| Account leverage | `multiplier` = 1 (no margin) for initial deployment |
| PDT protection | Enable for accounts < $25k |
| Stop-loss orders | Attach bracket orders to all entries |

### 2.3 Kill Switch Implementation

**Circuit Breaker Triggers:**
- Daily P&L exceeds -3% of equity
- Weekly P&L exceeds -5% of equity
- Consecutive losing trades > 5
- API connectivity lost > 30 seconds
- Position reconciliation mismatch detected

**Kill Switch Actions:**
1. Cancel all open orders via `DELETE /v2/orders`
2. Set internal `_halted = True`
3. Optionally liquidate all positions
4. Send alert notification
5. Require manual reset to resume

### 2.4 Leverage and Margin Constraints

| Account Type | Leverage Cap | Margin Requirement |
|--------------|--------------|-------------------|
| Cash Account | 1x | 100% |
| Margin Account (Initial) | 1x | 100% |
| Margin Account (Experienced) | 2x | 50% |

---

## 3. Latency and Reliability

### 3.1 RL Agent Inference Optimization

| Optimization | Implementation |
|--------------|----------------|
| Model quantization | INT8 inference for production |
| Batch inference | Accumulate signals over 100ms window |
| GPU pinning | Dedicate GPU for inference only |
| Warm-up | Pre-load model at market open |
| Caching | Cache feature computations (TTL: 1s) |

**Target Latency Budgets:**
- Feature extraction: < 10ms
- Model inference: < 20ms
- Risk check: < 5ms
- Order submission: < 100ms
- **Total signal-to-order: < 150ms**

### 3.2 Error Handling Matrix

| Error Type | Response | Retry |
|------------|----------|-------|
| Network timeout | Log, retry with backoff | Yes (3x) |
| API 4xx errors | Log, do not retry | No |
| API 5xx errors | Log, retry with backoff | Yes (5x) |
| Deserialization error | Log, skip update | No |
| State mismatch | Resync from REST | N/A |

### 3.3 WebSocket Persistence

**Connection Management:**
- Heartbeat interval: 30 seconds
- Missed heartbeat tolerance: 2
- Reconnection strategy: Immediate + exponential backoff
- Session recovery: Re-subscribe to all streams after reconnect

**Message Queue:**
- Buffer incoming messages during processing
- Max queue depth: 10,000 messages
- Overflow handling: Drop oldest, log warning

---

## 4. Compliance and Logging

### 4.1 Audit Log Schema

Every trading decision must log:

```json
{
  "timestamp": "ISO8601",
  "event_type": "signal|risk_check|order|fill|cancel",
  "signal_id": "uuid",
  "model_id": "string",
  "symbol": "string",
  "direction": "long|short|neutral",
  "quantity": "integer",
  "risk_check_result": {
    "passed": "boolean",
    "rules_evaluated": ["rule_ids"],
    "action_taken": "pass|reject|resize|halt"
  },
  "alpaca_order_id": "uuid|null",
  "execution_price": "float|null",
  "slippage_bps": "float|null",
  "latency_ms": "integer"
}
```

### 4.2 Required Log Streams

| Stream | Retention | Format |
|--------|-----------|--------|
| Signal generation | 90 days | JSON Lines |
| Risk check decisions | 1 year | JSON Lines |
| Order submissions | 1 year | JSON Lines |
| Fill confirmations | 7 years | JSON Lines |
| System health | 30 days | Structured logs |

### 4.3 Monitoring Dashboard Requirements

**Real-Time Metrics:**
- Current positions (from Alpaca `/v2/positions`)
- Unrealized P&L (calculated from positions)
- Daily realized P&L (from Alpaca `/v2/account/activities`)
- Open orders count and value
- Risk utilization (% of limits consumed)

**Alerting Thresholds:**
- P&L drawdown > 2%: Warning
- P&L drawdown > 3%: Critical (trigger kill switch)
- API latency p95 > 500ms: Warning
- Order rejection rate > 5%: Critical

### 4.4 Alpaca Data Endpoints for Compliance

| Data Need | Endpoint |
|-----------|----------|
| Account equity history | `GET /v2/account/portfolio/history` |
| Trade history | `GET /v2/account/activities?activity_types=FILL` |
| Order history | `GET /v2/orders?status=all` |
| Position snapshots | `GET /v2/positions` |

---

## 5. Environment Bridging

### 5.1 Data Feed Substitution

| TensorTrade Component | Production Replacement |
|----------------------|----------------------|
| `DataFeed` | `AlpacaDataFeed` wrapper |
| `Observer` | Alpaca quote/trade observers |
| `Exchange` simulation | Alpaca order routing |
| `Broker` simulation | Alpaca account/position sync |

**Implementation Pattern:**
```python
# Replace synthetic environment
env = TradingEnvironment(
    data_feed=AlpacaDataFeed(symbols=["AAPL", "MSFT"]),
    exchange=AlpacaExchange(client=alpaca_client),
    broker=AlpacaBroker(client=alpaca_client),
)
```

### 5.2 Microstructure Effects

**Real-World Considerations:**
- Slippage estimation: Apply historical slippage model (0.5-2 bps typical)
- Liquidity detection: Check quote depth before large orders
- Spread dynamics: Factor bid-ask spread into limit price calculation
- Fill probability: Estimate based on order type and market conditions

### 5.3 Pre-Production Validation

**Backtesting with Alpaca Historical Data:**
1. Fetch historical bars via `GET /v2/stocks/bars`
2. Run TensorTrade agent through historical replay
3. Compare simulated fills to what would have executed
4. Measure strategy drift under realistic microstructure

**Paper Trading Phase:**
- Minimum duration: 2 weeks continuous operation
- Use Alpaca paper trading environment (`paper-api.alpaca.markets`)
- Monitor for:
  - Order rejection patterns
  - Slippage vs. simulation assumptions
  - Risk rule trigger frequency
  - System stability under market volatility

---

## 6. Deployment Checklist

### Pre-Deployment

- [ ] RiskGuard rules configured and tested
- [ ] Alpaca API credentials secured (environment variables)
- [ ] WebSocket reconnection logic tested
- [ ] Audit logging pipeline verified
- [ ] Monitoring dashboard deployed
- [ ] Kill switch tested (manual trigger)
- [ ] Paper trading completed (2 weeks minimum)

### Go-Live

- [ ] Switch to live Alpaca API endpoints
- [ ] Enable real-time alerting
- [ ] Start with reduced position sizes (25% of target)
- [ ] Monitor first trading session manually
- [ ] Verify fill reports match expected execution

### Post-Deployment

- [ ] Daily P&L reconciliation
- [ ] Weekly strategy performance review
- [ ] Monthly risk parameter adjustment review
- [ ] Quarterly compliance audit

---

## 7. System Component Summary

```
+-------------------+     +------------------+     +------------------+
|   TensorTrade     |     |    RiskGuard     |     |   Alpaca API     |
|   RL Agent        |---->|   Risk Engine    |---->|   Broker         |
|   (Inference)     |     |   (Validation)   |     |   (Execution)    |
+-------------------+     +------------------+     +------------------+
        ^                         |                        |
        |                         v                        v
+-------------------+     +------------------+     +------------------+
|   Market Data     |     |   Audit Logger   |     |   Position Sync  |
|   (Alpaca WS)     |     |   (Compliance)   |     |   (REST + WS)    |
+-------------------+     +------------------+     +------------------+
```

**Data Flow:**
1. Alpaca WebSocket delivers market data
2. TensorTrade agent generates execution signal
3. RiskGuard validates signal against all rules
4. Approved signals route to Alpaca order submission
5. Execution updates flow back via WebSocket
6. All events logged to audit trail

---

*This specification defines the operational boundary conditions for production deployment. Implementation details for each component should reference the corresponding module documentation.*
