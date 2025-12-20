# First Build to Ground - Execution Plan

**Goal:** Run the complete trading workflow end-to-end
**Date:** 2025-12-01
**Strategy:** B + C (Real backtests + Paper trading)

---

## Phase B: Real Historical Backtests

### Step 1: Get Real Historical Data

**Options:**
1. Use IEX plugin (already implemented)
2. Use Polygon plugin (already implemented)
3. Load CSV data

**Recommended:** Start with IEX (simpler, free tier available)

```python
from plugins.market_data.iex import IEXDataPlugin

plugin = IEXDataPlugin(config={"api_key": "your_key_here"})
data = await plugin.fetch_historical_data(
    symbol="SPY",
    start_date="2024-01-01",
    end_date="2024-11-30"
)
```

### Step 2: Run Backtest with Real Data

**Strategy to test:** MovingAverageCrossover (50/200 SMA)
**Symbol:** SPY (liquid, reliable)
**Period:** 2024 YTD

```python
from engines.proofbench import SimulationEngine, SimulationConfig
from strategies.moving_average_crossover import MovingAverageCrossoverStrategy

# Configure
config = SimulationConfig(
    initial_capital=100000.0,
    bar_frequency="1d",
    risk_free_rate=0.02
)

# Run
engine = SimulationEngine(config)
engine.load_data("SPY", historical_data)
# ... integrate strategy
results = engine.run()
```

### Expected Outputs
- Total return %
- Sharpe ratio
- Max drawdown
- Win rate
- Number of trades
- Equity curve

---

## Phase C: Paper Trading

### Step 1: Build Paper Broker Adapter

**Location:** `src/engines/flowroute/adapters/paper.py` (partially exists)

**Needs Implementation:**
1. Order submission
2. Fill simulation (market/limit orders)
3. Position tracking
4. Balance management
5. Order status updates

**Interface:**
```python
class PaperBroker:
    def submit_order(order: Order) -> str  # order_id
    def get_order_status(order_id: str) -> OrderStatus
    def get_positions() -> list[Position]
    def get_account_balance() -> float
    def cancel_order(order_id: str) -> bool
```

### Step 2: Real-time Data Feed

**Options:**
1. Websocket from IEX/Polygon
2. Poll REST API every N seconds
3. Use paper trading mode data feed

**Recommended:** Start with polling (simpler)

```python
import asyncio

async def data_feed_loop():
    while trading_active:
        current_data = await plugin.fetch_quote("SPY")
        strategy.on_new_data(current_data)
        await asyncio.sleep(60)  # 1 minute bars
```

### Step 3: Strategy Execution Loop

```python
async def trading_loop():
    while market_open:
        # 1. Get current data
        quote = await broker.fetch_quote(symbol)

        # 2. Generate signal
        signal = strategy.generate_signal(historical_data, now)

        # 3. Check risk (RiskGuard)
        if risk_guard.evaluate(signal, portfolio):
            # 4. Submit order
            order_id = await broker.submit_order(signal.to_order())

        await asyncio.sleep(interval)
```

### Step 4: Run Paper Trading Session

**Duration:** 1 day (market hours)
**Symbol:** SPY
**Strategy:** MA Crossover
**Capital:** $100k virtual

**Monitoring:**
- Orders submitted
- Fills received
- Position updates
- P&L tracking

---

## Implementation Order

### Part 1: Historical Backtests (4-6 hours)

1. **Get API key** (30 min)
   - Sign up for IEX Cloud
   - Get API key
   - Add to .env

2. **Fetch historical data** (1 hour)
   - Test IEX plugin
   - Download SPY 2024 data
   - Validate data quality

3. **Run first backtest** (1 hour)
   - Integrate MA Crossover strategy
   - Run simulation
   - Generate report

4. **Analyze results** (1 hour)
   - Review metrics
   - Check for issues
   - Document findings

5. **Test multiple strategies** (1-2 hours)
   - RSI Mean Reversion
   - Momentum Breakout
   - Compare results

### Part 2: Paper Trading (8-12 hours)

6. **Complete PaperBroker** (4-6 hours)
   - Implement order submission
   - Implement fill logic
   - Add position tracking
   - Test thoroughly

7. **Build execution loop** (2-3 hours)
   - Real-time data feed
   - Strategy integration
   - Order management

8. **Add monitoring** (1-2 hours)
   - Log all actions
   - Track P&L
   - Status dashboard

9. **Run paper trading** (1 hour setup + runtime)
   - Start trading session
   - Monitor performance
   - Verify behavior

---

## Success Criteria

### Phase B: Backtests
- [ ] Real data fetched successfully
- [ ] Backtest runs without errors
- [ ] Metrics calculated correctly
- [ ] Results documented
- [ ] Multiple strategies tested

### Phase C: Paper Trading
- [ ] PaperBroker functional
- [ ] Orders submitted successfully
- [ ] Fills simulated correctly
- [ ] Positions tracked accurately
- [ ] P&L calculated correctly
- [ ] Can run for full trading day

---

## Files to Create/Modify

### New Files
1. `scripts/fetch_historical_data.py` - Data fetching utility
2. `scripts/run_backtest.py` - Backtest runner
3. `scripts/paper_trading_session.py` - Paper trading main loop
4. `tests/test_paper_broker.py` - Paper broker tests

### Files to Complete
1. `src/engines/flowroute/adapters/paper.py` - Complete implementation
2. `src/engines/flowroute/core/engine.py` - Integration

### Documentation
1. `docs/BACKTEST_RESULTS_2024.md` - Results documentation
2. `docs/PAPER_TRADING_GUIDE.md` - How to paper trade

---

## Risk Management

### During Backtests
- Validate data quality
- Check for look-ahead bias
- Verify transaction costs
- Test edge cases

### During Paper Trading
- START WITH SMALL CAPITAL ($10k)
- Use single symbol only
- Monitor closely first hour
- Have kill switch ready
- Log everything

---

## Timeline

**Week 1:**
- Day 1-2: Historical backtests (Phase B)
- Day 3-4: Paper broker implementation
- Day 5: Paper trading testing

**Week 2:**
- Day 1: Full paper trading session
- Day 2: Analysis and documentation
- Day 3+: Refinements

**Total Estimate:** 10-15 hours active work + runtime

---

## Next Immediate Action

**RIGHT NOW:**
1. Check if you have IEX/Polygon API key
2. If not, sign up (free tier works)
3. Add key to .env
4. Test data fetch

**Command to start:**
```bash
python -c "from plugins.market_data.iex import IEXDataPlugin; print('IEX import: OK')"
```

---

**Status:** Ready to begin
**Current Step:** Get API key and test data fetch
**Blocking:** Need API key from IEX or Polygon
