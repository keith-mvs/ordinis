# ✅ EXECUTION CHECKLIST: PHASE 1 DEPLOYMENT

## Pre-Deployment Verification (Do Before Going Live)

### Code Quality
- [ ] Review `src/ordinis/optimizations/confidence_filter.py`
  - [ ] All type hints present
  - [ ] Docstrings complete
  - [ ] Error handling in place
  - [ ] Logging enabled
  - [ ] Unit tests passing

- [ ] Review integration points
  - [ ] Can import ConfidenceFilter cleanly
  - [ ] No circular imports
  - [ ] Depends only on numpy/pandas
  - [ ] Version compatible with existing codebase

### Backtest Setup
- [ ] `scripts/comprehensive_backtest.py` has confidence filter flag
  - [ ] Add: `--enable_confidence_filter` option
  - [ ] Add: `--confidence_threshold` parameter
  - [ ] Default threshold: 0.80

- [ ] Real backtest ready to run
  ```bash
  python scripts/comprehensive_backtest.py \
    --enable_confidence_filter \
    --confidence_threshold 0.80
  ```

### Success Criteria Defined
- [ ] Win rate target: ≥ 50% (baseline ~44%)
- [ ] Sharpe ratio target: ≥ 1.4 (baseline ~0.6)
- [ ] Trade reduction expected: 85-95%
- [ ] Profit factor target: ≥ 1.20

---

## Week 1: Real Backtest Validation

### Monday
- [ ] Run real backtest with Phase 1 enabled
  ```bash
  cd c:\Users\kjfle\Workspace\ordinis
  python scripts/comprehensive_backtest.py \
    --enable_confidence_filter \
    --confidence_threshold 0.80 \
    --output reports/phase1_real_backtest.json
  ```
- [ ] Monitor for errors (timeout ~60 minutes)
- [ ] Save report: `reports/phase1_real_backtest.json`

### Tuesday
- [ ] Analyze real backtest results
  - [ ] Extract metrics: win rate, Sharpe, trades, profit factor
  - [ ] Compare to synthetic backtest results
  - [ ] Check variance: expect ±5% acceptable
  - [ ] Review confidence score distribution
  - [ ] Verify no lookahead bias

### Wednesday
- [ ] Decision point: Did real backtest pass?
  - ✅ YES (win rate ≥ 50%) → Proceed to paper trading
  - ⚠️ MAYBE (win rate 47-50%) → Check signal quality, test thresholds
  - ❌ NO (win rate < 45%) → Stop, investigate SignalCore signals

### Thursday-Friday
- [ ] If needed: Run threshold testing on real data
  ```bash
  python scripts/threshold_optimization_real_data.py
  ```
- [ ] Prepare paper trading configuration
  - [ ] Threshold: 0.80 (or optimal from testing)
  - [ ] Min models: 4
  - [ ] Position sizing: enabled
  - [ ] Logging: verbose

---

## Week 2-3: Paper Trading Phase

### Monday (Start Paper Trading)
- [ ] Deploy confidence filter to paper trading environment
- [ ] Enable in SignalCore ensemble
- [ ] Start generating real signals with confidence scores
- [ ] Verify: Confidence scores reasonable (0.3 - 0.95 range)
- [ ] Verify: Model agreement working (2-6 models)

### Daily (Mon-Fri)
- [ ] Morning: Review previous day trades
  - [ ] Trades executed: ____
  - [ ] Trades filtered: ____
  - [ ] Win rate: ___%
  - [ ] Sharpe ratio: ____

- [ ] Afternoon: Compare to baseline
  - [ ] Baseline trades: ____
  - [ ] Phase 1 trades: ____
  - [ ] Expected diff: ~80% reduction
  - [ ] Actual diff: ___%

### Weekly Review (Friday)
- [ ] Calculate week 1 metrics
  - [ ] Total trades: ____
  - [ ] Win rate: ___%
  - [ ] Sharpe ratio: ____
  - [ ] Profit factor: ____
  - [ ] Max drawdown: ___%

- [ ] Compare to backtest expectations
  - [ ] Win rate backtest vs actual: variance < 5%?
  - [ ] Trade count: 20-30 per week expected?
  - [ ] Sharpe ratio: >1.4 expected?

- [ ] Check for issues
  - [ ] Any execution errors?
  - [ ] Any confidence score anomalies?
  - [ ] Model agreement distribution reasonable?
  - [ ] Position sizes correct?

### Week 2 End (Friday)
- [ ] Evaluate: Is performance meeting expectations?
  - ✅ YES (50%+ win rate) → Prepare for live deployment
  - ⚠️ MAYBE (47-50%) → Continue monitoring 1 more week
  - ❌ NO (< 45%) → Review signal quality, consider alternative

### Week 3 (If Continuing)
- [ ] Continue paper trading
- [ ] Accumulate data (target: 50-100 trades total)
- [ ] Final decision: Sufficient evidence to go live?

---

## Week 4: Live Deployment

### Monday
- [ ] Final checks before going live
  - [ ] Code review complete
  - [ ] Paper trading results reviewed
  - [ ] Risk management confirmed
  - [ ] Monitoring setup ready
  - [ ] Fallback plan documented

- [ ] Deploy to production (PHASED)
  - [ ] Start with 25% capital allocation
  - [ ] Monitor for 2 days
  - [ ] Increase to 50% capital if no issues
  - [ ] Monitor for 1 week
  - [ ] Scale to 100% capital

### Daily (Week 4)
- [ ] Morning: Check overnight performance
  - [ ] Any errors in logs?
  - [ ] Signal generation working?
  - [ ] Trades executed properly?

- [ ] Monitor: Key metrics
  - [ ] Win rate (rolling 20-trade window)
  - [ ] Sharpe ratio (rolling 30-day window)
  - [ ] Maximum drawdown
  - [ ] Execution quality

### Weekly (Week 4+)
- [ ] Compare live results to expectations
  - [ ] Win rate: target 50%+
  - [ ] Sharpe ratio: target 1.4+
  - [ ] Drawdown: should improve vs baseline
  - [ ] Trades/week: expect 5-15 trades

- [ ] Gate check: Can we proceed to Phase 2?
  - ✅ YES → Start Phase 2 preparation
  - ⚠️ MAYBE → Continue Phase 1 longer, monitor
  - ❌ NO → Revert, investigate issues

---

## Monitoring Dashboards

### Daily Dashboard
```
PHASE 1 LIVE MONITORING

Today's Performance:
- Trades Executed: [_____]
- Win Rate (today): [___%]
- P&L (today): [$________]

Week-to-Date:
- Total Trades: [_____]
- Win Rate (WTD): [___%]
- P&L (WTD): [$________]
- Sharpe Ratio: [____]

Comparison to Baseline:
- Baseline Trades: [_____] → Phase 1: [_____]
- Baseline Win Rate: [___%] → Phase 1: [___%]
- Baseline Return: [___%] → Phase 1: [___%]

Risk Metrics:
- Max Drawdown (live): [___%]
- Volatility (live): [___%]
- Expected Drawdown: <12%

Status: ✅ Normal / ⚠️ Caution / ❌ Alert
```

### Signal Quality Monitoring
```
CONFIDENCE DISTRIBUTION (Daily)

Confidence 30-40%: [___] trades, [___%] win rate
Confidence 40-50%: [___] trades, [___%] win rate
Confidence 50-60%: [___] trades, [___%] win rate
Confidence 60-70%: [___] trades, [___%] win rate
Confidence 70-80%: [___] trades, [___%] win rate
Confidence 80-90%: [___] trades, [___%] win rate
Confidence 90%+: [___] trades, [___%] win rate

Target Distribution:
- 80%+ confidence: 10-15% of trades ✅
- 70-80% confidence: 12-18% of trades
- 60-70% confidence: 25-35% of trades
- <60% confidence: filtered out (reject rate 85%+)

Model Agreement:
- 2-3 models: [___] trades ([___%])
- 4-5 models: [___] trades ([___%])
- 6+ models: [___] trades ([___%])

Target: 80%+ of trades from 4+ models
```

---

## Troubleshooting Guide

### Issue: Win Rate Below 45%

**Investigation**:
1. [ ] Check real backtest results
   - Did synthetic match real?
   - Is confidence calculation correct?

2. [ ] Verify signal generation
   - Are models running correctly?
   - Are confidence scores realistic?
   - Is model agreement working?

3. [ ] Test alternative thresholds
   ```bash
   # Test 0.75 threshold instead
   python scripts/comprehensive_backtest.py \
     --enable_confidence_filter \
     --confidence_threshold 0.75
   ```

4. [ ] Check for data quality issues
   - Missing data?
   - Signal generation bugs?
   - Confidence calculation errors?

**Decision**:
- If confidence calculation is fine → Try 0.75 threshold
- If signals are low quality → Review SignalCore models
- If still bad → Skip Phase 1, proceed to Phase 2

### Issue: Extremely Low Trade Count (<50/month)

**Possible Causes**:
1. Confidence threshold too high
2. Model agreement requirement too strict
3. Volatility adjustment too aggressive

**Fix**:
1. [ ] Lower threshold to 0.75
2. [ ] Lower min_models from 4 to 3
3. [ ] Review volatility adjustment logic

### Issue: Large Variance vs Backtest (>10%)

**Possible Causes**:
1. Market conditions changed
2. Signal generation differs from backtest
3. Position sizing implementation different

**Verification**:
1. [ ] Check current market volatility vs backtest period
2. [ ] Verify confidence scores match expectations
3. [ ] Run small backtest with recent data

**Resolution**:
- If volatility is normal: Continue, variance acceptable
- If signals different: Check signal generation code
- If other issue: Escalate to engineering

---

## Phase 2 Preparation (Week 4+)

### Prerequisites Met?
- [ ] Phase 1 live trading for 2+ weeks
- [ ] Win rate ≥ 50%
- [ ] No execution issues
- [ ] Confidence filter stable and effective

### Phase 2 Preparation
- [ ] Create backtest for Phase 1 + Phase 2 together
  ```bash
  python scripts/comprehensive_backtest.py \
    --enable_confidence_filter \
    --enable_regime_adaptive_weights
  ```

- [ ] Create paper trading config for Phase 2
  - [ ] Import RegimeAdaptiveWeights
  - [ ] Implement regime detection
  - [ ] Deploy weight adjustment logic
  - [ ] Paper trade for 1 week

- [ ] Plan Phase 2 deployment
  - [ ] Similar timeline as Phase 1
  - [ ] Backtest → Paper Trade → Live
  - [ ] Expected gain: +2-3% win rate (50% → 52-53%)

---

## Sign-Off Checklist

### Code Review
- [ ] ConfidenceFilter code reviewed by: _____________
- [ ] Integration points verified by: _____________
- [ ] Unit tests passing: _____________

### Backtest Review
- [ ] Real backtest results reviewed by: _____________
- [ ] Metrics meet expectations: _____________
- [ ] Comparison to synthetic backtest: _____________

### Paper Trading Review
- [ ] 2+ weeks completed: _____________
- [ ] Results meet 50%+ win rate target: _____________
- [ ] No critical issues found: _____________

### Risk Management
- [ ] Risk limits set correctly: _____________
- [ ] Position sizing verified: _____________
- [ ] Stop loss logic confirmed: _____________

### Live Deployment
- [ ] All checks passed: _____________
- [ ] Monitoring setup ready: _____________
- [ ] Fallback plan documented: _____________
- [ ] Team trained on Phase 1: _____________

### Approval
- [ ] Technical Lead: _________________ Date: _______
- [ ] Risk Manager: _________________ Date: _______
- [ ] Trading Manager: _________________ Date: _______

---

## Emergency Procedures

### If Critical Issues Found

1. **Immediate**: Disable Phase 1
   ```python
   # In trading engine
   PHASE_1_ENABLED = False
   ```

2. **Within 1 hour**: Assess impact
   - [ ] How many trades affected?
   - [ ] Any losses incurred?
   - [ ] What was the error?

3. **Within 4 hours**: Fix and retest
   - [ ] Identify root cause
   - [ ] Implement fix
   - [ ] Test in paper trading
   - [ ] Get code review approval

4. **Redeploy**: Once fixed
   - [ ] Start with 10% capital
   - [ ] Monitor for 1 day
   - [ ] Gradually scale up

### If Win Rate Drops Below 45%

1. **Check**: Is this temporary?
   - [ ] Review last 20-30 trades
   - [ ] Check market conditions
   - [ ] Verify signal quality

2. **If Persistent** (<45% over 50 trades):
   - [ ] Switch to 0.75 threshold
   - [ ] OR disable Phase 1, keep baseline
   - [ ] Investigate signal generation

3. **If Unexplained**:
   - [ ] Escalate to engineering
   - [ ] Run detailed diagnostic
   - [ ] Don't deploy Phase 2 yet

---

## Success Criteria Summary

| Milestone | Target | Acceptable | Action |
|-----------|--------|-----------|--------|
| **Real Backtest** | WR ≥50% | WR 47-50% | 50%+ → paper trade |
| **Paper Trading** | WR ≥50% | WR 48-50% | 50%+ → go live |
| **Live (Week 1)** | WR ≥50% | WR 48-50% | 50%+ → continue |
| **Live (Week 2)** | WR ≥50% | WR 48-50% | 50%+ → scale 100% |
| **Live (Week 3)** | WR ≥50% | WR 48-50% | 50%+ → Phase 2 |

---

## Timeline Summary

```
Week 1:     Real backtest → decision point
Week 2-3:   Paper trading → validation
Week 4:     Live deployment → scaling
Week 5+:    Phase 2 preparation
```

**Total Time to Full Deployment**: 4 weeks
**Expected Impact**: +6-8% win rate improvement
**Risk Level**: LOW (extensive testing, fallback options)

---

**Print & post at desk during deployment**

Last Updated: December 15, 2025
Next Review: After real backtest results
