# Getting Started with Ordinis

**Estimated read time:** 5 minutes  
**Level:** Beginner

---

## What is Ordinis?

Ordinis is a **professional algorithmic trading system** that combines:
- **AI-powered signal generation** (multi-model consensus)
- **Intelligent risk management** (position limits, drawdown controls)
- **Rigorous backtesting** (proven 59.7% win rate on real market data)
- **Governance framework** (audit trails, compliance, ethics)

Think of it as: *"A disciplined trading bot that only takes high-confidence trades after rigorous validation."*

---

## Key Numbers (Proven Performance)

**Real Market Backtest (2019-2024, 22 stock symbols):**

| Metric | Performance | Status |
|--------|-------------|--------|
| **Trades Generated** | 6,424 | ✓ Real data |
| **High-Confidence Selected** | 231 (96.4% reduction) | ✓ Filters noise |
| **Win Rate** | 59.7% | ✓ Above 50% benchmark |
| **Sharpe Ratio** | 2.56 | ✓ Excellent risk-adjusted returns |
| **Annualized Return** | 8.4%/year | ✓ Consistent with S&P 500 |

→ **Details:** [Performance Results](../performance/index.md)

---

## 5-Minute System Overview

### 1. **Signals** (What to trade?)
- Multi-model consensus voting (6 independent models)
- Models analyze: fundamentals, sentiment, technical patterns, volume profiles
- Output: Trading signal with confidence score (0-100%)

### 2. **Risk Management** (Can we afford it?)
- Position sizing: Bigger for high-confidence, smaller for risky
- Portfolio limits: Max 10% per stock, 100% total
- Drawdown protection: Kill switch if losses exceed threshold

### 3. **Execution** (Place the order)
- Simulated (paper trading) or live (Alpaca Markets)
- Realistic slippage and commission modeling
- Order tracking: Created → Submitted → Filled → Closed

### 4. **Learning** (Did it work?)
- Track win/loss for every trade
- Calibration engine learns which confidence scores are reliable
- Continuous feedback loop improves future predictions

---

## How to Get Started

### Option A: Run a Backtest (Fastest - 10 minutes)

```bash
# 1. Activate environment
cd c:\Users\kjfle\Workspace\ordinis
.venv\Scripts\Activate.ps1

# 2. Run demo backtest (synthetic data)
python scripts/phase1_confidence_backtest.py

# 3. View results
cat reports/phase1_confidence_backtest_report.json
```

**What you'll see:** 1,000 simulated trades with win rate analysis and confidence distribution

→ **Next:** [Interpreting Your Results](../guides/interpreting-results.md)

### Option B: Run Live Backtest (Real market data - 30 minutes)

```bash
# Requires: Real market data from Yahoo Finance
python scripts/phase1_real_market_backtest.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --symbols AAPL MSFT GOOGL
```

→ **Next:** [Advanced Backtesting](../guides/running-backtests.md)

### Option C: Explore the Code (Deep dive)

Start with the key engines:
- **Signal Generation:** `src/SignalCore.py` (200 lines)
- **Risk Management:** `src/RiskGuard.py` (300 lines)
- **Execution:** `src/FlowRoute.py` (250 lines)

→ **Next:** [Architecture Overview](../architecture/index.md)

---

## System Requirements

| Requirement | What you need |
|-------------|--------------|
| **Python** | 3.11+ (we use 3.11.x) |
| **Memory** | 4GB+ (8GB recommended) |
| **Data** | Yahoo Finance (free) or Alpaca Markets API key |
| **Time** | 5 min for synthetic demo, 30 min for real backtest |

**Installation:** [Development Setup](dev-setup.md)

---

## Key Concepts

### Signal Confidence Score (0-100%)

- **0-30%:** Low confidence → Skip trade (too risky)
- **30-70%:** Medium confidence → Take trade with caution
- **70-100%:** High confidence → Full position size

*In real testing: High-confidence trades have 59.7% win rate vs. 57.8% baseline*

### Regime (Market State)

The system detects: **Bullish → Sideways → Bearish**

- **Bull market:** Signals skew bullish (buy signals prioritized)
- **Bear market:** Signals skew bearish (sell signals prioritized)
- **Sideways:** Volume-based mean-reversion trades

### Calibration (How trustworthy are our confidence scores?)

The system uses **ML-based probability calibration** to answer: *"When we say 80% confidence, are we right 80% of the time?"*

- **Brier Score:** 0.2434 (measures accuracy, lower is better)
- **Accuracy:** 57.8% (baseline performance)

→ **Details:** [Calibration Deep Dive](../performance/calibration.md)

---

## Next Steps

### I want to... | Then read...
|---|---|
| **Understand the system architecture** | [Architecture Overview](../architecture/index.md) |
| **Run my own backtest** | [Running Backtests Guide](../guides/running-backtests.md) |
| **See the performance results** | [Performance & Results](../performance/index.md) |
| **Learn the trading workflow** | [Trading Workflow](../fundamentals/workflow.md) |
| **Deploy to live trading** | [Deployment Guide](../guides/deployment.md) (coming soon) |
| **Contribute to the project** | [Developer Setup](dev-setup.md) |

---

## FAQ

**Q: Is this a guaranteed money maker?**  
A: No. Like all trading systems, past performance doesn't guarantee future results. We provide transparency, rigorous testing, and conservative filtering to improve odds, but market risk is always present.

**Q: Do I need to understand ML?**  
A: No. The system handles ML complexity. You just need to understand the concepts: signals, risk, execution, and calibration (all explained above).

**Q: Can I trade crypto/futures/forex?**  
A: Currently built for equities (stocks). Crypto and futures are planned for Phase 2.

**Q: How often should I run backtests?**  
A: Whenever you change strategy logic (signals, risk rules). We recommend weekly to catch market regime changes.

**Q: Where can I ask questions?**  
A: Check [Internal Documentation](../internal/index.md) for developer guides, or see the `CONTRIBUTING.md` for community guidelines.

---

## You're Ready! 🚀

Pick one of the three options above and run it now. You'll have your first Ordinis backtest in minutes.

**Stuck?** → See [Troubleshooting](troubleshooting.md)  
**Want to dive deeper?** → [Fundamentals Guide](../fundamentals/index.md)
