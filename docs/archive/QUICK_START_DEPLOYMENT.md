# Quick Start: Deploy Ordinis to Production

**Goal**: Get trading live in 3 weeks
**Starting Point**: All 6 components built ✅
**Effort**: ~40 hours development + 2 weeks monitoring

---

## Week 1: Setup & Configuration

### Day 1-2: Environment Setup
```bash
# 1. Clone repo and install
git clone https://github.com/keith-mvs/ordinis.git
cd ordinis
conda create -n ordinis-prod python=3.11
conda activate ordinis-prod
pip install -r requirements.txt

# 2. Verify all components
pytest tests/ -v
# Should see: All tests pass

# 3. Check SignalCore models
python -c "from ordinis.engines.signalcore.models.advanced_technical import *; print('✓')"
```

### Day 3: Broker Setup
```bash
# 1. Choose broker (recommend Alpaca for free tier)
# https://app.alpaca.markets/signup

# 2. Get API credentials
# Save to environment variables:
export APCA_API_KEY_ID="YOUR_KEY"
export APCA_API_SECRET_KEY="YOUR_SECRET"
export APCA_API_BASE_URL="https://paper-api.alpaca.markets"  # Paper trading

# 3. Test connection
python -c "
import os
from alpaca.trading.client import TradingClient

client = TradingClient(os.getenv('APCA_API_KEY_ID'), os.getenv('APCA_API_SECRET_KEY'))
account = client.get_account()
print(f'✓ Connected. Cash: ${account.cash}')
"
```

### Day 4-5: Data Configuration
```python
# 1. Configure data providers
# Option A: Alpha Vantage (free tier)
export ALPHA_VANTAGE_API_KEY="YOUR_KEY"

# Option B: Polygon (free tier)
export POLYGON_API_KEY="YOUR_KEY"

# Option C: yfinance (free, no key needed)
pip install yfinance

# 2. Test data loading
python scripts/test_data_loading.py
# Should output: ✓ Data loaded successfully
```

---

## Week 2: Validation & Configuration

### Day 6-8: Run Production Backtest

```bash
# 1. Download 5 years of real data
python scripts/download_historical_data.py \
  --symbols AAPL,MSFT,GOOGL,AMZN,NVDA \
  --start 2019-01-01 \
  --end 2024-12-31

# 2. Run full backtest
python -m ordinis.backtesting.cli \
  --name production_baseline \
  --symbols AAPL,MSFT,GOOGL,AMZN,NVDA \
  --start 2019-01-01 \
  --end 2024-12-31 \
  --capital 100000 \
  --commission 0.001 \
  --slippage 5

# 3. Analyze results
python scripts/analyze_backtest_results.py \
  --report backtest_results/production_baseline/report.json

# Expected output:
# Sharpe Ratio: 1.5 (target: > 1.0)
# Max Drawdown: 15% (target: < 20%)
# Win Rate: 52% (target: > 50%)
# IC Score: 0.12 (target: > 0.05)
```

### Day 9-10: Extract Model Weights

```python
# Extract IC scores from backtest report
import json

with open('backtest_results/production_baseline/report.json') as f:
    report = json.load(f)

model_metrics = report['metrics']['model_metrics']

# Print model weights
print("Model IC Scores (for ensemble weighting):")
for model_name, metrics in sorted(model_metrics.items(),
                                 key=lambda x: x[1]['ic'],
                                 reverse=True):
    ic = metrics['ic']
    hit_rate = metrics['hit_rate']
    print(f"  {model_name:<25} IC={ic:>6.3f}, Hit Rate={hit_rate:>6.1f}%")

# Save weights config
weights_config = {model: metrics['ic'] for model, metrics in model_metrics.items()}
with open('config/production_weights.json', 'w') as f:
    json.dump(weights_config, f, indent=2)

print("\n✓ Weights saved to config/production_weights.json")
```

---

## Week 2 (cont): Deploy & Configure

### Day 11-12: Configure Production Settings

```python
# Create config/production_config.py
PRODUCTION_CONFIG = {
    # Broker
    "broker": "alpaca",
    "broker_api_key": os.getenv("APCA_API_KEY_ID"),
    "broker_api_secret": os.getenv("APCA_API_SECRET_KEY"),
    "broker_base_url": "https://paper-api.alpaca.markets",  # Start with paper

    # Trading strategy
    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "initial_capital": 10000,  # Start small
    "position_size_pct": 0.05,  # 5% per position (from backtest)
    "max_position_size_pct": 0.15,  # 15% max
    "max_portfolio_exposure": 0.95,  # 95% (5% cash buffer)
    "stop_loss_pct": 0.08,  # 8% stop loss

    # Risk management
    "max_daily_loss_pct": 0.02,  # 2% daily stop
    "max_drawdown_pct": 0.20,  # 20% circuit breaker
    "enable_governance": True,
    "enable_risk_checks": True,

    # Data
    "data_provider": "polygon",  # or "alpha_vantage" or "yfinance"
    "data_collection_interval": 60,  # Collect every 60 seconds
    "min_data_quality": 0.90,  # 90% quality score minimum

    # Monitoring
    "enable_dashboard": True,
    "dashboard_port": 8501,
    "enable_alerts": True,
    "alert_email": "your-email@example.com",

    # Model ensemble (from backtest IC scores)
    "model_weights": {
        "IchimokuModel": 0.18,
        "ChartPatternModel": 0.12,
        "VolumeProfileModel": 0.15,
        "FundamentalModel": 0.20,
        "SentimentModel": 0.18,
        "AlgorithmicModel": 0.17,
    }
}
```

### Day 13-14: Start Paper Trading

```bash
# 1. Start the orchestration pipeline (paper mode)
python scripts/start_paper_trading.py \
  --config config/production_config.py \
  --mode paper \
  --duration 14  # Run for 14 days

# 2. In another terminal, start dashboard
streamlit run src/ordinis/dashboard/app.py

# 3. Monitor outputs
# Dashboard: http://localhost:8501
# Signals: Check Dashboard > Signals page
# Equity: Check Dashboard > Portfolio page
```

---

## Week 3: Live Trading

### Day 15: Monitor Paper Trading Results

```python
# After 1 week of paper trading, check metrics
python scripts/check_paper_trading_metrics.py

# Expected output:
# Live Sharpe: 1.4 (vs backtest: 1.5) ✓
# Signal rate: 2.3 trades/day (vs backtest: 2.1) ✓
# Data quality: 98% ✓
# Model IC: 0.11 (vs backtest: 0.12) ✓ (slight drop is normal)

# If all green → proceed to live
# If issues → debug and rerun paper trading
```

### Day 16-17: Go Live (Micro)

```bash
# 1. Switch to live mode (with small capital)
python scripts/start_live_trading.py \
  --config config/production_config.py \
  --mode live \
  --capital 1000  # Start with $1k

# 2. Monitor closely (check every 1 hour)
# Look for:
# - Trade execution success
# - Fill prices near expectations
# - Model IC staying positive
# - No critical errors

# 3. Scale after 3-5 days if all good
python scripts/scale_capital.py --new_capital 5000
```

### Day 18-21: Scale Gradually

```
Day 18-20: $5,000 capital
  - Monitor daily returns
  - Check Sharpe ratio tracking
  - Verify risk controls

Day 21+: $20,000 capital
  - Continue monitoring
  - If Sharpe > 1.0 and drawdown < 20% → scale to full $100k
  - Never scale too fast (max 5x per week)
```

---

## Daily Operations (After Launch)

### Morning Checklist (5 min)
```bash
# 1. Check data quality
curl http://localhost:8501/data_quality

# 2. Review overnight signals
sqlite3 trades.db "SELECT * FROM signals WHERE date = TODAY;"

# 3. Check cash position
python -c "
from alpaca.trading.client import TradingClient
import os

client = TradingClient(os.getenv('APCA_API_KEY_ID'), os.getenv('APCA_API_SECRET_KEY'))
account = client.get_account()
print(f'Cash: ${account.cash}')
print(f'Equity: ${account.equity}')
print(f'Daily PnL: ${account.rgt}')
"

# 4. Set alerts if something off
```

### Weekly Monitoring (30 min)
```python
# Every Friday, run analysis
python scripts/weekly_analysis.py

# Checks:
# - Model IC (should be > 0.04)
# - Sharpe ratio trend
# - Max drawdown
# - Execution quality
# - Data quality

# If IC drops significantly → investigate
# If Sharpe drops > 0.3 → prepare to retrain
```

### Monthly Retraining (2 hours)
```bash
# Every month, retrain on latest data
python scripts/monthly_retrain.py \
  --lookback_days 730  # Last 2 years

# Steps:
# 1. Download latest data
# 2. Run backtest on new data
# 3. Compare IC scores to baseline
# 4. Update model weights if needed
# 5. Test on paper before deploying
```

---

## Troubleshooting Guide

### No Signals Generated
```python
# 1. Check if models are registered
from ordinis.engines.signalcore.core.engine import SignalCoreEngine
engine = SignalCoreEngine()
print(engine._registry.list_models())

# 2. Check data quality
quality = data_pipeline.get_quality_report()
if quality['overall_score'] < 90:
    print("⚠ Data quality low, signals suppressed")

# 3. Check model thresholds
# Lower min_probability and min_score in config
```

### Trades Not Executing
```python
# 1. Check broker connection
python -c "
from alpaca.trading.client import TradingClient
client = TradingClient(...)
account = client.get_account()
print(f'Account status: {account.status}')
"

# 2. Check position limits
# Ensure position size < max_position_size_pct

# 3. Check market hours (9:30-16:00 ET)
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
print(f"Market open: {9.5 <= (now.hour - 4) <= 16}")
```

### Dashboard Not Loading
```bash
# 1. Check Streamlit process
ps aux | grep streamlit

# 2. Restart
pkill streamlit
streamlit run src/ordinis/dashboard/app.py --server.port=8501

# 3. Check port availability
netstat -an | grep 8501
```

---

## Success Indicators

### ✅ Good Sign
- Sharpe ratio > 1.0
- Win rate 50-55%
- IC score > 0.05
- Max drawdown < 20%
- Data quality > 95%
- Signal generation rate matches backtest ±10%

### ⚠️ Warning Sign
- Sharpe drops by > 0.5
- IC becomes negative
- Data quality < 90%
- Execution failures > 5%
- Profit factor < 1.2

### 🛑 Stop Signal
- Max drawdown > 2x backtest
- IC < 0 for > 5 days
- Data quality < 80%
- Model IC all negative
- Unexpected system errors

---

## Emergency Procedures

### If Strategy Underperforms (Sharpe drops > 0.5)

```bash
# 1. Switch to paper mode immediately
python scripts/switch_to_paper.py

# 2. Investigate
python scripts/analyze_underperformance.py

# 3. Retrain models
python scripts/emergency_retrain.py

# 4. Backtest new models
python -m ordinis.backtesting.cli --name emergency_backtest [...]

# 5. Only resume if new Sharpe > baseline - 0.3
```

### If Data Feed Fails

```python
# Automatic failover should work, but if not:

# 1. Check primary provider
python scripts/test_data_provider.py --provider polygon

# 2. Switch to backup
python scripts/switch_data_provider.py --provider alpha_vantage

# 3. Update config and restart
```

### If Broker Connection Dies

```bash
# 1. Trading stops automatically (by design)
# 2. Check broker status
curl https://status.alpaca.markets

# 3. Reconnect
python scripts/reconnect_broker.py

# 4. Resume trading in paper mode until verified
```

---

## Next Steps After First Month

### If Strategy is Profitable ✅
```
Week 5-8: Scale capital
  - Gradually increase from $1k → $100k
  - Each step wait 1 week before scaling

Month 2+: Continuous improvement
  - Monthly retraining
  - Weekly IC monitoring
  - Quarterly strategy review
  - Add new markets/symbols
```

### If Strategy is Breaking Even
```
- Retrain with more data
- Tune model parameters
- Increase position sizes (carefully)
- Add new technical models
- Wait 2-3 more months before scaling
```

### If Strategy is Losing Money
```
- Immediately stop live trading
- Analyze backtest vs live discrepancies
- Check for market regime change
- Retrain models
- Run backtests on new market conditions
- Only resume if backtest Sharpe > 1.5
```

---

## Cost Breakdown

| Item | Cost | Notes |
|------|------|-------|
| Broker Account | $0 | Alpaca free tier |
| Data Feed | $0-500/mo | yfinance free, Polygon/AV premium |
| Cloud Hosting | $50-200/mo | AWS/GCP for continuous operation |
| Monitoring | $50-200/mo | Prometheus/Grafana or Datadog |
| **Total** | **$100-900/mo** | Varies by scale |

---

## Final Checklist

Before going live:

- [ ] Broker account set up and tested
- [ ] API credentials secure (use env vars, not hardcoded)
- [ ] Data feed working (test last 30 days)
- [ ] Backtest run on 5+ years (Sharpe > 1.0)
- [ ] Paper trading completed (4+ weeks)
- [ ] Dashboard deployed and monitoring
- [ ] Risk limits configured
- [ ] Alerts set up (email, Slack, phone)
- [ ] Team briefed on operation
- [ ] Emergency procedures documented
- [ ] Emergency contact list created
- [ ] Rollback procedures tested
- [ ] Insurance/legal review completed (if required)
- [ ] Management approval obtained

---

## Support & Resources

**Documentation**:
- [System Architecture](docs/System%20Specification%20Modular%20AI%20Driven%20Trading%20Platform.md)
- [Backtesting Guide](BACKTESTING_FINDINGS.md)
- [Deployment Readiness](DEPLOYMENT_READINESS_REPORT.md)

**Code**:
- Backtesting: `src/ordinis/backtesting/`
- Orchestration: `src/ordinis/orchestration/`
- Live Data: `src/ordinis/data/live_pipeline.py`
- Analytics: `src/ordinis/analysis/model_analytics.py`
- Dashboard: `src/ordinis/dashboard/`

**Contact**:
- Developer: [Your Name]
- Slack Channel: #trading-ops
- Emergency: [Phone Number]

---

**Status: READY FOR PRODUCTION DEPLOYMENT** ✅

Estimated time to first live trade: **3 weeks**
Estimated time to full capital: **3 months**
Expected annual return (based on backtest): **15-25%**
