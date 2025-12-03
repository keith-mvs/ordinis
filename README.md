# Intelligent Investor

**Version:** 0.2.0-dev (Development Build)
**Status:** ✅ Ready for Testing

An AI-driven quantitative trading system with multi-source market data, paper trading simulation, and ML-driven signal generation.

## Project Overview

Complete end-to-end trading pipeline featuring:
- **4 Market Data APIs** (Alpha Vantage, Finnhub, Polygon/Massive, Twelve Data)
- **Paper Trading Engine** with realistic fill simulation
- **Backtesting Framework** (ProofBench)
- **5 Trading Strategies** ready to deploy
- **Real-time Dashboard** for monitoring
- **RiskGuard Framework** for risk management

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Knowledge Base & Strategy Design | ✅ 100% Complete |
| 2 | Code Implementation & Backtesting | ✅ 95% Complete |
| 3 | Paper Trading & Simulation | ✅ 90% Complete |
| 4 | Risk Management | ⚠️ 50% Complete |
| 5 | System Integration | ⚠️ 60% Complete |
| 6 | Production Preparation | ❌ 10% Planned |

## Quick Start

### Run Full System Demo
```bash
# Set up environment
cp .env.example .env
# Add your API keys to .env

# Run end-to-end demo
python scripts/demo_full_system.py
```

**Expected output:**
- 3 data sources initialized
- Live market data fetched (AAPL, MSFT, GOOGL)
- 2 trading signals generated
- Orders filled with realistic slippage
- Final P&L displayed

### Test Market Data APIs
```bash
python scripts/test_market_data_apis.py
```

### Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

## Repository Structure

```
intelligent-investor/
├── README.md                    # This file
├── PROJECT_STATUS_CARD.md       # Detailed status report
├── docs/                        # Documentation and knowledge base
│   ├── knowledge-base/          # Core KB sections (10 domains)
│   ├── strategies/              # Strategy specifications
│   └── architecture/            # System design documents
├── src/                         # Source code
│   ├── engines/                 # Core engines (ProofBench, RiskGuard, etc.)
│   ├── plugins/market_data/     # 4 API integrations
│   ├── strategies/              # 5 implemented strategies
│   └── dashboard/               # Streamlit monitoring dashboard
├── scripts/                     # Executable scripts and demos
│   ├── demo_full_system.py      # Full pipeline demonstration
│   ├── test_market_data_apis.py # API validation tests
│   └── test_live_trading.py     # Paper trading tests
├── tests/                       # Test suites (413 tests, 67% coverage)
└── data/                        # Sample market data
```

## Features

### Market Data Integration
- **Alpha Vantage**: 25 calls/day, comprehensive fundamentals
- **Finnhub**: 60 calls/min, real-time quotes + news
- **Polygon/Massive**: Market data and status
- **Twelve Data**: 800 calls/day, technical indicators

### Paper Trading
- Realistic order fill simulation with slippage (5 bps)
- Commission calculation ($0.005/share)
- Position tracking with real-time P&L
- Integration with live market data
- Pending order management

### Backtesting
- Event-driven simulation engine (ProofBench)
- Performance metrics: Sharpe, Sortino, Max Drawdown
- Equity curve tracking
- Walk-forward validation ready

### Strategies
1. Moving Average Crossover (50/200 SMA)
2. RSI Mean Reversion
3. Momentum Breakout
4. Bollinger Bands
5. MACD

### Dashboard
- Real-time position monitoring
- P&L visualization
- Trade history
- Performance metrics
- Multi-timeframe analysis

## Disclaimer

**IMPORTANT:**
- All trading involves risk of loss. There are no guarantees of profit.
- Past performance in backtests does not assure future results.
- This system is a research and engineering project, not personalized financial advice.
- The authors and contributors are not licensed financial advisors.

## License

[To be determined]
