# Expert Options Strategies Skill Card

## Quick Reference

**Skill Name:** options-strategies  
**Version:** 1.0.0  
**Category:** Quantitative Finance | Options Trading  
**Complexity:** Advanced  
**Integration:** Alpaca Markets API

## Overview

Expert-level framework for designing, modeling, and executing advanced multi-leg options strategies with institutional-grade quantitative analysis and risk management. Covers complete workflow from strategy design through automated execution and monitoring.

## Core Capabilities

### Strategy Design
- **Neutral Strategies:** Long straddle, long strangle
- **Volatility Compression:** Iron butterfly, iron condor
- **Time Decay:** Calendar spreads, diagonal spreads
- **Directional:** Bull/bear call spreads, bull/bear put spreads

### Quantitative Modeling
- **Pricing:** Black-Scholes model implementation
- **Greeks:** Delta, gamma, theta, vega, rho calculation
- **Analysis:** Payoff diagrams, breakeven calculation, profit/loss surfaces
- **Forecasting:** Expected move calculation, Monte Carlo simulation

### Risk Management
- **Monitoring:** Real-time position tracking, aggregate Greeks
- **Controls:** Automated stop-loss, delta hedging, position limits
- **Metrics:** Maximum profit/loss, risk/reward ratios, probability analysis

### Execution & Automation
- **API Integration:** Alpaca REST and WebSocket APIs
- **Order Management:** Multi-leg order submission, fill tracking
- **Backtesting:** Historical performance analysis, parameter optimization
- **Automation:** Strategy scanning, entry/exit logic, position management

## File Structure

```
options-strategies/
├── SKILL.md                   # Complete skill documentation (36KB)
├── README.md                  # Quick start and usage guide
├── SKILL-CARD.md             # This file - quick reference
├── references/               # Reference materials
│   ├── greeks.md            # Greeks calculation reference
│   ├── strategies.md        # Strategy payoff formulas
│   ├── volatility.md        # IV analysis and forecasting
│   └── sources.md           # Authoritative source catalog
└── scripts/                  # Python utilities
    ├── __init__.py          # Package initialization
    ├── option_pricing.py    # Black-Scholes and Greeks
    └── strategy_builder.py  # Multi-leg strategy construction
```

## Quick Start

### 1. Environment Setup
```bash
pip install alpaca-trade-api numpy pandas matplotlib scipy yfinance
```

### 2. API Configuration
```python
from alpaca_trade_api import REST
import os

api = REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url='https://paper-api.alpaca.markets'
)
```

### 3. Build a Strategy
```python
from strategy_builder import StrategyBuilder

# Long straddle on SPY
straddle = StrategyBuilder.long_straddle(
    underlying_price=450,
    strike=450,
    call_premium=8.0,
    put_premium=7.5
)

summary = straddle.summary()
print(f"Max Profit: ${summary['max_profit']:.2f}")
print(f"Max Loss: ${summary['max_loss']:.2f}")
print(f"Breakevens: {summary['breakeven_points']}")
```

### 4. Calculate Greeks
```python
from option_pricing import BlackScholesCalculator, OptionParameters

params = OptionParameters(
    S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type='call'
)

calc = BlackScholesCalculator()
greeks = calc.all_greeks(params)
print(f"Delta: {greeks['delta']:.4f}")
print(f"Theta: ${greeks['theta']:.4f} per day")
```

## Strategy Selection Guide

| Market Condition | IV Environment | Recommended Strategy | Key Characteristic |
|-----------------|----------------|---------------------|-------------------|
| Uncertain direction | IV Rank > 70 | Iron Butterfly | Max profit at ATM |
| Expect large move | IV Rank < 30 | Long Straddle | Unlimited profit potential |
| Range-bound | IV Rank > 60 | Iron Condor | Profit from stability |
| Bullish | IV Rank < 50 | Bull Call Spread | Limited risk directional |
| Bearish | IV Rank < 50 | Bear Put Spread | Limited risk directional |
| Time decay | Variable | Calendar Spread | Exploit theta differential |

## Key Metrics Reference

### Greeks Quick Reference
- **Delta (Δ):** Price sensitivity (±$1 stock move)
- **Gamma (Γ):** Delta acceleration
- **Theta (Θ):** Daily time decay ($/day)
- **Vega (ν):** IV sensitivity (per 1% IV change)
- **Rho (ρ):** Interest rate sensitivity

### Risk Calculations
- **Expected Move:** `Stock Price × IV × √(Days/365)`
- **Max Profit:** Depends on strategy structure
- **Max Loss:** Premium paid (debits) or Strike Width - Credit (spreads)
- **Breakeven:** Strike ± Total Premium (straddle/strangle)

## Integration Points

### With Other Ordinis-1 Skills
- **portfolio-management:** Track options positions, aggregate Greeks
- **due-diligence:** Research underlying companies before trading
- **benchmarking:** Compare strategy performance against indices

### With External Systems
- **Alpaca API:** Real-time execution and data
- **Yahoo Finance:** Historical data and research
- **TradingView:** Technical analysis and charting

## Learning Path

**Week 1:** Foundations & Environment Setup  
→ Options basics, Greeks, Alpaca API configuration

**Week 2:** Neutral Volatility Strategies  
→ Straddles, strangles, volatility analysis

**Week 3:** Multi-Leg Structures  
→ Butterflies, condors, spreads

**Week 4:** Quantitative Modeling  
→ Black-Scholes, Greeks calculation, simulations

**Week 5:** Risk Management  
→ Position monitoring, automated controls

**Week 6:** Automation & Backtesting  
→ Strategy deployment, performance optimization

## Authoritative Sources

### Primary References
- [Alpaca API Documentation](https://docs.alpaca.markets/)
- [Alpaca Learn Center](https://alpaca.markets/learn/)
- [Alpaca GitHub](https://github.com/alpacahq)
- [Options Industry Council](https://www.optionseducation.org/)
- [CBOE Education](https://www.cboe.com/education/)

### Academic References
- "Options, Futures, and Other Derivatives" - John Hull
- "Option Volatility and Pricing" - Sheldon Natenberg
- "Dynamic Hedging" - Nassim Taleb

## Risk Warnings

- Options trading involves substantial risk of loss
- Always start with paper trading
- Never risk more than 2% of capital per trade
- Understand maximum loss before entering positions
- Monitor positions regularly, especially near expiration
- Account for transaction costs in all calculations

## Support & Community

- **Alpaca Forum:** https://forum.alpaca.markets/
- **Alpaca Slack:** https://alpaca.markets/slack
- **GitHub Issues:** https://github.com/alpacahq/alpaca-trade-api-python/issues

## Usage in Claude

Import this skill in Claude conversations:

```
"Using the options-strategies skill, help me design an iron butterfly 
for SPY with current IV rank of 75%"
```

Claude will apply:
- Strategy selection logic based on market conditions
- Quantitative analysis using Black-Scholes and Greeks
- Risk assessment and position sizing recommendations
- Code generation for strategy implementation
- Integration with Alpaca API for execution

---

**Version:** 1.0.0  
**Last Updated:** December 2024  
**Maintainer:** Ordinis-1 Project  
**License:** Educational Use

*This skill is designed for educational and research purposes. All examples use Alpaca's paper trading environment.*
