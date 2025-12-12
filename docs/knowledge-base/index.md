# Ordinis Knowledge Base

Foundational trading knowledge for automated strategy development.

---

## Overview

This Knowledge Base (KB) provides foundational trading knowledge for automated strategy development. Content is organized by **strategy formulation workflow** for efficient retrieval during strategy design, backtesting, and deployment.

**Source Philosophy**: Academic/scholarly sources and peer-reviewed research for core logic. Supplement with credible industry publications and regulatory documents.

---

## Quick Navigation

| Phase | Folder | Use Case |
|-------|--------|----------|
| Learn Markets | `01_foundations/` | Understanding market mechanics |
| Generate Signals | `02_signals/` | Building entry/exit logic |
| Manage Risk | `03_risk/` | Position sizing, stop losses |
| Design Strategy | `04_strategy/` | Backtesting, validation |
| Build System | `05_execution/` | Implementation, automation |
| Trade Options | `06_options/` | Derivatives-specific |
| Cite Sources | `07_references/` | Academic validation |

---

## KB Structure

```
knowledge-base/
├── index.md                    # This file
├── 01_foundations/             # Market structure, microstructure, math
├── 02_signals/                 # Signal generation methods
│   ├── technical/              # Technical analysis indicators
│   ├── fundamental/            # Fundamental analysis
│   ├── volume/                 # Volume & liquidity
│   ├── sentiment/              # News & social sentiment
│   ├── events/                 # Event-driven strategies
│   ├── quantitative/           # Quant strategies & ML
│   └── 10_mathematical_foundations/  # Statistical & mathematical concepts
├── 03_risk/                    # Risk management & position sizing
├── 04_strategy/                # Strategy design & evaluation
├── 05_execution/               # System architecture & execution
├── 06_options/                 # Options & derivatives
└── 07_references/              # Academic sources & citations
```

---

## Section 1: Foundations

Understanding markets at mechanical and mathematical levels.

**Location**: `01_foundations/`

**Key Concepts**:
1. **Market Structure**: Exchange types, order routing, market makers, ECNs
2. **Order Types**: Market, limit, stop, stop-limit, trailing, conditional
3. **Price Formation**: Bid-ask spread, order book dynamics, price discovery
4. **Trade Execution**: Fills, partial fills, slippage, execution quality
5. **Market Sessions**: Pre-market, regular, after-hours, auction periods
6. **Settlement**: T+1/T+2 cycles, clearing, margin requirements
7. **Corporate Actions**: Dividends, splits, mergers, spinoffs
8. **Circuit Breakers**: Halts, limit up/down, volatility pauses
9. **Mathematical Foundations**: Probability, stochastic processes, time series
10. **Regulatory Framework**: SEC, FINRA, CFTC rules

**Academic References**:
- Market Microstructure Theory (O'Hara, 1995)
- Shreve - Stochastic Calculus for Finance I & II
- Hamilton - Time Series Analysis

---

## Section 2: Signal Generation

All methods for generating trading signals.

**Location**: `02_signals/`

### 2.1 Technical Analysis (`technical/`)

Chart-based methods for rule-based entry/exit signals.

```
technical/
├── README.md                    # Section overview
├── overlays/                    # Price overlay indicators
│   ├── moving_averages.md       # SMA, EMA, WMA, KAMA, VWAP
│   └── bollinger_bands.md       # BB, %B, Bandwidth
├── oscillators/                 # Bounded momentum indicators
│   ├── rsi.md                   # RSI, divergence, Stoch RSI
│   ├── stochastic.md            # %K, %D, crossovers
│   ├── cci.md                   # Commodity Channel Index
│   └── williams_r.md            # Williams %R
├── trend_indicators/            # Trend strength & direction
│   ├── adx_dmi.md               # ADX, +DI, -DI
│   ├── parabolic_sar.md         # Trailing stops
│   └── aroon.md                 # Time-based trend
├── volatility/                  # Volatility measures
│   ├── atr.md                   # Average True Range
│   └── implied_realized.md      # IV vs RV, VIX
├── composite/                   # Multi-component indicators
│   ├── macd.md                  # MACD, Signal, Histogram
│   └── momentum.md              # ROC, MOM, TRIX, TSI
├── patterns/                    # Price patterns
│   ├── candlestick.md           # Doji, Hammer, Engulfing
│   ├── chart_patterns.md        # H&S, Triangles, Flags
│   └── support_resistance.md    # S/R levels, Fibonacci
└── advanced/                    # Advanced techniques
    ├── multi_timeframe.md       # MTF analysis
    └── regime_detection.md      # Market regime identification
```

### 2.2 Fundamental Analysis (`fundamental/`)

Company fundamentals and macroeconomic context.

**Key Concepts**:
- Financial Statements: Income, balance sheet, cash flow
- Profitability: Margins, ROE, ROA, ROIC
- Growth: Revenue growth, earnings growth, guidance
- Valuation: P/E, P/B, P/S, EV/EBITDA, PEG
- Financial Health: Debt ratios, coverage, liquidity
- Quality: Earnings quality, accruals, red flags
- Sector Analysis: Rotation, relative strength
- Macro: Interest rates, inflation, GDP, employment

### 2.3 Volume & Liquidity (`volume/`)

Volume-based confirmation and liquidity filters.

**Key Concepts**:
- Volume Analysis: Absolute, relative, spikes
- Volume Confirmation: Breakout validation
- Liquidity Assessment: Spread, depth, ADV
- VWAP: Execution benchmarking
- Order Flow: OBV, money flow, tick analysis
- Liquidity Filters: Min ADV, spread constraints

### 2.4 Sentiment Analysis (`sentiment/`)

News and social sentiment signals.

**Key Concepts**:
- Lexicon-based: Loughran-McDonald financial dictionary
- Transformer-based: FinBERT, custom models
- Source Reliability: Tiered credibility scoring
- Social Filtering: Bot detection, quality filters
- SEC Filing Analysis: 10-K/10-Q sentiment, change detection
- Earnings Call Analysis: Management tone, analyst questions

### 2.5 Event-Driven (`events/`)

Trading strategies around corporate and macro events.

**Key Concepts**:
- Earnings: PEAD, surprise metrics, guidance trading
- Corporate Actions: Merger arbitrage, spinoffs, buybacks
- Macro Events: FOMC, NFP, CPI trading
- Event Calendars: Pre-event risk management
- Binary Event Risk: Position sizing around events

### 2.6 Quantitative (`quantitative/`)

Systematic quantitative strategies.

**Key Concepts**:
- **Algorithmic Strategies**: Index rebalancing, arbitrage, scalping, dark pools
- Statistical Arbitrage: Pairs trading, cointegration
- Factor Investing: Momentum, value, quality factors
- Machine Learning: Classification, regime detection
- Execution: TWAP, VWAP, Almgren-Chriss, Implementation Shortfall
- Portfolio: Mean-variance, risk parity, Black-Litterman
- Non-Ergodicity: Binomial evolution, predictive capacity assessment

---

## Section 3: Risk Management

Hard constraints for capital preservation.

**Location**: `03_risk/`

**Files**:
- `README.md` - Core risk management principles and methods
- `advanced_risk_methods.md` - Quantitative risk methods (VaR, stress testing)

**Key Concepts**:
1. **Risk Per Trade**: Fixed fractional, percentage-based limits
2. **Position Sizing**: ATR-based, volatility-adjusted, Kelly criterion
3. **Stop Loss Methods**: Fixed, ATR-based, support-based, time-based
4. **Portfolio Limits**: Max positions, sector concentration, correlation
5. **Drawdown Management**: Max drawdown triggers, equity curve trading
6. **Daily/Weekly Limits**: Loss limits that halt trading
7. **Risk-Adjusted Returns**: Sharpe, Sortino, Calmar ratios
8. **Correlation Risk**: Portfolio heat, correlated positions, correlation matrix
9. **Tail Risk**: VaR, CVaR, stress testing, black swan protection
10. **Capital Preservation**: Scaling down on losing streaks, adaptive sizing

**Advanced Methods**:
- **Correlation Matrix Engine**: Real-time correlation tracking
- **VaR/ES Calculations**: Historical, parametric, Cornish-Fisher
- **Stress Testing**: Historical scenarios, hypothetical shocks, reverse stress
- **Dynamic Risk Limits**: Regime-based adjustment, adaptive sizing

**Academic References**:
- "The Mathematics of Money Management" (Vince)
- Kelly Criterion literature
- Jorion - "Value at Risk"

---

## Section 4: Strategy Design

Constructing and validating trading strategies with NVIDIA AI integration.

**Location**: `04_strategy/`

**Files**:
- `backtesting-requirements.md` - Backtesting methodology and metrics
- `data-evaluation-requirements.md` - Comprehensive data and evaluation specs
- `nvidia_integration.md` - NVIDIA AI model integration patterns
- `strategy_formulation_framework.md` - Complete strategy development framework

**Supported Asset Classes**:
| Asset | Status | Notes |
|-------|--------|-------|
| Equities | Full | All strategies |
| Options | Full | Greeks, archetypes |
| Bonds | Planned | Duration/credit |
| Crypto | Placeholder | API pending |

**Key Concepts**:
1. **Strategy Specification**: Entry, exit, sizing, filters
2. **Edge Identification**: Market inefficiency being exploited
3. **Backtesting**: Train/test split, walk-forward, out-of-sample
4. **Transaction Costs**: Commission, spread, slippage modeling
5. **Performance Metrics**: CAGR, Sharpe, max DD, win rate, expectancy
6. **Overfitting Detection**: Degrees of freedom, parameter sensitivity
7. **Robustness Testing**: Monte Carlo, parameter variation

**NVIDIA Integration**:
- **Hypothesis Generation**: Llama 3.1 405B for strategy ideation
- **Signal Enhancement**: Llama 3.1 70B for signal interpretation
- **Regime Detection**: ML-based market regime classification
- **RAG Context**: Knowledge base retrieval for enhanced context

**Academic References**:
- "Advances in Financial Machine Learning" (de Prado)
- "Quantitative Trading" (Chan)

---

## Section 5: Execution

Technical infrastructure for automated trading.

**Location**: `05_execution/`

**Files**:
- `README.md` - System architecture overview
- `governance_engines.md` - Comprehensive governance framework
- `data_pipelines.md` - Complete data pipeline architecture
- `deployment_patterns.md` - Deployment architectures
- `monitoring.md` - Comprehensive monitoring

**Key Concepts**:
1. **Data Pipeline**: Market data ingestion, storage, preprocessing
2. **Signal Generation**: From indicators to actionable signals
3. **Risk Engine**: Pre-trade risk checks, position limits
4. **Order Management**: Creation, routing, status tracking
5. **Execution Algorithms**: Smart order routing, TWAP, VWAP
6. **Broker Integration**: API connectivity, rate limits
7. **Monitoring**: System health, anomaly detection
8. **Logging**: Complete trade history, decision logs
9. **Kill Switches**: Emergency shutdown, loss limits
10. **Deployment**: Cloud vs local, redundancy

**Governance Engines** (Implemented):
- **Audit Engine**: Immutable audit trails with hash chaining
- **Governance Engine**: Policy enforcement, approval workflows
- **PPI Engine**: Personal information detection, masking
- **Ethics Engine**: OECD AI Principles, ESG scoring
- **Broker Compliance Engine**: PDT rules, rate limits

Implementation: `src/engines/governance/`

---

## Section 6: Options & Derivatives

Options-specific knowledge for automated strategies.

**Location**: `06_options/`

**Key Concepts**:
1. **Options Fundamentals**: Calls, puts, strike, expiry, exercise
2. **Moneyness**: ITM, ATM, OTM definitions
3. **Implied Volatility**: IV rank, IV percentile, term structure
4. **Greeks**: Delta, gamma, theta, vega, rho
5. **Options Pricing**: Black-Scholes, binomial (conceptual)
6. **Strategy Archetypes**: Covered calls, verticals, condors, straddles
7. **Defined-Risk**: Max loss, max gain, break-even calculation
8. **Strike/Expiry Selection**: Delta-based, DTE targeting
9. **Volatility Trading**: IV vs RV, mean reversion
10. **Hedging**: Portfolio protection, tail risk

**Academic References**:
- "Options, Futures, and Other Derivatives" (Hull)
- "Option Volatility and Pricing" (Natenberg)

---

## Section 7: References

Academic sources and validation materials.

**Location**: `07_references/`

**Primary Academic Sources**:
- Journal of Finance
- Journal of Financial Economics
- Review of Financial Studies
- Journal of Portfolio Management
- Quantitative Finance

**Foundational Texts**:
1. Graham & Dodd - Security Analysis
2. Hull - Options, Futures, and Other Derivatives
3. Harris - Trading and Exchanges
4. de Prado - Advances in Financial Machine Learning
5. Aronson - Evidence-Based Technical Analysis

**Regulatory Sources**:
- SEC.gov - Securities regulations
- FINRA - Trading rules and compliance
- CFTC - Futures and derivatives

---

## Usage Notes

1. **Workflow-optimized**: Navigate by strategy development phase
2. **Rule-based**: All concepts expressible as programmatic logic
3. **Academic validated**: Cite sources for claimed edges
4. **Cross-reference**: Risk management touches all areas
5. **Regular updates**: Markets evolve, strategies decay
