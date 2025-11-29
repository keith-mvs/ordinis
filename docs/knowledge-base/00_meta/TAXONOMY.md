# Knowledge Base Domain Taxonomy

> **Formal classification system for trading knowledge domains**

---

## Overview

The Intelligent Investor Knowledge Base organizes trading and finance knowledge into **9 core domains**. This taxonomy ensures:
- **Consistent organization** across all artifacts (publications, concepts, strategies)
- **Clear ownership** for each topic area
- **Engine mapping** to system components
- **Searchability** and discoverability

---

## Domain Definitions

### Domain 1: Market Basics & Microstructure

**Scope:** How markets work at the mechanical level

**Topics:**
- Order types (market, limit, stop, etc.)
- Market structures (exchange, OTC, dark pools)
- Price formation and discovery
- Trading costs and liquidity
- Market impact models
- Regulation (Reg NMS, etc.)

**Primary Engine:** FlowRoute (execution), Cortex (understanding)

**Key Publications:**
- Harris, "Trading and Exchanges"
- Biais et al., "Market Microstructure Survey"
- Foucault et al., "Market Liquidity"
- Cartea et al., "Algorithmic and High-Frequency Trading"

**Example Concepts:**
- `concept_01_bid_ask_spread`
- `concept_01_market_impact`
- `concept_01_order_types`

---

### Domain 2: Technical / Chart-Based Analysis

**Scope:** Price and volume pattern analysis

**Topics:**
- Chart patterns (head & shoulders, triangles, flags)
- Technical indicators (MA, RSI, MACD, Bollinger Bands)
- Oscillators and momentum indicators
- Support and resistance
- Trend identification
- Intermarket analysis

**Primary Engine:** SignalCore (pattern detection), ProofBench (backtesting TA)

**Key Publications:**
- Kirkpatrick & Dahlquist, "Technical Analysis: The Complete Resource"
- Aronson, "Evidence-Based Technical Analysis"

**Example Concepts:**
- `concept_02_moving_average`
- `concept_02_rsi`
- `concept_02_bollinger_bands`

**Critical Note:** Must apply statistical rigor (Aronson) to avoid data mining bias

---

### Domain 3: Volume & Liquidity / Order-Flow Signals

**Scope:** Market depth, volume profiles, and order flow analysis

**Topics:**
- VWAP (Volume-Weighted Average Price)
- Volume profiles and distribution
- Market depth and order book analysis
- Bid-ask dynamics
- Large order detection
- Execution algorithms (TWAP, Implementation Shortfall)

**Primary Engine:** SignalCore (volume signals), FlowRoute (execution algos)

**Key Publications:**
- Johnson, "Algorithmic Trading and DMA"
- Foucault et al., "Market Liquidity"
- Cartea et al., "Algorithmic and High-Frequency Trading"
- Biais et al., "Market Microstructure Survey"

**Example Concepts:**
- `concept_03_vwap`
- `concept_03_order_flow`
- `concept_03_market_depth`

---

### Domain 4: Fundamental & Macro Analysis

**Scope:** Company fundamentals and macroeconomic factors

**Topics:**
- Valuation models (DCF, P/E, P/B, EV/EBITDA)
- Financial statement analysis
- Economic indicators (GDP, CPI, unemployment)
- Central bank policy
- Currency and commodity markets
- Sector rotation
- Factor investing (value, momentum, quality)

**Primary Engine:** Cortex (analysis), SignalCore (fundamental signals)

**Key Publications:**
- Ilmanen, "Expected Returns"
- Gliner, "Global Macro Trading"

**Example Concepts:**
- `concept_04_dcf_valuation`
- `concept_04_economic_indicators`
- `concept_04_sector_rotation`

---

### Domain 5: News, Headlines & Sentiment

**Scope:** News analysis, sentiment extraction, and event-driven trading

**Topics:**
- NLP for news analysis
- Sentiment scoring methodologies
- Event detection (earnings, M&A, FDA approvals)
- Social media sentiment
- Behavioral finance and market psychology
- Contrarian indicators

**Primary Engine:** Cortex (NLP), NewsPlugin (data), SignalCore (sentiment signals)

**Key Publications:**
- Mitra & Mitra, "The Handbook of News Analytics in Finance"
- Peterson, "Trading on Sentiment"

**Example Concepts:**
- `concept_05_sentiment_analysis`
- `concept_05_news_impact`
- `concept_05_event_driven`

---

### Domain 6: Options & Derivatives

**Scope:** Options pricing, Greeks, and derivatives strategies

**Topics:**
- Options fundamentals (calls, puts, payoffs)
- Black-Scholes and binomial models
- The Greeks (Delta, Gamma, Vega, Theta, Rho)
- Implied volatility and volatility surface
- Options strategies (spreads, straddles, iron condors)
- Exotic options
- Futures and swaps

**Primary Engine:** SignalCore (options models), RiskGuard (Greeks limits)

**Key Publications:**
- Hull, "Options, Futures, and Other Derivatives"
- Natenberg, "Option Volatility and Pricing"

**Example Concepts:**
- `concept_06_black_scholes`
- `concept_06_greeks`
- `concept_06_volatility_surface`
- `concept_06_iron_condor`

---

### Domain 7: Risk Management & Position Sizing

**Scope:** Portfolio risk, position sizing, and risk limits

**Topics:**
- Position sizing methodologies (fixed fractional, Kelly, ATR-based)
- Portfolio risk metrics (VaR, CVaR, max drawdown)
- Drawdown control and kill switches
- Risk-adjusted performance (Sharpe, Sortino, Calmar)
- Stop loss methods
- Correlation and concentration limits

**Primary Engine:** RiskGuard (enforcement), ProofBench (risk metrics)

**Key Publications:**
- Grant, "Trading Risk"
- López de Prado, "Advances in Financial Machine Learning"
- Gliner, "Global Macro Trading"
- Ilmanen, "Expected Returns"
- Peterson, "Trading on Sentiment"

**Example Concepts:**
- `concept_07_position_sizing`
- `concept_07_kelly_criterion`
- `concept_07_var`
- `concept_07_max_drawdown`
- `concept_07_kill_switch`

**Status:** ⭐ **Most mature domain** - Core RiskGuard implementation complete

---

### Domain 8: Strategy Design, Backtesting & Evaluation

**Scope:** Strategy specification, backtesting methodology, and performance evaluation

**Topics:**
- Strategy specification and documentation
- Backtest methodology (event-driven, vectorized)
- Cross-validation in finance (purged K-fold, CPCV)
- Performance metrics (returns, Sharpe, drawdown, win rate)
- Overfitting prevention
- Data mining bias and multiple testing
- Strategy robustness testing

**Primary Engine:** ProofBench (backtesting), Cortex (strategy design)

**Key Publications:**
- López de Prado, "Advances in Financial Machine Learning" ⭐ **CRITICAL**
- Aronson, "Evidence-Based Technical Analysis"
- Narang, "Inside the Black Box"
- Ilmanen, "Expected Returns"

**Example Concepts:**
- `concept_08_backtest_methodology`
- `concept_08_purged_cv`
- `concept_08_deflated_sharpe`
- `concept_08_overfitting`
- `concept_08_walk_forward`

**Critical Techniques:**
- Purged K-Fold CV (López de Prado Ch 7)
- Deflated Sharpe Ratio (López de Prado Ch 14)
- Probability of Backtest Overfitting (PBO)

---

### Domain 9: System Architecture & Automation

**Scope:** Trading system design, execution infrastructure, and automation

**Topics:**
- System architecture patterns (event-driven, microservices)
- Execution algorithms (VWAP, TWAP, implementation shortfall)
- Order routing and smart order routing
- Low-latency design
- System reliability and fault tolerance
- Broker API integration
- Deployment and DevOps for trading systems

**Primary Engine:** FlowRoute (execution), All engines (architecture)

**Key Publications:**
- Johnson, "Algorithmic Trading and DMA"
- Cartea et al., "Algorithmic and High-Frequency Trading"
- Narang, "Inside the Black Box"

**Example Concepts:**
- `concept_09_event_driven`
- `concept_09_execution_algorithms`
- `concept_09_smart_order_routing`
- `concept_09_system_reliability`

---

## Cross-Domain Mapping

Some topics span multiple domains. Here's how to handle them:

### Example: Backtesting an Options Strategy

**Primary Domain:** 8 (Strategy & Backtesting)
**Secondary Domains:** 6 (Options), 7 (Risk Management)

**Approach:**
- Strategy spec lives in Domain 8
- References options concepts from Domain 6
- Applies risk limits from Domain 7

### Example: Sentiment-Based Position Sizing

**Primary Domain:** 7 (Risk Management)
**Secondary Domains:** 5 (News & Sentiment)

**Approach:**
- Position sizing methodology in Domain 7
- Sentiment indicators from Domain 5
- Integration documented in both domains

---

## Domain Usage in Artifacts

### Publications

Each publication specifies `domains: [array]` in metadata:

```json
{
  "id": "pub_lopez_de_prado_advances",
  "domains": [7, 8],  // Risk Management & Backtesting
  "primary_domain": 7
}
```

**Rules:**
- List ALL relevant domains
- First domain is considered primary
- Publication file lives in primary domain's `publications/` folder
- Symlinks can reference from secondary domains

### Concepts

Each concept belongs to ONE primary domain:

```json
{
  "id": "concept_07_position_sizing",
  "domain": 7
}
```

The domain number is part of the concept ID.

### Strategies

Strategies reference domains they utilize:

```yaml
strategy:
  id: strat_ma_crossover_spy
  metadata:
    domains: [2, 7, 8]  # Technical, Risk, Backtesting
```

### Skills/Commands

Skills may operate across domains:

```json
{
  "id": "/position-size",
  "domains": [7],  // Primarily domain 7
  "related_domains": [1, 6]  // May consider microstructure, options
}
```

---

## Domain Evolution

### Adding New Domains

**Process:**
1. Proposal with justification
2. Review by Knowledge Engine team
3. Update TAXONOMY.md
4. Create domain folder structure
5. Update all schemas
6. Communicate to users

**Threshold:** Only add domain if >5 publications would fit there

### Merging Domains

If two domains have significant overlap and <3 publications each, consider merging.

### Deprecating Domains

Mark as deprecated, migrate content, maintain redirects for 6 months.

---

## Quick Reference

| # | Domain | Engine | Publications | Status |
|---|--------|--------|--------------|--------|
| **1** | Market Microstructure | FlowRoute | 4 | Active |
| **2** | Technical Analysis | SignalCore | 2 | Active |
| **3** | Volume & Liquidity | SignalCore, FlowRoute | 4 | Active |
| **4** | Fundamental & Macro | Cortex | 2 | Active |
| **5** | News & Sentiment | Cortex | 2 | Active |
| **6** | Options & Derivatives | SignalCore | 2 | Active |
| **7** | Risk Management | RiskGuard | 5 | ⭐ **Mature** |
| **8** | Strategy & Backtesting | ProofBench | 4 | Active |
| **9** | System Architecture | FlowRoute | 3 | Active |

**Total Publications:** 15 (some appear in multiple domains)

---

## Domain Naming Conventions

### Folder Names
```
0[domain_number]_[domain_name]/
```

Examples:
- `01_market_microstructure/`
- `07_risk_management/`
- `09_system_architecture/`

### Concept IDs
```
concept_[domain_number]_[concept_name]
```

Examples:
- `concept_01_bid_ask_spread`
- `concept_07_position_sizing`

### Publication IDs (domain-agnostic)
```
pub_[author]_[short_title]
```

Examples:
- `pub_harris_trading_exchanges`
- `pub_lopez_de_prado_advances`

---

## Governance

**Owner:** Knowledge Engine Team
**Review Cycle:** Quarterly
**Change Process:** Proposal → Review → Approval → Implementation
**Version:** v1.0.0
**Last Updated:** 2025-01-28

---

## Navigation

- [Knowledge Base Home](../README.md)
- [Metadata Schema](METADATA_SCHEMA.json)
- [Versioning Standards](VERSIONING.md)
- [Governance Process](GOVERNANCE.md)
