# Extensive Backtest Framework

## Overview

Comprehensive backtesting infrastructure designed for rigorous strategy validation across:
- **100 stocks** spanning small/mid/large cap
- **Bull and bear market performers** (cyclicals vs defensives)
- **All major sectors** (10 GICS sectors)
- **Multiple market regimes** (bull, bear, sideways)
- **20 years** of historical data (2005-2025)

---

## Dataset Universe

### Market Cap Distribution

| Category | Count | Characteristics |
|----------|-------|----------------|
| **Large Cap** | 43 stocks | >$200B market cap, high liquidity |
| **Mid Cap** | 25 stocks | $10B-$200B, growth potential |
| **Small Cap** | 20 stocks | $2B-$10B, high volatility |
| **ETFs** | 13 benchmarks | Market/sector benchmarks |
| **Total** | **101 symbols** | Complete market coverage |

### Performance Characteristics

| Type | Count | Description |
|------|-------|-------------|
| **Bull Performers** | 73 stocks | Cyclicals - outperform in bull markets |
| **Bear Performers** | 23 stocks | Defensives - outperform in bear markets |
| **Neutral** | 5 symbols | ETFs without performance bias |

**Bull Market Performers (Cyclicals)**:
- Technology (high growth)
- Financials (leverage expansion)
- Energy (commodity upturns)
- Consumer Discretionary (spending increases)
- Industrials (capex expansion)

**Bear Market Performers (Defensives)**:
- Utilities (stable cash flows)
- Consumer Staples (non-cyclical demand)
- Healthcare (essential services)
- REITs (dividend yield)
- Gold/Treasuries (safe havens)

### Sector Coverage

| Sector | Stocks | Volatility | Examples |
|--------|--------|------------|----------|
| **TECH** | 16 | High (1.3-2.2x) | AAPL, MSFT, GOOGL, NVDA, META, PLTR, CRWD |
| **FINANCE** | 14 | High (1.3-1.6x) | JPM, BAC, GS, MS, SCHW |
| **HEALTHCARE** | 14 | Moderate (0.9-1.5x) | UNH, JNJ, REGN, VRTX |
| **CONSUMER** | 18 | Low-Moderate (0.8-1.3x) | WMT, HD, MCD, COST, SBUX |
| **ENERGY** | 5 | High (1.4x) | XOM, CVX, COP, SLB, EOG |
| **INDUSTRIAL** | 13 | Moderate (1.0-1.4x) | BA, CAT, UNP, HON, GE |
| **UTILITIES** | 4 | Low (0.6x) | NEE, DUK, SO, D |
| **REAL_ESTATE** | 4 | Low (0.9x) | AMT, PLD, CCI, EQIX |
| **ETF** | 13 | Varied | SPY, QQQ, IWM, MDY, XLF, XLE, GLD, TLT |

---

## Dataset Organization

```
data/
├── historical/
│   ├── large_cap/           43 stocks (AAPL, MSFT, GOOGL, etc.)
│   ├── mid_cap/             25 stocks (PLTR, CRWD, SCHW, etc.)
│   ├── small_cap/           20 stocks (SMCI, FTNT, CYBR, etc.)
│   └── etfs/                13 ETFs (SPY, QQQ, IWM, sector ETFs)
├── synthetic/               25 stocks (GBM-generated test data)
├── macro/                   5 macro indicators (SPY, VIX, TNX, GLD, USO)
├── raw/                     Sample test datasets
└── enhanced_dataset_metadata.csv   Master index with metadata
```

### Dataset Features

Each historical dataset includes:
- **Core OHLCV**: open, high, low, close, volume, adj_close
- **Metadata**: symbol, sector, market_cap, bull_performer
- **Volatility Measures**: true_range, atr_14, hvol_20, parkinson_vol_20
- **Timeframe**: 20 years (~5,000 bars)
- **Source**: Yahoo Finance (via yfinance)

---

## Backtest Dimensions

### Test Matrix

```
Dimension 1: Strategies (6)
  ├── RSI Mean Reversion
  ├── MACD Crossover
  ├── Bollinger Bands
  ├── ADX Trend Filter (NEW)
  ├── Fibonacci Retracement (NEW)
  └── Parabolic SAR (NEW)

Dimension 2: Symbols (101)
  ├── Large Cap (43)
  ├── Mid Cap (25)
  ├── Small Cap (20)
  └── ETFs (13)

Dimension 3: Market Regimes (3)
  ├── BULL: +0.08% drift, 1.5% volatility
  ├── BEAR: -0.06% drift, 2.0% volatility
  └── SIDEWAYS: +0.01% drift, 1.2% volatility

Dimension 4: Timeframes (2)
  ├── DAILY: 500 bars (~2 years)
  └── WEEKLY: 100 bars (~2 years)

Total Test Scenarios: 6 × 101 × 3 × 2 = 3,636 backtests
```

### Performance Metrics Captured

**Returns**:
- Total return (%)
- Annualized return (%)
- Compound annual growth rate (CAGR)

**Risk**:
- Maximum drawdown (%)
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Beta (vs SPY)

**Trade Statistics**:
- Total trades
- Win rate (%)
- Average win/loss
- Profit factor
- Expectancy

**Market Cap Analysis**:
- Performance by cap size (large/mid/small)
- Cap rotation effects

**Market Regime Analysis**:
- Bull market performance
- Bear market performance
- Defensive vs cyclical effectiveness

---

## Execution Plan

### Phase 1: Dataset Fetch (IN PROGRESS)

```bash
# Fetching 101 historical datasets (20 years each)
python scripts/fetch_enhanced_datasets.py --years 20 --output data --format csv
```

**Progress**: Background process running (ETA: 5-10 minutes)

**Output**:
- `data/historical/large_cap/*.csv` (43 files)
- `data/historical/mid_cap/*.csv` (25 files)
- `data/historical/small_cap/*.csv` (20 files)
- `data/historical/etfs/*.csv` (13 files)
- `data/enhanced_dataset_metadata.csv` (master index)

### Phase 2: Comprehensive Backtest Suite

```bash
# Run extensive backtest suite with enhanced universe
python scripts/comprehensive_backtest_suite_enhanced.py \
  --data-source data/historical \
  --strategies ALL \
  --output results/extensive_backtest_$(date +%Y%m%d)
```

**Test Scenarios**: 3,636 backtests
**Estimated Runtime**: 4-6 hours
**Output**:
- Raw results CSV (3,636 rows)
- By-strategy aggregation
- By-market-cap aggregation
- By-sector aggregation
- By-regime aggregation
- By-performance-characteristic (bull/bear performers)
- Top 50 performers
- Robustness metrics

### Phase 3: Analysis & Reporting

```bash
# Generate comprehensive analysis report
python scripts/generate_enhanced_analysis_report.py \
  --input results/extensive_backtest_20251210 \
  --output reports/EXTENSIVE_BACKTEST_ANALYSIS.md
```

**Report Sections**:
1. Executive Summary
2. Strategy Rankings (overall, by cap, by regime)
3. Market Cap Effects (large vs mid vs small)
4. Sector Performance (10 GICS sectors)
5. Bull vs Bear Performers (cyclical vs defensive)
6. Regime Analysis (bull, bear, sideways)
7. Risk-Adjusted Returns (Sharpe, Sortino, Calmar)
8. Drawdown Analysis
9. Trade Distribution Analysis
10. Statistical Significance Testing
11. Recommendations

---

## Key Insights Expected

### Market Cap Effects

**Large Cap Characteristics**:
- Lower volatility
- Higher liquidity
- More efficient markets (harder to beat)
- Stable performance across regimes

**Mid Cap Characteristics**:
- Moderate volatility
- Growth potential
- Less analyst coverage (potential edge)
- Sensitive to economic cycles

**Small Cap Characteristics**:
- Higher volatility
- Higher potential returns
- Lower liquidity (execution challenges)
- More regime-dependent

### Bull vs Bear Performer Analysis

**In Bull Markets**:
- Cyclicals expected to outperform
- Tech/Finance/Energy leading
- Small/mid cap outperformance
- Momentum strategies work better

**In Bear Markets**:
- Defensives expected to outperform
- Utilities/Healthcare/Staples leading
- Large cap outperformance
- Mean reversion strategies work better

### Expected Strategy Differentiation

**By Market Cap**:
- Large cap: Lower signal-to-noise, need refined entries
- Mid cap: Sweet spot for technical strategies
- Small cap: Higher noise, need wider stops

**By Regime**:
- Bull: Trend-following strategies (MACD, PSAR)
- Bear: Mean reversion strategies (RSI, Bollinger)
- Sideways: Range-bound strategies (Bollinger, Fibonacci levels)

---

## Advanced Analysis Features

### Statistical Significance

- Bootstrap resampling (1,000 iterations)
- Sharpe ratio confidence intervals
- Win rate significance testing
- Multiple comparison corrections

### Robustness Metrics

- Cross-symbol consistency (% symbols profitable)
- Cross-regime consistency (performs in all regimes?)
- Parameter sensitivity analysis
- Drawdown duration distribution

### Portfolio Construction

**From Results**:
1. Identify top strategies per market cap
2. Identify top strategies per regime
3. Build market-cap-diversified portfolio
4. Build regime-diversified portfolio
5. Optimize allocations (max Sharpe, min drawdown)

---

## Usage Examples

### Example 1: Which Strategy Works Best for Small Cap in Bull Markets?

```python
import pandas as pd

# Load results
results = pd.read_csv("results/extensive_backtest_20251210/backtest_results_raw.csv")

# Filter
small_cap_bull = results[
    (results["market_cap"] == "SMALL") &
    (results["regime"] == "BULL")
]

# Rank by Sharpe ratio
top_strategies = small_cap_bull.groupby("strategy").agg({
    "sharpe_ratio": "mean",
    "total_return": "mean",
    "max_drawdown": "mean"
}).sort_values("sharpe_ratio", ascending=False)

print(top_strategies.head())
```

### Example 2: How Do Defensives Perform in Bear Markets?

```python
# Load metadata
metadata = pd.read_csv("data/enhanced_dataset_metadata.csv")
defensives = metadata[metadata["bull_performer"] == False]["symbol"].tolist()

# Filter results
defensive_bear = results[
    (results["symbol"].isin(defensives)) &
    (results["regime"] == "BEAR")
]

# Compare to cyclicals
cyclicals = metadata[metadata["bull_performer"] == True]["symbol"].tolist()
cyclical_bear = results[
    (results["symbol"].isin(cyclicals)) &
    (results["regime"] == "BEAR")
]

print(f"Defensives avg return (bear): {defensive_bear['total_return'].mean():.2%}")
print(f"Cyclicals avg return (bear): {cyclical_bear['total_return'].mean():.2%}")
```

### Example 3: Build Market-Cap-Diversified Portfolio

```python
# Find best strategy for each market cap
large_cap_best = results[results["market_cap"] == "LARGE"].nlargest(1, "sharpe_ratio")
mid_cap_best = results[results["market_cap"] == "MID"].nlargest(1, "sharpe_ratio")
small_cap_best = results[results["market_cap"] == "SMALL"].nlargest(1, "sharpe_ratio")

# Equal weight allocation
portfolio = pd.concat([large_cap_best, mid_cap_best, small_cap_best])
portfolio_return = portfolio["total_return"].mean()
portfolio_sharpe = portfolio["sharpe_ratio"].mean()

print(f"Diversified Portfolio Return: {portfolio_return:.2%}")
print(f"Diversified Portfolio Sharpe: {portfolio_sharpe:.2f}")
```

---

## Files Generated

### Scripts (Created Today)

1. `scripts/enhanced_dataset_config.py` (350 lines)
   - Symbol universe definition
   - Market cap classifications
   - Performance characteristics
   - Sector mappings

2. `scripts/fetch_enhanced_datasets.py` (200 lines)
   - Historical data fetch for 101 symbols
   - Organized by market cap
   - Metadata generation

3. `scripts/comprehensive_backtest_suite_enhanced.py` (PENDING)
   - 3,636 backtest execution
   - Market cap analysis
   - Bull/bear performer analysis

4. `scripts/generate_enhanced_analysis_report.py` (PENDING)
   - Statistical analysis
   - Visualization generation
   - Markdown report generation

### Documentation

1. `docs/EXTENSIVE_BACKTEST_FRAMEWORK.md` (THIS FILE)
2. `data/README.md` - Data folder organization
3. `docs/DATASET_MANAGEMENT_GUIDE.md` - Dataset generation guide
4. `docs/DATASET_QUICK_REFERENCE.md` - Quick commands

### Data Files (When Complete)

- 101 historical CSV files (~64 MB total)
- 1 enhanced metadata file
- Existing 25 synthetic datasets
- Existing 5 macro indicators

---

## Next Steps

**Immediate (Automated)**:
1. ✅ Dataset fetch in progress (background process 4493e0)
2. ⏳ Wait for fetch completion (~5-10 min)
3. ⏳ Verify all 101 datasets fetched successfully

**Short Term (Manual Decision Required)**:
4. ❓ Run comprehensive backtest suite (3,636 tests, 4-6 hours)
5. ❓ Generate analysis report
6. ❓ Review results and identify top strategies

**Medium Term**:
7. ❓ Parameter optimization for top strategies
8. ❓ Walk-forward validation
9. ❓ Out-of-sample testing (2024-2025 data)

---

## Expected Deliverables

**After Phase 2 Completion**:
- 3,636 backtest results
- Performance by market cap (large/mid/small)
- Performance by sector (10 sectors)
- Performance by regime (bull/bear/sideways)
- Performance by characteristic (cyclical/defensive)
- Statistical significance tests
- Top 50 strategy-symbol-regime combinations

**After Phase 3 Completion**:
- Comprehensive analysis report (50+ pages)
- Strategy rankings with confidence intervals
- Market cap effect analysis
- Sector rotation insights
- Bull/bear performer validation
- Portfolio construction recommendations

---

**Status**: Phase 1 in progress (dataset fetch running)
**Last Updated**: 2025-12-10
**Next Action**: Monitor dataset fetch completion
