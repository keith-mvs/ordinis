# Portfolio Management Skill

## Skill Package Location

The complete portfolio-management skill package has been created and is available for download from Claude's outputs.

**Download Link**: The `.skill` file is in your current Claude session outputs  
**File Name**: `portfolio-management.skill`

## Installation Instructions

### Option 1: Import to Claude (Recommended)
1. Download the `portfolio-management.skill` file from Claude outputs
2. Open Claude Settings > Skills  
3. Click "Import Skill"
4. Select the downloaded .skill file
5. The skill will be immediately available with all scripts and references

### Option 2: Manual Installation in Project
1. Download the `portfolio-management.skill` file
2. It's a ZIP file - rename to `.zip` and extract
3. Copy the extracted `portfolio-management` folder to:
   ```
   C:\Users\kjfle\.projects\intelligent-investor\skills\portfolio-management
   ```

## Skill Contents

### Structure
```
portfolio-management/
├── SKILL.md                      # Main skill documentation
├── scripts/                      # 5 Python scripts
│   ├── broker_connector.py       # Alpaca, IB, TD Ameritrade integration
│   ├── market_data.py            # Yahoo Finance, Alpha Vantage, Polygon.io
│   ├── portfolio_analytics.py   # Performance & risk metrics
│   ├── tax_tracker.py           # Lot accounting, wash sales, Form 8949
│   └── dividend_tracker.py      # Income tracking & projections
└── references/                   # 3 Reference files
    ├── formulas.md              # Mathematical formulas
    ├── benchmarks.md            # Market indices
    └── sectors.md               # GICS classification
```

### 5 Core Scripts (73KB total)

1. **broker_connector.py** (16KB)
   - Unified interface for Alpaca, Interactive Brokers, TD Ameritrade
   - Factory pattern for easy broker switching
   - Position, order, and transaction retrieval

2. **market_data.py** (14KB)
   - Yahoo Finance (free, no API key)
   - Alpha Vantage (API key required)
   - Polygon.io (API key required)
   - Real-time quotes and historical OHLCV data

3. **portfolio_analytics.py** (14KB)
   - Performance metrics: return, alpha, tracking error
   - Risk metrics: Sharpe, Sortino, beta, VaR, CVaR
   - Allocation analysis and drift monitoring
   - Position sizing alerts

4. **tax_tracker.py** (14KB)
   - 5 lot accounting methods (FIFO, LIFO, HIFO, Specific, Average)
   - Automatic wash sale detection
   - Short-term vs long-term classification
   - IRS Form 8949 export

5. **dividend_tracker.py** (14KB)
   - Qualified/non-qualified dividend tracking
   - Yield and growth rate analysis
   - Dividend Aristocrat identification
   - Future income projections
   - 1099-DIV export

### 3 Reference Files (23KB total)

1. **formulas.md** (5KB) - All calculation formulas
2. **benchmarks.md** (8KB) - Market indices reference
3. **sectors.md** (10KB) - GICS sector classification

## Quick Start

### Connect to Broker
```python
from scripts.broker_connector import create_connector

connector = create_connector('alpaca')
if connector.connect():
    account = connector.get_account()
    print(f"Portfolio Value: ${account.portfolio_value:,.2f}")
```

### Calculate Performance vs Benchmark
```python
from scripts.portfolio_analytics import PortfolioAnalytics

analytics = PortfolioAnalytics(risk_free_rate=0.04)
perf = analytics.calculate_performance(portfolio_returns, benchmark_returns)

print(f"Alpha: {perf.alpha:.2%}")
print(f"Sharpe Ratio: {risk.sharpe_ratio:.2f}")
```

### Track Tax Lots
```python
from scripts.tax_tracker import TaxTracker, LotMethod

tracker = TaxTracker(method=LotMethod.HIFO)
tracker.add_purchase('AAPL', 100, 150.00, datetime(2023, 1, 15))
tracker.add_sale('AAPL', 80, 170.00, datetime(2024, 10, 1))

form8949 = tracker.export_form8949(year=2024)
```

## Dependencies

```bash
pip install pandas numpy alpaca-trade-api ib_insync yfinance alpha_vantage polygon-api-client
```

## API Keys Setup

```bash
# Broker APIs
export BROKER_API_KEY="your_alpaca_key"
export BROKER_API_SECRET="your_alpaca_secret"

# Market Data APIs
export ALPHA_VANTAGE_API_KEY="your_av_key"
export POLYGON_API_KEY="your_polygon_key"
```

## Integration with Intelligent Investor Project

This skill is designed to work with the intelligent-investor project structure:

```
intelligent-investor/
├── skills/
│   ├── portfolio-management/      # This skill
│   ├── benchmarking/
│   └── due-diligence/
├── src/                           # Your Python source code
├── tests/                         # Your test files
└── data/                          # Your data files
```

### Usage in Project

Import the scripts in your Python code:

```python
# In your intelligent-investor project files
import sys
sys.path.append('skills/portfolio-management')

from scripts.broker_connector import create_connector
from scripts.portfolio_analytics import PortfolioAnalytics
from scripts.tax_tracker import TaxTracker

# Use the skills in your analysis workflows
```

## Documentation

- **SKILL.md**: Complete skill documentation (590 lines, 19KB)
- **PORTFOLIO-MANAGEMENT-README.md**: Quick reference guide (this file)
- **PORTFOLIO-MANAGEMENT-SKILL.md**: Feature overview and installation guide

All documentation is available in:
```
C:\Users\kjfle\.projects\intelligent-investor\skills\
```

## Project-Specific Features

Since this skill is part of the intelligent-investor project, it integrates with:

1. **Benchmarking Skill**: Use portfolio_analytics with benchmarking workflows
2. **Due Diligence Skill**: Combine position analysis with research
3. **Project Data**: Store portfolio data in project `/data` directory
4. **Project Tests**: Write tests for portfolio workflows in `/tests`

## Support

For detailed documentation:
1. See SKILL.md for complete API reference
2. See reference files for formulas, benchmarks, and sectors
3. Check examples in the Quick Start Workflows section

---

**Project**: Intelligent Investor  
**Skill**: Portfolio Management  
**Version**: 1.0  
**Created**: December 2024  
**Location**: `C:\Users\kjfle\.projects\intelligent-investor\skills\portfolio-management`
