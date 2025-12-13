# Dataset Manager - Quick Reference

## Commands

### Generate 20 Years of Historical Data

```bash
python scripts/dataset_manager.py \
  --mode historical \
  --years 20 \
  --sectors TECH,FINANCE,HEALTHCARE,ENERGY,CONSUMER \
  --symbols-per-sector 5 \
  --output data/historical_20yr
```

**Output**: 25 symbols × 20 years + 6 macro indicators

### Generate Synthetic Data for Testing

```bash
python scripts/dataset_manager.py \
  --mode synthetic \
  --years 10 \
  --sectors TECH,FINANCE \
  --symbols-per-sector 3 \
  --output data/synthetic_test \
  --no-windows
```

**Output**: 6 symbols × 10 years (fast, reproducible)

### Generate Complete Production Dataset

```bash
python scripts/dataset_manager.py \
  --mode combined \
  --years 20 \
  --window-months 3 \
  --sectors TECH,FINANCE,HEALTHCARE,ENERGY,CONSUMER,INDUSTRIAL \
  --symbols-per-sector 10 \
  --output data/production \
  --format parquet
```

**Output**: 60 symbols × 20 years + windows + macro + volatility features

---

## Dataset Specifications

### Historical Data Coverage

**Timeframe**: 20 years (2005-2025)
**Symbols**: 100+ across 10 sectors
**Bars**: ~5,000 per symbol (trading days)
**Features**: 11 columns per dataset

### Windowing Parameters

**Window Size**: 2-3 months (default: 3 months)
**Window Step**: 1 month (overlapping)
**Windows per Symbol**: ~237 (for 20 years, 3-month windows)
**Total Windows**: 23,700+ (for 100 symbols)

### Features Included

**Core OHLCV**:
- open, high, low, close, volume

**Metadata**:
- symbol, sector

**Volatility Measures** (if --no-volatility not set):
- true_range: True range (Wilder)
- atr_14: 14-day Average True Range
- hvol_20: 20-day historical volatility (annualized)
- parkinson_vol_20: 20-day Parkinson volatility (annualized)

### Sectors Available

| Code | Full Name | Volatility | Example Symbols |
|------|-----------|------------|-----------------|
| TECH | Technology | High (1.3x) | AAPL, MSFT, GOOGL, NVDA |
| FINANCE | Financials | Highest (1.5x) | JPM, BAC, WFC, GS |
| HEALTHCARE | Healthcare | Moderate (0.9x) | UNH, JNJ, PFE, ABBV |
| ENERGY | Energy | High (1.4x) | XOM, CVX, COP, SLB |
| CONSUMER | Consumer | Low (0.8x) | WMT, HD, MCD, NKE |
| INDUSTRIAL | Industrials | Average (1.0x) | BA, CAT, GE, UPS |
| MATERIALS | Materials | High (1.2x) | LIN, APD, ECL, SHW |
| REAL_ESTATE | Real Estate | Low (0.7x) | AMT, PLD, CCI, EQIX |
| UTILITIES | Utilities | Lowest (0.6x) | NEE, DUK, SO, D |
| COMMUNICATION | Communication | High (1.1x) | T, VZ, TMUS, CMCSA |

### Macro Indicators

- **SPY**: S&P 500 ETF (market benchmark)
- **^VIX**: CBOE Volatility Index
- **^TNX**: 10-Year Treasury Yield
- **DXY**: US Dollar Index
- **GLD**: Gold ETF
- **USO**: Oil ETF

---

## Output Structure

```
data/
└── production/
    ├── synthetic/                  # Synthetic datasets
    │   ├── AAPL_synthetic.csv
    │   └── ...
    ├── historical/                 # Historical datasets
    │   ├── AAPL_historical.csv
    │   ├── SPY_macro.csv
    │   └── ...
    ├── windows/                    # Windowed datasets
    │   ├── AAPL_window_000_20050101_20050401.csv
    │   ├── AAPL_window_001_20050201_20050501.csv
    │   └── ...
    ├── dataset_metadata.csv        # Dataset summary
    └── windows_metadata.csv        # Window index
```

---

## Usage Patterns

### Pattern 1: Quick Strategy Test

```bash
# 1. Generate small synthetic dataset
python scripts/dataset_manager.py \
  --mode synthetic \
  --years 5 \
  --sectors TECH \
  --symbols-per-sector 3 \
  --output data/quick_test \
  --no-windows

# 2. Run backtest
python scripts/comprehensive_backtest_suite.py \
  --data-source data/quick_test \
  --strategy RSI \
  --output results/quick_test
```

### Pattern 2: Historical Validation

```bash
# 1. Fetch real historical data
python scripts/dataset_manager.py \
  --mode historical \
  --years 15 \
  --sectors TECH,FINANCE \
  --symbols-per-sector 5 \
  --output data/historical_15yr

# 2. Run comprehensive suite
python scripts/comprehensive_backtest_suite.py \
  --data-source data/historical_15yr \
  --output results/historical_validation
```

### Pattern 3: Walk-Forward Analysis

```bash
# 1. Generate windowed dataset
python scripts/dataset_manager.py \
  --mode combined \
  --years 20 \
  --window-months 3 \
  --output data/walk_forward

# 2. Run walk-forward backtest
python scripts/comprehensive_backtest_suite.py \
  --data-source data/walk_forward/windows \
  --use-windows \
  --output results/walk_forward
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Historical fetch fails | Check internet connection, reduce symbols-per-sector |
| Too much disk space | Use `--format parquet` (5-10x compression) |
| Slow generation | Use `--no-windows` for testing, add windows later |
| Missing yfinance | `pip install yfinance` |

---

## Integration with Backtest Suite

The dataset manager is designed to work seamlessly with the comprehensive backtest suite:

```bash
# Step 1: Generate datasets
python scripts/dataset_manager.py \
  --mode combined \
  --years 20 \
  --sectors TECH,FINANCE,HEALTHCARE \
  --output data/backtest_input

# Step 2: Run comprehensive backtests (FIXED BUG - now works correctly!)
python scripts/comprehensive_backtest_suite.py \
  --output results/comprehensive_backtest

# Step 3: Generate consolidated report
python scripts/generate_consolidated_report.py \
  --input results/comprehensive_backtest \
  --output reports/CONSOLIDATED_REPORT.md
```

---

**Last Updated**: 2025-12-10
**Version**: 1.0.0
