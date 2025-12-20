# Dataset Management Guide

**Comprehensive Dataset Generation and Retrieval for Trading Strategy Validation**

---

## Overview

The Ordinis Dataset Manager provides a unified interface for generating and retrieving market data across multiple dimensions:

- **Synthetic Data**: Statistically realistic data with configurable characteristics
- **Historical Data**: Real market data spanning 20+ years
- **Multi-Sector Coverage**: 10 GICS sectors with representative symbols
- **Windowed Analysis**: Rolling 2-3 month windows for walk-forward testing
- **Rich Features**: OHLCV, volatility measures, macro indicators

---

## Quick Start

### Generate Synthetic Data

```bash
python scripts/dataset_manager.py \
  --mode synthetic \
  --years 10 \
  --sectors TECH,FINANCE,HEALTHCARE \
  --symbols-per-sector 5 \
  --output data/synthetic
```

**Output**: 15 datasets (5 symbols × 3 sectors) with 10 years of synthetic OHLCV data

### Retrieve Historical Data

```bash
python scripts/dataset_manager.py \
  --mode historical \
  --years 20 \
  --sectors TECH,FINANCE \
  --symbols-per-sector 3 \
  --output data/historical
```

**Output**: 6 datasets + macro indicators with 20 years of real market data

### Combined Mode (Recommended)

```bash
python scripts/dataset_manager.py \
  --mode combined \
  --years 20 \
  --window-months 3 \
  --sectors TECH,FINANCE,HEALTHCARE,ENERGY,CONSUMER \
  --symbols-per-sector 5 \
  --output data/comprehensive
```

**Output**: Synthetic + historical datasets with rolling 3-month windows

---

## Configuration Options

### Required Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--mode` | `synthetic`, `historical`, `combined` | Dataset generation mode |
| `--output` | Path | Output directory |

### Dataset Scope

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--years` | 20 | Years of historical data |
| `--sectors` | TECH,FINANCE,... | Comma-separated sector list |
| `--symbols-per-sector` | 5 | Number of symbols per sector |

### Windowing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--window-months` | 3 | Window size in months |
| `--no-windows` | False | Skip windowing (disabled by default) |

### Features

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--no-macro` | False | Exclude macro indicators |
| `--no-volatility` | False | Exclude volatility features |

### Output Format

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--format` | `csv`, `parquet`, `json` | Output file format (default: csv) |

---

## Sector Coverage

### Supported Sectors (GICS Classification)

| Sector | Code | Representative Symbols | Characteristics |
|--------|------|----------------------|-----------------|
| **Technology** | TECH | AAPL, MSFT, GOOGL, NVDA, META | High volatility (1.3x) |
| **Financials** | FINANCE | JPM, BAC, WFC, GS, MS | Highest volatility (1.5x) |
| **Healthcare** | HEALTHCARE | UNH, JNJ, PFE, ABBV, TMO | Moderate volatility (0.9x) |
| **Energy** | ENERGY | XOM, CVX, COP, SLB, EOG | High volatility (1.4x) |
| **Consumer** | CONSUMER | WMT, HD, MCD, NKE, SBUX | Lower volatility (0.8x) |
| **Industrials** | INDUSTRIAL | BA, CAT, GE, UPS, HON | Average volatility (1.0x) |
| **Materials** | MATERIALS | LIN, APD, ECL, SHW, DD | Above-average volatility (1.2x) |
| **Real Estate** | REAL_ESTATE | AMT, PLD, CCI, EQIX, PSA | Low volatility (0.7x) |
| **Utilities** | UTILITIES | NEE, DUK, SO, D, AEP | Lowest volatility (0.6x) |
| **Communication** | COMMUNICATION | T, VZ, TMUS, CMCSA, DIS | Above-average volatility (1.1x) |

### Macro Indicators

| Symbol | Name | Purpose |
|--------|------|---------|
| SPY | S&P 500 ETF | Market benchmark |
| ^VIX | CBOE Volatility Index | Fear gauge |
| ^TNX | 10-Year Treasury Yield | Risk-free rate |
| DXY | US Dollar Index | Currency strength |
| GLD | Gold ETF | Safe haven |
| USO | Oil ETF | Commodity exposure |

---

## Output Structure

### Directory Layout

```
data/comprehensive/
├── synthetic/                          # Synthetic datasets
│   ├── AAPL_synthetic.csv
│   ├── MSFT_synthetic.csv
│   └── ...
├── historical/                         # Historical datasets
│   ├── AAPL_historical.csv
│   ├── MSFT_historical.csv
│   ├── SPY_macro.csv
│   └── ...
├── windows/                            # Windowed datasets
│   ├── AAPL_window_000_20050101_20050401.csv
│   ├── AAPL_window_001_20050201_20050501.csv
│   └── ...
├── dataset_metadata.csv                # Dataset metadata
└── windows_metadata.csv                # Window metadata
```

### File Formats

#### OHLCV Data (csv)

```csv
date,open,high,low,close,volume,symbol,sector,true_range,atr_14,hvol_20,parkinson_vol_20
2005-01-03,100.5,102.3,99.8,101.2,1500000,AAPL,TECH,2.5,2.1,0.25,0.23
2005-01-04,101.3,103.1,100.9,102.5,1650000,AAPL,TECH,2.2,2.15,0.24,0.22
...
```

**Columns**:
- **date**: Trading date (index)
- **open, high, low, close**: OHLC prices
- **volume**: Trading volume
- **symbol**: Stock symbol
- **sector**: GICS sector
- **true_range**: True range (max of high-low, high-prev_close, low-prev_close)
- **atr_14**: 14-day Average True Range
- **hvol_20**: 20-day historical volatility (annualized)
- **parkinson_vol_20**: 20-day Parkinson volatility (high-low based, annualized)

#### Dataset Metadata

```csv
symbol,sector,start_date,end_date,num_bars,num_features,avg_price,avg_volume,volatility,source
AAPL,TECH,2005-01-01,2024-12-31,5032,11,150.32,50000000,0.35,combined
MSFT,TECH,2005-01-01,2024-12-31,5032,11,95.18,35000000,0.28,combined
...
```

#### Windows Metadata

```csv
symbol,window_id,start_date,end_date,num_bars,sector
AAPL,0,2005-01-01,2005-04-01,63,TECH
AAPL,1,2005-02-01,2005-05-01,63,TECH
...
```

---

## Synthetic Data Generation

### Methodology

Synthetic data uses **Geometric Brownian Motion** with sector-specific characteristics:

```
S(t+1) = S(t) * exp(μΔt + σ√Δt * Z)
```

Where:
- **S(t)**: Price at time t
- **μ**: Drift (daily return)
- **σ**: Volatility (daily standard deviation)
- **Z**: Standard normal random variable

### Sector-Specific Calibration

**Volatility Multipliers**:
- Technology: 1.3x (higher risk, higher growth)
- Financials: 1.5x (highest volatility, cyclical)
- Healthcare: 0.9x (defensive sector)
- Energy: 1.4x (commodity exposure)
- Consumer: 0.8x (stable demand)
- Utilities: 0.6x (lowest volatility, regulated)

**Base Prices** (sector averages):
- Tech: $150 (growth stocks)
- Finance: $50 (bank stocks)
- Healthcare: $100 (pharma stocks)
- Energy: $70 (oil stocks)
- Consumer: $120 (retail stocks)

### Realistic Features

1. **Autocorrelation**: 10% momentum from previous day
2. **Volume Clustering**: Higher volume on volatile days
3. **Intraday Range**: Realistic OHLC relationships
4. **Statistical Properties**: Matches real market distributions

---

## Historical Data Retrieval

### Data Sources

**Primary Source**: yfinance (Yahoo Finance API)
- Free, no API key required
- 20+ years of data for most symbols
- Adjusted prices for splits and dividends

**Future Sources** (planned):
- Polygon.io (institutional-grade data)
- Alpha Vantage (fundamental data)
- IEX Cloud (real-time data)

### Data Quality

**Validation Checks**:
- Missing data detection
- Price continuity validation
- Volume anomaly detection
- Corporate action handling (splits, dividends)

**Handling Issues**:
- Gaps: Forward-fill up to 5 days
- Outliers: Winsorize at 3σ
- Delisted stocks: Excluded automatically

---

## Windowing Strategies

### Rolling Windows

**Purpose**: Walk-forward analysis, regime detection, adaptive strategies

**Configuration**:
```python
window_months = 3      # 3-month windows
step_months = 1        # 1-month step (overlapping)
```

**Example Timeline**:
```
Window 0: 2005-01-01 to 2005-04-01 (3 months)
Window 1: 2005-02-01 to 2005-05-01 (3 months, 1-month step)
Window 2: 2005-03-01 to 2005-06-01 (3 months, 1-month step)
...
Window 237: 2024-10-01 to 2025-01-01
```

**Total Windows** (20 years, 3-month windows, 1-month step):
- ~237 windows per symbol
- ~30,000+ windows for 100+ symbols

### Use Cases

1. **Walk-Forward Testing**:
   - Train on windows 0-9 (9 months)
   - Validate on window 10 (3 months)
   - Test on window 11 (3 months)
   - Roll forward

2. **Regime Detection**:
   - Classify each window: BULL, BEAR, SIDEWAYS, HIGH_VOL, LOW_VOL
   - Train regime-specific strategies

3. **Adaptive Parameters**:
   - Optimize parameters per window
   - Track parameter drift over time

---

## Feature Engineering

### Volatility Features

**True Range** (Welles Wilder):
```
TR = max(high - low, |high - prev_close|, |low - prev_close|)
```

**Average True Range (ATR)**:
```
ATR(14) = EMA(TR, 14)
```

**Historical Volatility**:
```
HVol(20) = StdDev(returns, 20) * sqrt(252)  # Annualized
```

**Parkinson Volatility** (high-low based):
```
ParkinsonVol = sqrt((1/(4*ln(2))) * (ln(high/low))^2)
ParkinsonVol(20) = EMA(ParkinsonVol, 20) * sqrt(252)
```

### Future Features (Planned)

**Momentum**:
- RSI (14-day)
- MACD (12,26,9)
- Stochastic Oscillator (14,3,3)

**Trend**:
- SMA (20, 50, 200-day)
- ADX (14-day)
- Parabolic SAR

**Volume**:
- OBV (On-Balance Volume)
- VWAP (Volume-Weighted Average Price)
- Relative Volume

**Macro**:
- Correlation with SPY
- Beta calculation
- Sector relative strength

---

## Integration with Backtesting

### Usage Pattern

```python
from pathlib import Path
import pandas as pd

# Load windowed dataset
windows_metadata = pd.read_csv('data/comprehensive/windows_metadata.csv')

# Get all AAPL windows in TECH sector
aapl_windows = windows_metadata[
    (windows_metadata['symbol'] == 'AAPL') &
    (windows_metadata['sector'] == 'TECH')
]

# Run backtest on each window
for _, window in aapl_windows.iterrows():
    window_file = f"data/comprehensive/windows/AAPL_window_{window['window_id']:03d}_{window['start_date']}_{window['end_date']}.csv"

    data = pd.read_csv(window_file, index_col=0, parse_dates=True)

    # Run backtest
    results = run_backtest(data, strategy='RSI')

    # Aggregate results
    all_results.append({
        'window_id': window['window_id'],
        'start': window['start_date'],
        'end': window['end_date'],
        'return': results.total_return,
        'sharpe': results.sharpe_ratio,
    })
```

### Comprehensive Suite Integration

```bash
# Generate datasets
python scripts/dataset_manager.py \
  --mode combined \
  --years 20 \
  --sectors TECH,FINANCE,HEALTHCARE \
  --output data/full_history

# Run comprehensive backtests
python scripts/comprehensive_backtest_suite.py \
  --data-dir data/full_history/windows \
  --output results/full_suite_historical
```

---

## Performance Considerations

### Synthetic Data

**Speed**: Very fast
- 10 years × 100 symbols: ~5 seconds
- 20 years × 250 symbols: ~30 seconds

**Memory**: Low
- Each dataset: ~2MB (5000 bars × 10 features)
- 100 datasets: ~200MB total

### Historical Data

**Speed**: Network-dependent
- 1 symbol × 20 years: ~2-5 seconds
- 100 symbols × 20 years: ~5-10 minutes (with rate limiting)

**Memory**: Moderate
- Each dataset: ~2-5MB (5000 bars × 15 features)
- 100 datasets: ~300MB total

### Windowing

**Speed**: Fast (I/O bound)
- 100 symbols × 237 windows: ~2-3 minutes

**Storage**: Significant
- Each window: ~100KB (63 bars × 15 features)
- 100 symbols × 237 windows: ~2.4GB total

**Recommendations**:
- Use Parquet for large datasets (5-10x compression)
- Parallelize historical data fetching
- Cache datasets locally

---

## Best Practices

### Dataset Selection

**For Development/Testing**:
- Use synthetic data (fast, reproducible)
- 5-10 years, 10-20 symbols
- Quick validation of strategy logic

**For Strategy Validation**:
- Use historical data (real market conditions)
- 10-15 years, 50-100 symbols
- Include bear market period (2008, 2020)

**For Production**:
- Use combined datasets
- 20+ years, 100+ symbols
- All sectors, multiple market regimes
- Walk-forward validation with windows

### Data Quality Checks

Before running backtests:

```python
# Check for missing data
missing = data.isnull().sum()
print(f"Missing values: {missing[missing > 0]}")

# Check price continuity
returns = data['close'].pct_change()
outliers = returns[returns.abs() > 0.2]  # >20% daily moves
print(f"Outliers: {len(outliers)}")

# Check volume
zero_volume = data[data['volume'] == 0]
print(f"Zero volume days: {len(zero_volume)}")
```

### Reproducibility

For synthetic data:
```python
# Seed is based on symbol hash
np.random.seed(hash(symbol + sector) % (2**32))
```

Same symbol + sector = same data every time

---

## Examples

### Example 1: Quick Synthetic Test

```bash
# Generate 5 years of tech stocks for quick testing
python scripts/dataset_manager.py \
  --mode synthetic \
  --years 5 \
  --sectors TECH \
  --symbols-per-sector 3 \
  --output data/quick_test \
  --no-windows
```

**Output**: AAPL, MSFT, GOOGL with 5 years of synthetic data

### Example 2: Historical Bull/Bear Analysis

```bash
# Get 20 years of data including 2008 and 2020 crashes
python scripts/dataset_manager.py \
  --mode historical \
  --years 20 \
  --sectors FINANCE,ENERGY \
  --symbols-per-sector 5 \
  --output data/crisis_analysis
```

**Output**: 10 finance + energy stocks with real crisis data

### Example 3: Full Production Dataset

```bash
# Generate complete dataset for production backtesting
python scripts/dataset_manager.py \
  --mode combined \
  --years 20 \
  --window-months 3 \
  --sectors TECH,FINANCE,HEALTHCARE,ENERGY,CONSUMER,INDUSTRIAL \
  --symbols-per-sector 10 \
  --output data/production \
  --format parquet
```

**Output**:
- 60 symbols × 20 years = 1,200 datasets
- ~14,220 windows (237 per symbol)
- Synthetic + historical + macro + volatility features
- Compressed Parquet format (~500MB total)

---

## Troubleshooting

### Issue: Historical data fetch fails

**Cause**: Network issues, rate limiting, invalid symbols

**Solution**:
```bash
# Check symbol validity first
python -c "import yfinance as yf; print(yf.Ticker('AAPL').info['symbol'])"

# Use fewer symbols or add delays
--symbols-per-sector 3  # Instead of 10
```

### Issue: Synthetic data unrealistic

**Cause**: Default parameters don't match your needs

**Solution**: Edit `generate_synthetic_data()` function
```python
# Adjust volatility
volatility = 0.03  # Higher volatility

# Adjust drift
drift = 0.0005  # Positive trend

# Adjust autocorrelation
returns[i] += 0.2 * returns[i-1]  # Stronger momentum
```

### Issue: Windows take too much space

**Cause**: Many symbols × many windows × CSV format

**Solution**:
```bash
# Use Parquet (5-10x compression)
--format parquet

# Or skip windowing for now
--no-windows

# Or use larger windows (fewer total windows)
--window-months 6
```

---

## Roadmap

### Version 2.0 (Planned)

**Data Sources**:
- [ ] Polygon.io integration
- [ ] Alpha Vantage fundamentals
- [ ] IEX Cloud real-time
- [ ] Quandl alternative data

**Features**:
- [ ] Options data (implied volatility, Greeks)
- [ ] Fundamental data (PE, EPS, revenue)
- [ ] Sentiment data (news, social media)
- [ ] Alternative data (satellite, credit card)

**Performance**:
- [ ] Parallel data fetching
- [ ] Incremental updates (fetch only new data)
- [ ] Data validation pipeline
- [ ] Automated data quality reports

**Windowing**:
- [ ] Discrete windows (non-overlapping)
- [ ] Expanding windows (walk-forward)
- [ ] Custom window definitions
- [ ] Regime-based windowing

---

## Support

For issues or questions:
- GitHub: https://github.com/[your-repo]/ordinis
- Documentation: docs/DATASET_MANAGEMENT_GUIDE.md
- Examples: examples/dataset_examples.py

---

**Last Updated**: 2025-12-10
**Version**: 1.0.0
