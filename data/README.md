# Data Directory Structure

## Overview

Organized dataset repository for Ordinis trading system backtesting and analysis.

## Directory Structure

```
data/
├── historical/          25 stocks with real market data from yfinance
├── synthetic/           25 stocks with synthetic GBM-generated data
├── macro/               5 macro economic indicators
├── raw/                 Sample and test datasets
├── chromadb/            Vector database for RAG system
├── historical_cache/    Cache for historical data fetches
└── dataset_metadata.csv Master index of all datasets
```

## Dataset Details

### Historical Data (`historical/`)

**Source**: Yahoo Finance (yfinance API)
**Coverage**: 20 years (2005-12-15 to 2025-12-10)
**Bars**: ~5,028 per symbol
**Features**: open, high, low, close, volume, adj_close, symbol, true_range, atr_14, hvol_20, parkinson_vol_20

**Symbols by Sector**:
- **TECH** (5): AAPL, MSFT, GOOGL, NVDA, META
- **FINANCE** (5): JPM, BAC, WFC, GS, MS
- **HEALTHCARE** (5): UNH, JNJ, PFE, ABBV, TMO
- **ENERGY** (5): XOM, CVX, COP, SLB, EOG
- **CONSUMER** (5): WMT, HD, MCD, NKE, SBUX

**File Format**: `{SYMBOL}_historical.csv`

### Synthetic Data (`synthetic/`)

**Source**: Geometric Brownian Motion simulation
**Coverage**: 20 years (2005-12-15 to 2025-12-10)
**Bars**: ~5,215 per symbol
**Features**: open, high, low, close, volume, symbol, sector

**Methodology**:
- Sector-specific volatility multipliers (TECH: 1.3x, FINANCE: 1.5x, HEALTHCARE: 0.9x, etc.)
- Realistic OHLC relationships (high >= max(open, close), low <= min(open, close))
- Volume with realistic variation

**File Format**: `{SYMBOL}_synthetic.csv`

### Macro Indicators (`macro/`)

**Source**: Yahoo Finance
**Coverage**: 20 years
**Symbols**:
- SPY: S&P 500 ETF (market benchmark)
- ^VIX: CBOE Volatility Index
- ^TNX: 10-Year Treasury Yield
- GLD: Gold ETF
- USO: Oil ETF

**File Format**: `{SYMBOL}_macro.csv` or `INDEX_{NAME}_macro.csv`

### Raw/Sample Data (`raw/`)

**Contents**:
- `real_spy_daily.csv`: Original SPY data for testing
- `sample_spy_trending_up.csv`: Uptrend test scenario
- `sample_abc_trending_down.csv`: Downtrend test scenario
- `sample_xyz_sideways.csv`: Sideways market test scenario
- `sample_qqq_volatile.csv`: High volatility test scenario

## Metadata

### `dataset_metadata.csv`

Master index containing:
- symbol: Stock ticker
- sector: GICS sector classification
- start_date: First timestamp
- end_date: Last timestamp
- num_bars: Total number of bars
- num_features: Number of columns
- avg_price: Mean close price
- avg_volume: Mean daily volume
- volatility: Historical volatility
- source: Data source (synthetic, yfinance)

## Usage Examples

### Load Historical Data

```python
import pandas as pd

# Load single symbol
aapl_hist = pd.read_csv("data/historical/AAPL_historical.csv", index_col=0, parse_dates=True)

# Load all TECH stocks
import glob
tech_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
tech_data = {sym: pd.read_csv(f"data/historical/{sym}_historical.csv", index_col=0, parse_dates=True)
             for sym in tech_symbols}
```

### Compare Synthetic vs Historical

```python
# Load both versions
aapl_hist = pd.read_csv("data/historical/AAPL_historical.csv", index_col=0, parse_dates=True)
aapl_synth = pd.read_csv("data/synthetic/AAPL_synthetic.csv", index_col=0, parse_dates=True)

# Compare volatility
print(f"Historical volatility: {aapl_hist['close'].pct_change().std() * (252**0.5):.2%}")
print(f"Synthetic volatility: {aapl_synth['close'].pct_change().std() * (252**0.5):.2%}")
```

### Backtest with Historical Data

```bash
python -m src.cli backtest \
  --data data/historical/AAPL_historical.csv \
  --strategy rsi \
  --capital 100000
```

## Maintenance

### Regenerate Historical Data

```bash
python scripts/dataset_manager.py \
  --mode historical \
  --years 20 \
  --sectors TECH,FINANCE,HEALTHCARE,ENERGY,CONSUMER \
  --symbols-per-sector 5 \
  --output data \
  --format csv \
  --no-windows
```

### Regenerate Synthetic Data

```bash
python scripts/dataset_manager.py \
  --mode synthetic \
  --years 20 \
  --sectors TECH,FINANCE,HEALTHCARE,ENERGY,CONSUMER \
  --symbols-per-sector 5 \
  --output data \
  --format csv \
  --no-windows
```

### Update Metadata

Metadata is automatically regenerated when running `dataset_manager.py`. To manually update:

```python
from scripts.dataset_manager import generate_metadata
import pathlib

output_dir = pathlib.Path("data")
metadata = generate_metadata(datasets_dict, output_dir)
```

## Data Quality

### Historical Data Validation

All historical datasets include:
- Adjusted close prices (split/dividend adjusted)
- Data quality checks (no gaps, valid OHLC relationships)
- Minimum 3,000 bars for reliable backtesting

### Synthetic Data Calibration

Synthetic data is calibrated to match:
- Sector-appropriate volatility levels
- Realistic price movements (no negative prices)
- Consistent OHLC bar structure

## Storage

**Total Size**: ~32 MB
- Historical: ~16 MB (25 stocks × ~640 KB avg)
- Synthetic: ~14 MB (25 stocks × ~540 KB avg)
- Macro: ~3 MB (5 indicators × ~640 KB avg)
- Raw: ~1 MB (5 sample files)

## Notes

- Historical data fetched via yfinance may have slight variations on re-fetch due to data revisions
- DXY (US Dollar Index) is not available via yfinance and was excluded from macro datasets
- All timestamps are in UTC
- CSV format uses comma delimiters with headers

---

**Last Updated**: 2025-12-10
**Version**: 1.0.0
