# Ordinis Examples

Example scripts demonstrating various Ordinis features and strategies.

## Options Strategies

### Covered Call Strategy
- **covered-call-backtest.py** - Historical backtest of covered call income generation strategy
- **covered-call-analysis.py** - Interactive scenario analysis for covered calls
- **covered-call-demo.py** - Quick demonstration of covered call mechanics

### Married Put Strategy
- **married-put-backtest.py** - Historical backtest of married put protective strategy

### General Options
- **optionscore-demo.py** - Demonstration of OptionsCore pricing engine

## General Backtesting
- **complete-backtest-demo.py** - Comprehensive backtesting framework demonstration
- **technical_phase3_demo.py** - Phase 3 technical analytics (Ichimoku, patterns, composite, multi-timeframe)

## Running Examples

All examples can be run from the project root:

```bash
# Activate virtual environment first
.\.venv\Scripts\Activate.ps1

# Run any example
python examples/covered-call-backtest.py
python examples/married-put-backtest.py
python examples/optionscore-demo.py
```

## Data Requirements

Most backtest examples require historical data in `data/historical/`:
- AAPL_historical.csv
- MSFT_historical.csv
- etc.

Generate historical data:
```bash
python scripts/dataset_manager.py --mode historical
```

## Results

Backtest results are saved to:
- `data/backtest_results/` - CSV files with equity curves and trade logs
- Summary text files with performance metrics

## Documentation

Strategy guides and detailed documentation:
- See `docs/strategies/` for strategy documentation
- See `reports/` for backtest reports and analysis

---

**Last Updated**: 2025-12-11
