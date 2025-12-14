# CLI Usage Guide

Command-line interface for the Ordinis trading system. Run backtests, analyze strategies, and generate performance reports directly from your terminal.

## Installation

After installing the package, the `ordinis` command will be available:

```bash
pip install -e .
```

Verify installation:

```bash
ordinis --help
```

## Quick Start

### 1. List Available Strategies

```bash
ordinis list
```

Output:
```
Available Strategies:

  rsi          - RSI Mean Reversion
               Counter-trend using RSI indicator

  ma           - Moving Average Crossover
               Trend following with MA signals

  momentum     - Momentum Breakout
               Volatility breakouts with confirmation
```

### 2. Run a Simple Backtest

```bash
ordinis backtest --data data/SPY_2024.csv --strategy rsi
```

This runs a backtest using:
- RSI Mean Reversion strategy
- Default parameters (RSI period=14, oversold=30, overbought=70)
- Initial capital $100,000
- 10% position size

### 3. Customize Strategy Parameters

```bash
ordinis backtest \
  --data data/SPY_2024.csv \
  --strategy rsi \
  --params rsi_period=14 oversold_threshold=25 overbought_threshold=75
```

### 4. Run Technical Analysis (Phase 3)

Generate Ichimoku snapshot, candlestick/breakout signals, composite bias, and multi-timeframe alignment on an OHLCV CSV:

```bash
ordinis analyze --data data/AAPL_historical.csv --breakout-lookback 20 --breakout-volume-mult 1.5
```

Outputs include:
- Trend/bias with Ichimoku context
- Active candlestick patterns and breakout status (with volume confirmation)
- Multi-timeframe majority trend and agreement score
- Composite score combining bias + Ichimoku trend

### 5. Portfolio Optimization (QPO)

Run Mean-CVaR optimization using the NVIDIA QPO blueprint:

```bash
ordinis optimize \
  --returns examples/data/sample_returns.csv \
  --target-return 0.001 \
  --max-weight 0.2 \
  --risk-aversion 0.5 \
  --api cvxpy \
  --dry-run
```

Notes:
- `--dry-run` validates the environment without solving. Remove it to execute.
- Requires RAPIDS/cuOpt + cvxpy (GPU stack); see QPO blueprint docs.
- You can override the QPO source path via `--qpo-src` if the blueprint lives elsewhere.

## Commands

### backtest

Run a strategy backtest on historical data.

**Usage:**
```bash
ordinis backtest [OPTIONS]
```

**Required Arguments:**

- `--data PATH`: Path to CSV file with market data (must have OHLCV columns and timestamp/date)
- `--strategy {rsi,ma,momentum}`: Strategy to backtest

**Optional Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--params` | KEY=VALUE ... | - | Strategy-specific parameters |
| `--capital` | float | 100000 | Initial capital |
| `--position-size` | float | 0.1 | Position size as fraction of capital (0.0-1.0) |
| `--signal-frequency` | int | 1 | Check for signals every N bars |
| `--risk-free` | float | 0.02 | Risk-free rate for Sharpe ratio |
| `--ai` | flag | False | Enable AI-powered analysis |
| `--nvidia-key` | string | - | NVIDIA API key for AI features |
| `--suggestions` | flag | False | Generate optimization suggestions (requires --ai) |
| `--focus` | choice | general | Focus area: returns, risk, consistency, general |
| `--output` | PATH | - | Save results to CSV file |

**Examples:**

```bash
# Basic RSI backtest
ordinis backtest --data data.csv --strategy rsi

# MA crossover with custom parameters
ordinis backtest \
  --data data.csv \
  --strategy ma \
  --params fast_period=50 slow_period=200 ma_type=SMA

# Momentum with AI analysis
ordinis backtest \
  --data data.csv \
  --strategy momentum \
  --ai \
  --nvidia-key nvapi-... \
  --suggestions \
  --focus returns

# Save results to file
ordinis backtest \
  --data data.csv \
  --strategy rsi \
  --output results.csv
```

### list

List all available strategies with descriptions.

**Usage:**
```bash
ordinis list
```

## Data Format

The CLI expects CSV files with the following columns:

### Required Columns

- `timestamp` or `date`: DateTime column (will be set as index)
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

### Optional Columns

- `symbol`: Stock ticker symbol (otherwise uses filename)

### Example CSV

```csv
timestamp,open,high,low,close,volume
2024-01-01 09:30:00,100.00,101.50,99.50,101.00,1000000
2024-01-02 09:30:00,101.00,102.00,100.50,101.50,1200000
2024-01-03 09:30:00,101.50,103.00,101.00,102.50,1500000
```

## Strategy Parameters

### RSI Mean Reversion

**Parameters:**
- `rsi_period` (int, default=14): Period for RSI calculation
- `oversold_threshold` (int, default=30): RSI level for buy signals
- `overbought_threshold` (int, default=70): RSI level for sell signals
- `extreme_oversold` (int, default=20): Extreme oversold level
- `extreme_overbought` (int, default=80): Extreme overbought level

**Example:**
```bash
ordinis backtest \
  --data data.csv \
  --strategy rsi \
  --params rsi_period=14 oversold_threshold=30 overbought_threshold=70
```

### Moving Average Crossover

**Parameters:**
- `fast_period` (int, default=50): Fast MA period
- `slow_period` (int, default=200): Slow MA period
- `ma_type` (str, default='SMA'): Type of MA - 'SMA' or 'EMA'

**Example:**
```bash
ordinis backtest \
  --data data.csv \
  --strategy ma \
  --params fast_period=50 slow_period=200 ma_type=EMA
```

### Momentum Breakout

**Parameters:**
- `lookback_period` (int, default=20): Period for high/low range
- `atr_period` (int, default=14): ATR period
- `volume_multiplier` (float, default=1.5): Volume confirmation threshold
- `breakout_threshold` (float, default=0.02): Breakout distance threshold (2%)

**Example:**
```bash
ordinis backtest \
  --data data.csv \
  --strategy momentum \
  --params lookback_period=20 atr_period=14 volume_multiplier=1.5
```

## AI-Powered Features

Enable AI analysis with NVIDIA models for enhanced insights:

### Performance Narration

Get AI-generated analysis of backtest results:

```bash
ordinis backtest \
  --data data.csv \
  --strategy rsi \
  --ai \
  --nvidia-key nvapi-...
```

Output includes:
- Overall assessment
- Key strengths and weaknesses
- Risk analysis
- Natural language performance summary

### Optimization Suggestions

Get AI-powered recommendations for improving strategy performance:

```bash
ordinis backtest \
  --data data.csv \
  --strategy rsi \
  --ai \
  --nvidia-key nvapi-... \
  --suggestions \
  --focus returns
```

**Focus Areas:**
- `returns`: Maximize returns
- `risk`: Minimize risk and drawdown
- `consistency`: Improve win rate and stability
- `general`: Balanced optimization

## Output Format

### Console Output

```
================================================================================
INTELLIGENT INVESTOR - BACKTEST
================================================================================

[1/5] Loading market data from data.csv...
  Symbol: SPY
  Bars: 252
  Period: 2024-01-01 to 2024-12-31

[2/5] Creating rsi strategy...
  Strategy: rsi-backtest
  Required bars: 34

[3/5] Configuring backtest...
  Initial Capital: $100,000
  Risk-Free Rate: 2.0%

[4/5] Running backtest...
  Processing 252 bars...

  Signals Generated: 45
  Signals Executed: 12

[5/5] Analyzing results...

================================================================================
BACKTEST RESULTS
================================================================================

Performance:
  Total Return: 15.23%
  Annualized Return: 15.23%
  Sharpe Ratio: 1.45
  Sortino Ratio: 2.10
  Calmar Ratio: 1.88

Risk:
  Max Drawdown: -8.12%
  Volatility: 12.45%
  Downside Deviation: 6.78%

Trades:
  Total Trades: 12
  Win Rate: 58.3%
  Profit Factor: 1.85
  Avg Win: $1,250.00
  Avg Loss: -$680.00
```

### CSV Output (--output)

When using `--output results.csv`, the following metrics are saved:

| Metric | Value |
|--------|-------|
| total_return | 0.1523 |
| annualized_return | 0.1523 |
| sharpe_ratio | 1.45 |
| max_drawdown | -0.0812 |
| win_rate | 0.583 |
| num_trades | 12 |

## Advanced Usage

### Batch Processing

Run multiple backtests with different parameters:

```bash
#!/bin/bash
# backtest_sweep.sh

for rsi_period in 10 14 20; do
  for threshold in 25 30 35; do
    ordinis backtest \
      --data data.csv \
      --strategy rsi \
      --params rsi_period=$rsi_period oversold_threshold=$threshold \
      --output results_${rsi_period}_${threshold}.csv
  done
done
```

### Pipeline Integration

Use the CLI in data pipelines:

```bash
# Download data
python scripts/download_data.py --symbol SPY --start 2024-01-01

# Run backtest
ordinis backtest \
  --data data/SPY.csv \
  --strategy rsi \
  --output results.csv

# Analyze results
python scripts/analyze_results.py --input results.csv
```

### Docker Usage

Run backtests in Docker:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e .

ENTRYPOINT ["ordinis"]
```

```bash
# Build
docker build -t ordinis .

# Run
docker run -v $(pwd)/data:/app/data ordinis backtest \
  --data /app/data/SPY.csv \
  --strategy rsi
```

## Troubleshooting

### Common Issues

**1. Command not found**
```bash
# Solution: Reinstall package
pip install -e .
```

**2. Data format error**
```
ValueError: Missing required columns: ['open', 'high', 'low', 'close', 'volume']
```
Solution: Ensure CSV has all required OHLCV columns.

**3. Insufficient data**
```
[WARNING] Insufficient data: 50 < 100
```
Solution: Provide more historical data or use a strategy with lower requirements.

**4. NVIDIA API key error**
```
[ERROR] NVIDIA API key required for AI features
```
Solution: Set `--nvidia-key` or environment variable:
```bash
export NVIDIA_API_KEY='nvapi-...'
```

### Debug Mode

Enable verbose output:

```bash
# Add debug logging (if implemented)
LOGLEVEL=DEBUG ordinis backtest --data data.csv --strategy rsi
```

## Performance Tips

### 1. Data Quality

- Use clean, validated data
- Ensure no gaps or missing values
- Verify timestamp ordering

### 2. Signal Frequency

- Increase `--signal-frequency` for faster backtests
- Lower frequency for more thorough testing

### 3. Position Sizing

- Start with small positions (0.05-0.10)
- Increase gradually based on results
- Never risk more than 1-2% per trade

### 4. Parameter Optimization

- Test range of parameters systematically
- Avoid overfitting to historical data
- Use walk-forward analysis

## Examples

### Example 1: Quick Test

Test a strategy on recent data:

```bash
ordinis backtest \
  --data data/recent.csv \
  --strategy rsi \
  --capital 10000 \
  --position-size 0.05
```

### Example 2: Full Analysis

Complete backtest with AI insights:

```bash
ordinis backtest \
  --data data/SPY_5years.csv \
  --strategy ma \
  --params fast_period=50 slow_period=200 \
  --capital 100000 \
  --position-size 0.1 \
  --ai \
  --nvidia-key $NVIDIA_API_KEY \
  --suggestions \
  --focus returns \
  --output results/ma_50_200.csv
```

### Example 3: Strategy Comparison

Compare different strategies:

```bash
for strategy in rsi ma momentum; do
  ordinis backtest \
    --data data/SPY.csv \
    --strategy $strategy \
    --output results/${strategy}_results.csv
done
```

## Next Steps

- Review the strategy implementations in `src/strategies/` for detailed documentation
- Check the examples in `scripts/` for Python API usage
- Explore [NVIDIA Integration](../architecture/nvidia-integration.md) for AI features

---

**Version:** 1.0.0
**Last Updated:** 2025-11-29
