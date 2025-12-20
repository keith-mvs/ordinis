# Technical Indicators Skill Package

## Overview

Comprehensive skill package for mastering twelve core technical indicators used in market analysis and trading. This package provides expert-level guidance on calculation, interpretation, and practical application of trend, momentum, volatility, and volume indicators.

## Package Contents

### Core Files

- **SKILL.md**: Main skill document with YAML frontmatter, quick reference tables, workflows, and navigation to detailed resources
- **README.md**: This file - package overview and usage instructions

### Reference Materials (`references/`)

Detailed methodology, calculation formulas, and interpretation guidelines organized by indicator category:

- **trend_indicators.md**: ADX, Ichimoku Cloud, Moving Averages, Parabolic SAR
- **momentum_indicators.md**: CCI, MACD, RSI, Stochastic Oscillator
- **volatility_indicators.md**: ATR, Bollinger Bands
- **volume_indicators.md**: OBV, RVOL, VWAP, A/D Line
- **static_levels.md**: Fibonacci Retracement and Extension levels

### Implementation Scripts (`scripts/`)

- **calculate_indicators.py**: Production-ready Python implementation of all twelve indicators with CLI interface

### Practical Examples (`examples/`)

- **trend_following_cases.md**: Real-world case studies demonstrating indicator application across different market conditions

## Installation

### Prerequisites

```bash
pip install numpy pandas matplotlib scipy
pip install yfinance  # For data fetching
pip install ta  # Optional: Technical Analysis library
```

### Directory Structure

```
technical-indicators/
â"œâ"€â"€ SKILL.md
â"œâ"€â"€ README.md
â"œâ"€â"€ references/
â"‚   â"œâ"€â"€ trend_indicators.md
â"‚   â"œâ"€â"€ momentum_indicators.md
â"‚   â"œâ"€â"€ volatility_indicators.md
â"‚   â"œâ"€â"€ volume_indicators.md
â"‚   â""â"€â"€ static_levels.md
â"œâ"€â"€ scripts/
â"‚   â""â"€â"€ calculate_indicators.py
â""â"€â"€ examples/
    â""â"€â"€ trend_following_cases.md
```

## Usage

### Command-Line Interface

Calculate RSI:
```bash
python scripts/calculate_indicators.py --symbol SPY --indicator RSI --period 14
```

Calculate MACD:
```bash
python scripts/calculate_indicators.py --symbol QQQ --indicator MACD
```

Calculate Bollinger Bands:
```bash
python scripts/calculate_indicators.py --symbol AAPL --indicator BOLLINGER --period 20 --std-dev 2.0
```

Save results to file:
```bash
python scripts/calculate_indicators.py --symbol SPY --indicator RSI --output results.csv
```

### Supported Indicators

| Indicator | Command Flag | Description |
|-----------|-------------|-------------|
| RSI | `--indicator RSI` | Relative Strength Index |
| MACD | `--indicator MACD` | Moving Average Convergence Divergence |
| BOLLINGER | `--indicator BOLLINGER` | Bollinger Bands |
| ATR | `--indicator ATR` | Average True Range |
| ADX | `--indicator ADX` | Average Directional Index |
| STOCHASTIC | `--indicator STOCHASTIC` | Stochastic Oscillator |
| CCI | `--indicator CCI` | Commodity Channel Index |
| OBV | `--indicator OBV` | On-Balance Volume |
| MA | `--indicator MA` | Moving Averages (SMA & EMA) |
| PSAR | `--indicator PSAR` | Parabolic SAR |
| FIBONACCI | `--indicator FIBONACCI` | Fibonacci Retracement Levels |

### Python API

Import and use directly in your code:

```python
from scripts.calculate_indicators import TechnicalIndicators
import pandas as pd

# Load your OHLCV data
data = pd.read_csv('market_data.csv')

# Initialize calculator
calc = TechnicalIndicators()

# Calculate indicators
rsi = calc.calculate_rsi(data['Close'], period=14)
macd = calc.calculate_macd(data['Close'])
bb = calc.calculate_bollinger_bands(data['Close'], period=20, std_dev=2.0)
```

## Learning Path

### Beginner
1. Read SKILL.md overview and quick reference
2. Study one indicator category (start with momentum indicators)
3. Run calculations using provided script
4. Review trend following case study

### Intermediate
1. Study all indicator categories in references/
2. Implement indicators in personal trading analysis
3. Practice multi-indicator confirmation workflows
4. Analyze historical trades using indicator framework

### Advanced
1. Combine indicators for custom trading systems
2. Optimize parameters for specific securities
3. Develop backtesting frameworks
4. Create regime-adaptive indicator strategies

## Skill Activation in Claude

This skill is automatically activated when:
- Analyzing price charts or market data
- Evaluating technical trading signals
- Implementing quantitative trading strategies
- Discussing trend strength, momentum, volatility, or volume
- Mentioning specific indicators by name

## Authoritative Sources

All methodologies derive from established references:
- CMT Association curriculum
- Edwards & Magee: Technical Analysis of Stock Trends
- Bloomberg Market Concepts
- Murphy: Technical Analysis of the Financial Markets
- Pring: Technical Analysis Explained

## Ethical Considerations

- Technical analysis does not guarantee profits
- Past performance does not predict future results
- Implement robust risk management protocols
- Be aware of market impact and behavioral influences
- Consider information asymmetry in high-frequency environments

## Validation Framework

Test indicator effectiveness using:
- Backtesting across multiple market regimes
- Win rate and profit factor metrics
- Risk-adjusted returns (Sharpe ratio)
- Maximum drawdown analysis
- Parameter sensitivity testing

## Integration with Other Skills

This skill integrates with:
- Portfolio management (position sizing using ATR)
- Risk management (volatility-based stop losses)
- Options strategies (volatility indicators for options pricing)
- Quantitative analysis (indicator-based factor models)

## Contributing

Improvements welcome:
- Additional case studies
- New indicator implementations
- Enhanced visualization functions
- Optimization algorithms
- Backtesting frameworks

## License

Educational and research use. Not financial advice.

## Version

Version 1.0.0 - December 2024

## Support

For questions or issues:
- Review detailed methodology in references/
- Study case examples in examples/
- Check calculation accuracy against authoritative sources
- Validate results with multiple data sources

---

**Remember**: Indicators are tools for analysis, not crystal balls. Always combine technical analysis with fundamental research, risk management, and sound judgment.
