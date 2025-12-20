# Scripts Directory

Utility and operational scripts organized by purpose.

## Directory Structure

```
scripts/
├── data/              # Data fetching and management
├── backtesting/       # Backtesting and performance analysis
├── trading/           # Live and paper trading execution
├── demo/              # Demonstration and example scripts
├── analysis/          # Analysis and reporting tools
├── skills/            # Claude Code skills management
├── docs/              # Documentation generation
├── rag/               # RAG system indexing and querying
├── utils/             # Platform-specific utilities (PowerShell)
└── README.md          # This file
```

## Scripts by Category

### Data Management (`data/`)

**Purpose**: Fetch, manage, and generate market data

| Script | Description |
|--------|-------------|
| `dataset_manager.py` | Dataset management utilities and metadata generation |
| `fetch_real_data.py` | Fetch real market data for specific symbols |
| `fetch_enhanced_datasets.py` | Bulk fetch with enhanced metadata |
| `fetch_parallel.py` | Parallel data fetching for performance |
| `generate_sample_data.py` | Create synthetic test datasets |
| `test_data_fetch.py` | Test data API connections |
| `test_training_data.py` | Validate training data quality |
| `enhanced_dataset_config.py` | Dataset configuration and validation |

**Usage Examples**:
```bash
# Fetch real data for a symbol
python scripts/data/fetch_real_data.py --symbol AAPL --start 2020-01-01

# Generate synthetic test data
python scripts/data/generate_sample_data.py --symbols 10 --years 5

# Manage datasets
python scripts/data/dataset_manager.py --check --update-metadata
```

### Backtesting (`backtesting/`)

**Purpose**: Run backtests and analyze performance

| Script | Description |
|--------|-------------|
| `comprehensive_backtest_suite.py` | Full suite of strategy backtests |
| `run_backtest_demo.py` | Quick backtest demonstration |
| `run_real_backtest.py` | Backtest with real historical data |
| `run_adaptive_backtest.py` | Adaptive strategy backtesting |
| `run_multi_market_backtest.py` | Multi-symbol/market backtesting |
| `run_regime_backtest.py` | Market regime-based backtesting |
| `backtest_new_indicators.py` | Test new technical indicators |
| `analyze_backtest_results.py` | Analyze backtest performance |
| `monitor_backtest_suite.py` | Monitor long-running backtest suites |
| `debug_parabolic_sar.py` | Debug specific indicator implementations |

**Usage Examples**:
```bash
# Run quick backtest demo
python scripts/backtesting/run_backtest_demo.py --strategy bollinger_bands

# Run comprehensive suite
python scripts/backtesting/comprehensive_backtest_suite.py --all-strategies

# Analyze results
python scripts/backtesting/analyze_backtest_results.py --results-dir data/backtest_results/
```

### Trading (`trading/`)

**Purpose**: Execute live and paper trading

| Script | Description |
|--------|-------------|
| `run_paper_trading.py` | Run paper trading simulation |
| `run_risk_managed_trading.py` | Trade with RiskGuard integration |
| `test_paper_broker.py` | Test paper broker connectivity |
| `test_live_trading.py` | Test live trading pipeline |
| `test_market_data_apis.py` | Verify market data API connections |

**Usage Examples**:
```bash
# Run paper trading
python scripts/trading/run_paper_trading.py --strategy rsi --capital 100000

# Test broker connection
python scripts/trading/test_paper_broker.py --broker alpaca

# Test market data APIs
python scripts/trading/test_market_data_apis.py --all
```

### Demo (`demo/`)

**Purpose**: Demonstration and showcase scripts

| Script | Description |
|--------|-------------|
| `comprehensive_demo.py` | Full system capabilities demonstration |
| `demo_full_system.py` | End-to-end trading system demo |

**Usage Examples**:
```bash
# Run full system demo
python scripts/demo/demo_full_system.py

# Run comprehensive showcase
python scripts/demo/comprehensive_demo.py --verbose
```

### Analysis (`analysis/`)

**Purpose**: Analysis and reporting tools

| Script | Description |
|--------|-------------|
| `extended_analysis.py` | Extended performance analysis |
| `generate_consolidated_report.py` | Generate comprehensive reports |
| `wait_and_report.py` | Wait for completion and report results |

**Usage Examples**:
```bash
# Run extended analysis
python scripts/analysis/extended_analysis.py --results-dir results/

# Generate consolidated report
python scripts/analysis/generate_consolidated_report.py --format pdf
```

### Skills Management (`skills/`)

**Purpose**: Claude Code skills development and maintenance

| Script | Description |
|--------|-------------|
| `audit_assets.py` | Audit skill asset files |
| `audit_references.py` | Audit skill reference documentation |
| `audit_skills.py` | Comprehensive skill compliance audit |
| `check_skill_compliance.py` | Verify Claude Code skill standards |
| `consolidate_dependencies.py` | Consolidate skill dependencies |
| `generate_all_references.py` | Generate reference documentation |
| `generate_skill_assets.py` | Generate skill asset files |
| `refactor_skills.py` | Refactor skill structure |

**Usage Examples**:
```bash
# Audit all skills
python scripts/skills/audit_skills.py --verbose

# Generate assets
python scripts/skills/generate_skill_assets.py --all

# Check compliance
python scripts/skills/check_skill_compliance.py --strict
```

### Documentation (`docs/`)

**Purpose**: Documentation generation and processing

| Script | Description |
|--------|-------------|
| `add_frontmatter.py` | Add frontmatter to documentation files |
| `process_docs.py` | Process and build documentation |

**Usage Examples**:
```bash
# Add frontmatter to docs
python scripts/docs/add_frontmatter.py --dir docs/

# Build documentation
python scripts/docs/process_docs.py --output site/
```

### RAG System (`rag/`)

**Purpose**: RAG (Retrieval-Augmented Generation) system operations

| Script | Description |
|--------|-------------|
| `index_knowledge_base.py` | Index trading knowledge base |
| `index_kb_minimal.py` | Minimal knowledge base indexing |
| `start_rag_server.py` | Start RAG query server |
| `test_cortex_rag.py` | Test CortexRAG functionality |

**Usage Examples**:
```bash
# Index knowledge base
python scripts/rag/index_knowledge_base.py --kb-dir docs/knowledge-base/

# Start RAG server
python scripts/rag/start_rag_server.py --port 8000

# Test RAG queries
python scripts/rag/test_cortex_rag.py --query "What is a covered call?"
```

### Utilities (`utils/`)

**Purpose**: Platform-specific utility scripts

| Script | Description |
|--------|-------------|
| `generate_reference_files.ps1` | PowerShell script for reference generation |

**Usage Examples**:
```powershell
# Run PowerShell utility (Windows)
.\scripts\utils\generate_reference_files.ps1 -Skill married-put
```

## Common Patterns

### Script Arguments
Most scripts accept standard arguments:
- `--verbose, -v`: Enable verbose logging
- `--config PATH`: Specify configuration file
- `--output DIR`: Set output directory
- `--help, -h`: Show help message

### Environment Variables
Scripts respect environment variables from `.env`:
- `ALPHA_VANTAGE_API_KEY`: Alpha Vantage API key
- `FINNHUB_API_KEY`: Finnhub API key
- `MASSIVE_API_KEY`: Massive API key
- `TWELVE_DATA_API_KEY`: Twelve Data API key
- `ALPACA_API_KEY`: Alpaca trading API key
- `ALPACA_SECRET_KEY`: Alpaca secret key

### Logging
Scripts use structured logging:
```python
from loguru import logger

logger.info("Processing data for {symbol}", symbol="AAPL")
logger.error("Failed to fetch data: {error}", error=str(e))
```

Logs are written to `logs/` directory (gitignored).

## Development Guidelines

### Creating New Scripts
1. Choose appropriate category directory
2. Follow naming convention: `verb_noun.py`
3. Include docstring with usage examples
4. Add argument parsing with `argparse`
5. Implement proper error handling
6. Add logging with `loguru`
7. Write tests in `tests/`

### Script Template
```python
"""
Brief description of what this script does.

Usage:
    python script_name.py --option value
"""

import argparse
from pathlib import Path

from loguru import logger


def main(args: argparse.Namespace) -> None:
    """Main execution function."""
    logger.info("Starting {script}", script=__file__)

    # Implementation here

    logger.success("Completed successfully")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option", type=str, help="Option description")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
```

## Troubleshooting

### Script Fails to Import
```bash
# Ensure you're in the project root
cd /path/to/ordinis

# Install in development mode
pip install -e .
```

### Permission Errors (PowerShell)
```powershell
# Set execution policy (Windows)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### API Key Errors
```bash
# Check environment variables
python -c "import os; print(os.getenv('ALPHA_VANTAGE_API_KEY'))"

# Or check .env file
cat .env | grep API_KEY
```

### Data Not Found
```bash
# Verify data directory exists
ls data/

# Fetch required data
python scripts/data/fetch_real_data.py --symbol AAPL
```

## Performance Tips

1. **Use parallel scripts**: `fetch_parallel.py` for bulk operations
2. **Cache results**: Enable caching in configuration
3. **Batch operations**: Process multiple symbols at once
4. **Monitor memory**: Use `--low-memory` flag for large datasets
5. **Optimize queries**: Use indexed data sources

## Best Practices

- Always run scripts from project root
- Use virtual environment
- Check `.env` for required API keys
- Review script documentation (`--help`)
- Check logs in `logs/` for errors
- Validate output before using in production
- Keep scripts focused on single responsibility
- Add comprehensive error handling

## Future Enhancements

- Web UI for script execution
- Script scheduler/cron integration
- Enhanced parallel processing
- Real-time monitoring dashboard
- Automated testing for all scripts
- Docker containerization

## Support

For script-related questions:
- Check script `--help` documentation
- Review code comments in script files
- Consult `CONTRIBUTING.md` for development guidelines
- Open an issue on GitHub

---

**Maintained by**: Ordinis Development Team
**Last Updated**: 2025-12-12
