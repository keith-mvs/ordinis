# Ordinis AI Coding Agent Instructions

**Ordinis** is an AI-driven quantitative trading system built with clean architecture principles. Version 0.2.0-dev implements production-ready infrastructure with event-driven engines communicating via StreamingBus.

---

## Architecture Overview

### Engine-Based Design
The system is composed of independent, event-driven **engines** that inherit from `BaseEngine[ConfigT]`:

```
Market Data → StreamingBus → SignalEngine → RiskEngine → ExecutionEngine → PortfolioEngine → AnalyticsEngine
                              ↑             ↑             ↑                ↑
                              └─────── GovernanceEngine (pre-flight checks & audit) ──┘
```

**Core principle**: All engines extend [BaseEngine](src/ordinis/engines/base/engine.py:37-53) and follow a standard lifecycle:
1. Initialize (`_do_initialize()`)
2. Health checks (`_do_health_check()`)
3. Governance preflight (`self._governance.preflight()`)
4. Shutdown (`_do_shutdown()`)

### Standard Engine Structure
Every engine follows this template (see [base/__init__.py](src/ordinis/engines/base/__init__.py:7-18)):
```
engines/{engine_name}/
├── __init__.py              # Public API exports
├── core/
│   ├── engine.py            # Main engine class (extends BaseEngine)
│   ├── config.py            # Engine config (extends BaseEngineConfig)
│   └── models.py            # Data models/schemas
├── hooks/
│   └── governance.py        # Engine-specific governance
└── {domain_specific}/       # E.g., "strategies/", "models/"
```

**Examples**:
- [SignalCore](src/ordinis/engines/signalcore/) - ML-based signal generation
- [RiskGuard](src/ordinis/engines/riskguard/) - Risk policy enforcement
- [Cortex](src/ordinis/engines/cortex/) - LLM reasoning engine

---

## Development Workflows

### Running Tests
```bash
# All tests (requires 50% coverage minimum)
pytest

# Specific marker
pytest -m unit           # Fast, isolated tests
pytest -m integration    # External dependencies
pytest -m slow           # >1 second tests

# With coverage report
pytest --cov=src --cov-report=html
```

**Coverage target**: >80% for business logic (see [pyproject.toml](pyproject.toml:163))

### Code Quality
Pre-commit hooks run automatically ([.pre-commit-config.yaml](.pre-commit-config.yaml)):
- **Ruff**: Linting + formatting (replaces Black, isort, Flake8)
- **MyPy**: Type checking (disabled by default for large commits)

```bash
# Manual checks
ruff check --fix src/
ruff format src/
mypy src/ordinis/engines/new_feature/
```

### Environment Setup
```bash
# Conda (primary - GPU/CUDA 12.1 enabled)
conda activate ordinis-env

# Verify GPU
pwsh -File C:\Users\kjfle\verify-conda-gpu.ps1

# Install dependencies
pip install -e ".[dev]"        # Development
pip install -e ".[all]"        # Everything
```

**Python**: 3.11+ required. Type hints mandatory for new code.

---

## Critical Patterns

### 1. Async/Await Everywhere
All engine methods are async. Use `async with` for resource management:
```python
async with engine.managed_lifecycle():
    result = await engine.process_event(event)
```
See [BaseEngine.managed_lifecycle](src/ordinis/engines/base/engine.py:200-210)

### 2. Pydantic Models for Data
Use Pydantic v2 for all data structures (validation + serialization):
```python
from pydantic import BaseModel, Field

class SignalConfig(BaseEngineConfig):
    model_type: str = Field(default="xgboost")
    lookback_days: int = Field(default=30, ge=1, le=365)
```

### 3. Governance Before Action
**Critical**: All trading operations must pass governance preflight:
```python
context = PreflightContext(
    operation="signal_generation",
    metadata={"symbol": "AAPL", "model": "v2.3"}
)
result = await self._governance.preflight(context)
if not result.allowed:
    raise EngineError(f"Governance denied: {result.reasons}")
```

### 4. Error Handling with EngineError
```python
from ordinis.engines.base import EngineError

try:
    data = await fetch_market_data(symbol)
except Exception as e:
    self._logger.error(f"Data fetch failed: {e}")
    raise EngineError(f"MarketData unavailable for {symbol}") from e
```

### 5. Configuration via YAML + Pydantic
Configs live in [configs/](configs/) and load via `config.from_yaml()`:
```python
# configs/strategies/momentum.yaml
strategy:
  name: momentum_breakout
  lookback: 20
  threshold: 0.02

# Load in code
from ordinis.application.strategies import MomentumConfig
config = MomentumConfig.from_yaml("configs/strategies/momentum.yaml")
```

---

## Integration Points

### StreamingBus (Event Communication)
All engines publish/subscribe to events via StreamingBus:
```python
# Publishing
await bus.publish("signals.generated", signal_event)

# Subscribing
async def handle_signal(event: SignalEvent):
    await risk_engine.evaluate(event.signal)

await bus.subscribe("signals.generated", handle_signal)
```

**Schemas**: Events are Pydantic models with strict validation.

### Market Data Adapters
Located in [adapters/market_data/](src/ordinis/adapters/market_data/):
- AlphaVantage, Finnhub, Massive, TwelveData
- All return normalized `MarketDataEvent` models
- Rate limiting and retry logic built-in

### Storage Layer
Repository pattern with SQLite (WAL mode):
```python
from ordinis.adapters.storage import SQLiteRepository

repo = SQLiteRepository(db_path="data/ordinis.db")
await repo.save_position(position)
positions = await repo.get_all_positions()
```

---

## NVIDIA Integration (GPU-Accelerated)

### Portfolio Optimization
[PortfolioOptEngine](src/ordinis/engines/portfolioopt/) uses NVIDIA cuOpt for mean-CVaR optimization:
```python
# GPU-accelerated (falls back to CPU if no GPU)
result = await opt_engine.optimize(returns_data, constraints)
```

### LLM Services (NVIDIA NIM APIs)
Unified via [Helix](src/ordinis/services/helix/) facade:
```python
# Helix dispatches to appropriate model
response = await helix.generate(
    messages=[{"role": "user", "content": prompt}],
    model_id="nemotron-super-49b-v1.5"  # Optional, uses default if omitted
)
```

**Model mapping** (see [README.md](README.md:73-92)):
- `nemotron-super-49b-v1.5` - Cortex reasoning, code review
- `nemotron-8b-v3.1` - Fast inference, analytics reports
- `meta/llama-3.3-70b-instruct` - Risk explanations
- `nvidia/llama-3.1-nemoguard-8b-content-safety` - Safety guardrails

### RAG System (Synapse)
Indexes docs/code with NVIDIA EmbedLM-300M:
```python
# Retrieve relevant context
snippets = await synapse.retrieve(
    query="How do I add a futures adapter?",
    context={"domain": "architecture"}
)
```

---

## Testing Conventions

### File Structure
```
tests/
├── test_engines/           # Engine unit tests
├── test_adapters/          # Adapter integration tests
├── integration/            # End-to-end tests
└── conftest.py             # Shared fixtures
```

### Fixtures
Use `conftest.py` fixtures for common setups:
```python
@pytest.fixture
async def signal_engine():
    config = SignalEngineConfig(name="test_signals")
    engine = SignalEngine(config)
    async with engine.managed_lifecycle():
        yield engine
```

### Mocking External APIs
```python
from pytest_mock import mocker

@pytest.mark.integration
async def test_market_data_fetch(mocker):
    mock_response = {"price": 150.0}
    mocker.patch("aiohttp.ClientSession.get", return_value=mock_response)
    # ... test logic
```

---

## Common Tasks

### Adding a New Engine
1. Create structure under `src/ordinis/engines/new_engine/` (follow [template](src/ordinis/engines/base/__init__.py:7-18))
2. Extend `BaseEngine[YourConfig]` in `core/engine.py`
3. Implement abstract methods: `_do_initialize`, `_do_shutdown`, `_do_health_check`
4. Add tests in `tests/test_engines/test_new_engine/`
5. Register with orchestrator in `src/ordinis/orchestration/`

### Adding a New Strategy
Place in [src/ordinis/application/strategies/](src/ordinis/application/strategies/) with:
- `StrategyConfig` (extends `BaseModel`)
- `generate_signals(data: pd.DataFrame) -> List[Signal]` method
- Unit tests with synthetic data

See [strategies/README.md](src/ordinis/application/strategies/README.md) for examples.

### Debugging
```bash
# Logs to stdout + file (configs via ORDINIS_LOG_LEVEL)
export ORDINIS_LOG_LEVEL=DEBUG
python scripts/demo_dev_system.py

# Check logs
tail -f logs/ordinis_$(date +%Y%m%d).log
```

---

## Security & Compliance

1. **Never commit secrets**: Use `.env` files (see `.env.example`)
2. **Governance audits**: All trading actions logged immutably to `data/audit/`
3. **Content safety**: LLM outputs filtered via NemoGuard ([RiskGuard](src/ordinis/engines/riskguard/))
4. **Pre-commit hooks**: Detect private keys automatically

---

## Key Files Reference

| File | Purpose |
|------|---------|
| [pyproject.toml](pyproject.toml) | Dependencies, pytest/ruff/mypy config |
| [README.md](README.md) | Project overview, architecture diagram, KPIs |
| [src/ordinis/engines/base/](src/ordinis/engines/base/) | Engine framework (start here for new engines) |
| [docs/architecture/production-architecture.md](docs/architecture/production-architecture.md) | Full Phase 1 implementation details |
| [configs/default.yaml](configs/default.yaml) | Default system configuration |
| [scripts/README.md](scripts/README.md) | Development scripts documentation |

---

## Quick Wins

- **Start engine development**: Copy [engines/base/](src/ordinis/engines/base/) template
- **Add market data source**: See [adapters/market_data/alpha_vantage.py](src/ordinis/adapters/market_data/alpha_vantage.py)
- **Write strategy**: Use [strategies/moving_average.py](src/ordinis/application/strategies/moving_average.py) as reference
- **Run backtest**: `python scripts/backtesting/run_backtest.py --config configs/backtest.yaml`

---

## Avoiding Common Mistakes

1. **Don't bypass governance**: Never execute trades without `preflight()` check
2. **Don't block async loops**: Use `await` for I/O, never `time.sleep()`
3. **Don't ignore type hints**: Mypy will catch errors early
4. **Don't skip tests**: Pre-commit enforces 50% coverage minimum
5. **Don't use generic exceptions**: Raise `EngineError` with context

---

*Last Updated: 2025-12-17 | Ordinis v0.2.0-dev*
