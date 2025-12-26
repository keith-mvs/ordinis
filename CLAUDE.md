# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ordinis is an AI-driven quantitative trading system with event-driven engines communicating via StreamingBus. The system follows clean architecture principles with business logic isolated in engines and external concerns handled by adapters.

## Build, Test, and Development Commands

```bash
# Environment setup
conda activate ordinis-env           # Primary: GPU/CUDA 12.1 support
pip install -e ".[dev]"              # Alternative: pure Python venv

# Quality checks (run before commits)
make check          # fmt + lint + test
make test-quick     # Skip slow tests (-m "not slow")
make coverage       # HTML report in htmlcov/

# Run single test file
pytest tests/test_engines/test_signalcore/test_signal.py -v

# Run single test function
pytest tests/test_engines/test_signalcore/test_signal.py::test_function_name -v

# Linting and formatting
ruff format .
ruff check .
mypy src/ --ignore-missing-imports

# Demo
python scripts/demo_dev_system.py    # Mock data demo
```

## Architecture

```
Market Data → StreamingBus → SignalCore → RiskGuard → FlowRoute → Portfolio → Analytics
                               ↑            ↑           ↑           ↑
                               └── GovernanceEngine (preflight checks & audit) ──┘
```

### Engine Structure

All engines extend `BaseEngine[ConfigT]` at `src/ordinis/engines/base/engine.py`. Each engine implements:
- `_do_initialize()`, `_do_shutdown()`, `_do_health_check()`
- Governance preflight via `await self._governance.preflight(context)`

Engine directory layout (`src/ordinis/engines/{name}/`):
```
core/engine.py    # Extends BaseEngine
core/config.py    # Extends BaseEngineConfig (Pydantic v2)
core/models.py    # Domain models
hooks/governance.py
```

### Key Engines

| Engine | Purpose |
|--------|---------|
| SignalCore | Generates trading signals from market data |
| RiskGuard | Enforces risk policies (exposure, leverage, stop-loss) |
| FlowRoute | Order routing and execution |
| Portfolio | Position management and rebalancing |
| Governance | Cross-cutting policy enforcement and audit logging |
| ProofBench | Backtesting framework |
| Cortex | LLM reasoning and code analysis |
| PortfolioOpt | GPU-accelerated portfolio optimization |

### Key Directories

| Path | Purpose |
|------|---------|
| `src/ordinis/engines/` | Core trading engines |
| `src/ordinis/domain/` | Enums & models: `OrderSide`, `OrderType`, `OrderStatus`, `Position` |
| `src/ordinis/adapters/` | Market data, storage (SQLite repos), broker, alerting |
| `src/ordinis/ai/helix/` | Unified LLM facade (NVIDIA, OpenAI, Mistral) |
| `src/ordinis/application/strategies/` | Trading strategy implementations |
| `src/ordinis/safety/` | Kill switch and circuit breaker |
| `configs/` | YAML configs loaded via Pydantic |

## Critical Patterns

### Async Lifecycle (Mandatory)

```python
async with engine.managed_lifecycle():
    result = await engine.process_event(event)
```

### Governance Before Trading (Mandatory)

```python
result = await self._governance.preflight(PreflightContext(operation="signal", metadata={"symbol": "AAPL"}))
if not result.allowed:
    raise EngineError(f"Denied: {result.reasons}")
```

### Signal Flow

Signals are probabilistic assessments (NOT orders). See `src/ordinis/engines/signalcore/core/signal.py`:

```python
Signal(symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, direction=Direction.LONG, probability=0.65)
```

Use `signal.is_actionable(min_probability=0.6)` before processing.

### Safety Controls

- **Kill switch**: `src/ordinis/safety/kill_switch.py` - file-based fallback at `data/KILL_SWITCH`
- **Circuit breaker**: `src/ordinis/safety/circuit_breaker.py` - check `circuit_breaker.check_health()` before operations

## Adding a Strategy

Extend `BaseStrategy` in `src/ordinis/application/strategies/`:

```python
class MyStrategy(BaseStrategy):
    def configure(self):
        # Set up parameters
        pass

    async def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        # Data: DatetimeIndex with lowercase columns (open, high, low, close, volume)
        return Signal(symbol=data.name, timestamp=timestamp, signal_type=SignalType.ENTRY, ...)

    def get_description(self) -> str:
        return "My strategy description"
```

## Adding an Engine

1. Create `src/ordinis/engines/my_engine/core/{engine,config,models}.py`
2. Extend `BaseEngine[MyConfig]`
3. Implement `_do_initialize`, `_do_shutdown`, `_do_health_check`
4. Add tests in `tests/test_engines/test_my_engine/`

## Coding Style

- Python 3.11+, 100-character line limit
- Full type hints (mypy enforced)
- Google-style docstrings
- `snake_case` for functions/variables, `PascalCase` for classes
- Imports sorted via ruff/isort

## Common Mistakes to Avoid

- Bypassing `governance.preflight()` before trades
- Using `time.sleep()` in async code (use `await asyncio.sleep()`)
- Catching generic `Exception` instead of raising `EngineError` with context
- Skipping `circuit_breaker.check_health()` before cascading operations

## Environment Variables

Required: `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `NVIDIA_API_KEY`
Optional: `ORDINIS_ENVIRONMENT=dev`, `ORDINIS_LOG_LEVEL=INFO`

## Test Markers

```bash
pytest -m unit           # Fast, isolated tests
pytest -m integration    # External dependencies
pytest -m slow           # Tests > 1 second
pytest -m requires_api   # Needs API keys
```
- optimizer script shall save individual trial reports for each symbol for each symbol in addition to the summary report json with the optimized parameters.
- always leverage local hardware (CPU, GPU, VRAM, etc.) to maximum extent possible
- fetch historical data from MASSIVE <https://massive.com/docs/flat-files/stocks/overview>