# Ordinis AI Coding Agent Instructions

AI-driven quantitative trading system with event-driven engines communicating via StreamingBus.

## Architecture (Start Here)

```
Market Data → StreamingBus → SignalCore → RiskGuard → FlowRoute → Portfolio → Analytics
                               ↑            ↑           ↑           ↑
                               └── GovernanceEngine (preflight checks & audit) ──┘
```

**All engines extend `BaseEngine[ConfigT]`** at [src/ordinis/engines/base/engine.py](src/ordinis/engines/base/engine.py). Every engine implements:
- `_do_initialize()`, `_do_shutdown()`, `_do_health_check()`
- Governance preflight via `await self._governance.preflight(context)`

**Engine structure** (`src/ordinis/engines/{name}/`):
```
core/engine.py    # Extends BaseEngine
core/config.py    # Extends BaseEngineConfig (Pydantic v2)
core/models.py    # Domain models
hooks/governance.py
```

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
Signals are probabilistic assessments (NOT orders). See [src/ordinis/engines/signalcore/core/signal.py](src/ordinis/engines/signalcore/core/signal.py):
```python
Signal(symbol="AAPL", timestamp=dt, signal_type=SignalType.ENTRY, direction=Direction.LONG, probability=0.65)
```
Use `signal.is_actionable(min_probability=0.6)` before processing.

### Safety Controls
- **Kill switch**: [src/ordinis/safety/kill_switch.py](src/ordinis/safety/kill_switch.py) - file-based fallback at `data/KILL_SWITCH`
- **Circuit breaker**: [src/ordinis/safety/circuit_breaker.py](src/ordinis/safety/circuit_breaker.py) - check `circuit_breaker.check_health()` before operations

## Development Commands

```bash
# Environment setup (conda preferred, venv alternative)
conda activate ordinis-env           # Primary: GPU/CUDA 12.1 support
# OR: .venv\Scripts\Activate.ps1     # Alternative: pure Python venv

# Quality checks
make check          # fmt + lint + test (run before commits)
make test-quick     # Skip slow tests (-m "not slow")
make coverage       # HTML report in htmlcov/

# Demo
python scripts/demo_dev_system.py  # Mock data demo
```

**Environment details**: Conda env at `environment.yml`, venv via `pip install -e ".[dev]"`

## Key Directories

| Path | Purpose |
|------|---------|
| `src/ordinis/engines/` | 14 core engines (signalcore, riskguard, flowroute, etc.) |
| `src/ordinis/domain/` | Enums & models: `OrderSide`, `OrderType`, `OrderStatus`, `Position` |
| `src/ordinis/adapters/` | Market data, storage (SQLite repos), broker, alerting |
| `src/ordinis/ai/helix/` | Unified LLM facade (NVIDIA, OpenAI, Mistral) |
| `configs/` | YAML configs loaded via Pydantic |

## Adding a Strategy

Extend `BaseStrategy` in `src/ordinis/application/strategies/`:
```python
class MyStrategy(BaseStrategy):
    async def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        # Data: DatetimeIndex with lowercase columns (open, high, low, close, volume)
        return Signal(symbol=data.name, timestamp=timestamp, signal_type=SignalType.ENTRY, ...)
```

## Adding an Engine

1. Create `src/ordinis/engines/my_engine/core/{engine,config,models}.py`
2. Extend `BaseEngine[MyConfig]`
3. Implement `_do_initialize`, `_do_shutdown`, `_do_health_check`
4. Add tests in `tests/test_engines/test_my_engine/`

## Common Mistakes

- Bypassing `governance.preflight()` before trades
- Using `time.sleep()` in async code (use `await asyncio.sleep()`)
- Catching generic `Exception` instead of raising `EngineError` with context
- Skipping `circuit_breaker.check_health()` before cascading operations

## Environment Variables

Required: `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `NVIDIA_API_KEY`
Optional: `ORDINIS_ENVIRONMENT=dev`, `ORDINIS_LOG_LEVEL=INFO`

---
*See [AGENTS.md](../AGENTS.md) for coding style and [README.md](../README.md) for full architecture.*
