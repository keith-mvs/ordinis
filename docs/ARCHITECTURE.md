# Ordinis Architecture

High-level system architecture for the AI-driven quantitative trading system.

## System Overview

Ordinis is a modular, event-driven quantitative trading system designed for research, backtesting, and paper trading. The system integrates multiple market data sources, ML-driven signal generation, and robust risk management.

## Core Principles

1. **Modularity**: Each component is independent and replaceable
2. **Extensibility**: Easy to add new strategies, data sources, or engines
3. **Type Safety**: Comprehensive type hints and runtime validation
4. **Testability**: All components are unit and integration tested
5. **Performance**: Optimized for backtesting and real-time analysis

## System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   CLI Tool   │  │  Dashboard   │  │  REST API    │      │
│  │  (src/cli)   │  │(src/dashboard)│  │   (future)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Strategies  │  │  Monitoring  │  │ Visualization │      │
│  │(src/strategies)│ │(src/monitoring)│ │(src/visualization)│ │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      Engine Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ SignalCore   │  │ OptionsCore  │  │  ProofBench  │      │
│  │   (Signals)  │  │  (Options)   │  │ (Backtesting)│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │  RiskGuard   │  │   CortexRAG  │                         │
│  │    (Risk)    │  │     (RAG)    │                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                       Data Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Data Loaders │  │    Plugins   │  │   Storage    │      │
│  │  (src/data)  │  │(src/plugins) │  │    (data/)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                     External Services                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Alpha Vantage│  │   Finnhub    │  │  Polygon.io  │      │
│  │              │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ Twelve Data  │  │  Alpaca API  │                         │
│  │              │  │ (Paper Trade)│                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Layer (`src/data/`)

**Purpose**: Unified interface for market data from multiple sources

**Components**:
- `connectors/`: API clients for each data provider
- `loaders/`: Data loading and caching
- `validators/`: Data quality validation
- `transformers/`: Data normalization and feature engineering

**Key Abstractions**:
```python
class MarketDataProvider(Protocol):
    """Protocol for market data sources."""

    async def get_quote(self, symbol: str) -> Quote: ...
    async def get_bars(self, symbol: str, timeframe: str) -> pd.DataFrame: ...
    async def get_historical(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...
```

### 2. Engine Layer

#### SignalCore (`src/engines/signalcore/`)

**Purpose**: ML-driven signal generation and feature engineering

**Architecture**:
```
signalcore/
├── core/           # Base model, signal, and feature abstractions
├── features/       # Feature engineering (technical indicators, fundamentals)
├── models/         # Trading models (SMA, RSI, MACD, Bollinger, etc.)
└── strategies/     # Signal combination strategies
```

**Key Components**:
- `Model`: Base class for all trading models
- `Signal`: Standardized signal output (type, direction, score, probability)
- `TechnicalIndicators`: Feature engineering for price data
- `LLMEnhancedModel`: AI-augmented signal generation

#### OptionsCore (`src/engines/optionscore/`)

**Purpose**: Options pricing, Greeks calculation, and strategy analysis

**Components**:
- `pricing/`: Black-Scholes and Greeks calculators
- `strategies/`: Multi-leg options strategy builders
- `enrichment/`: Options data enrichment and validation

#### ProofBench (`src/engines/proofbench/`)

**Purpose**: Backtesting framework with realistic simulation

**Features**:
- Event-driven backtesting
- Realistic fill simulation (slippage, partial fills)
- Multiple timeframe support
- Performance metrics and reporting

#### RiskGuard (`src/engines/riskguard/`)

**Purpose**: Risk management and position sizing

**Components**:
- Position sizing algorithms
- Portfolio risk metrics (VaR, Sharpe, Sortino)
- Drawdown monitoring
- Risk limit enforcement

#### CortexRAG (`src/rag/`)

**Purpose**: Retrieval-Augmented Generation for trading knowledge

**Components**:
- Knowledge base indexing
- Semantic search over trading documentation
- Context-aware query answering

### 3. Strategy Layer (`src/strategies/`)

**Purpose**: Trading strategy implementations

**Organization**:
```
strategies/
├── options/        # Options strategies (covered call, married put, spreads)
├── technical/      # Technical analysis strategies
├── fundamental/    # Fundamental analysis strategies (future)
└── hybrid/         # Multi-signal strategies (future)
```

**Base Strategy Pattern**:
```python
class Strategy(ABC):
    """Base strategy interface."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """Generate trading signals from market data."""

    @abstractmethod
    def size_position(self, signal: Signal, portfolio: Portfolio) -> Order:
        """Determine position size for signal."""
```

### 4. Monitoring Layer (`src/monitoring/`)

**Purpose**: Performance tracking and KPI monitoring

**Components**:
- `kpi.py`: Key Performance Indicator tracking
- `metrics/`: Performance metric calculations
- `alerts/`: Alert system for anomalies (future)

### 5. Visualization Layer (`src/visualization/`)

**Purpose**: Data visualization and dashboard components

**Components**:
- `dashboard.py`: Performance dashboard
- `charts/`: Chart generation utilities
- `reports/`: PDF/HTML report generation (future)

## Data Flow

### Backtesting Flow
```
1. Historical Data → Data Loader
2. Data → SignalCore → Signals
3. Signals → Strategy → Orders
4. Orders → ProofBench → Fills
5. Fills → Portfolio → Performance Metrics
6. Metrics → Dashboard → User
```

### Live Trading Flow (Paper)
```
1. Real-time Data → Data Connector
2. Data → SignalCore → Signals
3. Signals → RiskGuard → Risk Check
4. Approved Signals → Strategy → Orders
5. Orders → Alpaca Paper API → Fills
6. Fills → Monitoring → Alerts
```

## Configuration Management

Configuration is managed through multiple layers:

1. **Environment Variables** (`.env`): API keys, secrets
2. **YAML Configuration** (`config/`): Strategy parameters, data sources
3. **Code Configuration** (`pyproject.toml`): Project metadata, dependencies
4. **Runtime Configuration**: CLI arguments, programmatic overrides

## Extensibility Points

### Adding a New Data Source
1. Implement `MarketDataProvider` protocol
2. Add connector to `src/data/connectors/`
3. Register in data loader factory
4. Add tests in `tests/test_data/`

### Adding a New Strategy
1. Inherit from `Strategy` base class
2. Implement `generate_signals()` and `size_position()`
3. Add to `src/strategies/` with tests
4. Register in strategy registry

### Adding a New Model
1. Inherit from `Model` base class in SignalCore
2. Implement `generate()` method
3. Define model parameters and validation
4. Add comprehensive tests

## Testing Strategy

### Unit Tests (`tests/unit/`)
- Test individual components in isolation
- Mock external dependencies
- Fast execution (<1s per test)
- >80% coverage target

### Integration Tests (`tests/integration/`)
- Test component interactions
- Use test data sources
- Validate end-to-end flows

### Performance Tests (future)
- Benchmark backtesting speed
- Memory profiling
- Load testing for live systems

## Deployment

### Development
```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run backtest
python scripts/backtesting/run_backtest_demo.py
```

### Production (Future)
- Docker containerization
- Kubernetes orchestration
- Monitoring with Prometheus/Grafana
- Logging with ELK stack

## Security Considerations

1. **API Keys**: Stored in environment variables, never in code
2. **Data Validation**: All external data validated before use
3. **Input Sanitization**: User inputs sanitized
4. **Dependency Scanning**: Regular security audits with `safety`
5. **Rate Limiting**: API request throttling
6. **Audit Logging**: All trades and decisions logged

## Performance Characteristics

- **Backtesting Speed**: ~10,000 days/second (single symbol)
- **Signal Generation**: <100ms per symbol
- **Memory Usage**: ~500MB for typical backtest
- **Data Storage**: Compressed CSV for historical data

## Future Enhancements

1. **Real-time Trading**: Live market integration
2. **Multi-asset Support**: Futures, forex, crypto
3. **Advanced ML**: Deep learning models
4. **Distributed Computing**: Spark for large-scale backtests
5. **Web Dashboard**: React-based UI
6. **Portfolio Optimization**: Modern portfolio theory implementation

## References

For detailed documentation:
- **API Documentation**: `docs/api/`
- **User Guides**: `docs/guides/`
- **Architecture Details**: `docs/architecture/`
- **Knowledge Base**: `docs/knowledge-base/`

For questions or contributions, see `CONTRIBUTING.md`.
