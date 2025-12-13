# System Architecture - Ordinis Trading System

## Purpose

This document describes the technical architecture of the Ordinis algorithmic trading system, including all engines, data pipelines, strategy frameworks, and execution components.

**Last Updated**: December 7, 2025
**Version**: 0.3.0-dev

---

## 1. High-Level Architecture

### 1.1 System Overview

```
+-----------------------------------------------------------------------------+
|                           ORDINIS TRADING SYSTEM                             |
+-----------------------------------------------------------------------------+
|                                                                              |
|  +------------------+    +------------------+    +------------------+        |
|  |   DATA LAYER     |--->|   SIGNALCORE     |--->|   RISKGUARD      |        |
|  | TrainingDataGen  |    | Signal Engine    |    | Risk Engine      |        |
|  | HistoricalFetcher|    | Strategies       |    | Position Sizing  |        |
|  +------------------+    +------------------+    +------------------+        |
|          |                      |                       |                    |
|          v                      v                       v                    |
|  +------------------+    +------------------+    +------------------+        |
|  |   PROOFBENCH     |    | REGIME ADAPTIVE  |    |   FLOWROUTE      |        |
|  | Backtesting      |    | Strategy Manager |    | Order Execution  |        |
|  | Simulation       |    | Regime Detector  |    | Broker Adapters  |        |
|  +------------------+    +------------------+    +------------------+        |
|          |                      |                       |                    |
|          +----------------------+-----------------------+                    |
|                                 |                                            |
|                                 v                                            |
|  +-----------------------------------------------------------------------+   |
|  |                        CORTEX (AI/RAG Layer)                          |   |
|  |   LLM Integration  |  Knowledge Base  |  Enhanced Analytics           |   |
|  +-----------------------------------------------------------------------+   |
|                                 |                                            |
|                                 v                                            |
|  +-----------------------------------------------------------------------+   |
|  |                    DASHBOARD & MONITORING                             |   |
|  |   Streamlit UI  |  Real-time Metrics  |  Performance Analytics        |   |
|  +-----------------------------------------------------------------------+   |
|                                                                              |
+-----------------------------------------------------------------------------+
```

---

### 1.2 Engine Responsibilities

| Engine | Location | Responsibility |
|--------|----------|----------------|
| **SignalCore** | `src/engines/signalcore/` | Signal generation, ML models, feature engineering |
| **ProofBench** | `src/engines/proofbench/` | Backtesting simulation, portfolio tracking, metrics |
| **RiskGuard** | `src/engines/riskguard/` | Risk management, position limits, kill switches |
| **FlowRoute** | `src/engines/flowroute/` | Order routing, broker adapters (Paper, Alpaca) |
| **Cortex** | `src/engines/cortex/` | LLM integration, RAG system, AI-enhanced analysis |

---

## 2. Data Layer

### 2.1 Training Data Generator

Location: `src/data/training_data_generator.py`

Generates regime-balanced, multi-timeframe datasets for strategy development.

```python
# Key Classes
TrainingConfig          # Configuration for data generation
MarketRegimeClassifier  # Classifies data into regime types
HistoricalDataFetcher   # Fetches data from Yahoo Finance
TrainingDataGenerator   # Main generator with balanced sampling
DataChunk               # Individual training data chunk
```

**Features:**
- Variable chunk sizes: 2, 3, 4, 6, 8, 10, 12 months
- Random start points within 5, 10, 15, 20 year windows
- Regime-balanced sampling across market conditions
- Automatic timezone handling for yfinance data

**Market Regimes:**
| Regime | Criteria |
|--------|----------|
| BULL | Annualized return > 15% |
| BEAR | Annualized return < -15% |
| SIDEWAYS | Return between -5% and +5% |
| VOLATILE | Volatility > 25% annualized |
| RECOVERY | Post-crash bounce (>5% recent gain) |
| CORRECTION | 10-20% decline from peak |

---

### 2.2 Data Flow

```
+-------------+     +---------------+     +----------------+     +-------------+
|   Yahoo     |---->| Historical    |---->|   Regime      |---->|   Data      |
|   Finance   |     | DataFetcher   |     |   Classifier  |     |   Chunks    |
+-------------+     +---------------+     +----------------+     +-------------+
                           |
                           v
                    +-------------+
                    |   Cache     |
                    | (CSV files) |
                    +-------------+
```

---

## 3. SignalCore Engine

### 3.1 Architecture

Location: `src/engines/signalcore/`

```
signalcore/
├── core/
│   ├── signal.py       # Signal, Direction, SignalType definitions
│   └── model.py        # ModelConfig, base model interfaces
├── features/
│   └── technical.py    # Feature engineering
└── models/
    ├── sma_crossover.py
    ├── rsi_mean_reversion.py
    ├── bollinger_bands.py
    ├── macd.py
    └── llm_enhanced.py
```

### 3.2 Signal Definition

```python
@dataclass
class Signal:
    symbol: str
    timestamp: datetime
    signal_type: SignalType      # ENTRY, EXIT, HOLD
    direction: Direction          # LONG, SHORT, NEUTRAL
    probability: float            # 0-1 confidence
    expected_return: float
    confidence_interval: tuple
    score: float
    model_id: str
    model_version: str
    metadata: dict
```

---

## 4. Strategy Framework

### 4.1 Base Strategy

Location: `src/strategies/base.py`

```python
class BaseStrategy(ABC):
    def configure(self): ...
    def generate_signal(self, data, timestamp) -> Signal: ...
    def get_description(self) -> str: ...
    def get_required_bars(self) -> int: ...
    def validate_data(self, data) -> tuple[bool, str]: ...
```

### 4.2 Legacy Strategies

| Strategy | File | Description |
|----------|------|-------------|
| Moving Average Crossover | `moving_average_crossover.py` | Golden/death cross signals |
| RSI Mean Reversion | `rsi_mean_reversion.py` | Oversold/overbought reversals |
| Bollinger Bands | `bollinger_bands.py` | Volatility-based mean reversion |
| MACD | `macd.py` | Momentum crossovers |
| Momentum Breakout | `momentum_breakout.py` | Price breakouts with volume |

### 4.3 Regime-Adaptive Framework

Location: `src/strategies/regime_adaptive/`

```
regime_adaptive/
├── __init__.py
├── regime_detector.py      # Market condition classification
├── trend_following.py      # Bull market strategies
├── mean_reversion.py       # Sideways market strategies
├── volatility_trading.py   # High volatility strategies
└── adaptive_manager.py     # Unified strategy orchestrator
```

#### Regime Detector

```python
class RegimeDetector:
    """Multi-indicator regime classification."""

    # Indicators used:
    # - ADX for trend strength
    # - Price vs moving averages for direction
    # - Bollinger Band width for volatility
    # - RSI for momentum extremes
    # - Historical volatility percentile

    def detect(self, data: pd.DataFrame) -> RegimeSignal: ...
```

#### Strategy Pools

**Trend-Following (Bull Markets):**
- MACrossoverStrategy (10/30 EMA with 200 trend filter)
- BreakoutStrategy (Donchian channel breakouts)
- ADXTrendStrategy (ADX-based with DI crossovers)

**Mean-Reversion (Sideways Markets):**
- BollingerFadeStrategy (Fade moves to band extremes)
- RSIReversalStrategy (RSI reversal confirmation)
- KeltnerChannelStrategy (ATR-based channels)
- StatisticalArbitrageStrategy (Z-score based)

**Volatility Trading (High Volatility):**
- ScalpingStrategy (Quick in-and-out trades)
- VolatilityBreakoutStrategy (BB squeeze breakouts)
- ATRTrailingStrategy (Chandelier exit concept)

#### Adaptive Manager

```python
# Default Regime Weights
REGIME_WEIGHTS = {
    BULL:         {trend: 80%, reversion: 10%, volatility: 5%,  cash: 5%},
    BEAR:         {trend: 20%, reversion: 20%, volatility: 10%, cash: 50%},
    SIDEWAYS:     {trend: 10%, reversion: 70%, volatility: 10%, cash: 10%},
    VOLATILE:     {trend: 20%, reversion: 20%, volatility: 40%, cash: 20%},
    TRANSITIONAL: {trend: 15%, reversion: 15%, volatility: 10%, cash: 60%},
}
```

---

## 5. Technical Indicators Library

Location: `src/ordinis/analysis/technical/indicators/`

```
indicators/
├── __init__.py
├── moving_averages.py    # SMA, EMA, WMA, VWAP, Hull MA, KAMA
├── oscillators.py        # RSI, Stochastic, CCI, Williams %R, MFI
├── volatility.py         # ATR, Bollinger Bands, Keltner, Donchian
├── volume.py             # OBV, A/D Line, CMF, Force Index, VWAP
└── combined.py           # TechnicalIndicators unified interface
```

### 5.1 Combined Interface

```python
class TechnicalIndicators:
    """Unified technical analysis interface."""

    def analyze(self, data: pd.DataFrame) -> TechnicalSnapshot:
        # Returns comprehensive analysis including:
        # - Trend direction and strength
        # - RSI, Stochastic, MACD
        # - ATR, Bollinger position
        # - Volume confirmation
        # - Overall bias (strong_buy to strong_sell)
```

---

## 6. ProofBench (Backtesting Engine)

Location: `src/engines/proofbench/`

```
proofbench/
├── core/
│   ├── simulator.py      # SimulationEngine, SimulationConfig
│   ├── execution.py      # Order, OrderSide, OrderType
│   ├── portfolio.py      # Portfolio tracking
│   └── events.py         # Event system
└── analytics/
    ├── performance.py    # Metrics calculation
    └── llm_enhanced.py   # AI-powered analysis
```

### 6.1 Simulation Engine

```python
class SimulationEngine:
    def __init__(self, config: SimulationConfig): ...
    def load_data(self, symbol: str, data: pd.DataFrame): ...
    def set_strategy(self, callback: Callable): ...
    def run(self) -> SimulationResults: ...
    def submit_order(self, order: Order): ...
```

### 6.2 Strategy Callback Pattern

```python
def strategy_callback(engine, symbol: str, bar):
    """Called on each bar during simulation."""
    data = engine.data[symbol].loc[:bar.timestamp]

    # Generate signals
    # Submit orders via engine.submit_order()
    # Access portfolio via engine.portfolio
```

### 6.3 Performance Metrics

```python
@dataclass
class SimulationMetrics:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    # ... additional metrics
```

---

## 7. RiskGuard Engine

Location: `src/engines/riskguard/`

```
riskguard/
├── core/
│   ├── engine.py         # Main risk engine
│   ├── rules.py          # Rule definitions
│   └── llm_enhanced.py   # AI-enhanced risk
└── rules/
    └── standard.py       # Standard risk rules
```

### 7.1 Risk Checks

- Daily loss limit (default: 3%)
- Maximum drawdown (default: 15%)
- Position concentration limits
- Sector exposure limits
- Correlation checks
- Kill switch triggers

---

## 8. FlowRoute (Execution Engine)

Location: `src/engines/flowroute/`

```
flowroute/
├── core/
│   ├── engine.py         # Order routing logic
│   └── orders.py         # Order management
└── adapters/
    ├── __init__.py
    ├── paper.py          # Paper trading adapter
    └── alpaca.py         # Alpaca broker adapter
```

### 8.1 Broker Adapters

**Paper Trading:**
- Simulated order execution
- No real money at risk
- Full order lifecycle simulation

**Alpaca Integration:**
- Real broker connectivity
- Paper and live trading modes
- Market data access
- Position management

---

## 9. Cross-Validation Framework

Location: `src/data/regime_cross_validator.py`

### 9.1 Regime-Stratified Validation

```python
class RegimeCrossValidator:
    """Ensures strategies tested across all market regimes."""

    def validate(self, chunks: list[DataChunk]) -> CrossValidationReport:
        # Tests strategy on each chunk
        # Aggregates results by regime
        # Computes consistency score
```

### 9.2 Walk-Forward Validation

```python
class WalkForwardValidator:
    """Walk-forward with expanding or rolling windows."""

    def generate_folds(self, data, min_train_bars) -> list[tuple]:
        # Returns (train_data, test_data, fold_info) tuples
```

---

## 10. Dashboard

Location: `src/dashboard/app.py`

Streamlit-based monitoring dashboard with:
- Real-time portfolio metrics
- Strategy performance visualization
- Regime detection status
- Trade history
- Risk metrics

---

## 11. Directory Structure

```
ordinis-1/
├── src/
│   ├── engines/
│   │   ├── signalcore/       # Signal generation
│   │   ├── proofbench/       # Backtesting
│   │   ├── riskguard/        # Risk management
│   │   ├── flowroute/        # Order execution
│   │   └── cortex/           # AI/RAG integration
│   ├── strategies/
│   │   ├── base.py           # Base strategy class
│   │   ├── moving_average_crossover.py
│   │   ├── rsi_mean_reversion.py
│   │   ├── bollinger_bands.py
│   │   ├── macd.py
│   │   ├── momentum_breakout.py
│   │   └── regime_adaptive/  # Adaptive strategy framework
│   ├── analysis/
│   │   └── technical/
│   │       └── indicators/   # Technical indicator library
│   ├── data/
│   │   ├── training_data_generator.py
│   │   └── regime_cross_validator.py
│   └── dashboard/
│       └── app.py
├── scripts/
│   ├── run_adaptive_backtest.py
│   ├── run_regime_backtest.py
│   ├── run_real_backtest.py
│   └── run_paper_trading.py
├── tests/
├── docs/
│   ├── knowledge-base/
│   └── architecture/
└── data/
    └── historical_cache/
```

---

## 12. Current Status & Known Issues

### 12.1 Working Components
- Training data generation with regime labeling
- Technical indicators library
- ProofBench simulation engine
- Legacy strategy backtesting
- Paper trading via Alpaca adapter
- Dashboard (basic)

### 12.2 Known Issues (as of Dec 7, 2025)

| Issue | Impact | Status |
|-------|--------|--------|
| Strategy returns 0% | Signals not crossing threshold | Needs tuning |
| Bull market 0% win rate | Trend strategies exit too early | Needs work |
| Regime detection 25% accuracy | Detector vs labels mismatch | Needs calibration |

### 12.3 Backtest Results Summary

**Legacy Strategies (no regime awareness):**
- Best performer: Bollinger Bands (34.7% beat B&H)
- None reliably beat buy-and-hold

**Adaptive System (current):**
- Bear markets: 100% beat B&H (+9% avg alpha)
- Recovery: 100% beat B&H (+12% avg alpha)
- Volatile: 52% beat B&H
- Bull/Sideways/Correction: Underperforming

---

## 13. Configuration

### 13.1 Environment Variables

```bash
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or live
```

### 13.2 Strategy Configuration

```python
AdaptiveConfig(
    regime_lookback=200,
    regime_confirm_bars=3,
    use_ensemble=True,
    base_position_size=0.8,
    min_position_size=0.3,
    confidence_scaling=True,
    volatility_scaling=True,
    max_drawdown_threshold=0.10,
)
```

---

## 14. References

- **Architecture Docs**: `docs/architecture/`
- **Strategy Docs**: `docs/strategies/`
- **Analysis Framework**: `docs/ANALYSIS_FRAMEWORK.md`
- **Backtest Results**: `docs/BACKTEST_RESULTS_20251207.md`
- **ProofBench Guide**: `docs/PROOFBENCH_GUIDE.md`

---

## Key Principles

1. **Modular Engines**: Each engine has single responsibility
2. **Regime Awareness**: Strategies adapt to market conditions
3. **Risk First**: RiskGuard enforces all constraints
4. **Paper Before Live**: Always validate in simulation
5. **Data Quality**: Validate and normalize all inputs
6. **Audit Trail**: Log all decisions for review
