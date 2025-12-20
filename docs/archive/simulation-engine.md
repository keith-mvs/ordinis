# Simulation Engine Architecture

## Overview

The simulation engine is the **critical validation layer** that tests strategies against historical data before any capital is risked. It must accurately replicate market conditions, execution realities, and risk constraints.

**Priority**: This component is foundational - no strategy proceeds to paper/live trading without simulation validation.

---

## Design Goals

1. **Accuracy**: Realistic simulation of fills, slippage, and costs
2. **Speed**: Fast enough for parameter optimization and walk-forward testing
3. **Flexibility**: Support equities, options, futures, forex, crypto
4. **Reproducibility**: Deterministic results for the same inputs
5. **Extensibility**: Easy to add new data sources and strategy types

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SIMULATION ENGINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Data Layer  │───▶│ Strategy     │───▶│ Execution    │          │
│  │              │    │ Engine       │    │ Simulator    │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Market Data  │    │ Signal       │    │ Portfolio    │          │
│  │ Provider     │    │ Generator    │    │ Manager      │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Data         │    │ Risk         │    │ Performance  │          │
│  │ Validation   │    │ Manager      │    │ Analytics    │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Data Layer

```python
class DataProvider:
    """
    Abstract base for all data sources.
    """
    def get_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> pd.DataFrame:
        """Return OHLCV data for symbol."""
        pass

    def get_fundamentals(
        self,
        symbol: str,
        fields: List[str],
        as_of: datetime
    ) -> Dict:
        """Return point-in-time fundamental data."""
        pass

    def get_options_chain(
        self,
        symbol: str,
        as_of: datetime
    ) -> pd.DataFrame:
        """Return options chain snapshot."""
        pass


class DataManager:
    """
    Manages data loading, caching, and preprocessing.
    """
    def __init__(self, providers: List[DataProvider]):
        self.providers = providers
        self.cache = DataCache()

    def load_universe(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        fields: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Load data for entire universe."""
        pass

    def apply_corporate_actions(
        self,
        data: pd.DataFrame,
        actions: pd.DataFrame
    ) -> pd.DataFrame:
        """Adjust for splits, dividends, etc."""
        pass

    def validate_data(self, data: pd.DataFrame) -> ValidationReport:
        """Check for gaps, errors, survivorship bias."""
        pass
```

---

### 2. Event-Driven Simulation Core

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import heapq

class EventType(Enum):
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    BAR_UPDATE = "bar_update"
    SIGNAL = "signal"
    ORDER_SUBMIT = "order_submit"
    ORDER_FILL = "order_fill"
    ORDER_CANCEL = "order_cancel"
    POSITION_UPDATE = "position_update"
    RISK_CHECK = "risk_check"
    EOD_SETTLEMENT = "eod_settlement"


@dataclass
class Event:
    timestamp: datetime
    event_type: EventType
    data: dict
    priority: int = 0

    def __lt__(self, other):
        if self.timestamp == other.timestamp:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


class EventQueue:
    """
    Priority queue for simulation events.
    """
    def __init__(self):
        self.queue = []

    def push(self, event: Event):
        heapq.heappush(self.queue, event)

    def pop(self) -> Optional[Event]:
        if self.queue:
            return heapq.heappop(self.queue)
        return None

    def peek(self) -> Optional[Event]:
        if self.queue:
            return self.queue[0]
        return None


class SimulationEngine:
    """
    Main event-driven simulation loop.
    """
    def __init__(
        self,
        strategy: Strategy,
        data_manager: DataManager,
        execution_simulator: ExecutionSimulator,
        portfolio: Portfolio,
        risk_manager: RiskManager
    ):
        self.strategy = strategy
        self.data_manager = data_manager
        self.execution = execution_simulator
        self.portfolio = portfolio
        self.risk = risk_manager
        self.event_queue = EventQueue()
        self.results = SimulationResults()

    def run(
        self,
        start: datetime,
        end: datetime,
        initial_capital: float
    ) -> SimulationResults:
        """
        Run simulation from start to end date.
        """
        self.portfolio.initialize(initial_capital)
        self._load_data(start, end)
        self._generate_market_events(start, end)

        while event := self.event_queue.pop():
            if event.timestamp > end:
                break

            self._process_event(event)
            self._record_state(event.timestamp)

        return self._finalize_results()

    def _process_event(self, event: Event):
        if event.event_type == EventType.BAR_UPDATE:
            self._on_bar(event)
        elif event.event_type == EventType.SIGNAL:
            self._on_signal(event)
        elif event.event_type == EventType.ORDER_FILL:
            self._on_fill(event)
        # ... handle other events

    def _on_bar(self, event: Event):
        """Process new price bar."""
        symbol = event.data['symbol']
        bar = event.data['bar']

        # Update positions with new prices
        self.portfolio.mark_to_market(symbol, bar)

        # Check stops and targets
        self._check_exits(symbol, bar)

        # Generate signals
        signals = self.strategy.on_bar(symbol, bar, self.portfolio)

        for signal in signals:
            self.event_queue.push(Event(
                timestamp=event.timestamp,
                event_type=EventType.SIGNAL,
                data={'signal': signal},
                priority=1
            ))

    def _on_signal(self, event: Event):
        """Process trading signal."""
        signal = event.data['signal']

        # Risk check
        if not self.risk.validate_signal(signal, self.portfolio):
            return

        # Generate order
        order = self.strategy.signal_to_order(signal, self.portfolio)

        if order:
            self.event_queue.push(Event(
                timestamp=event.timestamp,
                event_type=EventType.ORDER_SUBMIT,
                data={'order': order},
                priority=2
            ))
```

---

### 3. Execution Simulator

```python
@dataclass
class Order:
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: str  # 'MARKET', 'LIMIT', 'STOP'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'DAY'


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    timestamp: datetime


class ExecutionSimulator:
    """
    Simulates order execution with realistic fills.
    """
    def __init__(self, config: ExecutionConfig):
        self.config = config

    def simulate_fill(
        self,
        order: Order,
        bar: Bar,
        timestamp: datetime
    ) -> Optional[Fill]:
        """
        Simulate order fill with slippage and costs.
        """
        if order.order_type == 'MARKET':
            return self._fill_market_order(order, bar, timestamp)
        elif order.order_type == 'LIMIT':
            return self._fill_limit_order(order, bar, timestamp)
        elif order.order_type == 'STOP':
            return self._fill_stop_order(order, bar, timestamp)

    def _fill_market_order(
        self,
        order: Order,
        bar: Bar,
        timestamp: datetime
    ) -> Fill:
        """
        Fill market order with slippage model.
        """
        # Base price
        if order.side == 'BUY':
            base_price = bar.open  # Assume fill at open
        else:
            base_price = bar.open

        # Apply slippage
        slippage = self._calculate_slippage(order, bar)
        fill_price = base_price * (1 + slippage) if order.side == 'BUY' else base_price * (1 - slippage)

        # Calculate commission
        commission = self._calculate_commission(order, fill_price)

        return Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            timestamp=timestamp
        )

    def _calculate_slippage(self, order: Order, bar: Bar) -> float:
        """
        Calculate realistic slippage based on:
        - Order size relative to volume
        - Bid-ask spread (estimated)
        - Volatility
        """
        # Fixed component (half spread)
        spread_slippage = self.config.estimated_spread / 2

        # Variable component (market impact)
        volume_ratio = order.quantity / bar.volume
        impact_slippage = self.config.impact_coefficient * volume_ratio

        # Volatility component
        volatility = (bar.high - bar.low) / bar.close
        vol_slippage = volatility * self.config.volatility_factor

        total_slippage = spread_slippage + impact_slippage + vol_slippage
        return min(total_slippage, self.config.max_slippage)

    def _calculate_commission(self, order: Order, fill_price: float) -> float:
        """
        Calculate commission based on broker model.
        """
        if self.config.commission_model == 'per_share':
            return order.quantity * self.config.per_share_rate
        elif self.config.commission_model == 'per_trade':
            return self.config.per_trade_rate
        elif self.config.commission_model == 'percentage':
            return order.quantity * fill_price * self.config.percentage_rate
        else:
            return 0


@dataclass
class ExecutionConfig:
    """Configuration for execution simulation."""
    # Slippage model
    estimated_spread: float = 0.001  # 10 bps
    impact_coefficient: float = 0.1
    volatility_factor: float = 0.1
    max_slippage: float = 0.005  # 50 bps max

    # Commission model
    commission_model: str = 'per_share'
    per_share_rate: float = 0.005
    per_trade_rate: float = 5.00
    percentage_rate: float = 0.0001

    # Execution assumptions
    fill_at: str = 'open'  # 'open', 'close', 'vwap'
    partial_fills: bool = False
    reject_if_no_volume: bool = True
```

---

### 4. Portfolio Manager

```python
@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    side: str
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class Portfolio:
    """
    Tracks positions, cash, and portfolio state.
    """
    def __init__(self):
        self.cash: float = 0.0
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.equity_curve: List[EquityPoint] = []

    def initialize(self, capital: float):
        self.cash = capital
        self.initial_capital = capital

    @property
    def equity(self) -> float:
        position_value = sum(
            p.quantity * p.current_price for p in self.positions.values()
        )
        return self.cash + position_value

    @property
    def buying_power(self) -> float:
        # For cash account
        return self.cash

    def mark_to_market(self, symbol: str, bar: Bar):
        """Update position with current prices."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.current_price = bar.close
            pos.unrealized_pnl = (bar.close - pos.avg_cost) * pos.quantity
            if pos.side == 'SHORT':
                pos.unrealized_pnl *= -1

    def update_position(self, fill: Fill):
        """Update position based on fill."""
        symbol = fill.symbol

        if symbol not in self.positions:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=fill.quantity,
                avg_cost=fill.price,
                side='LONG' if fill.side == 'BUY' else 'SHORT',
                entry_time=fill.timestamp,
                current_price=fill.price
            )
            self.cash -= fill.quantity * fill.price + fill.commission

        else:
            pos = self.positions[symbol]

            if fill.side == 'BUY' and pos.side == 'LONG':
                # Add to long position
                total_cost = pos.avg_cost * pos.quantity + fill.price * fill.quantity
                pos.quantity += fill.quantity
                pos.avg_cost = total_cost / pos.quantity
                self.cash -= fill.quantity * fill.price + fill.commission

            elif fill.side == 'SELL' and pos.side == 'LONG':
                # Close/reduce long position
                realized = (fill.price - pos.avg_cost) * fill.quantity
                pos.realized_pnl += realized
                pos.quantity -= fill.quantity
                self.cash += fill.quantity * fill.price - fill.commission

                if pos.quantity == 0:
                    self._close_position(symbol)

            # Handle short positions similarly...

    def _close_position(self, symbol: str):
        """Record closed trade and remove position."""
        pos = self.positions.pop(symbol)
        self.trade_history.append(Trade(
            symbol=symbol,
            entry_time=pos.entry_time,
            exit_time=datetime.now(),
            side=pos.side,
            quantity=pos.quantity,
            entry_price=pos.avg_cost,
            exit_price=pos.current_price,
            pnl=pos.realized_pnl
        ))
```

---

### 5. Risk Manager (Simulation)

```python
class SimulationRiskManager:
    """
    Enforces risk rules during simulation.
    """
    def __init__(self, config: RiskConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.peak_equity = 0.0

    def validate_signal(self, signal: Signal, portfolio: Portfolio) -> bool:
        """
        Check if signal passes all risk checks.
        """
        checks = [
            self._check_position_limit(signal, portfolio),
            self._check_sector_concentration(signal, portfolio),
            self._check_daily_loss_limit(portfolio),
            self._check_drawdown_limit(portfolio),
            self._check_position_size(signal, portfolio),
        ]

        return all(checks)

    def _check_position_limit(self, signal: Signal, portfolio: Portfolio) -> bool:
        current_positions = len(portfolio.positions)
        return current_positions < self.config.max_positions

    def _check_sector_concentration(self, signal: Signal, portfolio: Portfolio) -> bool:
        sector = get_sector(signal.symbol)
        sector_exposure = portfolio.get_sector_exposure(sector)
        return sector_exposure < self.config.max_sector_concentration

    def _check_daily_loss_limit(self, portfolio: Portfolio) -> bool:
        daily_pnl_pct = self.daily_pnl / portfolio.initial_capital
        return daily_pnl_pct > -self.config.max_daily_loss

    def _check_drawdown_limit(self, portfolio: Portfolio) -> bool:
        if portfolio.equity > self.peak_equity:
            self.peak_equity = portfolio.equity
        drawdown = (self.peak_equity - portfolio.equity) / self.peak_equity
        return drawdown < self.config.max_drawdown

    def _check_position_size(self, signal: Signal, portfolio: Portfolio) -> bool:
        position_value = signal.quantity * signal.price
        position_pct = position_value / portfolio.equity
        return position_pct < self.config.max_position_pct
```

---

### 6. Performance Analytics

```python
@dataclass
class PerformanceMetrics:
    # Returns
    total_return: float
    cagr: float
    monthly_returns: pd.Series

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdowns
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float

    # Risk
    volatility: float
    downside_deviation: float
    var_95: float
    cvar_95: float


class PerformanceAnalyzer:
    """
    Calculate comprehensive performance metrics.
    """
    def analyze(
        self,
        equity_curve: pd.Series,
        trades: List[Trade],
        benchmark: Optional[pd.Series] = None
    ) -> PerformanceMetrics:

        returns = equity_curve.pct_change().dropna()

        return PerformanceMetrics(
            total_return=self._total_return(equity_curve),
            cagr=self._cagr(equity_curve),
            monthly_returns=self._monthly_returns(returns),
            sharpe_ratio=self._sharpe_ratio(returns),
            sortino_ratio=self._sortino_ratio(returns),
            calmar_ratio=self._calmar_ratio(equity_curve, returns),
            max_drawdown=self._max_drawdown(equity_curve),
            max_drawdown_duration=self._max_drawdown_duration(equity_curve),
            avg_drawdown=self._avg_drawdown(equity_curve),
            total_trades=len(trades),
            winning_trades=len([t for t in trades if t.pnl > 0]),
            losing_trades=len([t for t in trades if t.pnl < 0]),
            win_rate=self._win_rate(trades),
            avg_win=self._avg_win(trades),
            avg_loss=self._avg_loss(trades),
            profit_factor=self._profit_factor(trades),
            expectancy=self._expectancy(trades),
            volatility=returns.std() * np.sqrt(252),
            downside_deviation=self._downside_deviation(returns),
            var_95=self._var(returns, 0.95),
            cvar_95=self._cvar(returns, 0.95)
        )

    def _sharpe_ratio(self, returns: pd.Series, risk_free: float = 0.02) -> float:
        excess_returns = returns - risk_free / 252
        if returns.std() == 0:
            return 0
        return excess_returns.mean() / returns.std() * np.sqrt(252)

    def _sortino_ratio(self, returns: pd.Series, risk_free: float = 0.02) -> float:
        excess_returns = returns - risk_free / 252
        downside = returns[returns < 0].std()
        if downside == 0:
            return 0
        return excess_returns.mean() / downside * np.sqrt(252)

    def _max_drawdown(self, equity_curve: pd.Series) -> float:
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return drawdown.min()

    def _profit_factor(self, trades: List[Trade]) -> float:
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        return gross_profit / gross_loss

    def _expectancy(self, trades: List[Trade]) -> float:
        if not trades:
            return 0
        win_rate = self._win_rate(trades)
        avg_win = self._avg_win(trades)
        avg_loss = abs(self._avg_loss(trades))
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
```

---

## Backtesting Methodologies

### Walk-Forward Testing

```python
class WalkForwardOptimizer:
    """
    Walk-forward analysis to detect overfitting.
    """
    def __init__(
        self,
        strategy_class: Type[Strategy],
        param_grid: Dict,
        in_sample_pct: float = 0.70,
        n_splits: int = 5
    ):
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.in_sample_pct = in_sample_pct
        self.n_splits = n_splits

    def run(
        self,
        data: pd.DataFrame,
        start: datetime,
        end: datetime
    ) -> WalkForwardResults:
        """
        Run walk-forward optimization.
        """
        results = []
        total_days = (end - start).days
        window_size = total_days // self.n_splits

        for i in range(self.n_splits):
            # Define windows
            window_start = start + timedelta(days=i * window_size)
            window_end = window_start + timedelta(days=window_size)
            in_sample_end = window_start + timedelta(
                days=int(window_size * self.in_sample_pct)
            )

            # Optimize on in-sample
            best_params = self._optimize(
                data, window_start, in_sample_end
            )

            # Test on out-of-sample
            oos_result = self._test(
                data, in_sample_end, window_end, best_params
            )

            results.append({
                'window': i,
                'best_params': best_params,
                'oos_result': oos_result
            })

        return self._aggregate_results(results)
```

### Monte Carlo Analysis

```python
class MonteCarloSimulator:
    """
    Monte Carlo simulation for robustness testing.
    """
    def run(
        self,
        trades: List[Trade],
        n_simulations: int = 1000,
        initial_capital: float = 100000
    ) -> MonteCarloResults:
        """
        Simulate many possible trade sequences.
        """
        results = []

        for _ in range(n_simulations):
            # Shuffle trade order
            shuffled = random.sample(trades, len(trades))

            # Simulate equity curve
            equity = [initial_capital]
            for trade in shuffled:
                equity.append(equity[-1] + trade.pnl)

            results.append({
                'final_equity': equity[-1],
                'max_drawdown': self._calc_drawdown(equity),
                'sharpe': self._calc_sharpe(equity)
            })

        return MonteCarloResults(
            median_return=np.median([r['final_equity'] for r in results]),
            percentile_5=np.percentile([r['final_equity'] for r in results], 5),
            percentile_95=np.percentile([r['final_equity'] for r in results], 95),
            median_drawdown=np.median([r['max_drawdown'] for r in results]),
            worst_drawdown=min([r['max_drawdown'] for r in results])
        )
```

---

## Data Requirements

### Required Data Fields

```python
REQUIRED_DATA = {
    'equities': {
        'ohlcv': ['open', 'high', 'low', 'close', 'volume'],
        'adjustments': ['split_factor', 'dividend'],
        'metadata': ['sector', 'industry', 'market_cap']
    },
    'options': {
        'chain': ['strike', 'expiry', 'type', 'bid', 'ask', 'volume', 'oi'],
        'greeks': ['delta', 'gamma', 'theta', 'vega', 'iv'],
        'underlying': ['price', 'dividend_yield']
    },
    'fundamentals': {
        'quarterly': ['revenue', 'eps', 'guidance'],
        'annual': ['full financial statements']
    }
}
```

### Data Sources (Priority Order)

1. **Historical Data**:
   - Polygon.io (comprehensive, point-in-time)
   - Alpha Vantage (free tier available)
   - Yahoo Finance (free, but quality issues)
   - Quandl/Nasdaq (fundamentals)

2. **Options Data**:
   - CBOE DataShop
   - OptionMetrics (academic)
   - Polygon.io

---

## Configuration

```yaml
simulation:
  # Execution settings
  execution:
    fill_model: "open"  # open, close, vwap
    slippage_model: "proportional"
    base_slippage_bps: 5
    impact_coefficient: 0.1
    commission_per_share: 0.005

  # Capital settings
  capital:
    initial: 20000
    margin_enabled: false
    margin_rate: 0.0

  # Risk settings
  risk:
    max_position_pct: 0.10
    max_positions: 10
    max_daily_loss: 0.03
    max_drawdown: 0.20

  # Reporting
  output:
    equity_curve: true
    trade_log: true
    daily_summary: true
    performance_report: true
    format: ["json", "csv", "html"]
```

---

## Implementation Roadmap

### Phase 1: Core Engine (Weeks 1-2)
- [ ] Event-driven simulation loop
- [ ] Basic order execution
- [ ] Portfolio tracking
- [ ] Equity curve generation

### Phase 2: Realistic Execution (Week 3)
- [ ] Slippage models
- [ ] Commission models
- [ ] Partial fills
- [ ] Order rejection logic

### Phase 3: Analytics (Week 4)
- [ ] Performance metrics
- [ ] Trade analysis
- [ ] Visualization
- [ ] Report generation

### Phase 4: Advanced Features (Weeks 5-6)
- [ ] Walk-forward optimization
- [ ] Monte Carlo simulation
- [ ] Parameter sensitivity
- [ ] Multi-asset support

### Phase 5: Options Support (Weeks 7-8)
- [ ] Options chain simulation
- [ ] Greeks calculation
- [ ] Options strategy support
- [ ] IV modeling

---

## Technology Stack (Recommended)

```
Language: Python 3.10+

Core:
  - pandas: Data manipulation
  - numpy: Numerical computing
  - numba: JIT compilation for speed

Backtesting:
  - vectorbt: Vectorized backtesting (fast)
  - backtrader: Event-driven (flexible)
  - Custom engine: Maximum control

Visualization:
  - plotly: Interactive charts
  - matplotlib: Static plots

Storage:
  - parquet: Efficient data storage
  - sqlite: Trade logs
  - redis: Caching (optional)

Testing:
  - pytest: Unit tests
  - hypothesis: Property-based testing
```

---

## Quality Assurance

### Required Tests

1. **Unit Tests**: All components
2. **Integration Tests**: Full simulation runs
3. **Smoke Tests**: Known strategies with known results
4. **Benchmark Tests**: Compare to buy-and-hold
5. **Edge Case Tests**: No data, missing data, extreme values

### Validation Criteria

```python
VALIDATION_CRITERIA = {
    'min_trades': 100,  # Statistical significance
    'out_of_sample_required': True,
    'walk_forward_required': True,
    'monte_carlo_passes': True,
    'parameter_stability': True  # Results don't degrade with small param changes
}
```
