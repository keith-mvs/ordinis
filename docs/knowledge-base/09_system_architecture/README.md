# System Architecture & Automation - Knowledge Base

## Purpose

This section documents the technical architecture required to build a robust, automated trading system. It covers data pipelines, signal generation, order management, execution, and monitoring.

---

## 1. High-Level Architecture

### 1.1 System Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       INTELLIGENT INVESTOR SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐  │
│  │   DATA     │───▶│  SIGNAL    │───▶│   RISK     │───▶│ EXECUTION  │  │
│  │  LAYER     │    │  ENGINE    │    │  ENGINE    │    │  ENGINE    │  │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘  │
│        │                 │                 │                 │          │
│        ▼                 ▼                 ▼                 ▼          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    PERSISTENCE & LOGGING                          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│        │                                                     │          │
│        ▼                                                     ▼          │
│  ┌────────────┐                                      ┌────────────┐    │
│  │ MONITORING │                                      │  ALERTING  │    │
│  └────────────┘                                      └────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 1.2 Component Responsibilities

| Component | Responsibility | Key Functions |
|-----------|----------------|---------------|
| **Data Layer** | Ingest, store, serve market data | Real-time quotes, historical data, news |
| **Signal Engine** | Generate trading signals | Indicator calculation, pattern detection |
| **Risk Engine** | Enforce risk constraints | Position sizing, limit checks, approvals |
| **Execution Engine** | Order management | Order creation, routing, fill handling |
| **Persistence** | Store system state | Positions, orders, logs |
| **Monitoring** | Track system health | Metrics, dashboards, alerts |

---

## 2. Data Pipeline

### 2.1 Data Flow Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SOURCES   │───▶│  INGESTION  │───▶│  PROCESSING │───▶│   STORAGE   │
│             │    │             │    │             │    │             │
│ • Exchanges │    │ • WebSocket │    │ • Normalize │    │ • Time-series│
│ • Vendors   │    │ • REST API  │    │ • Validate  │    │ • Cache     │
│ • News      │    │ • FTP/SFTP  │    │ • Transform │    │ • Archive   │
│ • Filings   │    │ • Webhooks  │    │ • Enrich    │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

### 2.2 Data Layer Implementation

```python
# data/pipeline.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict
import asyncio

@dataclass
class MarketData:
    """Standardized market data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str

    def validate(self) -> bool:
        """Validate data integrity."""
        return (
            self.open > 0 and
            self.high >= self.open and
            self.high >= self.close and
            self.low <= self.open and
            self.low <= self.close and
            self.volume >= 0
        )


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> MarketData:
        """Get real-time quote."""
        pass

    @abstractmethod
    async def get_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> List[MarketData]:
        """Get historical data."""
        pass


class DataPipeline:
    """Main data pipeline orchestrator."""

    def __init__(self):
        self.providers: Dict[str, DataProvider] = {}
        self.cache = DataCache()
        self.validators = [
            PriceValidator(),
            TimestampValidator(),
            VolumeValidator()
        ]

    async def ingest(self, data: MarketData) -> Optional[MarketData]:
        """Ingest and validate data."""
        # Validate
        for validator in self.validators:
            if not validator.validate(data):
                self.log_validation_error(data, validator)
                return None

        # Cache
        await self.cache.set(data.symbol, data)

        # Persist
        await self.store(data)

        return data
```

---

### 2.3 Data Storage Schema

```python
# data/storage.py

from sqlalchemy import Column, Integer, Float, String, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class OHLCVBar(Base):
    """OHLCV bar storage."""
    __tablename__ = 'ohlcv_bars'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    timeframe = Column(String(10), nullable=False)  # '1m', '5m', '1h', '1d'
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    source = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_symbol_timeframe', 'symbol', 'timeframe'),
    )


class Quote(Base):
    """Real-time quote storage."""
    __tablename__ = 'quotes'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    bid = Column(Float)
    ask = Column(Float)
    bid_size = Column(Integer)
    ask_size = Column(Integer)
    last = Column(Float)
    last_size = Column(Integer)
    volume = Column(Integer)


class Trade(Base):
    """Trade storage for order history."""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # 'BUY', 'SELL'
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    strategy = Column(String(50))
    signal_id = Column(String(50))
```

---

## 3. Signal Engine

### 3.1 Signal Generation Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   DATA      │───▶│ INDICATORS  │───▶│   RULES     │───▶│  SIGNALS    │
│             │    │             │    │             │    │             │
│ • OHLCV     │    │ • MA, RSI   │    │ • Entry     │    │ • BUY/SELL  │
│ • Volume    │    │ • MACD, ATR │    │ • Exit      │    │ • CLOSE     │
│ • News      │    │ • Custom    │    │ • Filters   │    │ • ADJUST    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

### 3.2 Signal Engine Implementation

```python
# signals/engine.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict
import pandas as pd

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    HOLD = "HOLD"

@dataclass
class Signal:
    """Trading signal."""
    type: SignalType
    symbol: str
    timestamp: datetime
    strategy: str
    strength: float  # 0.0 to 1.0
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = None

    def to_dict(self) -> dict:
        return {
            'type': self.type.value,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'strategy': self.strategy,
            'strength': self.strength,
            'price_target': self.price_target,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'metadata': self.metadata
        }


class Strategy(ABC):
    """Abstract base strategy class."""

    name: str
    description: str

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate required indicators."""
        pass

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """Generate trading signal from data."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict:
        """Return strategy parameters."""
        pass


class SignalEngine:
    """Main signal generation engine."""

    def __init__(self):
        self.strategies: Dict[str, Strategy] = {}
        self.active_strategies: List[str] = []
        self.signal_history: List[Signal] = []

    def register_strategy(self, strategy: Strategy) -> None:
        """Register a strategy with the engine."""
        self.strategies[strategy.name] = strategy

    def activate_strategy(self, name: str) -> bool:
        """Activate a registered strategy."""
        if name in self.strategies:
            self.active_strategies.append(name)
            return True
        return False

    async def process(self, symbol: str, data: pd.DataFrame) -> List[Signal]:
        """Process data through all active strategies."""
        signals = []

        for strategy_name in self.active_strategies:
            strategy = self.strategies[strategy_name]

            try:
                # Calculate indicators
                data_with_indicators = strategy.calculate_indicators(data.copy())

                # Generate signal
                signal = strategy.generate_signal(data_with_indicators)

                if signal and signal.type != SignalType.HOLD:
                    signals.append(signal)
                    self.signal_history.append(signal)

            except Exception as e:
                self.log_error(f"Strategy {strategy_name} error: {e}")

        return signals
```

---

### 3.3 Example Strategy Implementation

```python
# strategies/ma_crossover.py

class MACrossoverStrategy(Strategy):
    """Simple moving average crossover strategy."""

    name = "MA_Crossover"
    description = "Buy when fast MA crosses above slow MA"

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data['ma_fast'] = data['close'].rolling(self.fast_period).mean()
        data['ma_slow'] = data['close'].rolling(self.slow_period).mean()
        data['ma_fast_prev'] = data['ma_fast'].shift(1)
        data['ma_slow_prev'] = data['ma_slow'].shift(1)
        return data

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        if len(data) < self.slow_period + 1:
            return None

        current = data.iloc[-1]
        symbol = data['symbol'].iloc[-1] if 'symbol' in data else 'UNKNOWN'

        # Golden cross: fast crosses above slow
        if (current['ma_fast'] > current['ma_slow'] and
            current['ma_fast_prev'] <= current['ma_slow_prev']):
            return Signal(
                type=SignalType.BUY,
                symbol=symbol,
                timestamp=datetime.now(),
                strategy=self.name,
                strength=0.7,
                stop_loss=current['close'] * 0.98,
                take_profit=current['close'] * 1.06
            )

        # Death cross: fast crosses below slow
        if (current['ma_fast'] < current['ma_slow'] and
            current['ma_fast_prev'] >= current['ma_slow_prev']):
            return Signal(
                type=SignalType.SELL,
                symbol=symbol,
                timestamp=datetime.now(),
                strategy=self.name,
                strength=0.7
            )

        return Signal(type=SignalType.HOLD, symbol=symbol,
                     timestamp=datetime.now(), strategy=self.name, strength=0.0)

    def get_parameters(self) -> Dict:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period
        }
```

---

## 4. Risk Engine

### 4.1 Risk Check Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   SIGNAL    │───▶│ PRE-TRADE   │───▶│  POSITION   │───▶│  APPROVED   │
│             │    │   CHECKS    │    │   SIZING    │    │   ORDER     │
│             │    │             │    │             │    │             │
│ • BUY/SELL  │    │ • Limits    │    │ • ATR-based │    │ • Quantity  │
│ • Symbol    │    │ • Balance   │    │ • Kelly     │    │ • Stops     │
│ • Price     │    │ • Risk      │    │ • Fixed %   │    │ • Target    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

### 4.2 Risk Engine Implementation

```python
# risk/engine.py

from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum

class RiskDecision(Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"

@dataclass
class RiskCheckResult:
    """Result of risk check."""
    decision: RiskDecision
    reason: Optional[str] = None
    original_quantity: Optional[int] = None
    approved_quantity: Optional[int] = None
    modifications: List[str] = None


class RiskEngine:
    """Main risk management engine."""

    def __init__(self, config: dict):
        self.config = config
        self.limits = RiskLimits(config)
        self.portfolio = PortfolioState()

    async def check_trade(self, signal: Signal, portfolio: Portfolio) -> RiskCheckResult:
        """
        Run all pre-trade risk checks.
        Returns approval/rejection with reasons.
        """
        checks = [
            self.check_daily_loss_limit,
            self.check_drawdown_limit,
            self.check_position_limit,
            self.check_sector_concentration,
            self.check_correlation,
            self.check_buying_power,
            self.check_kill_switch
        ]

        for check in checks:
            result = await check(signal, portfolio)
            if result.decision == RiskDecision.REJECTED:
                return result

        # All checks passed, calculate position size
        position_size = self.calculate_position_size(signal, portfolio)

        return RiskCheckResult(
            decision=RiskDecision.APPROVED,
            approved_quantity=position_size
        )

    async def check_daily_loss_limit(self, signal: Signal, portfolio: Portfolio) -> RiskCheckResult:
        """Check if daily loss limit is breached."""
        daily_pnl = portfolio.daily_pnl
        daily_limit = portfolio.equity * self.config['daily_loss_limit_pct']

        if daily_pnl <= -daily_limit:
            return RiskCheckResult(
                decision=RiskDecision.REJECTED,
                reason=f"Daily loss limit reached: {daily_pnl:.2f}"
            )

        return RiskCheckResult(decision=RiskDecision.APPROVED)

    async def check_position_limit(self, signal: Signal, portfolio: Portfolio) -> RiskCheckResult:
        """Check maximum position limits."""
        current_positions = len(portfolio.positions)
        max_positions = self.config['max_positions']

        if current_positions >= max_positions:
            return RiskCheckResult(
                decision=RiskDecision.REJECTED,
                reason=f"Max positions ({max_positions}) reached"
            )

        return RiskCheckResult(decision=RiskDecision.APPROVED)

    def calculate_position_size(self, signal: Signal, portfolio: Portfolio) -> int:
        """Calculate position size based on risk parameters."""
        equity = portfolio.equity
        risk_per_trade = self.config['risk_per_trade_pct']
        risk_amount = equity * risk_per_trade

        if signal.stop_loss:
            entry_price = signal.price_target or portfolio.get_current_price(signal.symbol)
            stop_distance = abs(entry_price - signal.stop_loss)

            if stop_distance > 0:
                shares = int(risk_amount / stop_distance)
                return shares

        # Fallback: fixed percentage of equity
        entry_price = portfolio.get_current_price(signal.symbol)
        max_position_value = equity * self.config['max_position_pct']
        shares = int(max_position_value / entry_price)

        return shares
```

---

## 5. Execution Engine

### 5.1 Order Management Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  APPROVED   │───▶│   ORDER     │───▶│   BROKER    │───▶│    FILL     │
│   SIGNAL    │    │  CREATION   │    │    API      │    │  HANDLING   │
│             │    │             │    │             │    │             │
│ • Size      │    │ • Type      │    │ • Submit    │    │ • Update    │
│ • Stops     │    │ • TIF       │    │ • Route     │    │ • Confirm   │
│ • Target    │    │ • Validate  │    │ • Monitor   │    │ • Log       │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

### 5.2 Execution Engine Implementation

```python
# execution/engine.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List
import asyncio

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'DAY'
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    strategy: Optional[str] = None
    signal_id: Optional[str] = None


class BrokerConnector(ABC):
    """Abstract broker connector."""

    @abstractmethod
    async def connect(self) -> bool:
        pass

    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """Submit order, return order ID."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        pass

    @abstractmethod
    async def get_account(self) -> Account:
        pass


class ExecutionEngine:
    """Main execution engine."""

    def __init__(self, broker: BrokerConnector, config: dict):
        self.broker = broker
        self.config = config
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []

    async def execute_signal(self, signal: Signal, quantity: int) -> Optional[Order]:
        """Execute a trading signal."""

        # Create order from signal
        order = self.create_order(signal, quantity)

        # Pre-flight validation
        if not self.validate_order(order):
            self.log_error(f"Order validation failed: {order}")
            return None

        # Submit to broker
        try:
            order_id = await self.broker.submit_order(order)
            order.id = order_id
            order.status = OrderStatus.SUBMITTED
            self.pending_orders[order_id] = order

            # Create stop loss and take profit orders if specified
            await self.create_bracket_orders(order, signal)

            return order

        except Exception as e:
            self.log_error(f"Order submission failed: {e}")
            return None

    def create_order(self, signal: Signal, quantity: int) -> Order:
        """Create order from signal."""
        side = 'BUY' if signal.type == SignalType.BUY else 'SELL'

        # Determine order type
        if signal.price_target:
            order_type = OrderType.LIMIT
            limit_price = signal.price_target
        else:
            order_type = OrderType.MARKET
            limit_price = None

        return Order(
            id='',  # Will be assigned by broker
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            strategy=signal.strategy,
            signal_id=str(id(signal))
        )

    async def create_bracket_orders(self, parent_order: Order, signal: Signal) -> None:
        """Create stop loss and take profit orders."""
        if signal.stop_loss:
            stop_order = Order(
                id='',
                symbol=parent_order.symbol,
                side='SELL' if parent_order.side == 'BUY' else 'BUY',
                quantity=parent_order.quantity,
                order_type=OrderType.STOP,
                stop_price=signal.stop_loss,
                strategy=parent_order.strategy
            )
            # Store as contingent order
            self.contingent_orders[parent_order.id] = [stop_order]

        if signal.take_profit:
            profit_order = Order(
                id='',
                symbol=parent_order.symbol,
                side='SELL' if parent_order.side == 'BUY' else 'BUY',
                quantity=parent_order.quantity,
                order_type=OrderType.LIMIT,
                limit_price=signal.take_profit,
                strategy=parent_order.strategy
            )
            self.contingent_orders[parent_order.id].append(profit_order)

    def validate_order(self, order: Order) -> bool:
        """Pre-flight order validation."""
        # Check symbol is valid
        if not self.is_valid_symbol(order.symbol):
            return False

        # Check quantity is reasonable
        if order.quantity <= 0:
            return False

        # Check price reasonableness
        if order.limit_price:
            current_price = self.get_current_price(order.symbol)
            if abs(order.limit_price - current_price) / current_price > 0.10:
                return False  # Price deviation > 10%

        return True
```

---

## 6. Monitoring & Alerting

### 6.1 Monitoring Architecture

```python
# monitoring/system.py

from dataclasses import dataclass
from typing import Dict, List
from enum import Enum
import asyncio

class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class HealthCheck:
    """System health check result."""
    component: str
    status: str  # 'healthy', 'degraded', 'down'
    latency_ms: float
    message: Optional[str] = None

@dataclass
class Alert:
    """System alert."""
    severity: AlertSeverity
    component: str
    message: str
    timestamp: datetime
    data: Dict = None


class MonitoringSystem:
    """System monitoring and alerting."""

    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alerts: List[Alert] = []
        self.alert_handlers: List[AlertHandler] = []

    async def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        checks = {
            'data_feed': self.check_data_feed,
            'broker_connection': self.check_broker,
            'database': self.check_database,
            'signal_engine': self.check_signal_engine,
            'risk_engine': self.check_risk_engine
        }

        results = {}
        for name, check_func in checks.items():
            try:
                results[name] = await check_func()
            except Exception as e:
                results[name] = HealthCheck(
                    component=name,
                    status='down',
                    latency_ms=0,
                    message=str(e)
                )

        self.health_checks = results
        return results

    async def check_data_feed(self) -> HealthCheck:
        """Check data feed health."""
        start = time.time()
        try:
            quote = await self.data_provider.get_quote('SPY')
            latency = (time.time() - start) * 1000

            if latency > 1000:
                return HealthCheck('data_feed', 'degraded', latency, 'High latency')

            return HealthCheck('data_feed', 'healthy', latency)
        except Exception as e:
            return HealthCheck('data_feed', 'down', 0, str(e))

    async def send_alert(self, alert: Alert) -> None:
        """Send alert through all handlers."""
        self.alerts.append(alert)

        for handler in self.alert_handlers:
            try:
                await handler.send(alert)
            except Exception as e:
                self.log_error(f"Alert handler failed: {e}")

    def register_alert_handler(self, handler: AlertHandler) -> None:
        """Register an alert handler."""
        self.alert_handlers.append(handler)
```

---

### 6.2 Metrics Collection

```python
# monitoring/metrics.py

from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime, timedelta
import statistics

@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    # Counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # P&L
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0

    # Ratios
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    def calculate(self, trades: List[Trade]) -> None:
        """Calculate metrics from trade history."""
        if not trades:
            return

        self.total_trades = len(trades)
        self.winning_trades = sum(1 for t in trades if t.pnl > 0)
        self.losing_trades = sum(1 for t in trades if t.pnl < 0)

        self.total_pnl = sum(t.pnl for t in trades)

        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades

        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl < 0]

        if wins:
            self.avg_win = statistics.mean(wins)
        if losses:
            self.avg_loss = statistics.mean(losses)

        if losses:
            self.profit_factor = sum(wins) / sum(losses) if sum(losses) > 0 else 0

        # Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
        self.expectancy = (self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)


class MetricsCollector:
    """Collect and aggregate system metrics."""

    def __init__(self):
        self.trading_metrics = TradingMetrics()
        self.system_metrics = {}
        self.daily_metrics = {}

    def update_from_trades(self, trades: List[Trade]) -> None:
        """Update metrics from trade history."""
        self.trading_metrics.calculate(trades)

    def record_execution(self, order: Order, fill: Fill) -> None:
        """Record execution metrics."""
        slippage = calculate_slippage(order, fill)

        self.system_metrics.setdefault('executions', []).append({
            'timestamp': fill.timestamp,
            'symbol': order.symbol,
            'slippage_bps': slippage,
            'fill_time_ms': fill.latency_ms
        })

    def get_dashboard_data(self) -> Dict:
        """Get data for monitoring dashboard."""
        return {
            'trading': self.trading_metrics.__dict__,
            'system': self.system_metrics,
            'health': self.get_health_summary()
        }
```

---

## 7. Logging & Audit Trail

### 7.1 Logging Configuration

```python
# utils/logging.py

import logging
import json
from datetime import datetime
from pathlib import Path

# Log levels by category
LOG_LEVELS = {
    'trades': logging.INFO,
    'signals': logging.INFO,
    'risk': logging.WARNING,
    'errors': logging.ERROR,
    'system': logging.DEBUG
}


def setup_logging(log_dir: str = 'logs') -> None:
    """Configure structured logging."""

    Path(log_dir).mkdir(exist_ok=True)

    # Trade log (JSON format for easy parsing)
    trade_handler = logging.FileHandler(f'{log_dir}/trades.jsonl')
    trade_handler.setFormatter(JSONFormatter())

    trade_logger = logging.getLogger('trades')
    trade_logger.addHandler(trade_handler)
    trade_logger.setLevel(logging.INFO)

    # Signal log
    signal_handler = logging.FileHandler(f'{log_dir}/signals.jsonl')
    signal_handler.setFormatter(JSONFormatter())

    signal_logger = logging.getLogger('signals')
    signal_logger.addHandler(signal_handler)
    signal_logger.setLevel(logging.INFO)

    # Error log
    error_handler = logging.FileHandler(f'{log_dir}/errors.log')
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    error_logger = logging.getLogger('errors')
    error_logger.addHandler(error_handler)
    error_logger.setLevel(logging.WARNING)


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }

        if hasattr(record, 'data'):
            log_data['data'] = record.data

        return json.dumps(log_data)


def log_trade(trade: Trade) -> None:
    """Log a completed trade."""
    logger = logging.getLogger('trades')
    logger.info('Trade executed', extra={
        'data': {
            'order_id': trade.order_id,
            'symbol': trade.symbol,
            'side': trade.side,
            'quantity': trade.quantity,
            'price': trade.price,
            'strategy': trade.strategy,
            'signal_id': trade.signal_id
        }
    })


def log_signal(signal: Signal) -> None:
    """Log a generated signal."""
    logger = logging.getLogger('signals')
    logger.info('Signal generated', extra={
        'data': signal.to_dict()
    })
```

---

### 7.2 Audit Trail

```python
# audit/trail.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict
from enum import Enum

class AuditEventType(Enum):
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    RISK_CHECK = "RISK_CHECK"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    STOP_TRIGGERED = "STOP_TRIGGERED"
    LIMIT_BREACH = "LIMIT_BREACH"
    SYSTEM_EVENT = "SYSTEM_EVENT"
    CONFIG_CHANGE = "CONFIG_CHANGE"

@dataclass
class AuditEvent:
    """Immutable audit event."""
    id: str
    timestamp: datetime
    event_type: AuditEventType
    component: str
    description: str
    data: Dict
    user: Optional[str] = None
    correlation_id: Optional[str] = None  # Link related events


class AuditTrail:
    """Audit trail storage and retrieval."""

    def __init__(self, storage: AuditStorage):
        self.storage = storage

    async def log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        await self.storage.store(event)

    async def get_trade_audit(self, order_id: str) -> List[AuditEvent]:
        """Get full audit trail for a trade."""
        return await self.storage.query(
            correlation_id=order_id
        )

    async def get_events(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: List[AuditEventType] = None
    ) -> List[AuditEvent]:
        """Query audit events."""
        return await self.storage.query(
            start_time=start_time,
            end_time=end_time,
            event_types=event_types
        )
```

---

## 8. Kill Switches & Safeguards

### 8.1 Kill Switch Implementation

```python
# safety/kill_switch.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable
from datetime import datetime

class KillSwitchState(Enum):
    ACTIVE = "ACTIVE"  # Trading enabled
    TRIGGERED = "TRIGGERED"  # Trading halted
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"  # Operator intervention

@dataclass
class KillSwitchTrigger:
    """Kill switch trigger condition."""
    name: str
    condition: Callable[[], bool]
    severity: str  # 'warning', 'halt', 'emergency'
    action: str  # 'stop_new_trades', 'close_all', 'cancel_all'


class KillSwitch:
    """Master kill switch for trading system."""

    def __init__(self):
        self.state = KillSwitchState.ACTIVE
        self.triggered_at: Optional[datetime] = None
        self.trigger_reason: Optional[str] = None
        self.triggers: List[KillSwitchTrigger] = []

        self._register_default_triggers()

    def _register_default_triggers(self) -> None:
        """Register default kill switch triggers."""
        self.triggers = [
            KillSwitchTrigger(
                name="daily_loss_limit",
                condition=lambda: self.portfolio.daily_pnl_pct <= -0.03,
                severity="halt",
                action="stop_new_trades"
            ),
            KillSwitchTrigger(
                name="max_drawdown",
                condition=lambda: self.portfolio.drawdown >= 0.20,
                severity="emergency",
                action="close_all"
            ),
            KillSwitchTrigger(
                name="broker_disconnected",
                condition=lambda: not self.broker.is_connected,
                severity="halt",
                action="cancel_all"
            ),
            KillSwitchTrigger(
                name="data_feed_stale",
                condition=lambda: self.data_age_seconds > 60,
                severity="halt",
                action="stop_new_trades"
            ),
            KillSwitchTrigger(
                name="error_threshold",
                condition=lambda: self.error_count > 10,
                severity="halt",
                action="stop_new_trades"
            )
        ]

    async def check(self) -> bool:
        """Check all kill switch triggers."""
        for trigger in self.triggers:
            try:
                if trigger.condition():
                    await self.trigger(trigger)
                    return True
            except Exception as e:
                self.log_error(f"Kill switch check error: {e}")
                # Fail safe: trigger on error
                await self.trigger_emergency("Kill switch check failed")
                return True

        return False

    async def trigger(self, trigger: KillSwitchTrigger) -> None:
        """Trigger kill switch."""
        self.state = KillSwitchState.TRIGGERED
        self.triggered_at = datetime.now()
        self.trigger_reason = trigger.name

        # Execute action
        if trigger.action == "stop_new_trades":
            await self.execution_engine.disable_new_trades()
        elif trigger.action == "cancel_all":
            await self.execution_engine.cancel_all_orders()
        elif trigger.action == "close_all":
            await self.execution_engine.close_all_positions()

        # Send alerts
        await self.alert_system.send_critical_alert(
            f"Kill switch triggered: {trigger.name}"
        )

    async def reset(self, user: str) -> None:
        """Reset kill switch (manual action required)."""
        if self.state == KillSwitchState.TRIGGERED:
            self.log_audit(f"Kill switch reset by {user}")
            self.state = KillSwitchState.ACTIVE
            self.triggered_at = None
            self.trigger_reason = None
```

---

## 9. Configuration Management

### 9.1 Configuration Schema

```python
# config/schema.py

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class TradingConfig:
    """Trading system configuration."""

    # Risk parameters
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.10
    max_positions: int = 10
    daily_loss_limit_pct: float = 0.03
    max_drawdown_pct: float = 0.15

    # Execution
    default_order_type: str = 'LIMIT'
    default_time_in_force: str = 'DAY'
    slippage_buffer_pct: float = 0.001

    # Data
    primary_data_provider: str = 'polygon'
    backup_data_provider: str = 'iex'

    # Broker
    broker: str = 'alpaca'
    paper_trading: bool = True

    # Strategies
    active_strategies: List[str] = None

    # Market hours
    trade_premarket: bool = False
    trade_afterhours: bool = False
    opening_delay_minutes: int = 15
    closing_cutoff_minutes: int = 15


@dataclass
class SystemConfig:
    """System configuration."""

    # Logging
    log_level: str = 'INFO'
    log_dir: str = 'logs'

    # Database
    database_url: str = 'sqlite:///trading.db'

    # API
    api_host: str = '0.0.0.0'
    api_port: int = 8000

    # Monitoring
    health_check_interval: int = 60
    metrics_retention_days: int = 90
```

---

## 10. Deployment Architecture

### 10.1 Component Deployment

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       DEPLOYMENT ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                         CORE SERVICES                               │ │
│  │                                                                     │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │ │
│  │  │   Data   │  │  Signal  │  │   Risk   │  │Execution │           │ │
│  │  │ Service  │  │  Engine  │  │  Engine  │  │  Engine  │           │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                         DATA STORES                                 │ │
│  │                                                                     │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │ │
│  │  │PostgreSQL│  │  Redis   │  │InfluxDB/ │  │   Logs   │           │ │
│  │  │ (State)  │  │ (Cache)  │  │TimescaleDB│ │  (ELK)   │           │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                       EXTERNAL SERVICES                             │ │
│  │                                                                     │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │ │
│  │  │ Broker   │  │  Market  │  │   News   │  │ Alerting │           │ │
│  │  │   API    │  │   Data   │  │   Feed   │  │ (PagerD) │           │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Academic References

1. **Aldridge, I.**: "High-Frequency Trading: A Practical Guide" - System design
2. **Narang, R.K.**: "Inside the Black Box" - Quantitative trading systems
3. **Chan, E.**: "Algorithmic Trading" - Practical system implementation
4. **De Prado, M.L.**: "Advances in Financial Machine Learning" - ML systems

---

## Key Takeaways

1. **Modular design**: Separate concerns for maintainability
2. **Data quality first**: Validate and clean all inputs
3. **Risk engine is non-negotiable**: Never bypass risk checks
4. **Execution quality matters**: Monitor slippage and fills
5. **Monitor everything**: Know system health at all times
6. **Kill switches save accounts**: Have multiple safeguards
7. **Audit trail is essential**: Log all decisions for review
8. **Paper trade first**: Validate before live trading
