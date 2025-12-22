from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ordinis.engines.proofbench.core.execution import Bar


@runtime_checkable
class StrategyProtocol(Protocol):
    """Protocol for strategy callback functions.

    Strategy callbacks receive the simulation engine, symbol, and current bar,
    and may submit orders via engine.submit_order().

    Example:
        def my_strategy(engine: SimulationEngine, symbol: str, bar: Bar) -> None:
            if bar.close > bar.open:
                engine.submit_order(Order(symbol=symbol, side=OrderSide.BUY, quantity=10))
    """

    def __call__(self, engine: "SimulationEngine", symbol: str, bar: "Bar") -> None:
        """Execute strategy logic for the given bar.

        Args:
            engine: The simulation engine (use engine.submit_order() to place orders)
            symbol: The symbol for this bar
            bar: The current OHLCV bar data
        """
        ...


# Forward reference for type checking
if TYPE_CHECKING:
    from ordinis.engines.proofbench.core.simulator import SimulationEngine


@dataclass
class Metric:
    """Base class for all metrics."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class TradeResult:
    """Represents the result of a trade execution."""

    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    order_id: str
    fees: float = 0.0
    pnl: float | None = None


@dataclass
class PerformanceReport:
    """Container for performance analysis results."""

    timestamp: datetime
    metrics: dict[str, float]
    narrative: str | None = None
    trade_count: int = 0
