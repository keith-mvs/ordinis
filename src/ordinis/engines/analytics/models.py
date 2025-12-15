from dataclasses import dataclass, field
from datetime import datetime


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
    """Aggregated performance report."""

    start_time: datetime
    end_time: datetime
    total_trades: int
    total_pnl: float
    win_rate: float
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    metrics: list[Metric] = field(default_factory=list)
