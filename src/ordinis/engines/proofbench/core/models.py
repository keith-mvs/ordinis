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
    """Container for performance analysis results."""

    timestamp: datetime
    metrics: dict[str, float]
    narrative: str | None = None
    trade_count: int = 0
