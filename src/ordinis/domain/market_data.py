"""
Market data domain models.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Bar(BaseModel):
    """
    OHLCV bar data for a single time period.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trade_count: int | None = None

    model_config = ConfigDict(frozen=True)  # Market data should be immutable
