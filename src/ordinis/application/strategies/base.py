"""
Base Strategy Class.

Provides a common interface for all trading strategies.
"""

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from ordinis.engines.signalcore.core.signal import Signal


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.

    All strategies must implement:
    - configure(): Set up strategy parameters
    - generate_signal(): Produce trading signals
    - get_description(): Return strategy description
    """

    def __init__(self, name: str, **params):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            **params: Strategy parameters
        """
        self.name = name
        self.params = params
        self.configure()

    @abstractmethod
    def configure(self):
        """Configure strategy parameters. Must be implemented by subclasses."""

    @abstractmethod
    async def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate trading signal from market data.

        Args:
            data: Historical OHLCV data
            timestamp: Current timestamp

        Returns:
            Signal object or None if no signal
        """

    @abstractmethod
    def get_description(self) -> str:
        """
        Get human-readable strategy description.

        Returns:
            Strategy description
        """

    def get_required_bars(self) -> int:
        """
        Get minimum number of bars required for signal generation.

        Returns:
            Minimum bars needed
        """
        return self.params.get("min_bars", 100)

    def validate_data(self, data: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate input data.

        Args:
            data: Market data to validate

        Returns:
            (is_valid, error_message) tuple
        """
        if data is None or data.empty:
            return False, "Data is empty"

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            return False, f"Missing columns: {missing}"

        if len(data) < self.get_required_bars():
            return False, f"Insufficient data: {len(data)} < {self.get_required_bars()}"

        return True, ""

    def __str__(self) -> str:
        """String representation."""
        return f"{self.name} Strategy"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"
