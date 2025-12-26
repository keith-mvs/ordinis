#!/usr/bin/env python3
"""
Equity Universe Management for Network Parity Optimization

Manages the construction and validation of the small-cap equity universe
with sector allocation and filtering.

Author: Ordinis Quantitative Research
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from config import SMALL_CAP_SECTORS, EquityUniverseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StockInfo:
    """Information about a stock in the universe."""

    symbol: str
    sector: str
    last_price: float | None = None
    avg_volume: float | None = None
    market_cap: float | None = None
    in_data: bool = False  # Whether data is available

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "sector": self.sector,
            "last_price": self.last_price,
            "avg_volume": self.avg_volume,
            "market_cap": self.market_cap,
            "in_data": self.in_data,
        }


@dataclass
class EquityUniverse:
    """
    Equity universe with sector allocation.

    Manages a collection of stocks organized by sector with
    support for filtering and validation.
    """

    config: EquityUniverseConfig
    stocks: dict[str, StockInfo] = field(default_factory=dict)
    available_symbols: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize stock info from config."""
        for sector, symbols in self.config.sectors.items():
            for symbol in symbols:
                self.stocks[symbol] = StockInfo(symbol=symbol, sector=sector)

    @property
    def symbols(self) -> list[str]:
        """All symbols in universe."""
        return list(self.stocks.keys())

    @property
    def n_stocks(self) -> int:
        """Total number of stocks."""
        return len(self.stocks)

    @property
    def sectors(self) -> list[str]:
        """List of sectors."""
        return list(self.config.sectors.keys())

    @property
    def n_sectors(self) -> int:
        """Number of sectors."""
        return len(self.sectors)

    def get_sector(self, symbol: str) -> str | None:
        """Get sector for a symbol."""
        if symbol in self.stocks:
            return self.stocks[symbol].sector
        return None

    def get_symbols_by_sector(self, sector: str) -> list[str]:
        """Get all symbols in a sector."""
        if sector in self.config.sectors:
            return self.config.sectors[sector]
        return []

    def get_sector_allocation(self) -> dict[str, int]:
        """Get count of stocks per sector."""
        return {sector: len(syms) for sector, syms in self.config.sectors.items()}

    def update_availability(self, available: list[str]) -> None:
        """
        Update which symbols have data available.

        Args:
            available: List of symbols with available data
        """
        self.available_symbols = available
        for symbol in self.stocks:
            self.stocks[symbol].in_data = symbol in available

    def get_available_stocks(self) -> list[StockInfo]:
        """Get stocks that have data available."""
        return [s for s in self.stocks.values() if s.in_data]

    def get_available_symbols(self) -> list[str]:
        """Get symbols that have data available."""
        return [s.symbol for s in self.stocks.values() if s.in_data]

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate universe meets requirements.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check minimum stocks
        n_available = len(self.get_available_symbols())
        if n_available < self.config.min_total_stocks:
            issues.append(
                f"Only {n_available} stocks available, need {self.config.min_total_stocks}"
            )

        # Check sector coverage
        for sector in self.sectors:
            sector_syms = self.get_symbols_by_sector(sector)
            available_in_sector = [s for s in sector_syms if self.stocks[s].in_data]
            if len(available_in_sector) == 0:
                issues.append(f"No data for any stock in sector: {sector}")

        return len(issues) == 0, issues

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Equity Universe: {self.config.market_cap_category}",
            f"  Total Stocks: {self.n_stocks}",
            f"  Available: {len(self.get_available_symbols())}",
            f"  Sectors: {self.n_sectors}",
            "  Allocation:",
        ]
        for sector, count in self.get_sector_allocation().items():
            available = len([
                s for s in self.get_symbols_by_sector(sector)
                if self.stocks[s].in_data
            ])
            lines.append(f"    {sector}: {available}/{count}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": self.config.to_dict(),
            "stocks": {sym: info.to_dict() for sym, info in self.stocks.items()},
            "available_symbols": self.available_symbols,
            "summary": {
                "total": self.n_stocks,
                "available": len(self.get_available_symbols()),
                "sectors": self.n_sectors,
                "allocation": self.get_sector_allocation(),
            },
        }


def create_default_universe() -> EquityUniverse:
    """
    Create the default small-cap universe.

    Returns:
        EquityUniverse with 34 small-cap stocks across 6 sectors
    """
    config = EquityUniverseConfig(
        market_cap_category="small_cap",
        sector_count=6,
        target_stocks_per_sector=5,
        min_total_stocks=30,
        sectors=SMALL_CAP_SECTORS.copy(),
    )
    return EquityUniverse(config=config)


def filter_by_price(
    universe: EquityUniverse,
    max_price: float = 50.0,
    price_data: dict[str, float] | None = None,
) -> list[str]:
    """
    Filter universe by maximum price.

    Args:
        universe: Equity universe
        max_price: Maximum share price
        price_data: Optional dict of symbol -> last price

    Returns:
        List of symbols meeting price criteria
    """
    if price_data is None:
        # Return all symbols if no price data
        return universe.symbols

    filtered = []
    for symbol in universe.symbols:
        if symbol in price_data and price_data[symbol] <= max_price:
            filtered.append(symbol)
            universe.stocks[symbol].last_price = price_data[symbol]

    return filtered


def filter_by_volume(
    universe: EquityUniverse,
    min_volume: float = 500_000,
    volume_data: dict[str, float] | None = None,
) -> list[str]:
    """
    Filter universe by minimum average volume.

    Args:
        universe: Equity universe
        min_volume: Minimum average daily volume
        volume_data: Optional dict of symbol -> average volume

    Returns:
        List of symbols meeting volume criteria
    """
    if volume_data is None:
        # Return all symbols if no volume data
        return universe.symbols

    filtered = []
    for symbol in universe.symbols:
        if symbol in volume_data and volume_data[symbol] >= min_volume:
            filtered.append(symbol)
            universe.stocks[symbol].avg_volume = volume_data[symbol]

    return filtered


if __name__ == "__main__":
    # Test equity universe
    universe = create_default_universe()

    print(universe.summary())
    print()

    # Test sector access
    for sector in universe.sectors:
        symbols = universe.get_symbols_by_sector(sector)
        print(f"{sector}: {symbols}")

    # Test validation (without data)
    is_valid, issues = universe.validate()
    print(f"\nValid: {is_valid}")
    if issues:
        print(f"Issues: {issues}")

    # Simulate some symbols being available
    test_available = ["RIOT", "MARA", "SOFI", "HOOD", "GME", "AMC", "PLUG", "FCEL"]
    universe.update_availability(test_available)

    print("\nAfter availability update:")
    print(universe.summary())

    is_valid, issues = universe.validate()
    print(f"\nValid: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
