"""
Enhanced Dataset Configuration with Market Cap and Performance Characteristics

Includes:
- Small, Mid, Large cap stocks
- Bull market performers (cyclicals)
- Bear market performers (defensives)
- All major sectors
"""

ENHANCED_SYMBOL_UNIVERSE = {
    # LARGE CAP (>$200B market cap)
    "LARGE_CAP_TECH": {
        "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
        "sector": "TECH",
        "market_cap": "LARGE",
        "volatility": 1.3,
        "bull_performer": True,  # Outperform in bull markets
    },
    "LARGE_CAP_FINANCE": {
        "symbols": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
        "sector": "FINANCE",
        "market_cap": "LARGE",
        "volatility": 1.5,
        "bull_performer": True,  # Cyclical
    },
    "LARGE_CAP_HEALTHCARE": {
        "symbols": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO"],
        "sector": "HEALTHCARE",
        "market_cap": "LARGE",
        "volatility": 0.9,
        "bull_performer": False,  # Defensive - bear market performers
    },
    "LARGE_CAP_CONSUMER_DEFENSIVE": {
        "symbols": ["WMT", "PG", "KO", "PEP", "COST"],
        "sector": "CONSUMER",
        "market_cap": "LARGE",
        "volatility": 0.8,
        "bull_performer": False,  # Defensive
    },
    "LARGE_CAP_CONSUMER_CYCLICAL": {
        "symbols": ["AMZN", "HD", "MCD", "NKE", "SBUX"],
        "sector": "CONSUMER",
        "market_cap": "LARGE",
        "volatility": 1.1,
        "bull_performer": True,  # Cyclical
    },
    "LARGE_CAP_ENERGY": {
        "symbols": ["XOM", "CVX", "COP", "SLB", "EOG"],
        "sector": "ENERGY",
        "market_cap": "LARGE",
        "volatility": 1.4,
        "bull_performer": True,  # Cyclical
    },
    "LARGE_CAP_INDUSTRIAL": {
        "symbols": ["BA", "CAT", "UNP", "HON", "GE"],
        "sector": "INDUSTRIAL",
        "market_cap": "LARGE",
        "volatility": 1.2,
        "bull_performer": True,  # Cyclical
    },
    "LARGE_CAP_UTILITIES": {
        "symbols": ["NEE", "DUK", "SO", "D"],
        "sector": "UTILITIES",
        "market_cap": "LARGE",
        "volatility": 0.6,
        "bull_performer": False,  # Defensive
    },
    # MID CAP ($10B - $200B market cap)
    "MID_CAP_TECH": {
        "symbols": ["PLTR", "CRWD", "SNOW", "NET", "DDOG"],
        "sector": "TECH",
        "market_cap": "MID",
        "volatility": 1.8,
        "bull_performer": True,  # High growth
    },
    "MID_CAP_FINANCE": {
        "symbols": ["SCHW", "TFC", "USB", "PNC"],
        "sector": "FINANCE",
        "market_cap": "MID",
        "volatility": 1.3,
        "bull_performer": True,  # Cyclical
    },
    "MID_CAP_HEALTHCARE": {
        "symbols": ["REGN", "VRTX", "IDXX", "DXCM"],
        "sector": "HEALTHCARE",
        "market_cap": "MID",
        "volatility": 1.1,
        "bull_performer": True,  # Growth healthcare
    },
    "MID_CAP_INDUSTRIAL": {
        "symbols": ["ETN", "EMR", "ITW", "PH"],
        "sector": "INDUSTRIAL",
        "market_cap": "MID",
        "volatility": 1.1,
        "bull_performer": True,  # Cyclical
    },
    "MID_CAP_CONSUMER": {
        "symbols": ["DG", "ROST", "ULTA", "YUM"],
        "sector": "CONSUMER",
        "market_cap": "MID",
        "volatility": 1.0,
        "bull_performer": True,  # Growth retail
    },
    "MID_CAP_REAL_ESTATE": {
        "symbols": ["AMT", "PLD", "CCI", "EQIX"],
        "sector": "REAL_ESTATE",
        "market_cap": "MID",
        "volatility": 0.9,
        "bull_performer": False,  # Defensive
    },
    # SMALL CAP ($2B - $10B market cap)
    "SMALL_CAP_TECH": {
        "symbols": ["SMCI", "FTNT", "CYBR", "ZS"],
        "sector": "TECH",
        "market_cap": "SMALL",
        "volatility": 2.2,
        "bull_performer": True,  # High volatility growth
    },
    "SMALL_CAP_FINANCE": {
        "symbols": ["EWBC", "WTFC", "GBCI", "UBSI"],
        "sector": "FINANCE",
        "market_cap": "SMALL",
        "volatility": 1.6,
        "bull_performer": True,  # Cyclical
    },
    "SMALL_CAP_HEALTHCARE": {
        "symbols": ["INCY", "EXAS", "TECH", "IONS"],
        "sector": "HEALTHCARE",
        "market_cap": "SMALL",
        "volatility": 1.5,
        "bull_performer": True,  # Biotech growth
    },
    "SMALL_CAP_INDUSTRIAL": {
        "symbols": ["GNRC", "AOS", "MIDD", "UFPI"],
        "sector": "INDUSTRIAL",
        "market_cap": "SMALL",
        "volatility": 1.4,
        "bull_performer": True,  # Cyclical
    },
    "SMALL_CAP_CONSUMER": {
        "symbols": ["FIVE", "BURL", "OLLI", "BJ"],
        "sector": "CONSUMER",
        "market_cap": "SMALL",
        "volatility": 1.3,
        "bull_performer": True,  # Retail growth
    },
    # ETFs for benchmarking
    "ETFS_MARKET_CAP": {
        "symbols": [
            "SPY",  # Large cap benchmark (S&P 500)
            "QQQ",  # Large cap tech (Nasdaq 100)
            "IWM",  # Small cap (Russell 2000)
            "MDY",  # Mid cap (S&P 400)
            "VTI",  # Total market
        ],
        "sector": "ETF",
        "market_cap": "N/A",
        "volatility": 1.0,
        "bull_performer": None,
    },
    "ETFS_DEFENSIVE": {
        "symbols": [
            "XLU",  # Utilities (defensive)
            "XLP",  # Consumer Staples (defensive)
            "GLD",  # Gold (safe haven)
            "TLT",  # Long-term Treasuries (safe haven)
        ],
        "sector": "ETF",
        "market_cap": "N/A",
        "volatility": 0.8,
        "bull_performer": False,  # Bear market outperformers
    },
    "ETFS_CYCLICAL": {
        "symbols": [
            "XLE",  # Energy (cyclical)
            "XLF",  # Financials (cyclical)
            "XLI",  # Industrials (cyclical)
            "XLY",  # Consumer Discretionary (cyclical)
        ],
        "sector": "ETF",
        "market_cap": "N/A",
        "volatility": 1.3,
        "bull_performer": True,  # Bull market outperformers
    },
}


def get_all_symbols_by_market_cap():
    """Return symbols grouped by market cap."""
    result = {
        "LARGE": [],
        "MID": [],
        "SMALL": [],
        "ETF": [],
    }

    for group_name, group_data in ENHANCED_SYMBOL_UNIVERSE.items():
        market_cap = group_data["market_cap"]
        symbols = group_data["symbols"]

        if market_cap in result:
            result[market_cap].extend(symbols)

    return result


def get_symbols_by_performance_characteristic():
    """Return symbols grouped by bull/bear performance."""
    result = {
        "BULL_PERFORMERS": [],  # Outperform in bull markets (cyclicals)
        "BEAR_PERFORMERS": [],  # Outperform in bear markets (defensives)
        "NEUTRAL": [],  # No clear bias
    }

    for group_name, group_data in ENHANCED_SYMBOL_UNIVERSE.items():
        bull_performer = group_data.get("bull_performer")
        symbols = group_data["symbols"]

        if bull_performer is True:
            result["BULL_PERFORMERS"].extend(symbols)
        elif bull_performer is False:
            result["BEAR_PERFORMERS"].extend(symbols)
        else:
            result["NEUTRAL"].extend(symbols)

    return result


def get_symbols_by_sector():
    """Return symbols grouped by sector."""
    result = {}

    for group_name, group_data in ENHANCED_SYMBOL_UNIVERSE.items():
        sector = group_data["sector"]
        symbols = group_data["symbols"]

        if sector not in result:
            result[sector] = []

        result[sector].extend(symbols)

    return result


def get_all_symbols_flat():
    """Return all symbols as a flat list (deduplicated)."""
    all_symbols = []
    for group_data in ENHANCED_SYMBOL_UNIVERSE.values():
        all_symbols.extend(group_data["symbols"])

    return list(set(all_symbols))  # Remove duplicates


def print_summary():
    """Print summary of enhanced dataset configuration."""
    by_cap = get_all_symbols_by_market_cap()
    by_perf = get_symbols_by_performance_characteristic()
    by_sector = get_symbols_by_sector()

    print("=" * 80)
    print("ENHANCED DATASET CONFIGURATION SUMMARY")
    print("=" * 80)

    print("\nBY MARKET CAP:")
    print(f"  Large Cap: {len(by_cap['LARGE'])} symbols")
    print(f"  Mid Cap: {len(by_cap['MID'])} symbols")
    print(f"  Small Cap: {len(by_cap['SMALL'])} symbols")
    print(f"  ETFs: {len(by_cap['ETF'])} symbols")

    print("\nBY PERFORMANCE CHARACTERISTIC:")
    print(f"  Bull Market Performers (Cyclicals): {len(by_perf['BULL_PERFORMERS'])} symbols")
    print(f"  Bear Market Performers (Defensives): {len(by_perf['BEAR_PERFORMERS'])} symbols")
    print(f"  Neutral: {len(by_perf['NEUTRAL'])} symbols")

    print("\nBY SECTOR:")
    for sector, symbols in sorted(by_sector.items()):
        print(f"  {sector}: {len(symbols)} symbols")

    print(f"\nTOTAL UNIQUE SYMBOLS: {len(get_all_symbols_flat())}")
    print("=" * 80)


if __name__ == "__main__":
    print_summary()

    print("\nSample commands for dataset generation:")
    print("\n# Generate all symbols (large + mid + small cap)")
    print("python scripts/dataset_manager.py --mode historical --years 20 \\")
    print("  --custom-symbols-file scripts/enhanced_dataset_config.py \\")
    print("  --output data --format csv")

    print("\n# Generate only large cap")
    symbols_str = ",".join(get_all_symbols_by_market_cap()["LARGE"][:10])  # First 10 for brevity
    print("python scripts/dataset_manager.py --mode historical --years 20 \\")
    print(f"  --symbols {symbols_str}... \\")
    print("  --output data/historical --format csv")
