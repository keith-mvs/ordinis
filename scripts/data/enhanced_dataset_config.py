"""
Enhanced Dataset Configuration V2 - Expanded Sector Coverage

Target: 200+ stocks across all GICS sectors with balanced representation
- 20+ stocks per major sector
- Complete small/mid/large cap coverage per sector
- All 11 GICS sectors included
"""

EXPANDED_SYMBOL_UNIVERSE = {
    # ========== TECHNOLOGY (30 stocks) ==========
    "LARGE_CAP_TECH": {
        "symbols": [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "AVGO",
            "ORCL",
            "CSCO",
            "ACN",
            "ADBE",
            "CRM",
            "INTC",
            "AMD",
        ],
        "sector": "TECH",
        "market_cap": "LARGE",
        "volatility": 1.3,
        "bull_performer": True,
    },
    "MID_CAP_TECH": {
        "symbols": ["PLTR", "CRWD", "SNOW", "NET", "DDOG", "ZS", "PANW", "FTNT", "MDB", "WDAY"],
        "sector": "TECH",
        "market_cap": "MID",
        "volatility": 1.8,
        "bull_performer": True,
    },
    "SMALL_CAP_TECH": {
        "symbols": ["SMCI", "CYBR", "TENB", "S", "GTLB"],
        "sector": "TECH",
        "market_cap": "SMALL",
        "volatility": 2.2,
        "bull_performer": True,
    },
    # ========== FINANCIALS (25 stocks) ==========
    "LARGE_CAP_FINANCE": {
        "symbols": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BX", "SPGI", "BLK", "AXP", "CME"],
        "sector": "FINANCE",
        "market_cap": "LARGE",
        "volatility": 1.5,
        "bull_performer": True,
    },
    "MID_CAP_FINANCE": {
        "symbols": ["SCHW", "TFC", "USB", "PNC", "COF", "AFL", "AIG", "MET"],
        "sector": "FINANCE",
        "market_cap": "MID",
        "volatility": 1.3,
        "bull_performer": True,
    },
    "SMALL_CAP_FINANCE": {
        "symbols": ["EWBC", "WTFC", "GBCI", "UBSI", "WAL", "CATY"],
        "sector": "FINANCE",
        "market_cap": "SMALL",
        "volatility": 1.6,
        "bull_performer": True,
    },
    # ========== HEALTHCARE (25 stocks) ==========
    "LARGE_CAP_HEALTHCARE": {
        "symbols": [
            "UNH",
            "JNJ",
            "LLY",
            "ABBV",
            "MRK",
            "TMO",
            "ABT",
            "DHR",
            "PFE",
            "AMGN",
            "BMY",
            "GILD",
        ],
        "sector": "HEALTHCARE",
        "market_cap": "LARGE",
        "volatility": 0.9,
        "bull_performer": False,  # Defensive
    },
    "MID_CAP_HEALTHCARE": {
        "symbols": ["REGN", "VRTX", "IDXX", "DXCM", "ISRG", "ALGN", "ZTS", "BIIB"],
        "sector": "HEALTHCARE",
        "market_cap": "MID",
        "volatility": 1.1,
        "bull_performer": True,  # Growth healthcare
    },
    "SMALL_CAP_HEALTHCARE": {
        "symbols": ["INCY", "EXAS", "TECH", "IONS", "NBIX"],
        "sector": "HEALTHCARE",
        "market_cap": "SMALL",
        "volatility": 1.5,
        "bull_performer": True,
    },
    # ========== CONSUMER DISCRETIONARY (25 stocks) ==========
    "LARGE_CAP_CONSUMER_CYCLICAL": {
        "symbols": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG"],
        "sector": "CONSUMER_DISCRETIONARY",
        "market_cap": "LARGE",
        "volatility": 1.1,
        "bull_performer": True,
    },
    "MID_CAP_CONSUMER_CYCLICAL": {
        "symbols": ["DG", "ROST", "ULTA", "YUM", "MAR", "HLT", "ORLY", "AZO", "POOL"],
        "sector": "CONSUMER_DISCRETIONARY",
        "market_cap": "MID",
        "volatility": 1.0,
        "bull_performer": True,
    },
    "SMALL_CAP_CONSUMER_CYCLICAL": {
        "symbols": ["FIVE", "BURL", "OLLI", "BJ", "BOOT", "DKS"],
        "sector": "CONSUMER_DISCRETIONARY",
        "market_cap": "SMALL",
        "volatility": 1.3,
        "bull_performer": True,
    },
    # ========== CONSUMER STAPLES (20 stocks) ==========
    "LARGE_CAP_CONSUMER_DEFENSIVE": {
        "symbols": ["WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "CL", "MDLZ", "KMB", "GIS", "K"],
        "sector": "CONSUMER_STAPLES",
        "market_cap": "LARGE",
        "volatility": 0.8,
        "bull_performer": False,  # Defensive
    },
    "MID_CAP_CONSUMER_DEFENSIVE": {
        "symbols": ["STZ", "TSN", "HSY", "CAG", "SJM"],
        "sector": "CONSUMER_STAPLES",
        "market_cap": "MID",
        "volatility": 0.7,
        "bull_performer": False,
    },
    "SMALL_CAP_CONSUMER_DEFENSIVE": {
        "symbols": ["CENTA", "JJSF", "HAIN"],
        "sector": "CONSUMER_STAPLES",
        "market_cap": "SMALL",
        "volatility": 0.9,
        "bull_performer": False,
    },
    # ========== ENERGY (20 stocks) ==========
    "LARGE_CAP_ENERGY": {
        "symbols": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "OXY", "HES"],
        "sector": "ENERGY",
        "market_cap": "LARGE",
        "volatility": 1.4,
        "bull_performer": True,
    },
    "MID_CAP_ENERGY": {
        "symbols": ["DVN", "FANG", "MRO", "APA", "HAL", "BKR"],
        "sector": "ENERGY",
        "market_cap": "MID",
        "volatility": 1.6,
        "bull_performer": True,
    },
    "SMALL_CAP_ENERGY": {
        "symbols": ["CTRA", "OVV", "MTDR", "SM"],
        "sector": "ENERGY",
        "market_cap": "SMALL",
        "volatility": 1.8,
        "bull_performer": True,
    },
    # ========== INDUSTRIALS (25 stocks) ==========
    "LARGE_CAP_INDUSTRIAL": {
        "symbols": ["BA", "CAT", "UNP", "HON", "GE", "UPS", "RTX", "LMT", "DE", "MMM", "FDX"],
        "sector": "INDUSTRIAL",
        "market_cap": "LARGE",
        "volatility": 1.2,
        "bull_performer": True,
    },
    "MID_CAP_INDUSTRIAL": {
        "symbols": ["ETN", "EMR", "ITW", "PH", "JCI", "CARR", "PCAR", "TT", "ROK"],
        "sector": "INDUSTRIAL",
        "market_cap": "MID",
        "volatility": 1.1,
        "bull_performer": True,
    },
    "SMALL_CAP_INDUSTRIAL": {
        "symbols": ["GNRC", "AOS", "MIDD", "UFPI", "WSO"],
        "sector": "INDUSTRIAL",
        "market_cap": "SMALL",
        "volatility": 1.4,
        "bull_performer": True,
    },
    # ========== MATERIALS (20 stocks) ==========
    "LARGE_CAP_MATERIALS": {
        "symbols": ["LIN", "APD", "ECL", "SHW", "DD", "NEM", "FCX", "NUE", "VMC", "MLM"],
        "sector": "MATERIALS",
        "market_cap": "LARGE",
        "volatility": 1.2,
        "bull_performer": True,
    },
    "MID_CAP_MATERIALS": {
        "symbols": ["ALB", "CE", "CF", "MOS", "FMC", "IFF"],
        "sector": "MATERIALS",
        "market_cap": "MID",
        "volatility": 1.4,
        "bull_performer": True,
    },
    "SMALL_CAP_MATERIALS": {
        "symbols": ["SLVM", "HCC", "CENX", "MP"],
        "sector": "MATERIALS",
        "market_cap": "SMALL",
        "volatility": 1.6,
        "bull_performer": True,
    },
    # ========== REAL ESTATE (15 stocks) ==========
    "LARGE_CAP_REAL_ESTATE": {
        "symbols": ["AMT", "PLD", "CCI", "EQIX", "PSA", "WELL", "DLR", "O"],
        "sector": "REAL_ESTATE",
        "market_cap": "LARGE",
        "volatility": 0.9,
        "bull_performer": False,
    },
    "MID_CAP_REAL_ESTATE": {
        "symbols": ["AVB", "EQR", "VTR", "ARE", "INVH"],
        "sector": "REAL_ESTATE",
        "market_cap": "MID",
        "volatility": 1.0,
        "bull_performer": False,
    },
    "SMALL_CAP_REAL_ESTATE": {
        "symbols": ["REXR", "CUBE"],
        "sector": "REAL_ESTATE",
        "market_cap": "SMALL",
        "volatility": 1.1,
        "bull_performer": False,
    },
    # ========== UTILITIES (15 stocks) ==========
    "LARGE_CAP_UTILITIES": {
        "symbols": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL"],
        "sector": "UTILITIES",
        "market_cap": "LARGE",
        "volatility": 0.6,
        "bull_performer": False,
    },
    "MID_CAP_UTILITIES": {
        "symbols": ["ED", "ES", "ETR", "FE", "PPL"],
        "sector": "UTILITIES",
        "market_cap": "MID",
        "volatility": 0.7,
        "bull_performer": False,
    },
    "SMALL_CAP_UTILITIES": {
        "symbols": ["NWE", "AVA"],
        "sector": "UTILITIES",
        "market_cap": "SMALL",
        "volatility": 0.8,
        "bull_performer": False,
    },
    # ========== COMMUNICATION SERVICES (20 stocks) ==========
    "LARGE_CAP_COMMUNICATION": {
        "symbols": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "CHTR", "TMUS"],
        "sector": "COMMUNICATION",
        "market_cap": "LARGE",
        "volatility": 1.1,
        "bull_performer": True,
    },
    "MID_CAP_COMMUNICATION": {
        "symbols": ["EA", "WBD", "MTCH", "PINS", "SNAP", "SPOT"],
        "sector": "COMMUNICATION",
        "market_cap": "MID",
        "volatility": 1.5,
        "bull_performer": True,
    },
    "SMALL_CAP_COMMUNICATION": {
        "symbols": ["LBRDK", "SIRI", "CABO", "MSG", "MSGM"],
        "sector": "COMMUNICATION",
        "market_cap": "SMALL",
        "volatility": 1.3,
        "bull_performer": True,
    },
    # ========== ETFs (20 benchmarks) ==========
    "ETFS_MARKET_CAP": {
        "symbols": ["SPY", "QQQ", "IWM", "MDY", "VTI", "DIA", "VOO"],
        "sector": "ETF",
        "market_cap": "N/A",
        "volatility": 1.0,
        "bull_performer": None,
    },
    "ETFS_SECTOR": {
        "symbols": ["XLE", "XLF", "XLI", "XLY", "XLP", "XLU", "XLV", "XLK", "XLB", "XLRE"],
        "sector": "ETF",
        "market_cap": "N/A",
        "volatility": 1.0,
        "bull_performer": None,
    },
    "ETFS_DEFENSIVE": {
        "symbols": ["GLD", "TLT", "SHY"],
        "sector": "ETF",
        "market_cap": "N/A",
        "volatility": 0.8,
        "bull_performer": False,
    },
}


def get_all_symbols_by_market_cap_v2():
    """Return symbols grouped by market cap."""
    result = {
        "LARGE": [],
        "MID": [],
        "SMALL": [],
        "ETF": [],
    }

    for group_name, group_data in EXPANDED_SYMBOL_UNIVERSE.items():
        market_cap = group_data["market_cap"]
        symbols = group_data["symbols"]

        if market_cap in result:
            result[market_cap].extend(symbols)

    return result


def get_symbols_by_sector_v2():
    """Return symbols grouped by sector."""
    result = {}

    for group_name, group_data in EXPANDED_SYMBOL_UNIVERSE.items():
        sector = group_data["sector"]
        symbols = group_data["symbols"]

        if sector not in result:
            result[sector] = []

        result[sector].extend(symbols)

    return result


def get_all_symbols_flat_v2():
    """Return all symbols as a flat list (deduplicated)."""
    all_symbols = []
    for group_data in EXPANDED_SYMBOL_UNIVERSE.values():
        all_symbols.extend(group_data["symbols"])

    return list(set(all_symbols))


def print_summary_v2():
    """Print summary of expanded configuration."""
    by_cap = get_all_symbols_by_market_cap_v2()
    by_sector = get_symbols_by_sector_v2()

    print("=" * 80)
    print("EXPANDED DATASET CONFIGURATION V2")
    print("=" * 80)

    print("\nBY MARKET CAP:")
    for cap, symbols in sorted(by_cap.items()):
        print(f"  {cap}: {len(symbols)} symbols")

    print("\nBY SECTOR (11 GICS Sectors):")
    for sector, symbols in sorted(by_sector.items()):
        print(f"  {sector}: {len(symbols)} symbols")

    print(f"\nTOTAL UNIQUE SYMBOLS: {len(get_all_symbols_flat_v2())}")
    print("=" * 80)


if __name__ == "__main__":
    print_summary_v2()
