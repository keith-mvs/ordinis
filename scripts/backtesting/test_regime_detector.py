#!/usr/bin/env python
"""
Test the Regime Detector on our 5 stocks to validate it correctly
identifies CRWD as QUIET_CHOPPY and DKNG as a tradeable regime.
"""

import os
import sys

import pandas as pd

# Add paths
sys.path.insert(0, "scripts/data")
sys.path.insert(0, "src")

from download_massive import load_massive_data

from ordinis.engines.signalcore.regime_detector import (
    RegimeDetector,
    regime_filter,
)


def load_all_data(symbol: str, data_dir: str = "data/massive") -> pd.DataFrame:
    """Load and concatenate all data for a symbol."""
    all_data = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".csv.gz"):
            df = load_massive_data(f"{data_dir}/{fname}", symbol)
            if not df.empty:
                all_data.append(df)

    if not all_data:
        raise ValueError(f"No data found for {symbol}")

    return pd.concat(all_data).sort_index()


def main():
    symbols = ["CRWD", "DKNG", "COIN", "AMD", "NET"]
    detector = RegimeDetector()

    print("=" * 70)
    print("🔍 REGIME DETECTOR VALIDATION")
    print("=" * 70)
    print()

    results = []

    for symbol in symbols:
        print(f"Loading {symbol}...")
        df = load_all_data(symbol)

        # Test on 5-minute data
        df_5min = (
            df.resample("5min")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        analysis = detector.analyze(df_5min, symbol, "5min")
        detector.print_analysis(analysis)

        results.append(
            {
                "symbol": symbol,
                "regime": analysis.regime.value,
                "confidence": analysis.confidence,
                "recommendation": analysis.trade_recommendation,
                "period_return": analysis.metrics.period_return,
                "dir_changes": analysis.metrics.direction_change_rate,
                "big_moves": analysis.metrics.big_move_frequency,
                "autocorr": analysis.metrics.autocorrelation,
            }
        )

    # Summary table
    print("\n" + "=" * 70)
    print("📊 REGIME SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Validate expectations
    print("\n" + "=" * 70)
    print("✅ VALIDATION")
    print("=" * 70)

    # Check CRWD is identified as problematic
    crwd = results_df[results_df["symbol"] == "CRWD"].iloc[0]
    if crwd["regime"] in ["quiet_choppy", "choppy"]:
        print("✓ CRWD correctly identified as CHOPPY or QUIET_CHOPPY")
    else:
        print(f"✗ CRWD misclassified as {crwd['regime']}")

    if crwd["recommendation"] in ["AVOID", "CAUTION"]:
        print("✓ CRWD correctly flagged as AVOID/CAUTION")
    else:
        print(f"✗ CRWD recommendation wrong: {crwd['recommendation']}")

    # Check DKNG is identified as tradeable
    dkng = results_df[results_df["symbol"] == "DKNG"].iloc[0]
    if dkng["recommendation"] == "TRADE":
        print("✓ DKNG correctly identified as TRADE-worthy")
    else:
        print(f"✗ DKNG recommendation wrong: {dkng['recommendation']}")

    # Test the quick filter function
    print("\n" + "=" * 70)
    print("🔧 REGIME FILTER TEST")
    print("=" * 70)

    for symbol in symbols:
        df = load_all_data(symbol)
        df_5min = (
            df.resample("5min")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )

        for strategy in ["rsi", "momentum", "trend"]:
            should_trade, reason = regime_filter(df_5min, strategy, symbol)
            status = "✓" if should_trade else "✗"
            print(f"{status} {symbol} + {strategy}: {reason}")

    print("\n✅ Regime detector validation complete!")


if __name__ == "__main__":
    import logging

    logging.disable(logging.INFO)  # Suppress data loading logs
    main()
