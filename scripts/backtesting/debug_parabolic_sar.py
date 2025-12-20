#!/usr/bin/env python3
"""
Debug Parabolic SAR Signal Generation.

Analyzes why PSAR model isn't generating signals on test data.
"""

from pathlib import Path
import sys

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engines.signalcore import ModelConfig
from engines.signalcore.models import ParabolicSARModel


def main():  # noqa: PLR0915
    """Debug PSAR signal generation."""
    print("\n" + "=" * 80)
    print("PARABOLIC SAR DEBUG ANALYSIS")
    print("=" * 80)

    # Load data
    data_file = Path(__file__).parent.parent / "data" / "real_spy_daily.csv"
    data = pd.read_csv(data_file)

    # Handle date column
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data.set_index("date", inplace=True)

    print(f"\nData loaded: {len(data)} bars")
    print(f"Period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Symbol: {data['symbol'].iloc[0]}")

    # Test different parameter configurations
    configs = [
        {"min_trend_bars": 2, "acceleration": 0.02, "maximum": 0.2},
        {"min_trend_bars": 1, "acceleration": 0.02, "maximum": 0.2},
        {"min_trend_bars": 3, "acceleration": 0.02, "maximum": 0.2},
        {"min_trend_bars": 2, "acceleration": 0.03, "maximum": 0.25},
    ]

    for i, params in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(
            f"Configuration {i}: min_trend_bars={params['min_trend_bars']}, "
            f"acceleration={params['acceleration']}, maximum={params['maximum']}"
        )
        print("=" * 80)

        # Create model
        config = ModelConfig(
            model_id=f"psar-test-{i}",
            model_type="trend",
            version="1.0.0",
            parameters=params,
        )
        model = ParabolicSARModel(config)

        # Test signals at different points
        entry_signals = 0
        exit_signals = 0
        hold_signals = 0
        reversals_detected = 0

        test_points = [100, 200, 300, 400, len(data) - 1]

        for bar_idx in test_points:
            historical = data.iloc[: bar_idx + 1]
            timestamp = historical.index[-1]

            try:
                signal = model.generate(historical, timestamp)

                if signal.signal_type.value == "entry":
                    entry_signals += 1
                    if signal.metadata.get("reversal_detected"):
                        reversals_detected += 1
                        print(
                            f"  Bar {bar_idx}: ENTRY signal (reversal={signal.metadata.get('reversal_detected')}, "
                            f"trend_bars={signal.metadata.get('trend_bars')}, direction={signal.direction.value})"
                        )
                elif signal.signal_type.value == "exit":
                    exit_signals += 1
                    print(f"  Bar {bar_idx}: EXIT signal")
                else:
                    hold_signals += 1

            except Exception as e:
                print(f"  Bar {bar_idx}: ERROR - {e}")

        print("\nSummary:")
        print(f"  Entry signals: {entry_signals}")
        print(f"  Exit signals: {exit_signals}")
        print(f"  Hold signals: {hold_signals}")
        print(f"  Reversals detected: {reversals_detected}")

    # Deep dive on last 100 bars
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS - Last 100 Bars")
    print("=" * 80)

    config = ModelConfig(
        model_id="psar-detail",
        model_type="trend",
        version="1.0.0",
        parameters={"min_trend_bars": 1, "acceleration": 0.02, "maximum": 0.2},
    )
    model = ParabolicSARModel(config)

    for bar_idx in range(len(data) - 100, len(data), 10):
        historical = data.iloc[: bar_idx + 1]
        timestamp = historical.index[-1]

        signal = model.generate(historical, timestamp)

        print(f"\nBar {bar_idx} ({timestamp.date()}):")
        print(f"  Signal: {signal.signal_type.value}")
        print(f"  Direction: {signal.direction.value}")
        print(f"  Score: {signal.score:.3f}")
        print(f"  SAR: {signal.metadata.get('current_sar'):.2f}")
        print(f"  Price: {signal.metadata.get('current_price'):.2f}")
        print(f"  Trend: {signal.metadata.get('trend')}")
        print(f"  Trend bars: {signal.metadata.get('trend_bars')}")
        print(f"  Reversal: {signal.metadata.get('reversal_detected')}")
        print(f"  Reversal bar: {signal.metadata.get('reversal_bar')}")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
