import pandas as pd

from ordinis.interface.cli.__main__ import analyze_market


def test_cli_analyze_runs(tmp_path):
    # Build synthetic OHLCV CSV
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=80, freq="D"),
            "open": 100 + pd.Series(range(80)) * 0.1,
            "high": 101 + pd.Series(range(80)) * 0.1,
            "low": 99 + pd.Series(range(80)) * 0.1,
            "close": 100 + pd.Series(range(80)) * 0.1,
            "volume": 100_000,
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    class Args:
        data = str(csv_path)
        breakout_lookback = 20
        breakout_volume_mult = 1.5
        save_ichimoku = None

    result = analyze_market(Args())
    assert result == 0
