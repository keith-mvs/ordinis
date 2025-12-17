"""
SignalCore Training/Validation CLI.

This module provides a CLI for training ML models and validating signal models.

Usage:
    python -m ordinis.engines.signalcore.train --model lstm --data data/historical --epochs 20
    python -m ordinis.engines.signalcore.train --model bb_v1 --days 100
"""

import argparse
import asyncio
from datetime import UTC, datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models import (
    ADXTrendModel,
    ATRBreakoutModel,
    BollingerBandsModel,
    FundamentalValueModel,
    LSTMModel,
    MACDModel,
    RSIMeanReversionModel,
    SentimentMomentumModel,
    StatisticalReversionModel,
    VolumeTrendModel,
)


def generate_mock_data(days: int = 100, symbol: str = "AAPL") -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    dates = pd.date_range(end=datetime.now(UTC), periods=days, tz="UTC")

    # Create a price series with trend and volatility
    t = np.linspace(0, 4 * np.pi, days)
    trend = np.linspace(100, 150, days)
    seasonality = 10 * np.sin(t)
    noise = np.random.normal(0, 5, days)  # Increased noise for breakouts

    prices = trend + seasonality + noise
    volatility = np.random.uniform(1.0, 3.0, days)

    # Fundamental Data (P/E Ratio)
    # Simulate P/E oscillating between 10 and 40
    pe_ratios = 25 + 15 * np.sin(t)

    # Sentiment Data (0.0 to 1.0)
    # Simulate sentiment oscillating
    sentiment = 0.5 + 0.4 * np.cos(t) + np.random.normal(0, 0.1, days)
    sentiment = np.clip(sentiment, 0.0, 1.0)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + volatility,
            "low": prices - volatility,
            "close": prices,
            "volume": np.random.randint(1000, 10000, days),
            "symbol": [symbol] * days,
            "pe_ratio": pe_ratios,
            "sentiment_score": sentiment,
        },
        index=dates,
    )

    return df


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from directory or file."""
    path = Path(data_path)
    if path.is_file():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path, parse_dates=True, index_col=0)
    elif path.is_dir():
        # Load first parquet or csv found
        files = list(path.glob("*.parquet")) + list(path.glob("*.csv"))
        if files:
            f = files[0]
            print(f"[INFO] Loading data from {f}")
            if f.suffix == ".parquet":
                return pd.read_parquet(f)
            return pd.read_csv(f, parse_dates=True, index_col=0)

    raise ValueError(f"No valid data found in {data_path}")


async def run_training(args):
    """Run training for ML models."""
    print(f"[INFO] Training model: {args.model}")

    if args.dry_run:
        print("[INFO] Dry run completed. Configuration valid.")
        return

    # 1. Load Data
    if args.data:
        try:
            data = load_data(args.data)
            print(f"[INFO] Loaded {len(data)} data points")
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            return
    else:
        print("[INFO] Generating mock data for training")
        data = generate_mock_data(args.days)

    # 2. Initialize Model
    config = ModelConfig(
        model_id=args.model,
        model_type="ml",
        parameters={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
    )

    if args.model == "lstm":
        model = LSTMModel(config)
    else:
        print(f"[ERROR] Unknown ML model: {args.model}")
        return

    # 3. Train
    if hasattr(model, "train"):
        model.train(data)

        # Save model
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"[INFO] Saving model to {args.output_dir}")
            import torch

            torch.save(model.model.state_dict(), Path(args.output_dir) / "model.pt")
    else:
        print(f"[ERROR] Model {args.model} does not support training")


async def run_validation(args):
    """Run validation for a specific model."""
    print(f"[INFO] Validating model: {args.model} over {args.days} days")

    # 1. Initialize Model
    config = ModelConfig(model_id=args.model, model_type="technical", min_data_points=30)

    if args.model == "bb_v1":
        model = BollingerBandsModel(config)
    elif args.model == "rsi_v1":
        model = RSIMeanReversionModel(config)
    elif args.model == "macd_v1":
        model = MACDModel(config)
    elif args.model == "atr_v1":
        model = ATRBreakoutModel(config)
    elif args.model == "adx_v1":
        model = ADXTrendModel(config)
    elif args.model == "stat_v1":
        model = StatisticalReversionModel(config)
    elif args.model == "vol_v1":
        model = VolumeTrendModel(config)
    elif args.model == "fund_v1":
        model = FundamentalValueModel(config)
    elif args.model == "sent_v1":
        model = SentimentMomentumModel(config)
    elif args.model == "lstm":
        # For validation, we use the LSTM model in inference mode (untrained if not loaded)
        model = LSTMModel(config)
    else:
        print(f"[ERROR] Unknown model: {args.model}")
        return

    # 2. Generate/Load Data
    if args.data:
        try:
            data = load_data(args.data)
            print(f"[INFO] Loaded {len(data)} data points")
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            return
    else:
        data = generate_mock_data(args.days)
        print(f"[INFO] Generated {len(data)} data points")

    # 3. Run Model
    # We simulate a rolling window execution
    signals = []
    start_idx = model.config.min_data_points

    print("[INFO] Running simulation...")
    for i in range(start_idx, len(data)):
        window = data.iloc[: i + 1]
        current_time = window.index[-1]

        try:
            signal = await model.generate(window, current_time)
            if signal:
                signals.append(signal)
        except Exception as e:
            # print(f"[ERROR] Failed at {current_time}: {e}")
            pass

    # 4. Report Results
    print("\n" + "=" * 40)
    print(f"VALIDATION RESULTS: {args.model}")
    print("=" * 40)
    print(f"Total Signals Generated: {len(signals)}")

    if not signals:
        print("No signals generated.")
        return

    # Distribution
    types = {}
    directions = {}

    for s in signals:
        types[s.signal_type] = types.get(s.signal_type, 0) + 1
        directions[s.direction] = directions.get(s.direction, 0) + 1

    print("\nSignal Types:")
    for k, v in types.items():
        print(f"  {k}: {v}")

    print("\nDirections:")
    for k, v in directions.items():
        print(f"  {k}: {v}")

    # Sample Signals
    print("\nSample Signals (Last 5):")
    for s in signals[-5:]:
        print(
            f"  {s.timestamp.date()} | {s.signal_type.name} | {s.direction.name} | Score: {s.score:.2f}"
        )
        if s.metadata:
            # Truncate metadata for display
            meta_str = str(s.metadata)
            if len(meta_str) > 100:
                meta_str = meta_str[:97] + "..."
            print(f"    Meta: {meta_str}")


def main():
    parser = argparse.ArgumentParser(description="SignalCore Training/Validation Tool")
    parser.add_argument(
        "--model", type=str, default="bb_v1", help="Model ID to validate/train (e.g., bb_v1, lstm)"
    )
    parser.add_argument("--days", type=int, default=100, help="Number of days of mock data")
    parser.add_argument("--data", type=str, help="Path to data directory or file")

    # Training args
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output-dir", type=str, help="Directory to save model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    parser.add_argument("--config", type=str, help="Path to config file")

    args = parser.parse_args()

    if args.model in ["lstm", "gbm", "xgboost", "transformer"]:
        asyncio.run(run_training(args))
    else:
        asyncio.run(run_validation(args))


if __name__ == "__main__":
    main()
