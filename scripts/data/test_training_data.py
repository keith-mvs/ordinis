"""Test the training data generator."""

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.training_data_generator import (
    TrainingConfig,
    TrainingDataGenerator,
)


def main():
    print("=" * 70)
    print("TRAINING DATA GENERATOR TEST")
    print("=" * 70)

    config = TrainingConfig(
        symbols=["SPY"],
        chunk_sizes_months=[2, 3, 4, 6],
        lookback_years=[5, 10],
        random_seed=42,
    )

    generator = TrainingDataGenerator(config)

    print("\nGenerating training chunks from historical data...")
    print("(Fetching from Yahoo Finance, may take a moment...)")

    chunks = generator.generate_chunks("SPY", num_chunks=30, balance_regimes=True)

    print(f"\nGenerated {len(chunks)} chunks")
    print("\nRegime Distribution:")
    for regime, count in generator.get_regime_distribution(chunks).items():
        pct = count / len(chunks) * 100 if chunks else 0
        print(f"  {regime.value:<12}: {count:>4} ({pct:.1f}%)")

    print("\nSample Chunks (first 10):")
    print("-" * 80)
    print(f"{'Symbol':<6} {'Regime':<12} {'Start':<12} {'End':<12} {'Months':>6} {'Return':>10}")
    print("-" * 80)

    for chunk in chunks[:10]:
        start = chunk.start_date.strftime("%Y-%m-%d")
        end = chunk.end_date.strftime("%Y-%m-%d")
        ret = chunk.metrics["total_return"]
        print(
            f"{chunk.symbol:<6} {chunk.regime.value:<12} {start:<12} {end:<12} {chunk.duration_months:>6} {ret:>+9.1%}"
        )

    # Summary statistics
    print("\n" + "=" * 70)
    print("CHUNK STATISTICS")
    print("=" * 70)

    returns = [c.metrics["total_return"] for c in chunks]
    volatilities = [c.metrics["volatility"] for c in chunks]
    drawdowns = [c.metrics["max_drawdown"] for c in chunks]

    print(f"\nReturn Range: {min(returns)*100:+.1f}% to {max(returns)*100:+.1f}%")
    print(f"Avg Volatility: {sum(volatilities)/len(volatilities)*100:.1f}%")
    print(f"Worst Drawdown: {min(drawdowns)*100:.1f}%")

    # Duration distribution
    durations = [c.duration_months for c in chunks]
    print("\nDuration Distribution:")
    for d in sorted(set(durations)):
        count = durations.count(d)
        print(f"  {d} months: {count} chunks")

    print("\n[OK] Training data generator working correctly")


if __name__ == "__main__":
    main()
