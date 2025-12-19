"""
Training Data Generator for ML Strategy Development.

Generates regime-balanced, multi-timeframe training datasets for robust
strategy development across diverse market conditions.

Key Features:
- Variable chunk sizes: 2, 3, 4, 6, 8, 10, 12 months
- Random start point selection within 5, 10, 15, 20 year windows
- Market regime labeling: Bull, Bear, Sideways, High Volatility
- Balanced sampling across all regime types
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


class MarketRegime(Enum):
    """Market regime classification."""

    BULL = "bull"  # Strong uptrend (>15% annualized)
    BEAR = "bear"  # Strong downtrend (<-15% annualized)
    SIDEWAYS = "sideways"  # Low directional movement (-5% to +5%)
    VOLATILE = "volatile"  # High volatility regime (VIX > 25 equivalent)
    RECOVERY = "recovery"  # Post-crash recovery phase
    CORRECTION = "correction"  # 10-20% decline from peak


@dataclass
class DataChunk:
    """Represents a chunk of training data."""

    data: pd.DataFrame
    regime: MarketRegime
    start_date: datetime
    end_date: datetime
    duration_months: int
    symbol: str
    metrics: dict


@dataclass
class TrainingConfig:
    """Configuration for training data generation."""

    symbols: list[str] = None
    chunk_sizes_months: list[int] = None
    lookback_years: list[int] = None
    min_samples_per_regime: int = 20
    random_seed: int = 42

    def __post_init__(self):
        self.symbols = self.symbols or ["SPY", "QQQ", "IWM", "DIA"]
        self.chunk_sizes_months = self.chunk_sizes_months or [2, 3, 4, 6, 8, 10, 12]
        self.lookback_years = self.lookback_years or [5, 10, 15, 20]


class MarketRegimeClassifier:
    """Classifies market data into regime types."""

    def __init__(
        self,
        bull_threshold: float = 0.15,
        bear_threshold: float = -0.15,
        sideways_band: float = 0.05,
        volatility_threshold: float = 0.25,
        correction_threshold: float = -0.10,
    ):
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.sideways_band = sideways_band
        self.volatility_threshold = volatility_threshold
        self.correction_threshold = correction_threshold

    def classify(self, data: pd.DataFrame) -> MarketRegime:
        """Classify a data chunk into a market regime."""
        if len(data) < 20:
            return MarketRegime.SIDEWAYS

        # Calculate metrics
        returns = data["close"].pct_change().dropna()
        total_return = (data["close"].iloc[-1] / data["close"].iloc[0]) - 1

        # Annualize return based on trading days
        trading_days = len(data)
        annualized_return = total_return * (252 / trading_days)

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)

        # Drawdown from peak
        rolling_max = data["close"].cummax()
        drawdown = (data["close"] / rolling_max - 1).min()

        # Classification logic
        if volatility > self.volatility_threshold:
            return MarketRegime.VOLATILE

        if drawdown < -0.20:
            # Check if recovering
            recent_return = (data["close"].iloc[-1] / data["close"].iloc[-20]) - 1
            if recent_return > 0.05:
                return MarketRegime.RECOVERY
            return MarketRegime.BEAR

        if self.correction_threshold > drawdown > -0.20:
            return MarketRegime.CORRECTION

        if annualized_return > self.bull_threshold:
            return MarketRegime.BULL
        if annualized_return < self.bear_threshold:
            return MarketRegime.BEAR
        return MarketRegime.SIDEWAYS

    def get_metrics(self, data: pd.DataFrame) -> dict:
        """Calculate comprehensive metrics for a data chunk."""
        returns = data["close"].pct_change().dropna()

        total_return = (data["close"].iloc[-1] / data["close"].iloc[0]) - 1
        trading_days = len(data)
        annualized_return = total_return * (252 / trading_days)
        volatility = returns.std() * np.sqrt(252)

        rolling_max = data["close"].cummax()
        max_drawdown = (data["close"] / rolling_max - 1).min()

        # Sharpe approximation (assume 0 risk-free rate)
        sharpe = annualized_return / volatility if volatility > 0 else 0

        # Trend strength (R-squared of linear fit)
        x = np.arange(len(data))
        y = data["close"].values
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation**2

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "trend_strength": r_squared,
            "trading_days": trading_days,
            "start_price": data["close"].iloc[0],
            "end_price": data["close"].iloc[-1],
        }


class HistoricalDataFetcher:
    """Fetches historical data from Yahoo Finance."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path("data/historical_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(
        self, symbol: str, start_date: datetime, end_date: datetime, use_cache: bool = True
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        cache_file = (
            self.cache_dir
            / f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        )

        if use_cache and cache_file.exists():
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Ensure timezone-naive index
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            return data

        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)

        if data.empty:
            raise ValueError(f"No data found for {symbol} between {start_date} and {end_date}")

        # Normalize column names
        data.columns = [c.lower() for c in data.columns]

        # Ensure required columns
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        data = data[required]

        # Remove timezone info from index to avoid datetime conversion issues
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        if use_cache:
            data.to_csv(cache_file)

        return data

    def fetch_full_history(self, symbol: str, years: int = 20) -> pd.DataFrame:
        """Fetch maximum available history for a symbol."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        cache_file = self.cache_dir / f"{symbol}_full_{years}y.csv"

        if cache_file.exists():
            # Check if cache is recent (within 1 day)
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.days < 1:
                data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                # Ensure timezone-naive index
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                return data

        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{years}y")

        data.columns = [c.lower() for c in data.columns]
        data = data[["open", "high", "low", "close", "volume"]]

        # Remove timezone info from index to avoid datetime conversion issues
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data.to_csv(cache_file)
        return data


class TrainingDataGenerator:
    """
    Generates balanced training datasets across market regimes.

    Implements the strategy of:
    - Variable chunk sizes (2, 3, 4, 6, 8, 10, 12 months)
    - Random start points within configurable lookback windows
    - Regime-balanced sampling for robust training
    """

    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self.fetcher = HistoricalDataFetcher()
        self.classifier = MarketRegimeClassifier()
        self.rng = np.random.default_rng(self.config.random_seed)

    def generate_chunks(
        self, symbol: str, num_chunks: int = 100, balance_regimes: bool = True
    ) -> list[DataChunk]:
        """
        Generate training data chunks with diverse market conditions.

        Args:
            symbol: Stock/ETF symbol
            num_chunks: Total number of chunks to generate
            balance_regimes: If True, ensure balanced regime representation

        Returns:
            List of DataChunk objects
        """
        # Fetch full history
        max_years = max(self.config.lookback_years)
        full_data = self.fetcher.fetch_full_history(symbol, years=max_years)

        if len(full_data) < 252:  # At least 1 year of data
            raise ValueError(f"Insufficient data for {symbol}: {len(full_data)} days")

        chunks = []
        regime_counts = dict.fromkeys(MarketRegime, 0)

        attempts = 0
        max_attempts = num_chunks * 10

        while len(chunks) < num_chunks and attempts < max_attempts:
            attempts += 1

            # Random chunk size
            chunk_months = self.rng.choice(self.config.chunk_sizes_months)
            chunk_days = int(chunk_months * 21)  # ~21 trading days per month

            # Random lookback window
            lookback_years = int(self.rng.choice(self.config.lookback_years))

            # Calculate valid date range
            earliest_date = full_data.index[-1] - timedelta(days=int(lookback_years * 365))
            latest_start = full_data.index[-1] - timedelta(days=chunk_days)

            # Filter to valid range
            valid_data = full_data[full_data.index >= earliest_date]
            if len(valid_data) < chunk_days:
                continue

            # Random start index
            max_start_idx = len(valid_data) - chunk_days
            if max_start_idx <= 0:
                continue

            start_idx = self.rng.integers(0, max_start_idx)

            # Extract chunk
            chunk_data = valid_data.iloc[start_idx : start_idx + chunk_days].copy()

            if len(chunk_data) < chunk_days * 0.9:  # Allow 10% tolerance
                continue

            # Classify regime
            regime = self.classifier.classify(chunk_data)

            # Balance check
            if balance_regimes:
                target_per_regime = num_chunks // len(MarketRegime)
                if regime_counts[regime] >= target_per_regime * 1.5:
                    continue  # Skip overrepresented regimes

            # Create chunk
            metrics = self.classifier.get_metrics(chunk_data)

            # Handle timezone-aware timestamps safely
            start_ts = chunk_data.index[0]
            end_ts = chunk_data.index[-1]

            # Remove timezone if present before converting to pydatetime
            if hasattr(start_ts, "tz") and start_ts.tz is not None:
                start_ts = start_ts.tz_localize(None)
            if hasattr(end_ts, "tz") and end_ts.tz is not None:
                end_ts = end_ts.tz_localize(None)

            chunk = DataChunk(
                data=chunk_data,
                regime=regime,
                start_date=start_ts.to_pydatetime(),
                end_date=end_ts.to_pydatetime(),
                duration_months=chunk_months,
                symbol=symbol,
                metrics=metrics,
            )

            chunks.append(chunk)
            regime_counts[regime] += 1

        return chunks

    def generate_multi_symbol_dataset(
        self, chunks_per_symbol: int = 50, balance_regimes: bool = True
    ) -> list[DataChunk]:
        """Generate training data across multiple symbols."""
        all_chunks = []

        for symbol in self.config.symbols:
            try:
                chunks = self.generate_chunks(
                    symbol, num_chunks=chunks_per_symbol, balance_regimes=balance_regimes
                )
                all_chunks.extend(chunks)
                print(f"[OK] {symbol}: {len(chunks)} chunks")
            except Exception as e:
                print(f"[WARN] {symbol}: {e}")

        return all_chunks

    def get_regime_distribution(self, chunks: list[DataChunk]) -> dict:
        """Get distribution of regimes in chunk list."""
        distribution = dict.fromkeys(MarketRegime, 0)
        for chunk in chunks:
            distribution[chunk.regime] += 1
        return distribution

    def export_dataset(
        self, chunks: list[DataChunk], output_dir: Path, format: str = "csv"
    ) -> None:
        """Export chunks to files for training."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata file
        metadata = []

        for i, chunk in enumerate(chunks):
            filename = f"chunk_{i:04d}_{chunk.symbol}_{chunk.regime.value}.{format}"
            filepath = output_dir / filename

            chunk.data.to_csv(filepath)

            metadata.append(
                {
                    "chunk_id": i,
                    "filename": filename,
                    "symbol": chunk.symbol,
                    "regime": chunk.regime.value,
                    "start_date": chunk.start_date.isoformat(),
                    "end_date": chunk.end_date.isoformat(),
                    "duration_months": chunk.duration_months,
                    **chunk.metrics,
                }
            )

        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(output_dir / "metadata.csv", index=False)

        # Summary report
        print(f"\nExported {len(chunks)} chunks to {output_dir}")
        print("\nRegime Distribution:")
        for regime, count in self.get_regime_distribution(chunks).items():
            pct = count / len(chunks) * 100
            print(f"  {regime.value:<12}: {count:>4} ({pct:.1f}%)")


def generate_training_dataset(
    output_dir: str = "data/training",
    symbols: list[str] | None = None,
    chunks_per_symbol: int = 100,
    random_seed: int = 42,
) -> list[DataChunk]:
    """
    Convenience function to generate a complete training dataset.

    Args:
        output_dir: Directory to save training data
        symbols: List of symbols (default: SPY, QQQ, IWM, DIA)
        chunks_per_symbol: Number of chunks per symbol
        random_seed: Random seed for reproducibility

    Returns:
        List of DataChunk objects
    """
    config = TrainingConfig(symbols=symbols, random_seed=random_seed)

    generator = TrainingDataGenerator(config)
    chunks = generator.generate_multi_symbol_dataset(
        chunks_per_symbol=chunks_per_symbol, balance_regimes=True
    )

    generator.export_dataset(chunks, Path(output_dir))

    return chunks


# Historical market regime reference periods
KNOWN_REGIMES = {
    MarketRegime.BULL: [
        ("2009-03-09", "2011-04-29", "Post-GFC Recovery"),
        ("2012-06-01", "2015-05-20", "QE Bull Market"),
        ("2016-11-09", "2018-01-26", "Trump Rally"),
        ("2020-03-23", "2021-12-31", "COVID Recovery"),
        ("2023-01-01", "2024-07-16", "AI Bull Market"),
    ],
    MarketRegime.BEAR: [
        ("2007-10-09", "2009-03-09", "Global Financial Crisis"),
        ("2020-02-19", "2020-03-23", "COVID Crash"),
        ("2022-01-03", "2022-10-12", "2022 Bear Market"),
    ],
    MarketRegime.SIDEWAYS: [
        ("2011-05-01", "2012-05-31", "2011 Consolidation"),
        ("2015-06-01", "2016-06-30", "2015-2016 Range"),
        ("2018-10-01", "2019-06-30", "Late 2018 Chop"),
    ],
    MarketRegime.VOLATILE: [
        ("2008-09-01", "2008-11-30", "Lehman Crisis"),
        ("2010-05-01", "2010-07-31", "Flash Crash Period"),
        ("2011-08-01", "2011-10-31", "Debt Ceiling Crisis"),
        ("2020-02-19", "2020-04-30", "COVID Volatility"),
    ],
    MarketRegime.CORRECTION: [
        ("2010-04-23", "2010-07-02", "2010 Correction"),
        ("2015-08-17", "2015-08-25", "China Deval Correction"),
        ("2018-01-26", "2018-02-08", "Volmageddon"),
        ("2018-09-20", "2018-12-24", "Q4 2018 Correction"),
    ],
}


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("TRAINING DATA GENERATOR")
    print("=" * 70)

    config = TrainingConfig(
        symbols=["SPY"],  # Start with just SPY for demo
        chunk_sizes_months=[2, 3, 4, 6],
        lookback_years=[5, 10],
        random_seed=42,
    )

    generator = TrainingDataGenerator(config)

    print("\nGenerating training chunks...")
    chunks = generator.generate_chunks("SPY", num_chunks=50, balance_regimes=True)

    print(f"\nGenerated {len(chunks)} chunks")
    print("\nRegime Distribution:")
    for regime, count in generator.get_regime_distribution(chunks).items():
        pct = count / len(chunks) * 100 if chunks else 0
        print(f"  {regime.value:<12}: {count:>4} ({pct:.1f}%)")

    print("\nSample Chunks:")
    for chunk in chunks[:5]:
        print(
            f"  {chunk.symbol} | {chunk.regime.value:<10} | "
            f"{chunk.start_date.strftime('%Y-%m-%d')} to {chunk.end_date.strftime('%Y-%m-%d')} | "
            f"{chunk.duration_months}mo | Return: {chunk.metrics['total_return']:+.1%}"
        )
