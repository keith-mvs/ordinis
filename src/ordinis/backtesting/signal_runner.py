"""Signal generation runner for historical backtesting."""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ordinis.engines.signalcore.core.engine import SignalCoreEngine
from ordinis.engines.signalcore.core.signal import Signal, SignalBatch


@dataclass
class SignalRunnerConfig:
    """Configuration for historical signal runner.

    Attributes:
        resampling_freq: How to resample (None=daily, '1H', '4H', etc.)
        models_to_use: Specific models (None=all enabled)
        ensemble_enabled: Enable ensemble consensus
        ensemble_strategy: Ensemble strategy (voting, weighted, etc.)
        cache_signals: Cache signal outputs
    """

    resampling_freq: str | None = None
    models_to_use: list[str] | None = None
    ensemble_enabled: bool = True
    ensemble_strategy: str = "voting"
    cache_signals: bool = True


class HistoricalSignalRunner:
    """Runs signals over historical data for all bars."""

    def __init__(self, engine: SignalCoreEngine, config: SignalRunnerConfig | None = None):
        """Initialize runner.

        Args:
            engine: Initialized SignalCoreEngine
            config: Runner configuration
        """
        self.engine = engine
        self.config = config or SignalRunnerConfig()
        self._signal_cache: dict[tuple[str, datetime], Signal] = {}
        self._batch_cache: dict[datetime, SignalBatch] = {}

    async def generate_signals_for_symbol(
        self,
        symbol: str,
        data: pd.DataFrame,
    ) -> list[Signal]:
        """Generate signals for each bar in historical data.

        Args:
            symbol: Stock ticker
            data: Historical OHLCV data

        Returns:
            List of signals, one per bar (or filtered by resampling_freq)
        """
        signals = []

        # Resample timestamps if needed
        if self.config.resampling_freq:
            timestamps = data.resample(self.config.resampling_freq).first().index
        else:
            timestamps = data.index

        for ts in timestamps:
            # Use data up to this timestamp (lookback)
            lookback_data = data[data.index <= ts]

            if len(lookback_data) < 10:  # Skip early bars
                continue

            # Generate signal using first enabled model
            signal = await self.engine.generate_signal(
                symbol=symbol,
                data=lookback_data,
                model_id=self.config.models_to_use[0] if self.config.models_to_use else None,
                timestamp=ts,
            )

            if signal:
                signals.append(signal)
                if self.config.cache_signals:
                    self._signal_cache[(symbol, ts)] = signal

        return signals

    async def generate_batch_signals(
        self,
        data: dict[str, pd.DataFrame],
    ) -> dict[datetime, SignalBatch]:
        """Generate ensemble signals for multiple symbols across all timestamps.

        Args:
            data: Dict of symbol -> OHLCV data

        Returns:
            Dict of timestamp -> SignalBatch
        """
        batches: dict[datetime, SignalBatch] = {}

        # Find all unique timestamps across all symbols
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df.index.tolist())

        all_timestamps = sorted(all_timestamps)

        for ts in all_timestamps:
            # Collect signals from all symbols for this timestamp
            signals = []

            for symbol, df in data.items():
                # Get data up to this timestamp
                lookback_data = df[df.index <= ts]

                if len(lookback_data) < 10:
                    continue

                signal = await self.engine.generate_signal(
                    symbol=symbol,
                    data=lookback_data,
                    timestamp=ts,
                )

                if signal:
                    signals.append(signal)

            if signals:
                batch = SignalBatch(
                    timestamp=ts,
                    signals=signals,
                    universe=list(data.keys()),
                )

                batches[ts] = batch
                if self.config.cache_signals:
                    self._batch_cache[ts] = batch

        return batches

    def get_cached_signal(self, symbol: str, ts: datetime) -> Signal | None:
        """Get cached signal for symbol at timestamp."""
        return self._signal_cache.get((symbol, ts))

    def get_cached_batch(self, ts: datetime) -> SignalBatch | None:
        """Get cached batch at timestamp."""
        return self._batch_cache.get(ts)

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._signal_cache.clear()
        self._batch_cache.clear()
