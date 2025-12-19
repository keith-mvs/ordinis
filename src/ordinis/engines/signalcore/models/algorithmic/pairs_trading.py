"""
Pairs Trading Model for SignalCore.

Generates trading signals based on cointegration and mean-reversion.
"""

from datetime import datetime

import pandas as pd

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType


class PairsTradingModel(Model):
    """
    Statistical arbitrage model using pairs trading strategy.

    Identifies mean-reversion opportunities in cointegrated pairs.
    """

    def __init__(self, config: ModelConfig = None):
        if config is None:
            config = ModelConfig(
                model_id="pairs_trading_model",
                model_type="algorithmic",
                version="1.0.0",
                parameters={
                    "entry_zscore": 2.0,
                    "exit_zscore": 0.5,
                    "lookback": 60,
                    "hedge_ratio": 1.0,  # Can be calculated dynamically
                },
                enabled=True,
                min_data_points=60,
            )
        super().__init__(config)

    def validate(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate data contains necessary columns."""
        is_valid, msg = super().validate(data)
        if not is_valid:
            return False, msg

        # Check for spread or pair data
        if "spread" not in data.columns and "pair_price" not in data.columns:
            # Will calculate from price if possible
            pass

        return True, ""

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """Generate pairs trading signal."""
        try:
            current_data = data.loc[timestamp]
        except KeyError:
            current_data = data.iloc[-1]

        # Calculate spread (or use provided spread)
        if "spread" in data.columns:
            spread = data["spread"]
        else:
            # Placeholder: use close price deviations
            spread = data["close"] - data["close"].rolling(window=20).mean()

        # Calculate z-score
        lookback = self.config.parameters.get("lookback", 60)
        if len(spread) >= lookback:
            mean = spread.iloc[-lookback:].mean()
            std = spread.iloc[-lookback:].std()
            zscore = (spread.iloc[-1] - mean) / std if std > 0 else 0.0
        else:
            zscore = 0.0

        # Determine direction
        entry_zscore = self.config.parameters.get("entry_zscore", 2.0)
        exit_zscore = self.config.parameters.get("exit_zscore", 0.5)

        if zscore > entry_zscore:
            direction = Direction.SHORT  # Spread too high, short spread
            probability = min(0.95, 0.5 + abs(zscore) / 10)
            score = -min(1.0, abs(zscore) / entry_zscore)
        elif zscore < -entry_zscore:
            direction = Direction.LONG  # Spread too low, long spread
            probability = min(0.95, 0.5 + abs(zscore) / 10)
            score = min(1.0, abs(zscore) / entry_zscore)
        elif abs(zscore) < exit_zscore:
            direction = Direction.NEUTRAL  # Close to mean, exit
            probability = 0.5
            score = 0.0
        else:
            direction = Direction.NEUTRAL  # Hold
            probability = 0.5
            score = 0.0

        return Signal(
            symbol=current_data.get("symbol", "UNKNOWN"),
            timestamp=timestamp,
            signal_type=SignalType.ENTRY if direction != Direction.NEUTRAL else SignalType.HOLD,
            direction=direction,
            probability=probability,
            score=score,
            expected_return=0.0,
            confidence_interval=(0.0, 0.0),
            model_id=self.config.model_id,
            model_version=self.config.version,
            metadata={
                "zscore": zscore,
                "spread_mean": mean if len(spread) >= lookback else 0.0,
                "spread_std": std if len(spread) >= lookback else 0.0,
                "entry_threshold": entry_zscore,
                "exit_threshold": exit_zscore,
            },
        )
