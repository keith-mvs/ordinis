"""
News Sentiment Model for SignalCore.

Generates trading signals based on news sentiment analysis using LLMs.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from ordinis.engines.signalcore.core.model import Model, ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType


class NewsSentimentModel(Model):
    """
    News sentiment model that scores assets based on recent news sentiment.

    Uses sentiment scores (if provided in data) or generates placeholder signals.
    In production, this would integrate with Helix for real-time news analysis.
    """

    def __init__(self, config: ModelConfig = None, helix=None):
        if config is None:
            config = ModelConfig(
                model_id="news_sentiment_model",
                model_type="sentiment",
                version="1.0.0",
                parameters={
                    "sentiment_weight": 1.0,
                    "lookback_days": 7,
                    "buy_threshold": 0.6,
                    "sell_threshold": -0.6,
                },
                enabled=True,
                min_data_points=1,
            )
        super().__init__(config)
        self.helix = helix  # Optional Helix integration

    def validate(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate that data contains necessary columns."""
        is_valid, msg = super().validate(data)
        if not is_valid:
            return False, msg

        # Check for sentiment column (optional)
        if "news_sentiment" not in data.columns:
            # Will use placeholder logic
            pass

        return True, ""

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """Generate news sentiment signal."""
        try:
            current_data = data.loc[timestamp]
        except KeyError:
            current_data = data.iloc[-1]

        # Extract sentiment score (if available)
        if "news_sentiment" in data.columns:
            sentiment_score = current_data.get("news_sentiment", 0.0)
        elif len(data) >= 7:
            price_change = (data["close"].iloc[-1] - data["close"].iloc[-7]) / data["close"].iloc[
                -7
            ]
            sentiment_score = np.tanh(price_change * 10)  # Normalize to [-1, 1]
        else:
            sentiment_score = 0.0

        # Determine direction
        buy_threshold = self.config.parameters.get("buy_threshold", 0.6)
        sell_threshold = self.config.parameters.get("sell_threshold", -0.6)

        if sentiment_score >= buy_threshold:
            direction = Direction.LONG
            probability = min(0.95, 0.5 + abs(sentiment_score) / 2)
        elif sentiment_score <= sell_threshold:
            direction = Direction.SHORT
            probability = min(0.95, 0.5 + abs(sentiment_score) / 2)
        else:
            direction = Direction.NEUTRAL
            probability = 0.5

        return Signal(
            symbol=current_data.get("symbol", "UNKNOWN"),
            timestamp=timestamp,
            signal_type=SignalType.ENTRY if direction != Direction.NEUTRAL else SignalType.HOLD,
            direction=direction,
            probability=probability,
            score=sentiment_score,  # Already in [-1, 1]
            expected_return=0.0,
            confidence_interval=(0.0, 0.0),
            model_id=self.config.model_id,
            model_version=self.config.version,
            metadata={
                "raw_sentiment": sentiment_score,
                "threshold_buy": buy_threshold,
                "threshold_sell": sell_threshold,
            },
        )
