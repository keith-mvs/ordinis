"""
Sentiment Momentum Model.

Trades based on aggregated news and social media sentiment scores.
Based on 'News Sentiment Analysis' from Knowledge Base.
"""

from datetime import datetime

import pandas as pd

from ..core.model import Model, ModelConfig
from ..core.signal import Direction, Signal, SignalType


class SentimentMomentumModel(Model):
    """
    Sentiment Momentum trading model.

    Uses aggregated sentiment scores (0.0 to 1.0) to identify market-moving news.

    Parameters:
        sentiment_threshold: Score above this is bullish (default 0.6)
        negative_threshold: Score below this is bearish (default 0.4)
        confidence_threshold: Minimum confidence in sentiment (default 0.8)

    Signals:
        - ENTRY/LONG: Sentiment > sentiment_threshold
        - ENTRY/SHORT: Sentiment < negative_threshold
    """

    def __init__(self, config: ModelConfig):
        """Initialize Sentiment Momentum model."""
        super().__init__(config)

        # Set default parameters
        params = self.config.parameters
        self.sentiment_threshold = params.get("sentiment_threshold", 0.6)
        self.negative_threshold = params.get("negative_threshold", 0.4)
        self.confidence_threshold = params.get("confidence_threshold", 0.8)

        self.config.min_data_points = 1

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        """
        Generate trading signal from Sentiment analysis.

        Args:
            data: Historical OHLCV + Sentiment data
            timestamp: Current timestamp

        Returns:
            Signal with Sentiment prediction
        """
        # Validate data
        is_valid, msg = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data: {msg}")

        if "sentiment_score" not in data.columns:
            return None

        if "symbol" in data:
            symbol_data = data["symbol"]
            symbol = symbol_data.iloc[0] if hasattr(symbol_data, "iloc") else str(symbol_data)
        else:
            symbol = "UNKNOWN"

        current_sentiment = data["sentiment_score"].iloc[-1]
        current_price = data["close"].iloc[-1]

        # Optional: Sentiment Confidence (mocked if not present)
        confidence = 0.9  # Default high confidence for mock

        # Determine signal
        signal_type = SignalType.HOLD
        direction = Direction.NEUTRAL
        score = 0.0
        probability = 0.5
        expected_return = 0.0

        # Logic
        # Bullish News
        if current_sentiment > self.sentiment_threshold and confidence >= self.confidence_threshold:
            signal_type = SignalType.ENTRY
            direction = Direction.LONG

            # Score scales with sentiment intensity (0.6 -> 1.0)
            score = (current_sentiment - self.sentiment_threshold) / (
                1.0 - self.sentiment_threshold
            )
            score = min(score, 1.0)

            probability = 0.6 + (score * 0.2)  # Max 0.8
            expected_return = 0.03  # Short term momentum

        # Bearish News
        elif (
            current_sentiment < self.negative_threshold and confidence >= self.confidence_threshold
        ):
            signal_type = SignalType.ENTRY
            direction = Direction.SHORT

            # Score scales with negative intensity (0.4 -> 0.0)
            score = -(self.negative_threshold - current_sentiment) / self.negative_threshold
            score = max(score, -1.0)

            probability = 0.6 + (abs(score) * 0.2)
            expected_return = -0.03

        return Signal(
            model_id=self.config.model_id,
            signal_type=signal_type,
            direction=direction,
            score=score,
            timestamp=timestamp,
            metadata={
                "sentiment_score": float(current_sentiment),
                "current_price": float(current_price),
            },
            symbol=str(symbol),
            probability=probability,
            expected_return=expected_return,
            confidence_interval=(expected_return - 0.01, expected_return + 0.01),
            model_version="1.0",
        )
