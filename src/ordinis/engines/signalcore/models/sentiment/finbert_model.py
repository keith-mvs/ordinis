"""
FinBERT sentiment analysis model.

Reference: https://github.com/ProsusAI/finBERT
Model: ProsusAI/finbert (Hugging Face)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from ...core.model import Model, ModelConfig
from ...core.signal import Direction, Signal, SignalType

# Optional transformers import with graceful degradation
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""

    text: str
    sentiment: str  # positive, negative, neutral
    score: float  # Confidence score
    positive_score: float
    negative_score: float
    neutral_score: float
    model_name: str
    analysis_timestamp: datetime


class FinBERTSentimentModel(Model):
    """
    FinBERT-based sentiment analysis model.

    Uses ProsusAI/finbert for financial sentiment classification.
    Analyzes news headlines, SEC filings, earnings calls, etc.
    """

    FINBERT_MODEL = "ProsusAI/finbert"

    def __init__(
        self,
        config: ModelConfig | None = None,
        device: int = -1,  # -1 for CPU, 0+ for GPU
        batch_size: int = 8,
        sentiment_threshold: float = 0.6,  # Min confidence for signal
    ):
        """
        Initialize FinBERT sentiment model.

        Args:
            config: Model configuration
            device: Device for inference (-1=CPU, 0+=GPU)
            batch_size: Batch size for inference
            sentiment_threshold: Minimum confidence for generating signal
        """
        if config is None:
            config = ModelConfig(
                model_id="finbert-sentiment",
                model_type="sentiment",
                version="1.0.0",
                parameters={
                    "model": self.FINBERT_MODEL,
                    "device": device,
                    "sentiment_threshold": sentiment_threshold,
                },
                min_data_points=1,  # Can work with single text
                lookback_period=1,
            )
        super().__init__(config)

        self.device = device
        self.batch_size = batch_size
        self.sentiment_threshold = sentiment_threshold
        self._pipeline = None
        self._initialized = False

    def _initialize(self) -> None:
        """Lazy initialization of the model pipeline."""
        if self._initialized or not TRANSFORMERS_AVAILABLE:
            return

        try:
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.FINBERT_MODEL,
                tokenizer=self.FINBERT_MODEL,
                device=self.device,
                return_all_scores=True,
            )
            self._initialized = True
        except Exception:
            self._initialized = False

    def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate trading signal based on sentiment analysis.

        Expects data to contain a 'text' or 'headline' column with text to analyze.
        Or 'news' column with list of news items.

        Args:
            data: DataFrame with text data
            timestamp: Current timestamp

        Returns:
            Signal with sentiment-based direction
        """
        symbol = data.attrs.get("symbol", "UNKNOWN")

        # Extract text from data
        texts = self._extract_texts(data)

        if not texts:
            # No text to analyze - return neutral
            return self._create_neutral_signal(symbol, timestamp, "no_text")

        # Analyze sentiment
        sentiment_result = self.analyze_texts(texts)

        # Determine signal direction based on aggregate sentiment
        if (
            sentiment_result.sentiment == "positive"
            and sentiment_result.score >= self.sentiment_threshold
        ):
            direction = Direction.LONG
            strength = sentiment_result.score
        elif (
            sentiment_result.sentiment == "negative"
            and sentiment_result.score >= self.sentiment_threshold
        ):
            direction = Direction.SHORT
            strength = sentiment_result.score
        else:
            direction = Direction.NEUTRAL
            strength = 0.5

        self._last_update = timestamp

        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=SignalType.ENTRY if direction != Direction.NEUTRAL else SignalType.HOLD,
            direction=direction,
            probability=min(1.0, sentiment_result.score),
            expected_return=0.0,  # Sentiment doesn't predict specific returns
            confidence_interval=(-0.05, 0.05),  # Conservative bounds
            score=strength
            if direction == Direction.LONG
            else (-strength if direction == Direction.SHORT else 0.0),
            model_id=self.config.model_id,
            model_version=self.config.version,
            metadata={
                "sentiment": sentiment_result.sentiment,
                "sentiment_score": sentiment_result.score,
                "positive_score": sentiment_result.positive_score,
                "negative_score": sentiment_result.negative_score,
                "neutral_score": sentiment_result.neutral_score,
                "model": sentiment_result.model_name,
                "texts_analyzed": len(texts),
            },
        )

    def analyze_texts(self, texts: list[str]) -> SentimentResult:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of text strings to analyze

        Returns:
            Aggregated SentimentResult
        """
        if not texts:
            return SentimentResult(
                text="",
                sentiment="neutral",
                score=0.5,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                model_name="no_input",
                analysis_timestamp=datetime.utcnow(),
            )

        if not TRANSFORMERS_AVAILABLE:
            return self._fallback_sentiment(texts)

        self._initialize()

        if not self._initialized or self._pipeline is None:
            return self._fallback_sentiment(texts)

        try:
            # Run inference in batches
            all_results = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                batch_results = self._pipeline(batch)
                all_results.extend(batch_results)

            # Aggregate scores
            positive_scores = []
            negative_scores = []
            neutral_scores = []

            for result in all_results:
                # result is list of dicts with label and score
                for item in result:
                    if item["label"].lower() == "positive":
                        positive_scores.append(item["score"])
                    elif item["label"].lower() == "negative":
                        negative_scores.append(item["score"])
                    else:
                        neutral_scores.append(item["score"])

            # Calculate averages
            avg_positive = sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
            avg_negative = sum(negative_scores) / len(negative_scores) if negative_scores else 0.0
            avg_neutral = sum(neutral_scores) / len(neutral_scores) if neutral_scores else 0.0

            # Determine dominant sentiment
            if avg_positive > avg_negative and avg_positive > avg_neutral:
                sentiment = "positive"
                score = avg_positive
            elif avg_negative > avg_positive and avg_negative > avg_neutral:
                sentiment = "negative"
                score = avg_negative
            else:
                sentiment = "neutral"
                score = avg_neutral

            return SentimentResult(
                text=texts[0] if len(texts) == 1 else f"{len(texts)} texts",
                sentiment=sentiment,
                score=score,
                positive_score=avg_positive,
                negative_score=avg_negative,
                neutral_score=avg_neutral,
                model_name="ProsusAI/finbert",
                analysis_timestamp=datetime.utcnow(),
            )

        except Exception:
            return self._fallback_sentiment(texts)

    def analyze_single(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text string to analyze

        Returns:
            SentimentResult for the text
        """
        return self.analyze_texts([text])

    def _extract_texts(self, data: pd.DataFrame) -> list[str]:
        """Extract text data from DataFrame."""
        texts = []

        # Check for common text column names
        text_columns = ["text", "headline", "title", "news", "content", "body"]

        for col in text_columns:
            if col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    # Handle case where column contains lists
                    for item in col_data:
                        if isinstance(item, list):
                            texts.extend([str(t) for t in item if t])
                        elif isinstance(item, str) and item.strip():
                            texts.append(item.strip())
                break

        return texts

    def _fallback_sentiment(self, texts: list[str]) -> SentimentResult:
        """
        Fallback sentiment analysis using simple lexicon.

        Uses basic positive/negative word counting.
        """
        # Simple financial sentiment lexicon (Loughran-McDonald style)
        positive_words = {
            "gain",
            "profit",
            "growth",
            "increase",
            "rise",
            "up",
            "higher",
            "strong",
            "positive",
            "beat",
            "exceed",
            "outperform",
            "upgrade",
            "bullish",
            "rally",
            "surge",
            "recover",
            "improve",
            "opportunity",
        }
        negative_words = {
            "loss",
            "decline",
            "fall",
            "drop",
            "down",
            "lower",
            "weak",
            "negative",
            "miss",
            "below",
            "underperform",
            "downgrade",
            "bearish",
            "crash",
            "plunge",
            "risk",
            "concern",
            "warning",
        }

        positive_count = 0
        negative_count = 0
        total_words = 0

        for text in texts:
            words = text.lower().split()
            total_words += len(words)
            for word in words:
                if word in positive_words:
                    positive_count += 1
                elif word in negative_words:
                    negative_count += 1

        if total_words == 0:
            return SentimentResult(
                text=texts[0] if texts else "",
                sentiment="neutral",
                score=0.5,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                model_name="lexicon_fallback",
                analysis_timestamp=datetime.utcnow(),
            )

        # Calculate sentiment scores
        pos_score = positive_count / (positive_count + negative_count + 1)
        neg_score = negative_count / (positive_count + negative_count + 1)
        neutral_score = 1 - pos_score - neg_score

        if pos_score > neg_score and pos_score > 0.3:
            sentiment = "positive"
            score = pos_score
        elif neg_score > pos_score and neg_score > 0.3:
            sentiment = "negative"
            score = neg_score
        else:
            sentiment = "neutral"
            score = neutral_score

        return SentimentResult(
            text=texts[0] if len(texts) == 1 else f"{len(texts)} texts",
            sentiment=sentiment,
            score=score,
            positive_score=pos_score,
            negative_score=neg_score,
            neutral_score=neutral_score,
            model_name="lexicon_fallback",
            analysis_timestamp=datetime.utcnow(),
        )

    def _create_neutral_signal(self, symbol: str, timestamp: datetime, reason: str) -> Signal:
        """Create neutral signal when sentiment cannot be determined."""
        return Signal(
            symbol=symbol,
            timestamp=timestamp,
            signal_type=SignalType.HOLD,
            direction=Direction.NEUTRAL,
            probability=0.5,
            expected_return=0.0,
            confidence_interval=(-0.01, 0.01),
            score=0.0,
            model_id=self.config.model_id,
            model_version=self.config.version,
            metadata={
                "sentiment": "neutral",
                "reason": reason,
            },
        )

    def describe(self) -> dict[str, Any]:
        """Get model description."""
        desc = super().describe()
        desc.update(
            {
                "finbert_model": self.FINBERT_MODEL,
                "device": self.device,
                "batch_size": self.batch_size,
                "sentiment_threshold": self.sentiment_threshold,
                "transformers_available": TRANSFORMERS_AVAILABLE,
                "initialized": self._initialized,
            }
        )
        return desc
