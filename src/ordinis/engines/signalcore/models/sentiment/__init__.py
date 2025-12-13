"""
Sentiment analysis models using ProsusAI FinBERT.

Reference: https://github.com/ProsusAI/finBERT
"""

from .finbert_model import FinBERTSentimentModel, SentimentResult

__all__ = [
    "FinBERTSentimentModel",
    "SentimentResult",
]
