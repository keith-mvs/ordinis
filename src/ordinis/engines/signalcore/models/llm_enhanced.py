"""
LLM-Enhanced SignalCore model with Helix integration.

Uses Helix (NVIDIA models) for:
- Signal interpretation and explanation
- Market pattern recognition
- Feature importance analysis
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from ordinis.ai.helix.models import ChatMessage

if TYPE_CHECKING:
    # Avoid runtime import to break circular dependency
    from ordinis.ai.helix.engine import Helix

from ..core.model import Model
from ..core.signal import Signal


class LLMEnhancedModel(Model):
    """
    LLM-enhanced model that wraps base models with AI interpretation.

    Adds natural language explanations and insights to signals using Helix.
    """

    def __init__(
        self,
        base_model: Model,
        helix: "Helix | None" = None,
        llm_enabled: bool = False,
    ):
        """
        Initialize LLM-enhanced model.

        Args:
            base_model: Underlying model to enhance
            helix: Helix engine instance
            llm_enabled: Enable LLM features
        """
        self.base_model = base_model
        self.helix = helix
        self.llm_enabled = llm_enabled

        # Inherit config from base model
        self.config = base_model.config
        self.config.metadata["llm_enhanced"] = llm_enabled

    def validate(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate using base model."""
        return self.base_model.validate(data)

    async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate signal with LLM enhancement.

        Args:
            data: Market data
            timestamp: Signal timestamp

        Returns:
            Enhanced signal with LLM interpretation
        """
        # Generate base signal
        signal = await self.base_model.generate(data, timestamp)

        # Add LLM interpretation if enabled
        if self.llm_enabled and self.helix:
            # Try to add interpretation, but continue without it on error
            enhanced_signal = await self._add_llm_interpretation(signal, data)
            if enhanced_signal is not None:
                signal = enhanced_signal

        return signal

    async def _add_llm_interpretation(self, signal: Signal, data: pd.DataFrame) -> Signal:
        """
        Add LLM-based interpretation to signal.

        Args:
            signal: Base signal
            data: Market data

        Returns:
            Signal with added interpretation
        """
        if not self.helix:
            return signal

        # Prepare market context
        latest_data = data.tail(5)
        market_context = self._format_market_context(latest_data, signal)

        try:
            prompt = f"""Analyze this trading signal and provide a concise interpretation.

Signal Details:
- Type: {signal.signal_type.value}
- Symbol: {signal.symbol}
- Probability: {signal.probability:.1%}
- Expected Return: {signal.expected_return:.2%}
- Model: {signal.model_id}

Recent Market Data:
{market_context}

Provide a 2-3 sentence interpretation focusing on:
1. What the signal suggests
2. Key market conditions supporting it
3. Main risk factor

Keep it concise and actionable."""

            response = await self.helix.generate(
                messages=[ChatMessage(role="user", content=prompt)],
                model="meta/llama-3.3-70b-instruct",
                temperature=0.3,
                max_tokens=512,
            )
            interpretation = response.content

            # Add interpretation to metadata
            signal.metadata["llm_interpretation"] = interpretation
            signal.metadata["llm_model"] = "meta-llama-3.3-70b"
            signal.metadata["interpretation_timestamp"] = datetime.now(tz=UTC).isoformat()

        except Exception:
            # Return original signal if LLM fails
            pass

        return signal

    def _format_market_context(self, data: pd.DataFrame, signal: Signal) -> str:
        """
        Format recent market data for LLM context.

        Args:
            data: Recent market data
            signal: Current signal

        Returns:
            Formatted market context string
        """
        lines = []

        for _, row in data.iterrows():
            close = row.get("close", 0)
            volume = row.get("volume", 0)
            lines.append(f"Close: ${close:.2f}, Volume: {volume:,}")

        # Add signal-specific context if features exist
        if hasattr(signal, "features") and signal.features:
            lines.append(f"\nKey Features: {signal.features}")

        return "\n".join(lines[-3:])  # Last 3 data points

    def describe(self) -> dict[str, Any]:
        """Describe model with LLM enhancement info."""
        base_desc = self.base_model.describe()
        base_desc["llm_enhanced"] = self.llm_enabled
        base_desc["llm_available"] = self.helix is not None
        base_desc["wrapper_type"] = "LLMEnhancedModel"
        return base_desc


class LLMFeatureEngineer:
    """
    LLM-powered feature engineering for SignalCore models.

    Uses Helix (NVIDIA models) to suggest and generate features from market data.
    """

    def __init__(self, helix: "Helix | None" = None):
        """
        Initialize feature engineer.

        Args:
            helix: Helix engine instance
        """
        self.helix = helix

    async def suggest_features(
        self, data: pd.DataFrame, strategy_type: str = "general"
    ) -> list[str]:
        """
        Suggest relevant features for strategy type.

        Args:
            data: Market data
            strategy_type: Strategy type (trend_following, mean_reversion, etc.)

        Returns:
            List of suggested feature names
        """
        if not self.helix:
            return self._get_basic_features()

        try:
            available_columns = list(data.columns)
            prompt = f"""Suggest 5-7 technical indicators or features for a {strategy_type} trading strategy.

Available data columns: {available_columns}

List feature names only, one per line. Focus on features that are:
1. Relevant for {strategy_type} strategies
2. Calculable from OHLCV data
3. Commonly used and proven effective

Format: Just list the feature names (e.g., "RSI_14", "SMA_50", "ATR_20")"""

            response = await self.helix.generate(
                messages=[ChatMessage(role="user", content=prompt)],
                model="meta/llama-3.3-70b-instruct",
                temperature=0.5,
                max_tokens=1024,
            )
            content = response.content

            # Parse features from response
            features = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            return features[:7] if features else self._get_basic_features()

        except Exception:
            return self._get_basic_features()

    def _get_basic_features(self) -> list[str]:
        """Get basic feature set as fallback."""
        return [
            "SMA_20",
            "SMA_50",
            "RSI_14",
            "MACD",
            "Bollinger_Bands",
            "ATR_14",
            "Volume_SMA_20",
        ]

    async def explain_feature(self, feature_name: str) -> str:
        """
        Explain what a feature measures and why it's useful.

        Args:
            feature_name: Name of feature

        Returns:
            Explanation string
        """
        if not self.helix:
            return f"Feature: {feature_name}"

        try:
            prompt = f"""Explain the {feature_name} technical indicator in 2-3 sentences.

Include:
1. What it measures
2. How traders use it
3. Key signal to watch for

Keep it concise and practical."""

            response = await self.helix.generate(
                messages=[ChatMessage(role="user", content=prompt)],
                model="meta/llama-3.3-70b-instruct",
                temperature=0.3,
                max_tokens=512,
            )
            return response.content

        except Exception:
            return f"Feature: {feature_name}"
