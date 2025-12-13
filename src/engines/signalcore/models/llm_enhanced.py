"""
LLM-Enhanced SignalCore model with NVIDIA integration.

Uses NVIDIA models for:
- Signal interpretation and explanation
- Market pattern recognition
- Feature importance analysis
"""

from datetime import datetime
from typing import Any

import pandas as pd

from ..core.model import Model
from ..core.signal import Signal

# Optional NVIDIA integration
try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    ChatNVIDIA = None  # type: ignore[misc, assignment]


class LLMEnhancedModel(Model):
    """
    LLM-enhanced model that wraps base models with AI interpretation.

    Adds natural language explanations and insights to signals using NVIDIA LLMs.
    """

    def __init__(
        self,
        base_model: Model,
        nvidia_api_key: str | None = None,
        llm_enabled: bool = False,
    ):
        """
        Initialize LLM-enhanced model.

        Args:
            base_model: Underlying model to enhance
            nvidia_api_key: NVIDIA API key for LLM access
            llm_enabled: Enable LLM features
        """
        self.base_model = base_model
        self.nvidia_api_key = nvidia_api_key
        self.llm_enabled = llm_enabled
        self._llm_client = None

        # Inherit config from base model
        self.config = base_model.config
        self.config.metadata["llm_enhanced"] = llm_enabled

    def _init_llm(self) -> Any:
        """Initialize NVIDIA LLM client."""
        if not self.nvidia_api_key:
            raise ValueError("NVIDIA API key required for LLM features")

        if not NVIDIA_AVAILABLE:
            raise ImportError("Install: pip install langchain-nvidia-ai-endpoints")

        return ChatNVIDIA(
            model="meta/llama-3.1-70b-instruct",  # Smaller model for speed
            nvidia_api_key=self.nvidia_api_key,
            temperature=0.3,  # Balanced between creativity and consistency
            max_tokens=512,  # Shorter responses for signal interpretation
        )

    def validate(self, data: pd.DataFrame) -> tuple[bool, str]:
        """Validate using base model."""
        return self.base_model.validate(data)

    def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal:
        """
        Generate signal with LLM enhancement.

        Args:
            data: Market data
            timestamp: Signal timestamp

        Returns:
            Enhanced signal with LLM interpretation
        """
        # Generate base signal
        signal = self.base_model.generate(data, timestamp)

        # Add LLM interpretation if enabled
        if self.llm_enabled and self.nvidia_api_key:
            # Try to add interpretation, but continue without it on error
            enhanced_signal = self._add_llm_interpretation(signal, data)
            if enhanced_signal is not None:
                signal = enhanced_signal

        return signal

    def _add_llm_interpretation(self, signal: Signal, data: pd.DataFrame) -> Signal:
        """
        Add LLM-based interpretation to signal.

        Args:
            signal: Base signal
            data: Market data

        Returns:
            Signal with added interpretation
        """
        if self._llm_client is None:
            try:
                self._llm_client = self._init_llm()
            except (ImportError, ValueError):
                return signal

        # Prepare market context
        latest_data = data.tail(5)
        market_context = self._format_market_context(latest_data, signal)

        # Generate interpretation (client is guaranteed to exist here)
        if self._llm_client is None:
            return signal

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

            response = self._llm_client.invoke(prompt)
            interpretation = response.content if hasattr(response, "content") else str(response)

            # Add interpretation to metadata
            signal.metadata["llm_interpretation"] = interpretation
            signal.metadata["llm_model"] = "nvidia-llama-3.1-70b"
            signal.metadata["interpretation_timestamp"] = datetime.utcnow().isoformat()

        except Exception:  # noqa: S110
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
        # Use vectorized string formatting instead of iterrows() for better performance
        lines = [
            f"Close: ${row['close']:.2f}, Volume: {int(row['volume']):,}"
            for row in data[["close", "volume"]].itertuples(index=False, name=None)
        ]

        # Add signal-specific context if features exist
        if hasattr(signal, "features") and signal.features:
            lines.append(f"\nKey Features: {signal.features}")

        return "\n".join(lines[-3:])  # Last 3 data points

    def describe(self) -> dict[str, Any]:
        """Describe model with LLM enhancement info."""
        base_desc = self.base_model.describe()
        base_desc["llm_enhanced"] = self.llm_enabled
        base_desc["llm_available"] = NVIDIA_AVAILABLE
        base_desc["wrapper_type"] = "LLMEnhancedModel"
        return base_desc


class LLMFeatureEngineer:
    """
    LLM-powered feature engineering for SignalCore models.

    Uses NVIDIA models to suggest and generate features from market data.
    """

    def __init__(self, nvidia_api_key: str | None = None):
        """
        Initialize feature engineer.

        Args:
            nvidia_api_key: NVIDIA API key
        """
        self.nvidia_api_key = nvidia_api_key
        self._llm_client = None

    def _init_llm(self) -> Any:
        """Initialize NVIDIA LLM client."""
        if not self.nvidia_api_key:
            raise ValueError("NVIDIA API key required")

        if not NVIDIA_AVAILABLE:
            raise ImportError("Install: pip install langchain-nvidia-ai-endpoints")

        return ChatNVIDIA(
            model="meta/llama-3.1-70b-instruct",
            nvidia_api_key=self.nvidia_api_key,
            temperature=0.5,  # More creative for feature suggestions
            max_completion_tokens=1024,
        )

    def suggest_features(self, data: pd.DataFrame, strategy_type: str = "general") -> list[str]:
        """
        Suggest relevant features for strategy type.

        Args:
            data: Market data
            strategy_type: Strategy type (trend_following, mean_reversion, etc.)

        Returns:
            List of suggested feature names
        """
        if not self.nvidia_api_key:
            # Return basic features
            return self._get_basic_features()

        if self._llm_client is None:
            try:
                self._llm_client = self._init_llm()
            except (ImportError, ValueError):
                return self._get_basic_features()

        # Client should exist now, but check to be safe
        if self._llm_client is None:
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

            response = self._llm_client.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

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

    def explain_feature(self, feature_name: str) -> str:
        """
        Explain what a feature measures and why it's useful.

        Args:
            feature_name: Name of feature

        Returns:
            Explanation string
        """
        if not self.nvidia_api_key:
            return f"Feature: {feature_name}"

        if self._llm_client is None:
            try:
                self._llm_client = self._init_llm()
            except (ImportError, ValueError):
                return f"Feature: {feature_name}"

        # Client should exist now, but check to be safe
        if self._llm_client is None:
            return f"Feature: {feature_name}"

        try:
            prompt = f"""Explain the {feature_name} technical indicator in 2-3 sentences.

Include:
1. What it measures
2. How traders use it
3. Key signal to watch for

Keep it concise and practical."""

            response = self._llm_client.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        except Exception:
            return f"Feature: {feature_name}"
